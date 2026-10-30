"""Bounded NFS/rsync operations adapted from neuro_tasks/interface/data_sync.py.

Remote filesystem probes run in children so a stale NFS mount cannot hang the UI.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import SyncConfig
from .experiments import ExperimentStore
from .file_access import share_output

NFS_FILESYSTEM_TYPES = frozenset({"nfs", "nfs4"})
FILESYSTEM_TYPE_PROBE_TIMEOUT_S = 5.0
REMOTE_DIRECTORY_PROBE_TIMEOUT_S = 5.0
RSYNC_IO_TIMEOUT_SECONDS = 30
PROCESS_KILL_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)
STALL_TIMEOUT_SECONDS = 60.0


_REMOTE_DIRECTORY_PROBE_SCRIPT = r"""
import json
import os
import sys

path = sys.argv[1]
if not os.path.isdir(path):
    result = {"is_directory": False, "has_entries": False, "error": None}
else:
    try:
        with os.scandir(path) as entries:
            try:
                next(entries)
            except StopIteration:
                has_entries = False
            else:
                has_entries = True
        result = {
            "is_directory": True,
            "has_entries": has_entries,
            "error": None,
        }
        destination = os.path.join(path, "pitracker")
        source = sys.argv[2]
        if os.path.islink(destination):
            raise OSError("pitracker destination must not be a symbolic link")
        if os.path.exists(destination) and not os.path.isdir(destination):
            raise OSError("pitracker destination must be a directory")
        remote = os.path.realpath(destination)
        local = os.path.realpath(source)
        if os.path.commonpath([remote, local]) in (remote, local):
            raise OSError("remote destination overlaps the local data directory")
    except OSError as exc:
        result = {
            "is_directory": True,
            "has_entries": False,
            "error": str(exc),
        }

sys.stdout.write(json.dumps(result))
"""


def run_command_bounded(
    command: Sequence[str],
    *,
    timeout_s: float,
    cwd: Path | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess:
    """Run a captured text command without an unbounded post-timeout wait.

    Unlike ``subprocess.run(timeout=...)`` on POSIX, this function does not
    wait forever after sending SIGKILL if the child is stuck in an
    uninterruptible filesystem syscall.  Such a child is handed to a daemon
    reaper and ``TimeoutExpired`` is raised promptly.
    """
    args = list(command)
    timeout_s = max(0.0, float(timeout_s))
    process = subprocess.Popen(
        args,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        terminate_process(process, process_group=os.name == "posix")
        _close_process_pipes(process)
        raise subprocess.TimeoutExpired(
            args,
            timeout_s,
            output=exc.output,
            stderr=exc.stderr,
        ) from None

    completed = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def mounted_filesystem_type(path: Path) -> str | None:
    """Return the containing filesystem type, preserving probe failure details."""
    try:
        result = run_command_bounded(
            ["findmnt", "--target", str(path), "--noheadings", "--output", "FSTYPE"],
            check=True,
            timeout_s=FILESYSTEM_TYPE_PROBE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"findmnt timed out after {exc.timeout:g} seconds") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"findmnt failed: {detail}") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not run findmnt: {exc}") from exc
    filesystem_type = result.stdout.strip().splitlines()
    return filesystem_type[-1].strip().lower() if filesystem_type else None


@dataclass(frozen=True)
class DirectoryProbeResult:
    """Bounded result of checking a possibly stale remote directory."""

    is_directory: bool
    has_entries: bool
    error: str | None = None


def _reap_process_in_background(process: subprocess.Popen) -> None:
    """Eventually reap an uninterruptible child without blocking its caller."""

    def reap() -> None:
        try:
            process.wait()
        except (OSError, subprocess.SubprocessError):
            pass

    threading.Thread(
        target=reap,
        name="remote-filesystem-process-reaper",
        daemon=True,
    ).start()


def _close_process_pipes(process: subprocess.Popen) -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, stream_name, None)
        if stream is None:
            continue
        try:
            stream.close()
        except OSError:
            pass


def probe_remote_directory(
    path: Path,
    *,
    source: Path,
    timeout_s: float = REMOTE_DIRECTORY_PROBE_TIMEOUT_S,
) -> DirectoryProbeResult:
    """Check directory existence/content without touching the path in this process.

    Remote filesystem calls may remain blocked in the kernel even after their
    process receives SIGKILL.  The probe therefore runs in a child, applies a
    deadline, and leaves any still-uninterruptible child with a daemon reaper
    rather than waiting indefinitely in the launcher/UI process.
    """
    timeout_s = max(0.0, float(timeout_s))
    try:
        completed = run_command_bounded(
            [sys.executable, "-c", _REMOTE_DIRECTORY_PROBE_SCRIPT, str(path), str(source)],
            timeout_s=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return DirectoryProbeResult(
            False,
            False,
            f"remote directory probe timed out after {timeout_s:g} seconds",
        )
    except (OSError, ValueError) as exc:
        return DirectoryProbeResult(False, False, f"could not start probe: {exc}")

    if completed.returncode != 0:
        detail = completed.stderr.strip() or (
            f"probe exited with status {completed.returncode}"
        )
        return DirectoryProbeResult(False, False, detail)

    try:
        payload = json.loads(completed.stdout)
        is_directory = payload["is_directory"]
        has_entries = payload["has_entries"]
        error = payload["error"]
        if not isinstance(is_directory, bool) or not isinstance(has_entries, bool):
            raise TypeError("probe booleans are invalid")
        if error is not None and not isinstance(error, str):
            raise TypeError("probe error is invalid")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return DirectoryProbeResult(False, False, f"invalid probe response: {exc}")

    return DirectoryProbeResult(is_directory, has_entries, error)


def _signal_process(
    process: subprocess.Popen,
    signal_number: int,
    *,
    process_group: bool,
) -> None:
    if process_group and hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal_number)
            return
        except OSError:
            pass
    if signal_number == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


def _kill_remaining_process_group(process: subprocess.Popen) -> None:
    """Best-effort cleanup for descendants left after the group leader exits."""
    if not hasattr(os, "killpg"):
        return
    try:
        os.killpg(process.pid, PROCESS_KILL_SIGNAL)
    except OSError:
        pass


def terminate_process(
    process: subprocess.Popen,
    *,
    timeout_s: float = 0.25,
    process_group: bool = False,
) -> bool:
    """Stop a process with bounded waits; return whether it was reaped."""
    if process.poll() is not None:
        if process_group:
            _signal_process(
                process,
                signal.SIGTERM,
                process_group=True,
            )
            _kill_remaining_process_group(process)
        return True
    timeout_s = max(0.0, float(timeout_s))
    try:
        _signal_process(
            process,
            signal.SIGTERM,
            process_group=process_group,
        )
        process.wait(timeout=timeout_s)
        if process_group:
            _kill_remaining_process_group(process)
        return True
    except OSError:
        if process.poll() is not None:
            if process_group:
                _kill_remaining_process_group(process)
            return True
    except subprocess.TimeoutExpired:
        pass

    try:
        _signal_process(
            process,
            PROCESS_KILL_SIGNAL,
            process_group=process_group,
        )
        process.wait(timeout=timeout_s)
        return True
    except OSError:
        if process.poll() is not None:
            if process_group:
                _kill_remaining_process_group(process)
            return True
    except subprocess.TimeoutExpired:
        pass

    _reap_process_in_background(process)
    return False


class SyncStatus(enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SyncReport:
    """One completed copy attempt, including errors retained for the interface/log."""

    status: SyncStatus
    detail: str
    removed: tuple[Path, ...] = ()
    cleanup_errors: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        if self.cleanup_errors:
            return "Data copied; some local experiments could not be removed"
        return {
            SyncStatus.SUCCESS: "Data sync completed",
            SyncStatus.FAILED: "Data sync failed; local data retained",
            SyncStatus.CANCELLED: "Data sync cancelled; local data retained",
        }[self.status]


class ExperimentDataSync:
    """Copy a fixed set of experiments, preserving all local data on transfer failure."""

    def __init__(self, store: ExperimentStore, config: SyncConfig):
        self.store = store
        self.config = config

    def run(
        self, cancel: threading.Event, progress: Callable[[str], None], *,
        exclude: tuple[Path, ...] = (),
    ) -> SyncReport:
        """Run outside tracking/UI threads; remove experiments only after full success."""
        try:
            report = self._run(cancel, progress, exclude)
        except Exception as exc:  # noqa: BLE001 - retain data and report filesystem/process failures
            report = SyncReport(SyncStatus.FAILED, str(exc))
        self._log(report)
        return report

    def _run(self, cancel, progress, exclude) -> SyncReport:
        """Validate the NFS root, transfer all planned directories, then prune that plan."""
        experiments = tuple(path for path in self.store.sync_candidates() if path not in exclude)
        if not experiments:
            return SyncReport(SyncStatus.SUCCESS, "No local experiments to copy")
        root = Path(self.config.remote_data_url.strip()).expanduser()
        destination = root / "pitracker"
        source = self.store.data_root
        if not self.config.remote_data_url.strip():
            raise RuntimeError("sync.remote_data_url is not configured")
        if source == destination or source in destination.parents or destination in source.parents:
            raise RuntimeError("The remote destination must not overlap the local data directory")
        progress(f"Checking NFS storage at {root}")
        filesystem = mounted_filesystem_type(root)
        if filesystem not in NFS_FILESYSTEM_TYPES:
            raise RuntimeError(f"Data destination is not on mounted NFS: {root} (type={filesystem})")
        directory = probe_remote_directory(root, source=source)
        if directory.error:
            raise RuntimeError(f"Could not read data destination {root}: {directory.error}")
        if not directory.is_directory or not directory.has_entries:
            raise RuntimeError(f"Data destination is unavailable or empty and may not be mounted: {root}")
        if cancel.is_set():
            return SyncReport(SyncStatus.CANCELLED, f"Copy to {destination} cancelled")
        progress(f"Syncing {len(experiments)} experiment(s) to {destination}")
        with tempfile.TemporaryDirectory(prefix="pi-tracker-sync-") as temporary:
            file_list = Path(temporary) / "experiments.txt"
            file_list.write_text("".join(f"{path.relative_to(source)}/\n" for path in experiments))
            command = [
                # A previous remote copy may have matching size/mtime but
                # damaged bytes. Refresh it before deleting the local copy.
                # Unlike a checksum prescan, copying keeps progress observable
                # to the watchdog even for large files on slow NFS mounts.
                "rsync", "--archive", "--recursive", "--ignore-times",
                f"--timeout={RSYNC_IO_TIMEOUT_SECONDS}",
                "--progress", "--no-owner", "--no-group", f"--files-from={file_list}",
                "--", f"{source}/", f"{destination}/",
            ]
            report = self._transfer(command, destination, cancel)
        if report.status is not SyncStatus.SUCCESS:
            return report
        if cancel.is_set():
            return SyncReport(SyncStatus.CANCELLED, f"Copy to {destination} cancelled; local data retained")
        # Preserve inherited calibration and allocated IDs before deleting even one directory.
        self.store.remember_synced(experiments)
        if cancel.is_set():
            return SyncReport(SyncStatus.CANCELLED, f"Copy to {destination} cancelled; local data retained")
        progress("Finalizing sync; removing uploaded local experiments")
        removed = []
        errors = []
        for path in experiments:
            try:
                if path.is_symlink() or path.parent.is_symlink():
                    raise RuntimeError(f"Local experiment path changed during sync: {path}")
                shutil.rmtree(path)
                removed.append(path)
            except (OSError, RuntimeError) as exc:
                errors.append(str(exc))
        detail = f"Copied {len(experiments)} experiment(s) to {destination}; removed {len(removed)} locally."
        if errors:
            detail += "\nSome synced local data could not be removed:\n" + "\n".join(errors)
        return SyncReport(SyncStatus.SUCCESS, detail, tuple(removed), tuple(errors))

    @staticmethod
    def _transfer(command: list[str], destination: Path, cancel: threading.Event) -> SyncReport:
        """Mirror neuro_tasks' progress watchdog and bounded process-group termination."""
        with tempfile.TemporaryFile(mode="w+b") as output, tempfile.TemporaryFile(mode="w+b") as errors:
            try:
                process = subprocess.Popen(
                    command, stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
                    start_new_session=os.name == "posix",
                )
            except OSError as exc:
                return SyncReport(SyncStatus.FAILED, f"Could not start rsync to {destination}: {exc}")
            size, last_progress = 0, time.monotonic()
            failure = None
            try:
                while process.poll() is None:
                    if cancel.wait(0.1):
                        failure = (SyncStatus.CANCELLED, "Copy cancelled; no local data removed")
                        break
                    current = os.fstat(output.fileno()).st_size
                    if current > size:
                        size, last_progress = current, time.monotonic()
                    if time.monotonic() - last_progress >= STALL_TIMEOUT_SECONDS:
                        failure = (SyncStatus.FAILED,
                                   f"rsync produced no progress for {STALL_TIMEOUT_SECONDS:g} seconds")
                        break
            finally:
                if process.poll() is None:
                    terminate_process(process, process_group=os.name == "posix")
            errors.seek(max(0, os.fstat(errors.fileno()).st_size - 65536))
            stderr = errors.read().decode(errors="replace").strip()
            if failure is None and process.returncode != 0:
                failure = (SyncStatus.FAILED, f"rsync exited with status {process.returncode}")
            if failure is not None:
                status, detail = failure
                return SyncReport(status, f"{destination}: {detail}\n{stderr}".strip())
            return SyncReport(SyncStatus.SUCCESS, f"Copied to {destination}")

    def _log(self, report: SyncReport) -> None:
        """Keep diagnostics locally, outside experiment directories that may be removed."""
        message = f"{report.summary}\n{report.detail}"
        logging.getLogger(__name__).log(
            logging.INFO if report.status is SyncStatus.SUCCESS and not report.cleanup_errors
            else logging.WARNING, message,
        )
        try:
            directory = self.store.data_root
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "interface_sync.log"
            if path.is_symlink():
                raise OSError("Refusing a symbolic link for the sync log")
            with path.open("a", encoding="utf-8") as stream:
                stream.write(f"{datetime.now(UTC).isoformat()}\n{message}\n\n")
            share_output(path)
        except (OSError, ValueError):
            logging.getLogger(__name__).exception("Could not save sync diagnostics")
