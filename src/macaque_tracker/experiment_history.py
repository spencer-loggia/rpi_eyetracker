"""Persistent local TSV summaries, updated only at experiment lifecycle boundaries."""

from __future__ import annotations

import csv
import fcntl
import os
import stat
import tempfile
import threading
from pathlib import Path

from .file_access import SHARED_FILE_MODE
from .protocol import validate_run_number, validate_subject_name

HISTORY_FILENAME = "experiment_history.tsv"
HISTORY_COLUMNS = (
    "experiment_name", "subject", "experiment_number", "date_utc",
    "started_utc", "ended_utc", "started_timestamp_ns", "ended_timestamp_ns",
    "runtime_seconds", "block_count", "status", "calibration_file",
    "calibration_source", "git_commit",
)
_LOCK = threading.RLock()


def _check_file(path: Path) -> None:
    """Reject redirected or non-regular logs without touching their targets."""
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError(f"Experiment history must be a regular file, not a symbolic link: {path}")


def read_history(path: Path) -> list[dict[str, str]]:
    """Read a complete snapshot; malformed history must never silently lose numbering."""
    _check_file(path)
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t", strict=True)
            if reader.fieldnames != list(HISTORY_COLUMNS):
                raise ValueError(f"Invalid experiment history header: {path}")
            rows = list(reader)
    except (csv.Error, UnicodeError) as exc:
        raise ValueError(f"Invalid experiment history in {path}: {exc}") from exc
    seen = set()
    for row in rows:
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Incomplete row")
            validate_subject_name(row["subject"])
            name = row["experiment_name"]
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError("Invalid experiment name")
            if row["experiment_number"]:
                validate_run_number(int(row["experiment_number"]))
            if len(row["date_utc"]) != 8 or not row["date_utc"].isascii() or not row["date_utc"].isdigit():
                raise ValueError("Invalid date")
            if int(row["started_timestamp_ns"]) < 0:
                raise ValueError("Invalid start time")
            key = (row["subject"], name)
            if key in seen:
                raise ValueError("Duplicate experiment")
            seen.add(key)
        except ValueError as exc:
            raise ValueError(f"Invalid experiment history row in {path}: {exc}") from exc
    return rows


def update_history(path: Path, updates: list[dict]) -> None:
    """Durably upsert rows without dropping old experiments, across threads/processes.

    Atomic replacement protects readers; the local lock file serializes writers.
    Both the TSV and its directory are flushed before synced data may be removed.
    """
    if not updates:
        return
    with _LOCK:
        lock_path = path.with_name(".experiment_history.lock")
        _check_file(lock_path)
        descriptor = os.open(
            lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, SHARED_FILE_MODE,
        )
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f"Invalid experiment history lock: {lock_path}")
            if stat.S_IMODE(info.st_mode) != SHARED_FILE_MODE:
                os.fchmod(descriptor, SHARED_FILE_MODE)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            rows = {(row["subject"], row["experiment_name"]): row for row in read_history(path)}
            for row in updates:
                rows[(row["subject"], row["experiment_name"])] = row
            _write_history(path, list(rows.values()))
        finally:
            os.close(descriptor)


def _write_history(path: Path, rows: list[dict]) -> None:
    """Replace the log only after its complete contents reach local storage."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=".experiment-history-", delete=False,
        ) as output:
            temporary = Path(output.name)
            writer = csv.DictWriter(output, HISTORY_COLUMNS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            output.flush()
            os.fchmod(output.fileno(), SHARED_FILE_MODE)
            os.fsync(output.fileno())
        _check_file(path)
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
