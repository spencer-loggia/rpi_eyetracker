"""Subject-scoped experiment storage, independent of cameras and the GUI."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .config import AppConfig, RoiLayout
from .experiment_history import HISTORY_FILENAME, read_history, update_history
from .file_access import save_shared_json, share_output
from .protocol import validate_run_number, validate_subject_name
from .system_state import capture_system_state

_EXPERIMENT_NAME = re.compile(r"[0-9]{8}T[0-9]{6}\.[0-9]{9}Z", re.ASCII)
_CALIBRATED_CAMERA_FIELDS = (
    "exposure_us",
    "analogue_gain",
    "brightness",
    "contrast",
    "sharpness",
)


class ExperimentError(ValueError):
    """An experiment cannot be created or updated safely."""


@dataclass(frozen=True)
class Experiment:
    subject: str
    directory: Path
    config_path: Path
    calibration_path: Path
    started_timestamp_ns: int
    number: int = 1

    @property
    def name(self) -> str:
        return self.directory.name


@dataclass(frozen=True)
class Block:
    """One continuous saved run and the exact settings used to acquire it."""

    number: int
    directory: Path

    @property
    def calibration_path(self) -> Path:
        return self.directory / "calibration.json"

    @property
    def config_path(self) -> Path:
        return self.directory / "config.json"


def _experiment_name(subject: str, timestamp_ns: int, number: int) -> str:
    """Name experiments by subject, UTC day, number, and UTC start time."""
    stamp = _timestamp_name(timestamp_ns)
    return f"exp_{subject}_{stamp[:8]}_{number}_{stamp[9:15]}"


def _timestamp_name(timestamp_ns: int) -> str:
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
        raise ExperimentError(
            "Experiment time must be a non-negative UTC timestamp in nanoseconds"
        )
    seconds, fraction = divmod(timestamp_ns, 1_000_000_000)
    try:
        stamp = datetime.fromtimestamp(seconds, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ExperimentError(
            "Experiment timestamp is outside the supported date range"
        ) from exc
    return f"{stamp.year:04d}{stamp.month:02d}{stamp.day:02d}T{stamp:%H%M%S}.{fraction:09d}Z"


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ExperimentError(f"Experiment storage cannot use a symbolic link: {path}")


def _regular_file(path: Path) -> bool:
    _reject_symlink(path)
    if path.exists() and not path.is_file():
        raise ExperimentError(f"Expected a regular file: {path}")
    return path.is_file()


def _save_manifest(path: Path, content: dict) -> None:
    _reject_symlink(path)
    save_shared_json(path, content)


def _copy_calibration_controls(source: Path, destination: Path) -> None:
    """Carry optional raw annotations with their calibration, including legacy files."""
    source = source.with_suffix(".controls.npz")
    if _regular_file(source):
        destination = destination.with_suffix(".controls.npz")
        shutil.copyfile(source, destination)
        share_output(destination)


def _utc_seconds(timestamp_ns: int) -> str:
    """Format system wall time as timezone-explicit ISO 8601, to the second."""
    return datetime.fromtimestamp(timestamp_ns // 1_000_000_000, UTC).isoformat(timespec="seconds")


def finish_block(directory: Path, *, error: Exception | None = None) -> None:
    """Close a block after its writers finish; standalone recordings have no block manifest."""
    _reject_symlink(directory)
    path = directory / "block.json"
    if not _regular_file(path):
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ExperimentError(f"Invalid block manifest: {path}")
    if manifest.get("ended_timestamp_ns") is not None:
        return
    ended = time.time_ns()
    manifest.update(
        ended_timestamp_ns=ended,
        ended_utc=_utc_seconds(ended),
        started_utc=_utc_seconds(manifest["started_timestamp_ns"]),
        status="closed" if error is None else "error",
    )
    if error is not None:
        manifest["error"] = str(error)
    _save_manifest(path, manifest)


class ExperimentStore:
    """Keep every experiment's calibration, configuration, and recordings together.

    Timestamps are UTC Unix nanoseconds. Experiments never overwrite another
    experiment; only an explicitly numbered block can replace an existing block.
    """

    def __init__(self, data_root: Path, base_config_path: Path) -> None:
        requested_root = Path(data_root).expanduser().absolute()
        _reject_symlink(requested_root)
        # Canonicalize ancestors (e.g. macOS /tmp) before enforcing our namespace.
        self.data_root = requested_root.parent.resolve() / requested_root.name
        self.base_config_path = Path(base_config_path).expanduser().resolve()
        # Keep the running launcher's identity even if its checkout later changes.
        self._launch_state = capture_system_state()
        # Backfill experiments created before the local TSV was introduced.
        self._record_history(self.sync_candidates())

    @property
    def history_path(self) -> Path:
        """Human-readable local history, outside every synchronized experiment."""
        return self.data_root / HISTORY_FILENAME

    def _record_history(self, experiments: tuple[Path, ...]) -> None:
        """Refresh summaries from disk before closing or deleting experiment folders."""
        summaries = []
        for directory in experiments:
            subject = directory.parent.name
            if directory.parent != self._subject_path(subject):
                raise ExperimentError(f"Experiment does not belong to this store: {directory}")
            _reject_symlink(directory)
            manifest_path = directory / "experiment.json"
            if not _regular_file(manifest_path):
                raise ExperimentError(f"Missing experiment manifest: {directory}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            started = manifest["started_timestamp_ns"]
            day = _timestamp_name(started)[:8]
            ended = manifest.get("ended_timestamp_ns")
            if ended is not None:
                _timestamp_name(ended)
            number = validate_run_number(manifest.get("number"))
            blocks = directory / "blocks"
            _reject_symlink(blocks)
            block_count = 0
            if blocks.exists():
                for block in blocks.iterdir():
                    if re.fullmatch(r"[1-9][0-9]*", block.name):
                        _reject_symlink(block)
                        block_count += block.is_dir()
            calibration = directory / "calibration.json"
            calibration_file = (str(calibration.relative_to(self.data_root))
                                if _regular_file(calibration) else "")
            state_path = directory / "state.json"
            state = (json.loads(state_path.read_text(encoding="utf-8"))
                     if _regular_file(state_path) else {})
            summaries.append({
                "experiment_name": directory.name, "subject": subject,
                "experiment_number": number, "date_utc": day,
                "started_utc": _utc_seconds(started),
                "ended_utc": "" if ended is None else _utc_seconds(ended),
                "started_timestamp_ns": started, "ended_timestamp_ns": ended,
                "runtime_seconds": "" if ended is None else f"{(ended - started) / 1e9:.6f}",
                "block_count": block_count, "status": manifest.get("status", ""),
                "calibration_file": calibration_file,
                "calibration_source": manifest.get("calibration_source"),
                "git_commit": state.get("git_commit"),
            })
        update_history(self.history_path, summaries)

    def _check_root(self) -> None:
        _reject_symlink(self.data_root)
        if self.data_root.exists() and not self.data_root.is_dir():
            raise ExperimentError(f"Experiment data root is not a directory: {self.data_root}")

    def _subject_path(self, subject: str) -> Path:
        self._check_root()
        try:
            validate_subject_name(subject)
        except ValueError as exc:
            raise ExperimentError(str(exc)) from exc
        directory = self.data_root / subject
        _reject_symlink(directory)
        if directory.exists() and not directory.is_dir():
            raise ExperimentError(f"Subject path is not a directory: {directory}")
        return directory

    def subjects(self) -> list[str]:
        self._check_root()
        if not self.data_root.exists():
            return []
        names = []
        for entry in self.data_root.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                continue
            try:
                validate_subject_name(entry.name)
            except ValueError:
                continue
            names.append(entry.name)
        return sorted(names)

    def add_subject(self, name: str) -> Path:
        directory = self._subject_path(name)
        directory.mkdir(parents=True, exist_ok=True)
        share_output(self.data_root, directory=True)
        share_output(directory, directory=True)
        return self._subject_path(name)

    def _history_directory(self, subject: str) -> Path:
        """Locate the small local cache that survives removal of uploaded experiments."""
        self._subject_path(subject)
        root = self.data_root / ".history"
        _reject_symlink(root)
        directory = root / subject
        _reject_symlink(directory)
        return directory

    def _history(self, subject: str) -> dict:
        """Read the numbering ledger and calibration pointer without remote filesystem I/O."""
        path = self._history_directory(subject) / "index.json"
        if not _regular_file(path):
            return {"version": 1, "experiments": {}}
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("experiments"), dict):
            raise ExperimentError(f"Invalid subject history: {path}")
        for numbers in value["experiments"].values():
            if not isinstance(numbers, list):
                raise ExperimentError(f"Invalid subject history: {path}")
            for number in numbers:
                if validate_run_number(number) is None:
                    raise ExperimentError(f"Invalid subject history: {path}")
        return value

    def sync_candidates(self) -> tuple[Path, ...]:
        """Snapshot real experiment directories; omit subjects, caches and diagnostics."""
        experiments = []
        for subject in self.subjects():
            pattern = re.compile(rf"exp_{re.escape(subject)}_[0-9]{{8}}_[1-9][0-9]*_.*")
            for path in self._subject_path(subject).iterdir():
                if not (_EXPERIMENT_NAME.fullmatch(path.name) or pattern.fullmatch(path.name)):
                    continue
                _reject_symlink(path)
                if not path.is_dir() or not _regular_file(path / "experiment.json"):
                    raise ExperimentError(f"Incomplete experiment directory: {path}")
                experiments.append(path)
        return tuple(sorted(experiments))

    def remember_synced(self, experiments: tuple[Path, ...]) -> None:
        """Persist summaries, daily numbering and calibration before deleting uploaded data."""
        grouped: dict[str, list[tuple[Path, dict]]] = {}
        for path in experiments:
            subject = path.parent.name
            if path.parent != self._subject_path(subject):
                raise ExperimentError(f"Experiment does not belong to this store: {path}")
            _reject_symlink(path)
            manifest = path / "experiment.json"
            if not _regular_file(manifest):
                raise ExperimentError(f"Missing experiment manifest: {path}")
            value = json.loads(manifest.read_text(encoding="utf-8"))
            _timestamp_name(value["started_timestamp_ns"])
            validate_run_number(value.get("number"))
            grouped.setdefault(subject, []).append((path, value))
        self._record_history(experiments)
        for subject, entries in grouped.items():
            directory = self._history_directory(subject)
            history = self._history(subject)
            directory.mkdir(parents=True, exist_ok=True)
            share_output(directory.parent, directory=True)
            share_output(directory, directory=True)
            old_cache = history.get("calibration")
            latest_key = tuple(history.get("calibration_key", (-1, 0)))
            latest = None
            for path, manifest in entries:
                number = manifest.get("number")
                stamp = manifest["started_timestamp_ns"]
                if number is not None:
                    day = _timestamp_name(stamp)[:8]
                    history["experiments"][day] = sorted(
                        set(history["experiments"].get(day, ())) | {number}
                    )
                key = (stamp, number or 0)
                if key > latest_key and _regular_file(path / "calibration.json"):
                    latest, latest_key = path, key
            prepared = None
            try:
                if latest is not None:
                    prepared = Path(tempfile.mkdtemp(prefix="calibration-", dir=directory))
                    share_output(prepared, directory=True)
                    for name in ("calibration.json", "config.json"):
                        if not _regular_file(latest / name):
                            raise ExperimentError(f"Missing {name} in {latest}")
                        shutil.copyfile(latest / name, prepared / name)
                        share_output(prepared / name)
                    _copy_calibration_controls(
                        latest / "calibration.json", prepared / "calibration.json",
                    )
                    history.update(calibration=prepared.name, calibration_key=latest_key)
                _save_manifest(directory / "index.json", history)
            except BaseException:
                if prepared is not None:
                    shutil.rmtree(prepared)
                raise
            if old_cache and old_cache != history.get("calibration"):
                old_path = self._cached_calibration_directory(subject, old_cache)
                shutil.rmtree(old_path)

    def _cached_calibration_directory(self, subject: str, name: str) -> Path:
        """Validate a local calibration-cache pointer before reading or removing it."""
        if not isinstance(name, str) or re.fullmatch(r"calibration-[A-Za-z0-9_-]+", name) is None:
            raise ExperimentError("Invalid cached calibration path")
        directory = self._history_directory(subject) / name
        _reject_symlink(directory)
        return directory

    def check_available(
        self, subject: str, timestamp_ns: int, number: int | None = None,
    ) -> int:
        """Resolve a free number for this subject/UTC day without writing anything."""
        stamp = _timestamp_name(timestamp_ns)
        validate_run_number(number)
        subject_directory = self._subject_path(subject)
        pattern = re.compile(rf"exp_{re.escape(subject)}_{stamp[:8]}_([1-9][0-9]*)(?:_.*)?$")
        used = set(self._history(subject)["experiments"].get(stamp[:8], ()))
        used.update(int(row["experiment_number"]) for row in read_history(self.history_path)
                    if row["subject"] == subject and row["date_utc"] == stamp[:8]
                    and row["experiment_number"])
        if subject_directory.exists():
            for entry in subject_directory.iterdir():
                match = pattern.fullmatch(entry.name)
                if match:
                    _reject_symlink(entry)
                    used.add(int(match[1]))
        if number is None:
            number = max(used, default=0) + 1
            validate_run_number(number)
        if number in used:
            raise ExperimentError(
                f"Experiment already exists: {subject}/{stamp[:8]} number {number}"
            )
        name = _experiment_name(subject, timestamp_ns, number)
        directory = subject_directory / name
        _reject_symlink(directory)
        if directory.exists():
            raise ExperimentError(f"Experiment already exists: {subject}/{name}")
        return number

    def _previous_calibration(
        self, subject_directory: Path, timestamp_ns: int, number: int,
    ) -> Path | None:
        """Find the latest prior calibration, including legacy timestamp directories."""
        candidates = []
        cutoff = (_timestamp_name(timestamp_ns), number)
        history = self._history(subject_directory.name)
        if history.get("calibration"):
            saved_time, saved_number = history["calibration_key"]
            key = (_timestamp_name(saved_time), saved_number)
            if key < cutoff:
                candidates.append((key, self._cached_calibration_directory(
                    subject_directory.name, history["calibration"],
                )))
        pattern = re.compile(rf"exp_{re.escape(subject_directory.name)}_[0-9]{{8}}_[1-9][0-9]*_.*")
        for entry in subject_directory.iterdir():
            legacy = _EXPERIMENT_NAME.fullmatch(entry.name)
            if not legacy and not pattern.fullmatch(entry.name):
                continue
            _reject_symlink(entry)
            if not entry.is_dir():
                continue
            if legacy:
                key = (entry.name, 0)
            else:
                manifest_path = entry / "experiment.json"
                if not _regular_file(manifest_path):
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                key = (_timestamp_name(manifest["started_timestamp_ns"]), manifest["number"])
            if key < cutoff:
                candidates.append((key, entry))
        for _, directory in sorted(candidates, reverse=True):
            calibration = directory / "calibration.json"
            if _regular_file(calibration):
                return calibration
        return None

    def create(
        self, subject: str, timestamp_ns: int | None = None, number: int | None = None,
    ) -> Experiment:
        """Create a numbered experiment and inherit its most recent calibration."""
        timestamp_ns = time.time_ns() if timestamp_ns is None else timestamp_ns
        number = self.check_available(subject, timestamp_ns, number)
        name = _experiment_name(subject, timestamp_ns, number)
        subject_directory = self.add_subject(subject)
        directory = subject_directory / name
        _reject_symlink(directory)
        if directory.exists():
            raise ExperimentError(f"Experiment already exists: {subject}/{name}")

        config = AppConfig.load(self.base_config_path)
        source = self._previous_calibration(subject_directory, timestamp_ns, number)
        if source is not None:
            previous_config_path = source.parent / "config.json"
            if _regular_file(previous_config_path):
                previous = AppConfig.load(previous_config_path)
                config = replace(
                    config,
                    camera=replace(
                        config.camera,
                        **{
                            key: getattr(previous.camera, key)
                            for key in _CALIBRATED_CAMERA_FIELDS
                        },
                    ),
                    tracker=previous.tracker,
                )
        else:
            source = Path(config.roi_config).expanduser()
            if not source.is_absolute():
                source = self.base_config_path.parent / source
            if not _regular_file(source):
                source = None
        if source is not None:
            RoiLayout.load(source).validate_for_frame(
                config.camera.analysis_width,
                config.camera.analysis_height,
                tracker_config=config.tracker,
            )

        experiment = Experiment(
            subject=subject,
            directory=directory,
            config_path=directory / "config.json",
            calibration_path=directory / "calibration.json",
            started_timestamp_ns=timestamp_ns,
            number=number,
        )
        config = replace(
            config,
            roi_config="calibration.json",
            recording=replace(
                config.recording, directory=str(directory), layout="hands_eyes",
                shared_permissions=True,
            ),
        )
        # Exclusive creation reserves this destination. On failure, remove only
        # this newly created directory; never touch an existing experiment.
        try:
            directory.mkdir()
        except FileExistsError as exc:
            raise ExperimentError(f"Experiment already exists: {subject}/{name}") from exc
        try:
            share_output(directory, directory=True)
            if source is not None:
                shutil.copyfile(source, experiment.calibration_path)
                share_output(experiment.calibration_path)
                _copy_calibration_controls(source, experiment.calibration_path)
            config.save(experiment.config_path, shared=True)
            _save_manifest(directory / "state.json", self._launch_state)
            _save_manifest(
                directory / "experiment.json",
                {
                    "version": 2,
                    "subject": subject,
                    "name": name,
                    "number": number,
                    "date_utc": _timestamp_name(timestamp_ns)[:8],
                    "started_timestamp_ns": timestamp_ns,
                    "status": "active",
                    "calibration_source": None if source is None else str(source),
                },
            )
            self._record_history((directory,))
        except BaseException:
            shutil.rmtree(directory)
            raise
        return experiment

    def _check_experiment(self, experiment: Experiment) -> Path:
        """Require an owned, real experiment directory before modifying its contents."""
        expected = self._subject_path(experiment.subject) / _experiment_name(
            experiment.subject, experiment.started_timestamp_ns, experiment.number,
        )
        if experiment.directory != expected:
            raise ExperimentError("Experiment does not belong to this store")
        _reject_symlink(expected)
        return expected

    def create_block(
        self, experiment: Experiment, config: AppConfig, layout: RoiLayout,
        number: int | None = None,
    ) -> tuple[Block, AppConfig]:
        """Snapshot loaded settings and reserve a block after its previous writer stops.

        Automatic numbering never overwrites. An explicit number replaces only
        that block; snapshots are prepared before removing any previous contents.
        """
        parent = self._check_experiment(experiment) / "blocks"
        _reject_symlink(parent)
        validate_run_number(number)
        parent.mkdir(exist_ok=True)
        share_output(parent, directory=True)
        explicit = number is not None
        if number is None:
            number = max((int(p.name) for p in parent.iterdir()
                          if re.fullmatch(r"[1-9][0-9]*", p.name)), default=0) + 1
            validate_run_number(number)
        block = Block(number, parent / str(number))
        _reject_symlink(block.directory)
        if block.directory.exists() and (not explicit or not block.directory.is_dir()):
            raise ExperimentError(f"Block destination already exists: {block.directory}")
        saved = replace(
            config, roi_config="calibration.json",
            recording=replace(config.recording, directory=str(block.directory), shared_permissions=True),
        )
        staging = Path(tempfile.mkdtemp(prefix=".block-", dir=parent))
        prepared = staging / "new"
        backup = staging / "previous"
        preserve_backup = False
        try:
            prepared.mkdir()
            share_output(prepared, directory=True)
            layout.save(prepared / "calibration.json", shared=True)
            saved.save(prepared / "config.json", shared=True)
            started = time.time_ns()
            _save_manifest(prepared / "block.json", {
                "version": 2, "number": number, "experiment": experiment.name,
                "status": "active", "started_timestamp_ns": started,
                "ended_timestamp_ns": None,
                "started_utc": _utc_seconds(started), "ended_utc": None,
            })
            _reject_symlink(block.directory)
            if block.directory.exists():
                if not explicit or not block.directory.is_dir():
                    raise ExperimentError(f"Block destination already exists: {block.directory}")
                block.directory.rename(backup)
            try:
                prepared.rename(block.directory)
            except BaseException as exc:
                if backup.exists():
                    try:
                        backup.rename(block.directory)
                    except OSError as restore_error:
                        preserve_backup = True
                        exc.add_note(f"Previous block retained at {backup}: {restore_error}")
                raise
        finally:
            if not preserve_backup:
                shutil.rmtree(staging)
        return block, saved

    def finish(self, experiment: Experiment) -> None:
        """Mark this experiment closed, preserving its original close time on retries."""
        expected = self._check_experiment(experiment)
        manifest_path = expected / "experiment.json"
        if not _regular_file(manifest_path):
            raise ExperimentError(f"Experiment manifest does not exist: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"Invalid experiment manifest: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise ExperimentError(f"Invalid experiment manifest: {manifest_path}")
        if manifest.get("status") != "closed":
            manifest.update(status="closed", ended_timestamp_ns=time.time_ns())
            _save_manifest(manifest_path, manifest)
        self._record_history((expected,))
