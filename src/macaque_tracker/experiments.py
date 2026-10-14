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
from .protocol import validate_subject_name

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

    @property
    def name(self) -> str:
        return self.directory.name


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
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".experiment-", delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            json.dump(content, output, indent=2)
            output.write("\n")
            output.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


class ExperimentStore:
    """Keep every experiment's calibration, configuration, and recordings together.

    Timestamps are UTC Unix nanoseconds. Creating an experiment never changes the
    system clock, and an existing subject/timestamp is never overwritten.
    """

    def __init__(self, data_root: Path, base_config_path: Path) -> None:
        requested_root = Path(data_root).expanduser().absolute()
        _reject_symlink(requested_root)
        # Canonicalize ancestors (e.g. macOS /tmp) before enforcing our namespace.
        self.data_root = requested_root.parent.resolve() / requested_root.name
        self.base_config_path = Path(base_config_path).expanduser().resolve()

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
        return self._subject_path(name)

    def check_available(self, subject: str, timestamp_ns: int) -> None:
        """Validate a requested destination without creating files or directories."""
        name = _timestamp_name(timestamp_ns)
        directory = self._subject_path(subject) / name
        _reject_symlink(directory)
        if directory.exists():
            raise ExperimentError(f"Experiment already exists: {subject}/{name}")

    def _previous_calibration(self, subject_directory: Path, name: str) -> Path | None:
        candidates = sorted(
            (
                entry
                for entry in subject_directory.iterdir()
                if _EXPERIMENT_NAME.fullmatch(entry.name) and entry.name < name
            ),
            reverse=True,
        )
        for directory in candidates:
            _reject_symlink(directory)
            if not directory.is_dir():
                continue
            calibration = directory / "calibration.json"
            if _regular_file(calibration):
                return calibration
        return None

    def create(self, subject: str, timestamp_ns: int | None = None) -> Experiment:
        timestamp_ns = time.time_ns() if timestamp_ns is None else timestamp_ns
        self.check_available(subject, timestamp_ns)
        name = _timestamp_name(timestamp_ns)
        subject_directory = self.add_subject(subject)
        directory = subject_directory / name
        _reject_symlink(directory)
        if directory.exists():
            raise ExperimentError(f"Experiment already exists: {subject}/{name}")

        config = AppConfig.load(self.base_config_path)
        source = self._previous_calibration(subject_directory, name)
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
        )
        config = replace(
            config,
            roi_config="calibration.json",
            recording=replace(config.recording, directory=str(directory), layout="hands_eyes"),
        )
        # Exclusive creation reserves the timestamp. On failure, remove only
        # this newly created directory; never touch an existing experiment.
        try:
            directory.mkdir()
        except FileExistsError as exc:
            raise ExperimentError(f"Experiment already exists: {subject}/{name}") from exc
        try:
            if source is not None:
                shutil.copyfile(source, experiment.calibration_path)
            config.save(experiment.config_path)
            _save_manifest(
                directory / "experiment.json",
                {
                    "version": 1,
                    "subject": subject,
                    "name": name,
                    "started_timestamp_ns": timestamp_ns,
                    "status": "active",
                    "calibration_source": None if source is None else str(source),
                },
            )
        except BaseException:
            shutil.rmtree(directory)
            raise
        return experiment

    def finish(self, experiment: Experiment) -> None:
        """Mark this experiment closed, preserving its original close time on retries."""
        expected = self._subject_path(experiment.subject) / _timestamp_name(
            experiment.started_timestamp_ns
        )
        if experiment.directory != expected:
            raise ExperimentError("Experiment does not belong to this store")
        _reject_symlink(expected)
        manifest_path = expected / "experiment.json"
        if not _regular_file(manifest_path):
            raise ExperimentError(f"Experiment manifest does not exist: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"Invalid experiment manifest: {manifest_path}") from exc
        if not isinstance(manifest, dict):
            raise ExperimentError(f"Invalid experiment manifest: {manifest_path}")
        if manifest.get("status") == "closed":
            return
        manifest.update(status="closed", ended_timestamp_ns=time.time_ns())
        _save_manifest(manifest_path, manifest)
