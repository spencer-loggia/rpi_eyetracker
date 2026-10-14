from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from macaque_tracker.config import (
    AppConfig,
    CameraConfig,
    RecordingConfig,
    RoiLayout,
    TrackerConfig,
    TransportConfig,
)
from macaque_tracker.experiments import ExperimentError, ExperimentStore
from macaque_tracker.models import EyeImageSettings, NormalizedRoi

_FIRST_TIME = 1_792_000_000_123_456_789
_SECOND_TIME = _FIRST_TIME + 1_000_000_000
_THIRD_TIME = _FIRST_TIME + 2_000_000_000


def _layout(*, gain: float = 1.0) -> RoiLayout:
    return RoiLayout(
        rois=(NormalizedRoi(0, "left eye", 0.55, 0.2, 0.1, 0.4, EyeImageSettings(gain=gain)),),
        source_width=5120,
        source_height=720,
    )


def _store(tmp_path: Path, *, calibrated: bool = True) -> ExperimentStore:
    config_path = tmp_path / "settings" / "eye_tracker.json"
    config = AppConfig(roi_config="initial-rois.json")
    config.save(config_path)
    if calibrated:
        _layout().save(config_path.parent / config.roi_config)
    return ExperimentStore(tmp_path / "data", config_path)


def test_subjects_are_explicit_validated_directories(tmp_path):
    store = _store(tmp_path)
    assert store.subjects() == []
    assert store.add_subject("subject_B") == store.data_root / "subject_B"
    assert store.add_subject("subject_A") == store.data_root / "subject_A"
    assert store.add_subject("subject_A") == store.data_root / "subject_A"
    (store.data_root / "notes.txt").write_text("Notes", encoding="utf-8")
    (store.data_root / ".temporary").mkdir()

    assert store.subjects() == ["subject_A", "subject_B"]


@pytest.mark.parametrize(
    "subject", ["", "..", "../escape", "/tmp/escape", "a/b", "a\\b", "x" * 33, "José"]
)
def test_subject_names_cannot_escape_storage(tmp_path, subject):
    store = _store(tmp_path)
    with pytest.raises(ExperimentError, match="Subject"):
        store.create(subject, _FIRST_TIME)
    assert not store.data_root.exists()


def test_new_experiment_owns_config_calibration_and_recording_directory(tmp_path):
    store = _store(tmp_path)
    base_before = store.base_config_path.read_bytes()
    calibration_before = (store.base_config_path.parent / "initial-rois.json").read_bytes()

    experiment = store.create("subject_A", _FIRST_TIME)

    assert experiment.subject == "subject_A"
    assert experiment.directory.parent == store.data_root / "subject_A"
    assert experiment.name == "20261014T174640.123456789Z"
    assert experiment.started_timestamp_ns == _FIRST_TIME
    config = AppConfig.load(experiment.config_path)
    assert config.roi_config == "calibration.json"
    assert Path(config.recording.directory) == experiment.directory
    assert experiment.calibration_path.read_bytes() == calibration_before
    assert store.base_config_path.read_bytes() == base_before
    manifest = json.loads((experiment.directory / "experiment.json").read_text())
    assert manifest["subject"] == "subject_A"
    assert manifest["started_timestamp_ns"] == _FIRST_TIME
    assert manifest["status"] == "active"


def test_calibration_carries_forward_but_hardware_recording_transport_stay_current(tmp_path):
    store = _store(tmp_path)
    first = store.create("subject_A", _FIRST_TIME)
    _layout(gain=1.7).save(first.calibration_path)
    first_config = AppConfig.load(first.config_path)
    replace(
        first_config,
        camera=replace(first_config.camera, exposure_us=8000, analogue_gain=2.0),
        tracker=TrackerConfig(min_confidence=0.7),
    ).save(first.config_path)
    prior_config_bytes = first.config_path.read_bytes()
    prior_calibration_bytes = first.calibration_path.read_bytes()
    base = AppConfig(
        camera=CameraConfig(fps=60, exposure_us=9000, buffer_count=8),
        tracker=TrackerConfig(min_confidence=0.6),
        recording=RecordingConfig(fps=30, crf=30, container="mp4"),
        transport=TransportConfig(backend="none"),
        roi_config="initial-rois.json",
    )
    base.save(store.base_config_path)

    second = store.create("subject_A", _SECOND_TIME)
    current = AppConfig.load(second.config_path)

    assert current.camera.fps == base.camera.fps
    assert current.camera.buffer_count == base.camera.buffer_count
    assert current.camera.exposure_us == 8000
    assert current.camera.analogue_gain == 2.0
    assert current.tracker.min_confidence == 0.7
    assert current.recording == replace(
        base.recording, directory=str(second.directory), layout="hands_eyes"
    )
    assert current.transport == base.transport
    assert second.calibration_path.read_bytes() == prior_calibration_bytes
    _layout(gain=2.0).save(second.calibration_path)
    assert first.calibration_path.read_bytes() == prior_calibration_bytes
    assert first.config_path.read_bytes() == prior_config_bytes
    assert AppConfig.load(store.base_config_path) == base


def test_experiments_select_configured_pose_cameras_even_from_legacy_full_layout(tmp_path):
    store = _store(tmp_path)
    base = AppConfig.load(store.base_config_path)
    base = replace(base, recording=replace(base.recording, layout="full", pose_cameras=(1, 3)))
    base.save(store.base_config_path)

    experiment = store.create("subject_A", _FIRST_TIME)
    configured = AppConfig.load(experiment.config_path)

    assert configured.recording.layout == "hands_eyes"
    assert configured.recording.pose_cameras == (1, 3)
    assert configured.recording.fps == base.recording.fps
    assert AppConfig.load(store.base_config_path) == base


def test_latest_calibration_uses_timestamp_order_and_never_another_subject(tmp_path):
    store = _store(tmp_path)
    first = store.create("subject_A", _FIRST_TIME)
    _layout(gain=1.2).save(first.calibration_path)
    second = store.create("subject_A", _SECOND_TIME)
    _layout(gain=1.8).save(second.calibration_path)
    os.utime(first.directory, (2_000_000_000, 2_000_000_000))
    os.utime(second.directory, (1, 1))
    other = store.create("subject_B", _SECOND_TIME)
    _layout(gain=3.0).save(other.calibration_path)

    third = store.create("subject_A", _THIRD_TIME)
    new_subject = store.create("subject_C", _THIRD_TIME)

    assert RoiLayout.load(third.calibration_path) == _layout(gain=1.8)
    assert RoiLayout.load(new_subject.calibration_path) == _layout()


def test_missing_initial_calibration_allows_calibrating_in_new_experiment(tmp_path):
    store = _store(tmp_path, calibrated=False)
    experiment = store.create("subject_A", _FIRST_TIME)

    assert experiment.config_path.is_file()
    assert not experiment.calibration_path.exists()
    assert AppConfig.load(experiment.config_path).roi_config == "calibration.json"
    assert (
        json.loads((experiment.directory / "experiment.json").read_text())["calibration_source"]
        is None
    )


def test_most_recent_incomplete_experiment_does_not_discard_existing_calibration(tmp_path):
    store = _store(tmp_path)
    first = store.create("subject_A", _FIRST_TIME)
    _layout(gain=1.5).save(first.calibration_path)
    second = store.create("subject_A", _SECOND_TIME)
    second.calibration_path.unlink()

    third = store.create("subject_A", _THIRD_TIME)

    assert RoiLayout.load(third.calibration_path) == _layout(gain=1.5)


def test_collision_cannot_overwrite_existing_experiment(tmp_path):
    store = _store(tmp_path)
    experiment = store.create("subject_A", _FIRST_TIME)
    marker = experiment.directory / "recording.mkv"
    marker.write_bytes(b"recorded data")

    with pytest.raises(ExperimentError, match="already exists"):
        store.create("subject_A", _FIRST_TIME)

    assert marker.read_bytes() == b"recorded data"
    assert len(list(experiment.directory.parent.iterdir())) == 1


def test_availability_is_read_only_and_checks_paths_timestamps_and_collisions(tmp_path):
    store = _store(tmp_path)
    store.check_available("subject_A", _FIRST_TIME)
    assert not store.data_root.exists()
    with pytest.raises(ExperimentError, match="time"):
        store.check_available("subject_A", 10**30)
    with pytest.raises(ExperimentError, match="Subject"):
        store.check_available("../outside", _FIRST_TIME)
    assert not store.data_root.exists()
    first = store.create("subject_A", _FIRST_TIME)
    with pytest.raises(ExperimentError, match="already exists"):
        store.check_available("subject_A", _FIRST_TIME)
    linked = store.data_root / "subject_B"
    linked.symlink_to(first.directory, target_is_directory=True)
    with pytest.raises(ExperimentError, match="symbolic link"):
        store.check_available("subject_B", _SECOND_TIME)


@pytest.mark.parametrize("timestamp", [-1, True, 1.5, 10**30])
def test_invalid_timestamp_creates_nothing(tmp_path, timestamp):
    store = _store(tmp_path)
    with pytest.raises(ExperimentError, match="time"):
        store.create("subject_A", timestamp)
    assert not store.data_root.exists()


def test_manual_create_uses_wall_clock_without_setting_it(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setattr("macaque_tracker.experiments.time.time_ns", lambda: _FIRST_TIME)
    assert store.create("subject_A").started_timestamp_ns == _FIRST_TIME


def test_symlink_roots_subjects_and_experiments_are_rejected(tmp_path):
    store = _store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ExperimentError, match="symbolic link"):
        ExperimentStore(linked, store.base_config_path)
    store.data_root.mkdir()
    subject_link = store.data_root / "subject_A"
    subject_link.symlink_to(outside, target_is_directory=True)
    assert store.subjects() == []
    with pytest.raises(ExperimentError, match="symbolic link"):
        store.create("subject_A", _FIRST_TIME)
    subject_link.unlink()
    subject_link.mkdir()
    (subject_link / "20261014T174640.123456789Z").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ExperimentError, match="symbolic link"):
        store.create("subject_A", _SECOND_TIME)
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("artifact", ["calibration.json", "config.json"])
def test_prior_artifact_symlinks_are_rejected(tmp_path, artifact):
    store = _store(tmp_path)
    first = store.create("subject_A", _FIRST_TIME)
    path = first.directory / artifact
    outside = tmp_path / artifact
    path.rename(outside)
    path.symlink_to(outside)

    with pytest.raises(ExperimentError, match="symbolic link"):
        store.create("subject_A", _SECOND_TIME)

    assert len(list(first.directory.parent.iterdir())) == 1


def test_partial_creation_removes_only_its_own_new_directory(tmp_path, monkeypatch):
    store = _store(tmp_path)
    first = store.create("subject_A", _FIRST_TIME)
    first_calibration = first.calibration_path.read_bytes()

    def failed_copy(source, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr("macaque_tracker.experiments.shutil.copyfile", failed_copy)
    with pytest.raises(OSError, match="disk full"):
        store.create("subject_A", _SECOND_TIME)

    assert list(first.directory.parent.iterdir()) == [first.directory]
    assert first.calibration_path.read_bytes() == first_calibration


def test_finish_records_time_once_and_rejects_other_store(tmp_path, monkeypatch):
    store = _store(tmp_path)
    experiment = store.create("subject_A", _FIRST_TIME)
    monkeypatch.setattr("macaque_tracker.experiments.time.time_ns", lambda: _SECOND_TIME)
    store.finish(experiment)
    manifest = json.loads((experiment.directory / "experiment.json").read_text())
    assert manifest["status"] == "closed"
    assert manifest["ended_timestamp_ns"] == _SECOND_TIME
    monkeypatch.setattr("macaque_tracker.experiments.time.time_ns", lambda: _THIRD_TIME)
    store.finish(experiment)
    assert json.loads((experiment.directory / "experiment.json").read_text()) == manifest
    other_store = ExperimentStore(tmp_path / "other-data", store.base_config_path)
    with pytest.raises(ExperimentError, match="does not belong"):
        other_store.finish(experiment)
