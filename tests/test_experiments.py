from __future__ import annotations

import json
import os
import shutil
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


def test_sync_history_preserves_latest_calibration_and_daily_numbers_after_removal(tmp_path):
    store = _store(tmp_path)
    first = store.create("subject_A", _FIRST_TIME, number=4)
    last = store.create("subject_A", _SECOND_TIME)
    _layout(gain=1.7).save(last.calibration_path)
    config = AppConfig.load(last.config_path)
    replace(config, camera=replace(config.camera, exposure_us=7_000)).save(last.config_path)
    store.remember_synced((first.directory, last.directory))
    for experiment in (first, last):
        shutil.rmtree(experiment.directory)
    assert store.subjects() == ["subject_A"]
    assert store.sync_candidates() == ()
    assert store.check_available("subject_A", _THIRD_TIME) == 6
    with pytest.raises(ExperimentError, match="already exists"):
        store.create("subject_A", _THIRD_TIME, 4)
    assert store.check_available("subject_A", _THIRD_TIME, 2) == 2
    # A newly opened store must recover everything from disk, without a remote read.
    store = ExperimentStore(store.data_root, store.base_config_path)
    new = store.create("subject_A", _THIRD_TIME)
    assert new.number == 6
    assert RoiLayout.load(new.calibration_path) == _layout(gain=1.7)
    assert AppConfig.load(new.config_path).camera.exposure_us == 7_000
    assert store.check_available("subject_A", _THIRD_TIME + 86_400_000_000_000) == 1


def test_cache_replacement_keeps_only_latest_calibration_and_survives_uncalibrated_sync(tmp_path):
    store = _store(tmp_path)
    first = store.create("subject_A", _FIRST_TIME)
    store.remember_synced((first.directory,))
    last = store.create("subject_A", _SECOND_TIME)
    _layout(gain=1.9).save(last.calibration_path)
    store.remember_synced((last.directory,))
    uncalibrated = store.create("subject_A", _THIRD_TIME)
    uncalibrated.calibration_path.unlink()
    store.remember_synced((uncalibrated.directory,))
    history_dir = store.data_root / ".history" / "subject_A"
    cached = list(history_dir.glob("calibration-*"))
    assert len(cached) == 1
    assert RoiLayout.load(cached[0] / "calibration.json") == _layout(gain=1.9)
    assert json.loads((history_dir / "index.json").read_text())["experiments"] == {
        "20261014": [1, 2, 3],
    }
    assert cached[0].stat().st_mode & 0o777 == 0o777
    assert (cached[0] / "calibration.json").stat().st_mode & 0o777 == 0o666


def test_sync_candidate_rejects_incomplete_experiment_and_history_symlinks(tmp_path):
    store = _store(tmp_path)
    experiment = store.create("subject_A", _FIRST_TIME)
    manifest = (experiment.directory / "experiment.json").read_bytes()
    (experiment.directory / "experiment.json").unlink()
    with pytest.raises(ExperimentError, match="Incomplete"):
        store.sync_candidates()
    (experiment.directory / "experiment.json").write_bytes(manifest)
    (store.data_root / ".history").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ExperimentError, match="symbolic link"):
        store.remember_synced((experiment.directory,))


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
    assert experiment.name == "exp_subject_A_20261014_1_174640"
    assert experiment.number == 1
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
        base.recording, directory=str(second.directory), layout="hands_eyes",
        shared_permissions=True,
    )
    assert current.transport == base.transport
    assert second.calibration_path.read_bytes() == prior_calibration_bytes
    _layout(gain=2.0).save(second.calibration_path)
    assert first.calibration_path.read_bytes() == prior_calibration_bytes
    assert first.config_path.read_bytes() == prior_config_bytes
    assert AppConfig.load(store.base_config_path) == base


def test_experiment_tree_and_replaced_manifest_are_shared_under_root_style_umask(tmp_path):
    store = _store(tmp_path)
    base_mode = store.base_config_path.stat().st_mode
    parent_mode = tmp_path.stat().st_mode
    previous = os.umask(0o077)
    try:
        experiment = store.create("shared_subject", _FIRST_TIME)
        for directory in (store.data_root, experiment.directory.parent, experiment.directory):
            assert directory.stat().st_mode & 0o777 == 0o777
        for artifact in experiment.directory.iterdir():
            assert artifact.stat().st_mode & 0o777 == 0o666
        store.finish(experiment)
        assert (experiment.directory / "experiment.json").stat().st_mode & 0o777 == 0o666
        assert AppConfig.load(experiment.config_path).recording.shared_permissions is True
    finally:
        os.umask(previous)
    assert store.base_config_path.stat().st_mode == base_mode
    assert tmp_path.stat().st_mode == parent_mode


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
        store.create("subject_A", _FIRST_TIME, number=experiment.number)

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
        store.check_available("subject_A", _FIRST_TIME, number=first.number)
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


def test_daily_numbers_include_remote_ids_and_survive_store_restart(tmp_path):
    store = _store(tmp_path)
    first = store.create("subject_A", _FIRST_TIME)
    second = store.create("subject_A", _FIRST_TIME)  # Same timestamp is still a new local run.
    remote = store.create("subject_A", _SECOND_TIME, number=12)
    reopened = ExperimentStore(store.data_root, store.base_config_path)
    assert [first.number, second.number, remote.number] == [1, 2, 12]
    assert reopened.create("subject_A", _THIRD_TIME).number == 13
    assert reopened.create("subject_B", _THIRD_TIME).number == 1
    assert reopened.create("subject_A", _FIRST_TIME + 86_400_000_000_000).number == 1
    with pytest.raises(ExperimentError, match="already exists"):
        reopened.create("subject_A", _THIRD_TIME, number=12)


def test_legacy_timestamp_directory_still_provides_calibration(tmp_path):
    store = _store(tmp_path)
    subject = store.add_subject("subject_A")
    legacy = subject / "20261014T174640.123456789Z"
    legacy.mkdir()
    _layout(gain=1.6).save(legacy / "calibration.json")
    current = store.create("subject_A", _SECOND_TIME)
    assert RoiLayout.load(current.calibration_path) == _layout(gain=1.6)


def test_block_snapshots_numbering_and_explicit_replacement_are_scoped(tmp_path):
    store = _store(tmp_path)
    experiment = store.create("subject_A", _FIRST_TIME)
    config = AppConfig.load(experiment.config_path)
    before = experiment.config_path.read_bytes()
    first, saved = store.create_block(experiment, config, _layout())
    marker = first.directory / "old-video.mkv"
    marker.write_bytes(b"old")
    second, _ = store.create_block(experiment, config, _layout(), number=6)
    (second.directory / "keep.mkv").write_bytes(b"keep")
    third, _ = store.create_block(experiment, config, _layout())
    assert [first.number, second.number, third.number] == [1, 6, 7]
    assert first.directory == experiment.directory / "blocks" / "1"
    assert Path(saved.recording.directory) == first.directory
    assert RoiLayout.load(first.calibration_path) == _layout()
    assert AppConfig.load(first.config_path) == saved
    replaced, _ = store.create_block(experiment, config, _layout(gain=2), number=1)
    assert replaced.directory == first.directory
    assert not marker.exists()
    assert RoiLayout.load(replaced.calibration_path) == _layout(gain=2)
    assert (second.directory / "keep.mkv").read_bytes() == b"keep"
    assert experiment.config_path.read_bytes() == before
    assert {p.name for p in first.directory.parent.iterdir()} == {"1", "6", "7"}
    assert first.directory.stat().st_mode & 0o777 == 0o777
    assert first.calibration_path.stat().st_mode & 0o777 == 0o666


def test_block_snapshot_failure_preserves_old_contents(tmp_path, monkeypatch):
    store = _store(tmp_path)
    experiment = store.create("subject_A", _FIRST_TIME)
    config = AppConfig.load(experiment.config_path)
    block, _ = store.create_block(experiment, config, _layout())
    marker = block.directory / "keep.mkv"
    marker.write_bytes(b"keep")

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(RoiLayout, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        store.create_block(experiment, config, _layout(), number=1)
    assert marker.read_bytes() == b"keep"
    assert list(block.directory.parent.iterdir()) == [block.directory]


def test_block_publish_failure_restores_previous_directory(tmp_path, monkeypatch):
    store = _store(tmp_path)
    experiment = store.create("subject_A", _FIRST_TIME)
    config = AppConfig.load(experiment.config_path)
    block, _ = store.create_block(experiment, config, _layout())
    marker = block.directory / "keep.mkv"
    marker.write_bytes(b"keep")
    rename = Path.rename

    def fail_publish(path, target):
        if path.name == "new":
            raise OSError("cannot publish block")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_publish)
    with pytest.raises(OSError, match="cannot publish"):
        store.create_block(experiment, config, _layout(gain=2), number=1)
    assert marker.read_bytes() == b"keep"
    assert RoiLayout.load(block.calibration_path) == _layout()
    assert list(block.directory.parent.iterdir()) == [block.directory]


@pytest.mark.parametrize("link_parent", [False, True])
def test_block_overwrite_cannot_follow_symlinks(tmp_path, link_parent):
    store = _store(tmp_path)
    experiment = store.create("subject_A", _FIRST_TIME)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    parent = experiment.directory / "blocks"
    if link_parent:
        parent.symlink_to(outside, target_is_directory=True)
    else:
        parent.mkdir()
        (parent / "1").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ExperimentError, match="symbolic link"):
        store.create_block(experiment, AppConfig.load(experiment.config_path), _layout(), number=1)
    assert (outside / "keep").read_text() == "keep"
