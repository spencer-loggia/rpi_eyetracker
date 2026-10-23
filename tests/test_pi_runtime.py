from __future__ import annotations

import json
import os
import pty
import select
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from macaque_tracker.config import AppConfig, RoiLayout, SyncConfig
from macaque_tracker.data_sync import SyncReport, SyncStatus
from macaque_tracker.experiments import ExperimentStore
from macaque_tracker.models import NormalizedRoi
from macaque_tracker.pi_runtime import CameraFeed, ExperimentRuntime
from macaque_tracker.protocol import (
    TRANSACTION_SIZE,
    CommandCode,
    CommandPacket,
    StatusCode,
    StatusFlag,
    StatusSnapshot,
    decode_status,
    encode_command,
)
from macaque_tracker.service import InvalidStateError
from macaque_tracker.transport import ProtocolEngine, UartServer

_TIME = 1_792_000_000_123_456_789


class FakeCamera:
    def __init__(self, harness, config, layout):
        self.harness = harness
        self.config = config
        self.layout = layout
        self.started = False
        self.closed = False

    def start(self):
        if not self.started:
            assert self.harness.owners == 0, "Two camera owners are active"
            self.harness.owners += 1
            self.started = True

    def capture_preview(self):
        return np.full((36, 256), 42, dtype=np.uint8), 1

    def stop(self):
        if self.started:
            self.harness.owners -= 1
            self.started = False

    def close(self):
        self.stop()
        self.closed = True


class FakeFeed:
    def __init__(self, camera):
        self.camera = camera
        self.closed = False
        self.fail_close = False
        self.error_message = None

    def start(self):
        self.camera.start()

    def close(self):
        if self.fail_close:
            raise RuntimeError("feed shutdown failed")
        self.camera.close()
        self.closed = True

    def latest(self):
        return self.camera.capture_preview()[0]


class FakePreview:
    def __init__(self, config, **kwargs):
        self.config = config
        self.options = kwargs
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.closed = True


class FakeService:
    def __init__(self, config, layout, *, camera, preview, recording_config_factory=None):
        self.config = config
        self.layout = layout
        self.camera = camera
        self.preview = preview
        self.tracking = False
        self.recording = False
        self.closed = False
        self.fail_close = False
        self.close_entered = None
        self.close_release = None
        self.start_calls = 0
        self.stop_calls = 0
        self.clear_calls = 0
        self.error_message = None
        self.recording_config_factory = recording_config_factory
        self.session_config = config

    def _allocate_session(self):
        if self.recording_config_factory:
            self.session_config = self.recording_config_factory(self.config, self.layout)

    def start_preview(self):
        self.camera.start()
        self.preview.start()

    def start_tracking(self):
        if self.tracking:
            return
        self.start_calls += 1
        self._allocate_session()
        self.tracking = True
        self.recording = self.config.recording.record_on_tracking

    def stop_tracking(self):
        self.stop_calls += 1
        self.tracking = False
        self.recording = False

    def start_recording(self):
        if self.recording:
            return
        self._allocate_session()
        self.recording = True

    def stop_recording(self):
        self.recording = False

    def clear_error(self):
        self.clear_calls += 1

    def snapshot(self):
        return StatusSnapshot(
            tracking=self.tracking,
            recording=self.recording,
            configured=True,
            configured_eye_count=len(self.layout.rois),
            result=None,
        )

    def close(self):
        if self.close_entered is not None:
            self.close_entered.set()
            assert self.close_release.wait(2), "Test did not release service shutdown"
        if self.fail_close:
            raise RuntimeError("recording finalization failed")
        self.stop_tracking()
        self.camera.close()
        self.preview.stop()
        self.closed = True


class FakeCalibration:
    def __init__(self, config, experiment, feed):
        self.config = config
        self.experiment = experiment
        self.feed = feed
        self.saved = False
        self.close_calls = 0

    def save(self):
        self.saved = True

    def close(self):
        assert not self.feed.closed, "Cancel calibration before shutting down its feed"
        self.close_calls += 1


class Harness:
    def __init__(self, store, **options):
        self.store = store
        self.owners = 0
        self.cameras = []
        self.services = []
        self.clock_calls = []
        self.clock_error = None
        self.shutdown_calls = 0
        self.shutdown_error = None
        self.runtime = ExperimentRuntime(
            store,
            clock_setter=self.set_clock,
            shutdown_executor=self.power_off,
            camera_factory=self.camera,
            service_factory=self.service,
            preview_factory=FakePreview,
            feed_factory=FakeFeed,
            **options,
        )

    def power_off(self):
        self.shutdown_calls += 1
        assert self.owners == 0
        assert self.runtime.view().experiment is None
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def set_clock(self, timestamp_ns):
        self.clock_calls.append(timestamp_ns)
        if self.clock_error is not None:
            raise self.clock_error

    def camera(self, config, layout=None):
        camera = FakeCamera(self, config, layout)
        self.cameras.append(camera)
        return camera

    def service(self, config, layout, **kwargs):
        service = FakeService(config, layout, **kwargs)
        self.services.append(service)
        return service

    def command(self, code, sequence, **kwargs):
        return self.runtime.execute(CommandPacket(code, sequence, **kwargs))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    config_path = tmp_path / "eye_tracker.json"
    AppConfig().save(config_path)
    RoiLayout(
        rois=(NormalizedRoi(0, "eye", 0.55, 0.2, 0.1, 0.4),),
        source_width=5120,
        source_height=720,
    ).save(tmp_path / "rois.json")
    monkeypatch.setitem(
        sys.modules,
        "macaque_tracker.pi_calibration",
        SimpleNamespace(CalibrationSession=FakeCalibration),
    )
    return Harness(ExperimentStore(tmp_path / "data", config_path))


@pytest.fixture
def sync_harness(harness):
    """Hold transfers at a barrier to exercise UART/UI transitions during cleanup."""
    configured = replace(AppConfig.load(harness.store.base_config_path),
                         sync=SyncConfig(remote_data_url="/mnt/test-data"))
    configured.save(harness.store.base_config_path)
    entered, release = threading.Event(), threading.Event()
    calls = []
    outcome = [SyncReport(SyncStatus.SUCCESS, "copied")]

    class Sync:
        def __init__(self, store, config):
            assert config == configured.sync

        def run(self, cancel, progress, *, exclude):
            assert controller.owners == 0
            assert all(service.closed for service in controller.services)
            calls.append(exclude)
            progress("Syncing test experiments")
            entered.set()
            assert release.wait(3), "Test did not release sync"
            return (SyncReport(SyncStatus.CANCELLED, "cancelled") if cancel.is_set()
                    else outcome[0])

    controller = Harness(harness.store, sync_factory=Sync)
    sync = SimpleNamespace(entered=entered, release=release, calls=calls, outcome=outcome)
    yield controller, sync
    release.set()
    controller.runtime.close()


def finish_sync(runtime, sync):
    sync.release.set()
    runtime._sync_thread.join(timeout=2)
    assert not runtime._sync_thread.is_alive()


def test_stop_experiment_queues_copy_after_writers_close_and_keeps_poll_responsive(sync_harness):
    harness, sync = sync_harness
    runtime = harness.runtime
    experiment = runtime.start_experiment("alpha", _TIME)
    assert harness.command(CommandCode.START_TRACKING, 1).status is StatusCode.OK
    response = harness.command(CommandCode.STOP_EXPERIMENT, 2)
    assert response.status is StatusCode.OK
    assert not response.tracking and not response.experiment_active
    assert sync.entered.wait(1)
    assert runtime.view().busy and runtime.view().sync_active
    assert not harness.command(CommandCode.POLL, 3).tracking
    assert harness.command(CommandCode.START_EXPERIMENT, 4, subject="beta",
                           controller_timestamp_ns=_TIME).status is StatusCode.INVALID_STATE
    runtime.ensure_feed()
    assert harness.owners == 0
    assert json.loads((experiment.directory / "experiment.json").read_text())["status"] == "closed"
    finish_sync(runtime, sync)
    assert not runtime.view().busy


def test_replacing_experiment_excludes_new_active_directory_from_sync(sync_harness):
    harness, sync = sync_harness
    runtime = harness.runtime
    runtime.start_experiment("alpha", _TIME)
    second = runtime.start_experiment("alpha", _TIME + 1_000_000_000)
    assert sync.entered.wait(1)
    assert sync.calls == [(second.directory,)]
    finish_sync(runtime, sync)
    runtime.open_tracking()
    assert runtime.view().experiment == second


@pytest.mark.parametrize("ack_first", [True, False])
def test_shutdown_waits_for_both_uart_ack_and_successful_sync(sync_harness, ack_first):
    harness, sync = sync_harness
    runtime = harness.runtime
    runtime.start_experiment("alpha", _TIME)
    assert harness.command(CommandCode.SHUTDOWN, 10).status is StatusCode.OK
    assert sync.entered.wait(1)
    assert runtime.snapshot().shutting_down
    if ack_first:
        runtime.after_response()
        assert harness.shutdown_calls == 0
        finish_sync(runtime, sync)
    else:
        finish_sync(runtime, sync)
        assert harness.shutdown_calls == 0
        runtime.after_response()
    assert harness.shutdown_calls == 1
    harness.command(CommandCode.SHUTDOWN, 10)
    runtime.after_response()
    assert harness.shutdown_calls == 1


@pytest.mark.parametrize("choice", ["retry", "shutdown", "cancel"])
def test_failed_shutdown_sync_requires_explicit_choice(sync_harness, choice):
    harness, sync = sync_harness
    runtime = harness.runtime
    experiment = runtime.start_experiment("alpha", _TIME)
    sync.outcome[0] = SyncReport(SyncStatus.FAILED, "server unavailable")
    harness.command(CommandCode.SHUTDOWN, 11)
    runtime.after_response()
    finish_sync(runtime, sync)
    assert harness.shutdown_calls == 0
    assert runtime.view().shutdown_decision and runtime.snapshot().error
    assert experiment.directory.exists()
    sync.outcome[0] = SyncReport(SyncStatus.SUCCESS, "copied")
    runtime.resolve_sync_failure(choice)
    if choice == "retry":
        finish_sync(runtime, sync)
        assert len(sync.calls) == 2
    if choice == "cancel":
        assert not runtime.snapshot().shutting_down
        assert harness.command(CommandCode.SHUTDOWN, 11).status is StatusCode.INVALID_STATE
        runtime.after_response()
        assert harness.shutdown_calls == 0
    else:
        assert harness.shutdown_calls == 1


def test_cancelling_sync_cancels_shutdown_and_identical_uart_retries(sync_harness):
    harness, sync = sync_harness
    runtime = harness.runtime
    runtime.start_experiment("alpha", _TIME)
    harness.command(CommandCode.SHUTDOWN, 12)
    runtime.after_response()
    runtime.cancel_sync()
    finish_sync(runtime, sync)
    assert not runtime.view().shutdown_requested
    assert not runtime.view().shutdown_decision
    assert harness.shutdown_calls == 0
    assert harness.command(CommandCode.SHUTDOWN, 12).status is StatusCode.INVALID_STATE
    assert runtime.view().sync_report.status is SyncStatus.CANCELLED


def test_stopping_without_an_active_experiment_retries_existing_data(sync_harness):
    harness, sync = sync_harness
    assert harness.command(CommandCode.STOP_EXPERIMENT, 13).status is StatusCode.OK
    assert sync.entered.wait(1)
    finish_sync(harness.runtime, sync)
    assert sync.calls == [()]


def test_closing_interface_after_failed_shutdown_sync_does_not_power_off(sync_harness):
    harness, sync = sync_harness
    runtime = harness.runtime
    sync.outcome[0] = SyncReport(SyncStatus.FAILED, "server unavailable")
    harness.command(CommandCode.SHUTDOWN, 14)
    runtime.after_response()
    finish_sync(runtime, sync)
    runtime.close()
    assert harness.shutdown_calls == 0


def test_navigation_and_tracking_keep_one_camera_owner_and_experiment_identity(harness):
    runtime = harness.runtime
    assert runtime.view().mode == "home"
    runtime.ensure_feed()
    home_feed = runtime.view().feed
    assert harness.owners == 1
    runtime.ensure_feed()
    assert len(harness.cameras) == 1
    runtime.show_subjects()
    assert runtime.view().mode == "subjects"
    assert runtime.view().feed is home_feed
    experiment = runtime.start_experiment("subject_A", _TIME)
    assert home_feed.closed
    assert harness.owners == 0
    runtime.ensure_feed()
    menu_feed = runtime.view().feed
    runtime.open_tracking()
    view = runtime.view()
    service = harness.services[-1]
    assert menu_feed.closed
    assert view.mode == "tracking"
    assert view.experiment is experiment
    assert view.preview is service.preview
    assert service.preview.options["embedded"] is True
    assert service.preview.started
    assert harness.owners == 1
    assert not runtime.snapshot().tracking
    assert not runtime.snapshot().recording
    runtime.toggle_tracking()
    assert runtime.snapshot().tracking and runtime.snapshot().recording
    runtime.toggle_tracking()
    assert not runtime.snapshot().tracking
    assert not service.preview.closed
    runtime.back()
    assert service.closed
    assert harness.owners == 0
    assert runtime.view().mode == "experiment"
    assert runtime.view().experiment is experiment
    runtime.begin_calibration()
    calibration = runtime.view().calibration
    assert runtime.view().mode == "calibration"
    assert runtime.view().experiment is experiment
    assert runtime.snapshot().calibrating
    assert harness.owners == 1
    runtime.save_calibration()
    assert calibration.saved
    assert calibration.close_calls == 1
    assert calibration.feed.closed
    assert runtime.view().mode == "experiment"
    assert harness.owners == 0
    runtime.back()
    assert runtime.view().mode == "subjects"
    assert runtime.view().experiment is None
    assert (
        json.loads((experiment.directory / "experiment.json").read_text())["status"] == "closed"
    )
    runtime.back()
    assert runtime.view().mode == "home"
    runtime.close()
    runtime.ensure_feed()
    assert harness.owners == 0
    assert harness.clock_calls == []


def test_back_from_calibration_cancels_without_saving(harness):
    runtime = harness.runtime
    experiment = runtime.start_experiment("subject_A", _TIME)
    original = experiment.calibration_path.read_bytes()
    runtime.begin_calibration()
    calibration = runtime.view().calibration

    runtime.back()

    assert not calibration.saved
    assert calibration.close_calls == 1
    assert calibration.feed.closed
    assert experiment.calibration_path.read_bytes() == original
    assert runtime.view().mode == "experiment"
    runtime.close()


@pytest.mark.parametrize("action", ["end_experiment", "close"])
def test_ending_calibration_cancels_optimization_before_discarding_session(harness, action):
    runtime = harness.runtime
    runtime.start_experiment("subject_A", _TIME)
    runtime.begin_calibration()
    calibration = runtime.view().calibration

    getattr(runtime, action)()

    assert calibration.close_calls == 1
    assert calibration.feed.closed
    assert runtime.view().calibration is None
    assert not calibration.saved
    runtime.close()
    assert calibration.close_calls == 1


def test_calibration_cancellation_survives_camera_shutdown_failure(harness):
    runtime = harness.runtime
    runtime.start_experiment("subject_A", _TIME)
    runtime.begin_calibration()
    calibration = runtime.view().calibration
    calibration.feed.fail_close = True

    with pytest.raises(RuntimeError, match="feed shutdown"):
        runtime.back()

    assert calibration.close_calls == 1
    assert runtime.view().calibration is calibration
    assert harness.owners == 1
    calibration.feed.fail_close = False
    runtime.close()
    assert calibration.close_calls == 2
    assert harness.owners == 0


def test_uart_experiment_tracking_and_stop_are_idempotent_and_ordered(harness):
    runtime = harness.runtime
    runtime.ensure_feed()
    command = CommandPacket(
        CommandCode.START_EXPERIMENT, 1, controller_timestamp_ns=_TIME, subject="subject_A"
    )
    result = runtime.execute(command)
    experiment = runtime.view().experiment
    assert result.status is StatusCode.OK
    assert result.experiment_active
    assert not result.tracking
    assert harness.clock_calls == [_TIME]
    assert experiment.started_timestamp_ns == _TIME
    assert experiment.subject == "subject_A"
    assert runtime.execute(command).status is StatusCode.OK
    assert runtime.view().experiment is experiment
    assert harness.clock_calls == [_TIME]
    conflict = runtime.execute(replace(command, subject="subject_B"))
    assert conflict.status is StatusCode.SEQUENCE_CONFLICT
    assert not (harness.store.data_root / "subject_B").exists()
    assert harness.command(CommandCode.START_TRACKING, 2).tracking
    service = harness.services[-1]
    harness.command(CommandCode.START_TRACKING, 2)
    assert service.start_calls == 1
    stopped = harness.command(CommandCode.STOP_TRACKING, 3)
    assert not stopped.tracking and not stopped.recording
    assert stopped.experiment_active
    assert runtime.view().mode == "tracking"
    assert harness.owners == 1
    assert not service.closed
    assert harness.command(CommandCode.START_TRACKING, 4).tracking
    assert not harness.command(CommandCode.STOP_RECORDING, 5).recording
    assert harness.command(CommandCode.START_RECORDING, 6).recording
    ended = harness.command(CommandCode.STOP_EXPERIMENT, 7)
    assert ended.status is StatusCode.OK
    assert not ended.experiment_active
    assert service.closed
    assert harness.owners == 0
    assert runtime.view().mode == "home"
    manifest = (experiment.directory / "experiment.json").read_bytes()
    harness.command(CommandCode.STOP_EXPERIMENT, 7)
    assert (experiment.directory / "experiment.json").read_bytes() == manifest
    runtime.ensure_feed()
    assert harness.owners == 1
    runtime.close()


def test_numbered_experiments_blocks_and_retry_safe_overwrite(harness):
    runtime = harness.runtime
    command = CommandPacket(CommandCode.START_EXPERIMENT, 101, _TIME,
                            subject="subject_A", experiment_number=4)
    assert runtime.execute(command).status is StatusCode.OK
    experiment = runtime.view().experiment
    assert experiment.name == "exp_subject_A_20261014_4_174640"
    start = CommandPacket(CommandCode.START_TRACKING, 102, block_number=8)
    assert runtime.execute(start).tracking
    block = runtime.view().block
    assert block.number == 8
    marker = block.directory / "video.mkv"
    marker.write_bytes(b"keep on retry")
    runtime.execute(start)
    assert marker.read_bytes() == b"keep on retry"
    conflict = runtime.execute(replace(start, block_number=9))
    assert conflict.status is StatusCode.SEQUENCE_CONFLICT
    assert marker.exists()
    service = harness.services[-1]
    stops = service.stop_calls
    assert runtime.execute(replace(start, sequence=103)).tracking
    assert service.stop_calls == stops + 1  # Writer stopped before explicit replacement.
    assert not marker.exists()
    assert block.calibration_path.exists()
    runtime.toggle_tracking()
    runtime.toggle_tracking()
    assert runtime.view().block.number == 9
    assert block.directory.exists()
    runtime.close()


def test_saved_run_restart_creates_next_block_and_snapshots_loaded_calibration(harness):
    runtime = harness.runtime
    experiment = runtime.start_experiment("subject_A", _TIME)
    runtime.open_tracking()
    loaded = harness.services[-1].layout
    # An external file edit after loading cannot mislabel the calibration used by this service.
    replace(loaded, camera_rotations=(90, 0, 0, 0)).save(experiment.calibration_path)
    runtime.toggle_tracking()
    first = runtime.view().block
    assert RoiLayout.load(first.calibration_path) == loaded
    runtime.execute(CommandPacket(CommandCode.STOP_RECORDING, 10))
    assert runtime.snapshot().tracking
    runtime.execute(CommandPacket(CommandCode.START_RECORDING, 11))
    assert runtime.view().block.number == first.number + 1
    assert first.calibration_path.exists()
    runtime.execute(CommandPacket(CommandCode.START_RECORDING, 12, block_number=5))
    assert runtime.view().block.number == 5
    runtime.back()
    assert runtime.view().block is None
    runtime.open_tracking()
    runtime.toggle_tracking()
    assert runtime.view().block.number == 6
    assert RoiLayout.load(runtime.view().block.calibration_path).camera_rotations[0] == 90
    runtime.close()


def test_failed_stop_cannot_overwrite_numbered_block(harness, monkeypatch):
    runtime = harness.runtime
    runtime.start_experiment("subject_A", _TIME)
    runtime.execute(CommandPacket(CommandCode.START_TRACKING, 1, block_number=3))
    block = runtime.view().block
    marker = block.directory / "keep.mkv"
    marker.write_bytes(b"keep")

    def failed_stop():
        raise OSError("writer still open")

    with monkeypatch.context() as patch:
        patch.setattr(harness.services[-1], "stop_tracking", failed_stop)
        status = runtime.execute(CommandPacket(CommandCode.START_TRACKING, 2, block_number=3))
        assert status.status is StatusCode.INTERNAL_ERROR
        assert marker.read_bytes() == b"keep"
        assert runtime.view().block is block
    runtime.close()


def test_uart_replaces_active_experiment_and_finishes_previous(harness):
    runtime = harness.runtime
    first = runtime.start_experiment("subject_A", _TIME)
    runtime.open_tracking()
    runtime.toggle_tracking()
    previous_service = harness.services[-1]

    result = harness.command(
        CommandCode.START_EXPERIMENT,
        1,
        controller_timestamp_ns=_TIME + 1_000_000_000,
        subject="subject_B",
    )

    assert result.status is StatusCode.OK
    assert result.experiment_active and not result.tracking
    assert previous_service.closed
    assert runtime.view().mode == "experiment"
    assert runtime.view().experiment.subject == "subject_B"
    assert json.loads((first.directory / "experiment.json").read_text())["status"] == "closed"
    runtime.close()


def test_uart_tracking_rejected_without_experiment_and_during_calibration(harness):
    assert harness.command(CommandCode.START_TRACKING, 1).status is StatusCode.INVALID_STATE
    assert harness.owners == 0
    runtime = harness.runtime
    runtime.start_experiment("subject_A", _TIME)
    runtime.begin_calibration()
    feed = runtime.view().feed

    for sequence, code in enumerate(
        (CommandCode.START_TRACKING, CommandCode.STOP_TRACKING, CommandCode.START_RECORDING),
        start=2,
    ):
        result = harness.command(code, sequence)
        assert result.status is StatusCode.INVALID_STATE
        assert result.calibrating and result.experiment_active
    assert runtime.view().feed is feed
    assert harness.services == []
    assert harness.command(CommandCode.STOP_EXPERIMENT, 5).status is StatusCode.OK
    assert feed.closed
    assert runtime.view().mode == "home"
    runtime.close()


def test_clock_failure_preserves_active_tracking_and_creates_nothing(harness):
    runtime = harness.runtime
    first = runtime.start_experiment("subject_A", _TIME)
    runtime.open_tracking()
    runtime.toggle_tracking()
    service = harness.services[-1]
    manifest = (first.directory / "experiment.json").read_bytes()
    harness.clock_error = PermissionError("clock permission denied")

    result = harness.command(
        CommandCode.START_EXPERIMENT,
        1,
        controller_timestamp_ns=_TIME + 1_000_000_000,
        subject="subject_B",
    )

    assert result.status is StatusCode.INTERNAL_ERROR
    assert result.tracking and result.experiment_active
    assert runtime.view().experiment is first
    assert runtime.view().mode == "tracking"
    assert not service.closed
    assert harness.owners == 1
    assert not (harness.store.data_root / "subject_B").exists()
    assert (first.directory / "experiment.json").read_bytes() == manifest
    assert "clock permission denied" in runtime.view().error_message
    assert harness.command(CommandCode.CLEAR_ERROR, 2).status is StatusCode.OK
    assert runtime.view().error_message is None
    assert service.clear_calls == 1
    runtime.close()


@pytest.mark.parametrize("timestamp", [0, -1, True, 10**30])
def test_invalid_remote_time_never_changes_clock_or_storage(harness, timestamp):
    result = harness.command(
        CommandCode.START_EXPERIMENT, 1, controller_timestamp_ns=timestamp, subject="subject_A"
    )
    assert result.status is StatusCode.INVALID_STATE
    assert harness.clock_calls == []
    assert not harness.store.data_root.exists()


def test_remote_collision_does_not_change_clock_or_active_experiment(harness):
    runtime = harness.runtime
    first = runtime.start_experiment("subject_A", _TIME)
    runtime.open_tracking()
    service = harness.services[-1]

    result = harness.command(
        CommandCode.START_EXPERIMENT, 1, controller_timestamp_ns=_TIME, subject="subject_A",
        experiment_number=first.number,
    )

    assert result.status is StatusCode.INVALID_STATE
    assert harness.clock_calls == []
    assert runtime.view().experiment is first
    assert not service.closed
    runtime.close()


def test_tracking_forces_recordings_under_experiment_even_after_config_edit(harness, tmp_path):
    runtime = harness.runtime
    experiment = runtime.start_experiment("subject_A", _TIME)
    config = AppConfig.load(experiment.config_path)
    external = tmp_path / "external"
    replace(
        config,
        roi_config=str(external / "calibration.json"),
        recording=replace(
            config.recording, directory=str(external), layout="full", pose_cameras=(1, 3)
        ),
    ).save(experiment.config_path)

    runtime.open_tracking()
    service = harness.services[-1]

    assert Path(service.config.recording.directory) == experiment.directory
    assert Path(service.camera.config.recording.directory) == experiment.directory
    assert service.config.recording.layout == "hands_eyes"
    assert service.config.recording.pose_cameras == (1, 3)
    assert service.layout == RoiLayout.load(experiment.calibration_path)
    assert not external.exists()
    runtime.close()


def test_video_input_disables_recording_without_disabling_tracking(harness, tmp_path):
    playback = Harness(harness.store, video_path=tmp_path / "source.mp4")
    runtime = playback.runtime
    runtime.start_experiment("subject_A", _TIME)
    runtime.open_tracking()
    runtime.toggle_tracking()

    assert runtime.snapshot().tracking
    assert not runtime.snapshot().recording
    assert playback.services[-1].config.recording.record_on_tracking is False
    runtime.close()


def test_feed_shutdown_failure_prevents_second_camera_owner(harness):
    runtime = harness.runtime
    runtime.ensure_feed()
    feed = runtime.view().feed
    feed.fail_close = True

    with pytest.raises(RuntimeError, match="feed shutdown"):
        runtime.start_experiment("subject_A", _TIME)

    runtime.ensure_feed()
    assert runtime.view().feed is feed
    assert harness.owners == 1
    assert len(harness.cameras) == 1
    assert not harness.store.data_root.exists()
    feed.fail_close = False
    runtime.close()


def test_service_shutdown_failure_retains_owner_and_blocks_replacement(harness):
    runtime = harness.runtime
    first = runtime.start_experiment("subject_A", _TIME)
    runtime.open_tracking()
    service = harness.services[-1]
    service.fail_close = True

    with pytest.raises(RuntimeError, match="recording finalization"):
        runtime.back()
    runtime.ensure_feed()
    with pytest.raises(RuntimeError, match="recording finalization"):
        runtime.begin_calibration()

    assert runtime.view().experiment is first
    assert runtime.view().mode == "tracking"
    assert harness.owners == 1
    assert len(harness.cameras) == 1
    service.fail_close = False
    runtime.close()
    assert harness.owners == 0


def test_poll_and_view_do_not_wait_for_recording_shutdown(harness):
    runtime = harness.runtime
    runtime.start_experiment("subject_A", _TIME)
    runtime.open_tracking()
    service = harness.services[-1]
    service.close_entered = threading.Event()
    service.close_release = threading.Event()

    with ThreadPoolExecutor(max_workers=2) as pool:
        transition = pool.submit(runtime.back)
        assert service.close_entered.wait(1)
        try:
            poll = pool.submit(harness.command, CommandCode.POLL, 50)
            assert poll.result(timeout=0.5).experiment_active
            assert runtime.view().mode == "tracking"
            assert not transition.done()
        finally:
            service.close_release.set()
        transition.result(timeout=1)
    assert runtime.view().mode == "experiment"
    runtime.close()


def test_camera_feed_returns_frames_and_releases_camera(harness):
    camera = harness.camera(harness.runtime.base_config)
    feed = CameraFeed(camera)

    feed.start()
    try:
        assert np.all(feed.latest() == 42)
        assert harness.owners == 1
    finally:
        feed.close()

    assert camera.closed
    assert harness.owners == 0


def test_camera_feed_capture_error_is_reported_and_owner_can_be_closed(harness):
    camera = harness.camera(harness.runtime.base_config)

    def broken_capture():
        raise RuntimeError("sensor disconnected")

    camera.capture_preview = broken_capture
    feed = CameraFeed(camera)
    with pytest.raises(RuntimeError, match="sensor disconnected"):
        feed.start()
    feed.close()
    assert camera.closed
    assert harness.owners == 0


def test_no_tracking_or_calibration_without_selected_experiment(harness):
    with pytest.raises(InvalidStateError, match="experiment"):
        harness.runtime.open_tracking()
    with pytest.raises(InvalidStateError, match="experiment"):
        harness.runtime.begin_calibration()
    with pytest.raises(InvalidStateError, match="tracking view"):
        harness.runtime.toggle_tracking()
    assert harness.owners == 0


def test_closed_runtime_rejects_new_uart_actions(harness):
    harness.runtime.close()

    result = harness.command(
        CommandCode.START_EXPERIMENT, 1, controller_timestamp_ns=_TIME, subject="subject_A"
    )

    assert result.status is StatusCode.INVALID_STATE
    assert harness.clock_calls == []
    assert not harness.store.data_root.exists()


def test_service_constructor_failure_closes_newly_allocated_camera(harness):
    runtime = harness.runtime
    runtime.start_experiment("subject_A", _TIME)

    def failed_service(*args, **kwargs):
        raise RuntimeError("detector construction failed")

    runtime._service_factory = failed_service
    with pytest.raises(RuntimeError, match="detector construction"):
        runtime.open_tracking()

    assert harness.cameras[-1].closed
    assert runtime.view().mode == "experiment"
    assert runtime.view().preview is None


def test_feed_constructor_failure_closes_newly_allocated_camera(harness):
    runtime = harness.runtime

    def failed_feed(camera):
        raise RuntimeError("feed construction failed")

    runtime._feed_factory = failed_feed
    with pytest.raises(RuntimeError, match="feed construction"):
        runtime.ensure_feed()

    assert harness.cameras[-1].closed
    assert runtime.view().feed is None


def test_uart_pty_experiment_lifecycle_returns_64_byte_acks_and_scopes_artifacts(
    harness, monkeypatch
):
    from macaque_tracker import transport

    runtime = harness.runtime
    runtime.ensure_feed()
    master, slave = pty.openpty()
    ready = threading.Event()
    stop = threading.Event()
    errors = []
    configure = transport.configure_raw_uart

    def configured(*args, **kwargs):
        configure(*args, **kwargs)
        ready.set()

    monkeypatch.setattr(transport, "configure_raw_uart", configured)

    def recording_service(config, layout, **kwargs):
        service = harness.service(config, layout, **kwargs)
        start = service.start_tracking

        def start_and_save_artifacts():
            start()
            output = Path(service.session_config.recording.directory)
            (output / "test-video.mkv").write_bytes(b"fake encoded frames")
            (output / "test-tracking.csv").write_text("timestamp,x,y\n", encoding="utf-8")

        service.start_tracking = start_and_save_artifacts
        return service

    runtime._service_factory = recording_service
    server = UartServer(os.ttyname(slave), 115_200, ProtocolEngine(runtime))

    def serve():
        try:
            server.serve(stop)
        except BaseException as exc:  # noqa: BLE001 - surface failures in the test thread
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    def transact(command):
        packet = encode_command(command)
        assert len(packet) == TRANSACTION_SIZE == 64
        # Exercise UART framing with a fragmented record as well as real tty I/O.
        os.write(master, packet[:17])
        os.write(master, packet[17:])
        response = bytearray()
        deadline = time.monotonic() + 3
        while len(response) < TRANSACTION_SIZE:
            remaining = deadline - time.monotonic()
            assert remaining > 0 and select.select([master], [], [], remaining)[0]
            response.extend(os.read(master, TRANSACTION_SIZE - len(response)))
        decoded = decode_status(bytes(response))
        assert len(response) == 64
        assert decoded.request_sequence == command.sequence
        assert decoded.status is StatusCode.OK
        assert not decoded.flags & StatusFlag.ERROR
        return decoded, bytes(response)

    try:
        assert ready.wait(3)
        initial, _ = transact(CommandPacket(CommandCode.POLL, 1))
        assert not initial.flags & StatusFlag.EXPERIMENT_ACTIVE
        start = CommandPacket(
            CommandCode.START_EXPERIMENT,
            2,
            controller_timestamp_ns=_TIME,
            subject="subject_UART",
            experiment_number=3,
        )
        opened, opened_packet = transact(start)
        experiment = runtime.view().experiment
        assert opened.flags & StatusFlag.EXPERIMENT_ACTIVE
        assert opened.flags & StatusFlag.CONFIGURED
        assert not opened.flags & (StatusFlag.TRACKING | StatusFlag.RECORDING)
        assert experiment.directory.parent == harness.store.data_root / "subject_UART"
        assert experiment.started_timestamp_ns == _TIME
        assert experiment.number == 3
        assert experiment.calibration_path.is_file()
        assert AppConfig.load(experiment.config_path).recording.directory == str(
            experiment.directory
        )
        _, duplicate_packet = transact(start)
        assert duplicate_packet == opened_packet
        assert harness.clock_calls == [_TIME]
        assert len(list(experiment.directory.parent.iterdir())) == 1

        tracking, _ = transact(CommandPacket(CommandCode.START_TRACKING, 3, block_number=7))
        assert tracking.flags & StatusFlag.TRACKING
        assert tracking.flags & StatusFlag.RECORDING
        assert tracking.configured_eye_count == 1
        service = harness.services[-1]
        block = experiment.directory / "blocks" / "7"
        assert (block / "test-video.mkv").read_bytes() == b"fake encoded frames"
        assert (block / "test-tracking.csv").is_file()
        assert (block / "calibration.json").is_file()

        idle, _ = transact(CommandPacket(CommandCode.STOP_TRACKING, 4))
        assert idle.flags & StatusFlag.EXPERIMENT_ACTIVE
        assert not idle.flags & (StatusFlag.TRACKING | StatusFlag.RECORDING)
        assert runtime.view().mode == "tracking"
        assert service.preview.started and not service.preview.closed
        assert harness.owners == 1
        ended, _ = transact(CommandPacket(CommandCode.STOP_EXPERIMENT, 5))
        assert not ended.flags & StatusFlag.EXPERIMENT_ACTIVE
        assert runtime.view().mode == "home"
        assert service.closed
        assert harness.owners == 0
        manifest = json.loads((experiment.directory / "experiment.json").read_text())
        assert manifest["status"] == "closed"
        assert manifest["subject"] == "subject_UART"
        assert manifest["started_timestamp_ns"] == _TIME

        next_time = _TIME + 1_000_000_000
        second_open, _ = transact(
            CommandPacket(
                CommandCode.START_EXPERIMENT,
                6,
                controller_timestamp_ns=next_time,
                subject="subject_UART",
            )
        )
        second = runtime.view().experiment
        assert second.number == 4
        assert second_open.flags & StatusFlag.EXPERIMENT_ACTIVE
        assert second.directory != experiment.directory
        assert second.calibration_path.read_bytes() == experiment.calibration_path.read_bytes()
        assert not (second.directory / "test-video.mkv").exists()
        assert harness.clock_calls == [_TIME, next_time]
        transact(CommandPacket(CommandCode.STOP_EXPERIMENT, 7))
    finally:
        stop.set()
        thread.join(timeout=2)
        os.close(slave)
        os.close(master)
        runtime.close()
    assert not thread.is_alive()
    assert not errors


@pytest.mark.parametrize("mode", ["home", "experiment", "tracking", "calibration"])
def test_shutdown_finalizes_activity_before_poweroff_and_blocks_reopening(harness, mode):
    runtime = harness.runtime
    runtime.ensure_feed()
    experiment = None
    calibration = None
    if mode != "home":
        experiment = runtime.start_experiment("shutdown_subject", _TIME)
    if mode == "tracking":
        harness.command(CommandCode.START_TRACKING, 1)
        assert harness.services[-1].recording
    elif mode == "calibration":
        runtime.begin_calibration()
        calibration = runtime.view().calibration
    command = CommandPacket(CommandCode.SHUTDOWN, 10)
    engine = ProtocolEngine(runtime)
    status = decode_status(engine.handle(encode_command(command)))
    assert status.status is StatusCode.OK
    assert status.flags & StatusFlag.SHUTTING_DOWN
    assert not status.flags & (StatusFlag.TRACKING | StatusFlag.RECORDING)
    assert harness.owners == 0
    if calibration is not None:
        assert calibration.close_calls == 1
        assert runtime.view().calibration is None
    assert harness.shutdown_calls == 0  # Wait for the transport's reply attempt.
    if experiment is not None:
        manifest = json.loads((experiment.directory / "experiment.json").read_text())
        assert manifest["status"] == "closed"
    runtime.ensure_feed()
    assert harness.owners == 0
    for action in (runtime.show_subjects, runtime.begin_calibration, runtime.open_tracking,
                   runtime.back, runtime.toggle_tracking,
                   lambda: runtime.start_experiment("late_click", _TIME + 1)):
        with pytest.raises(InvalidStateError, match="shutting down"):
            action()
    assert harness.command(CommandCode.START_TRACKING, 11).status is StatusCode.INVALID_STATE
    engine.after_response()
    assert harness.shutdown_calls == 1
    assert runtime.view().shutdown_submitted
    assert runtime.execute(command).status is StatusCode.OK
    engine.after_response()
    assert harness.shutdown_calls == 1
    assert harness.command(CommandCode.START_EXPERIMENT, 10, subject="conflict",
                           controller_timestamp_ns=_TIME).status is StatusCode.SEQUENCE_CONFLICT


def test_shutdown_cleanup_failure_never_powers_off_and_new_sequence_can_retry(harness):
    runtime = harness.runtime
    runtime.start_experiment("shutdown_subject", _TIME)
    runtime.open_tracking()
    service = harness.services[-1]
    service.fail_close = True
    assert harness.command(CommandCode.SHUTDOWN, 1).status is StatusCode.INTERNAL_ERROR
    runtime.after_response()
    assert harness.shutdown_calls == 0
    assert not runtime.view().shutdown_requested
    assert runtime.view().experiment is not None
    service.fail_close = False
    assert harness.command(CommandCode.SHUTDOWN, 1).status is StatusCode.INTERNAL_ERROR
    assert harness.command(CommandCode.SHUTDOWN, 2).status is StatusCode.OK
    runtime.after_response()
    assert harness.shutdown_calls == 1


def test_poweroff_failure_is_visible_and_duplicate_does_not_retry(harness):
    runtime = harness.runtime
    harness.shutdown_error = PermissionError("Access denied")
    assert harness.command(CommandCode.SHUTDOWN, 1).status is StatusCode.OK
    runtime.after_response()
    assert harness.shutdown_calls == 1
    assert not runtime.view().shutdown_requested
    assert not runtime.view().shutdown_submitted
    assert "Access denied" in runtime.view().error_message
    assert runtime.snapshot().status is StatusCode.INTERNAL_ERROR
    assert harness.command(CommandCode.SHUTDOWN, 1).status is StatusCode.INTERNAL_ERROR
    runtime.after_response()
    assert harness.shutdown_calls == 1
    harness.shutdown_error = None
    assert harness.command(CommandCode.SHUTDOWN, 2).status is StatusCode.OK
    runtime.after_response()
    assert harness.shutdown_calls == 2


def test_corrupted_shutdown_packet_cannot_power_off(harness):
    packet = bytearray(encode_command(CommandPacket(CommandCode.SHUTDOWN, 1)))
    packet[-1] ^= 1
    engine = ProtocolEngine(harness.runtime)
    status = decode_status(engine.handle(bytes(packet)))
    assert status.status is StatusCode.BAD_CRC
    engine.after_response()
    assert harness.shutdown_calls == 0
    assert not harness.runtime.view().shutdown_requested


@pytest.mark.parametrize("reply_fails", [False, True])
def test_uart_shutdown_replies_and_drains_before_poweroff_even_if_reader_disappears(
    harness, monkeypatch, reply_fails
):
    from macaque_tracker import transport

    runtime = harness.runtime
    runtime.ensure_feed()
    master, slave = pty.openpty()
    ready, stop, submitted = threading.Event(), threading.Event(), threading.Event()
    events, errors = [], []
    configure, drain = transport.configure_raw_uart, transport.termios.tcdrain

    def configured(*args, **kwargs):
        configure(*args, **kwargs)
        ready.set()

    def drained(fd):
        events.append("drain")
        drain(fd)

    def power_off():
        harness.power_off()
        events.append("poweroff")
        stop.set()
        submitted.set()

    runtime._shutdown_executor = power_off
    monkeypatch.setattr(transport, "configure_raw_uart", configured)
    monkeypatch.setattr(transport.termios, "tcdrain", drained)
    server = UartServer(os.ttyname(slave), 115_200, ProtocolEngine(runtime))
    write_all = server._write_all

    def write_reply(fd, data):
        status = decode_status(data)
        if status.flags & StatusFlag.SHUTTING_DOWN:
            events.append("reply")
            if reply_fails:
                raise BrokenPipeError("reader disappeared")
        write_all(fd, data)

    monkeypatch.setattr(server, "_write_all", write_reply)

    def serve():
        try:
            server.serve(stop)
        except Exception as exc:  # noqa: BLE001 - surface test worker failures
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        assert ready.wait(2)
        os.write(master, encode_command(CommandPacket(CommandCode.POLL, 1)))
        assert select.select([master], [], [], 2)[0]
        assert decode_status(os.read(master, 64)).request_sequence == 1
        assert events == []  # No drain/power-control work for normal telemetry.
        os.write(master, encode_command(CommandPacket(CommandCode.SHUTDOWN, 2)))
        if not reply_fails:
            assert select.select([master], [], [], 2)[0]
            status = decode_status(os.read(master, 64))
            assert status.request_sequence == 2
            assert status.flags & StatusFlag.SHUTTING_DOWN
        assert submitted.wait(2)
        thread.join(2)
        assert not thread.is_alive()
        assert harness.shutdown_calls == 1
        if reply_fails:
            assert events == ["reply", "poweroff"]
            assert len(errors) == 1 and isinstance(errors[0], BrokenPipeError)
        else:
            assert events == ["reply", "drain", "poweroff"]
            assert not errors
    finally:
        stop.set()
        thread.join(2)
        os.close(master)
        os.close(slave)
        runtime.close()


def test_menu_feed_and_tracking_preview_load_persisted_camera_rotations(harness):
    runtime = harness.runtime
    path = harness.store.base_config_path.parent / runtime.base_config.roi_config
    layout = RoiLayout.load(path)
    replace(layout, camera_rotations=(90, 180, 270, 0)).save(path)
    runtime.ensure_feed()
    assert runtime.view().camera_rotations == (90, 180, 270, 0)
    runtime.start_experiment("rotated", _TIME)
    runtime.ensure_feed()
    assert runtime.view().camera_rotations == (90, 180, 270, 0)
    runtime.open_tracking()
    assert runtime.view().preview.options["camera_rotations"] == (90, 180, 270, 0)
    runtime.close()
