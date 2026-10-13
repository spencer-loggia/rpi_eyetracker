from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import numpy as np

from macaque_tracker.config import AppConfig, RecordingConfig, RoiLayout
from macaque_tracker.models import (
    AnalysisFrame,
    EyeMeasurement,
    FrameResult,
    NormalizedRoi,
)
from macaque_tracker.protocol import CommandCode, CommandPacket, StatusCode
from macaque_tracker.service import EyeTrackingService


class FakeCamera:
    def __init__(self) -> None:
        self.started = False
        self.recording = False
        self.closed = False
        self.sequence = 0
        self.start_calls = 0

    def start(self) -> None:
        self.start_calls += 1
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True

    def capture_analysis(self) -> AnalysisFrame:
        if not self.started:
            raise RuntimeError("not started")
        time.sleep(0.003)
        self.sequence += 1
        return AnalysisFrame(
            self.sequence,
            time.monotonic_ns(),
            ((0, np.zeros((40, 60), dtype=np.uint8)),),
        )

    def start_recording(self, destination):
        self.recording = True
        return destination

    def stop_recording(self) -> None:
        self.recording = False

class FakeTracker:
    def __init__(self) -> None:
        self.processed: list[int] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def process(self, frame: AnalysisFrame, dropped_frames: int = 0) -> FrameResult:
        self.processed.append(frame.frame_sequence)
        now = time.monotonic_ns()
        return FrameResult(
            frame_sequence=frame.frame_sequence,
            sensor_timestamp_ns=frame.sensor_timestamp_ns,
            produced_timestamp_ns=now,
            eyes=(EyeMeasurement(0, 10.0, 20.0, 12.0, 0.9, True, False),),
            dropped_analysis_frames=dropped_frames,
            processing_time_us=50,
        )


class FailingCaptureCamera(FakeCamera):
    def capture_analysis(self) -> AnalysisFrame:
        raise RuntimeError("sensor disconnected")


class FailingRecordingCamera(FakeCamera):
    def start_recording(self, destination):
        raise RuntimeError("encoder unavailable")


class FakePreview:
    def __init__(self, *, fail_publish: bool = False) -> None:
        self.started = False
        self.closed = False
        self.error_message = None
        self.start_calls = 0
        self.stop_calls = 0
        self.published = threading.Event()
        self.fail_publish = fail_publish
        self.frames: list[tuple[AnalysisFrame, FrameResult | None]] = []

    def start(self) -> None:
        self.start_calls += 1
        self.started = True
        self.closed = False

    def publish(self, frame: AnalysisFrame, result: FrameResult) -> None:
        self.frames.append((frame, result))
        self.published.set()
        if self.fail_publish:
            raise RuntimeError("display disconnected")

    def stop(self) -> None:
        self.stop_calls += 1
        self.started = False
        self.closed = True


def _service() -> EyeTrackingService:
    config = replace(
        AppConfig(),
        recording=RecordingConfig(record_on_tracking=False),
    )
    layout = RoiLayout(
        rois=(NormalizedRoi(0, "eye", 0.1, 0.1, 0.2, 0.2),),
        source_width=config.camera.analysis_width,
        source_height=config.camera.analysis_height,
    )
    return EyeTrackingService(config, layout, camera=FakeCamera(), tracker=FakeTracker())


def _service_with_camera(
    camera: FakeCamera,
    *,
    record_on_tracking: bool = False,
    recording_directory: str = "recordings",
) -> EyeTrackingService:
    config = replace(
        AppConfig(),
        recording=RecordingConfig(
            record_on_tracking=record_on_tracking,
            directory=recording_directory,
        ),
    )
    layout = RoiLayout(
        rois=(NormalizedRoi(0, "eye", 0.1, 0.1, 0.2, 0.2),),
        source_width=config.camera.analysis_width,
        source_height=config.camera.analysis_height,
    )
    return EyeTrackingService(config, layout, camera=camera, tracker=FakeTracker())


def test_start_poll_stop_state_flow() -> None:
    service = _service()
    try:
        started = service.execute(CommandPacket(CommandCode.START_TRACKING, 10))
        assert started.tracking
        deadline = time.monotonic() + 1.0
        while service.snapshot().result is None and time.monotonic() < deadline:
            time.sleep(0.005)
        polled = service.execute(CommandPacket(CommandCode.POLL, 12))
        assert polled.result is not None
        stopped = service.execute(CommandPacket(CommandCode.STOP_TRACKING, 11))
        assert not stopped.tracking
    finally:
        service.close()


def test_same_sequence_different_command_is_conflict() -> None:
    service = _service()
    try:
        service.execute(CommandPacket(CommandCode.START_TRACKING, 4))
        conflict = service.execute(CommandPacket(CommandCode.STOP_TRACKING, 4))
        assert conflict.status is StatusCode.SEQUENCE_CONFLICT
        assert conflict.error
    finally:
        service.close()


def test_delayed_duplicate_is_idempotent_after_an_intervening_command() -> None:
    service = _service()
    try:
        start = CommandPacket(CommandCode.START_TRACKING, 30)
        service.execute(start)
        service.execute(CommandPacket(CommandCode.STOP_TRACKING, 31))

        replay = service.execute(start)
        assert replay.status is StatusCode.OK
        assert not replay.tracking
    finally:
        service.close()


def test_invalid_command_retry_preserves_error_outcome() -> None:
    service = _service()
    try:
        command = CommandPacket(CommandCode.START_RECORDING, 7)
        first = service.execute(command)
        retry = service.execute(command)
        assert first.status is StatusCode.INVALID_STATE
        assert retry.status is StatusCode.INVALID_STATE
        assert first.error and retry.error
    finally:
        service.close()


def test_capture_failure_stops_camera_and_sets_internal_error() -> None:
    camera = FailingCaptureCamera()
    service = _service_with_camera(camera)
    try:
        service.start_tracking()
        deadline = time.monotonic() + 1.0
        while service.tracking and time.monotonic() < deadline:
            time.sleep(0.005)

        snapshot = service.snapshot()
        assert not snapshot.tracking
        assert not camera.started
        assert snapshot.status is StatusCode.INTERNAL_ERROR
        assert snapshot.error
        assert "sensor disconnected" in (service.error_message or "")

        stopped = service.execute(CommandPacket(CommandCode.STOP_TRACKING, 21))
        assert stopped.status is StatusCode.OK
        assert stopped.error
        assert not stopped.tracking

        polled = service.execute(CommandPacket(CommandCode.POLL, 23))
        assert polled.status is StatusCode.INTERNAL_ERROR

        cleared = service.execute(CommandPacket(CommandCode.CLEAR_ERROR, 22))
        assert cleared.status is StatusCode.OK
        assert not cleared.error
    finally:
        service.close()


def test_recording_start_failure_stops_camera(tmp_path) -> None:
    camera = FailingRecordingCamera()
    service = _service_with_camera(
        camera,
        record_on_tracking=True,
        recording_directory=str(tmp_path),
    )
    try:
        try:
            service.start_tracking()
        except RuntimeError as exc:
            assert "encoder unavailable" in str(exc)
        else:  # pragma: no cover - makes a missing failure explicit
            raise AssertionError("start_tracking unexpectedly succeeded")

        assert not service.tracking
        assert not camera.started
        assert not camera.recording
    finally:
        service.close()


def test_preview_stays_live_across_tracking_and_recording_transitions(tmp_path) -> None:
    config = replace(AppConfig(), recording=RecordingConfig(directory=str(tmp_path / "records")))
    layout = RoiLayout(
        rois=(NormalizedRoi(0, "eye", 0.1, 0.1, 0.2, 0.2),),
        source_width=config.camera.analysis_width,
        source_height=config.camera.analysis_height,
    )
    preview = FakePreview()
    camera = FakeCamera()
    tracker = FakeTracker()
    service = EyeTrackingService(
        config,
        layout,
        camera=camera,
        tracker=tracker,
        preview=preview,
    )
    try:
        service.start_preview()
        _wait_for_preview(preview, idle=True)
        assert camera.started and not camera.recording
        assert not service.tracking and not service.recording
        assert service.snapshot().result is None
        assert tracker.processed == []
        assert not (tmp_path / "records").exists()

        service.start_tracking()
        _wait_for_preview(preview, idle=False)
        assert service.tracking
        assert service.recording and camera.recording
        assert preview.start_calls == 1

        service.stop_recording()
        previous_frame = preview.frames[-1][0].frame_sequence
        _wait_for_preview(preview, idle=False, after=previous_frame)
        assert service.tracking and not service.recording
        assert preview.stop_calls == 0
        service.start_recording()

        service.stop_tracking()
        assert not service.tracking and not service.recording
        assert not camera.recording and camera.started
        assert service.snapshot().result is None
        assert preview.stop_calls == 0 and not preview.closed
        _wait_for_preview(preview, idle=True)
        processed_count = len(tracker.processed)
        previous_frame = preview.frames[-1][0].frame_sequence
        _wait_for_preview(preview, idle=True, after=previous_frame + 2)
        assert len(tracker.processed) == processed_count
        for metadata in (tmp_path / "records").glob("*.jsonl"):
            rows = [json.loads(line) for line in metadata.read_text().splitlines()]
            assert rows[-1]["type"] == "session_end"
            assert all(
                row["frame_sequence"] <= tracker.processed[-1]
                for row in rows if row["type"] == "frame"
            )

        service.start_tracking()
        _wait_for_preview(preview, idle=False, after=previous_frame)
        assert service.tracking and service.recording
        assert service.snapshot().result.frame_sequence > previous_frame
        assert camera.start_calls == 1
        assert tracker.resets == 2
        assert preview.start_calls == 1
    finally:
        service.close()
    assert preview.stop_calls == 1 and preview.closed
    assert not camera.started and camera.closed


def _wait_for_preview(preview, *, idle: bool, after: int = -1) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if preview.frames:
            frame, result = preview.frames[-1]
            if frame.frame_sequence > after and (result is None) == idle:
                return
        time.sleep(0.005)
    raise AssertionError("Preview did not publish the expected fresh frame")


def test_start_tracking_discards_an_idle_capture_already_in_flight() -> None:
    entered = threading.Event()
    release = threading.Event()

    class HeldCamera(FakeCamera):
        def capture_analysis(self):
            if self.sequence == 0:
                entered.set()
                assert release.wait(1.0)
            return super().capture_analysis()

    service = _service_with_camera(HeldCamera())
    preview = FakePreview()
    service._preview = preview
    try:
        service.start_preview()
        assert entered.wait(1.0)
        service.start_tracking()
        release.set()
        _wait_for_preview(preview, idle=False)
        assert service._tracker.processed[0] > 1
    finally:
        release.set()
        service.close()


def test_stop_waits_for_current_result_then_publishes_only_idle_frames() -> None:
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    errors = []

    class HeldTracker(FakeTracker):
        def process(self, *args, **kwargs):
            entered.set()
            assert release.wait(1.0)
            return super().process(*args, **kwargs)

    service = _service()
    preview = FakePreview()
    service._preview = preview
    service._tracker = HeldTracker()

    def stop():
        try:
            service.stop_tracking()
        except Exception as exc:  # noqa: BLE001 - surface worker failures in the test
            errors.append(exc)
        finally:
            completed.set()

    thread = threading.Thread(target=stop)
    try:
        service.start_tracking()
        assert entered.wait(1.0)
        thread.start()
        assert not completed.wait(0.02)
        release.set()
        assert completed.wait(1.0)
        _wait_for_preview(preview, idle=True)
        count = len(preview.frames)
        _wait_for_preview(preview, idle=True, after=preview.frames[-1][0].frame_sequence + 2)
        assert all(result is None for _, result in preview.frames[count:])
        assert service.snapshot().result is None
        assert not errors
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(1.0)
        service.close()


def test_preview_failure_does_not_stop_tracking() -> None:
    config = replace(AppConfig(), recording=RecordingConfig(record_on_tracking=False))
    layout = RoiLayout(
        rois=(NormalizedRoi(0, "eye", 0.1, 0.1, 0.2, 0.2),),
        source_width=config.camera.analysis_width,
        source_height=config.camera.analysis_height,
    )
    preview = FakePreview(fail_publish=True)
    service = EyeTrackingService(
        config,
        layout,
        camera=FakeCamera(),
        tracker=FakeTracker(),
        preview=preview,
    )
    try:
        service.start_tracking()
        assert preview.published.wait(1.0)
        assert service.tracking
        assert "display disconnected" in (service.preview_error_message or "")
        assert service.error_message is None
    finally:
        service.close()
