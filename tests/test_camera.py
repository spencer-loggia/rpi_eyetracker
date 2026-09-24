from __future__ import annotations

import threading

import pytest

from macaque_tracker.camera import Picamera2Camera
from macaque_tracker.config import CameraConfig, RecordingConfig


class _UnderlyingCamera:
    def __init__(self, *, fail_controls: bool = False, fail_stop: bool = False) -> None:
        self.fail_controls = fail_controls
        self.fail_stop = fail_stop
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def set_controls(self, _controls) -> None:
        if self.fail_controls:
            raise RuntimeError("controls failed")

    def stop(self) -> None:
        self.stop_calls += 1
        if self.fail_stop:
            raise RuntimeError("stop failed")

    def close(self) -> None:
        self.close_calls += 1


def _camera(underlying: _UnderlyingCamera) -> Picamera2Camera:
    camera = object.__new__(Picamera2Camera)
    camera.config = CameraConfig(ir_led_pin=None, ir_led_warmup_seconds=0.0)
    camera._camera = underlying
    camera._lock = threading.RLock()
    camera._started = False
    camera._recording = False
    camera._led = None
    camera._closed = False
    camera._camera_controls = dict
    return camera


def test_start_stops_camera_if_applying_controls_fails() -> None:
    underlying = _UnderlyingCamera(fail_controls=True)
    camera = _camera(underlying)

    with pytest.raises(RuntimeError, match="controls failed"):
        camera.start()

    assert underlying.start_calls == 1
    assert underlying.stop_calls == 1
    assert not camera.started


def test_close_releases_camera_and_is_idempotent_after_stop_failure() -> None:
    underlying = _UnderlyingCamera(fail_stop=True)
    camera = _camera(underlying)
    camera._started = True

    with pytest.raises(RuntimeError, match="stop failed"):
        camera.close()

    assert underlying.close_calls == 1
    assert not camera.started
    camera.close()
    assert underlying.close_calls == 1


def test_recording_encoder_uses_full_field_main_stream(tmp_path) -> None:
    class RecordingCamera(_UnderlyingCamera):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_stream = None

        def start_encoder(self, _encoder, _output, *, name: str) -> None:
            self.encoder_stream = name

    underlying = RecordingCamera()
    camera = _camera(underlying)
    camera.recording_config = RecordingConfig(minimum_free_gib=0.0)
    camera._H264Encoder = lambda **_kwargs: object()
    camera._FileOutput = lambda _path: object()
    camera._PyavOutput = lambda _path: object()
    camera._started = True

    destination = tmp_path / "full-field.mkv"
    assert camera.start_recording(destination) == destination.resolve()
    assert underlying.encoder_stream == "main"
