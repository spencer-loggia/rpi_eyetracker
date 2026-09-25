import json

import pytest

from video import recorder as recorder_module
from video.live_preview import DEFAULT_PREVIEW_CONFIG_PATH, preview_command
from video.recorder import RecorderConfig, RecordingError


def _assert_zero_saturation(command: list[str]) -> None:
    index = command.index("--saturation")
    assert command[index + 1] == "0"


def _value_after(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_standalone_recorder_forces_grayscale_camera_output(tmp_path) -> None:
    config = RecorderConfig(cpu_affinity=(), nice=0)
    command = config.command(tmp_path / "test.h264")

    _assert_zero_saturation(command)
    assert _value_after(command, "--shutter") == str(config.exposure_us)
    assert _value_after(command, "--gain") == str(config.analogue_gain)
    assert _value_after(command, "--codec") == "libav"
    assert _value_after(command, "--libav-format") == "h264"
    assert _value_after(command, "--libav-video-codec") == "libx264"
    assert _value_after(command, "--libav-video-codec-opts") == (
        "preset=ultrafast;crf=24;maxrate=64000000;bufsize=128000000"
    )
    assert "--bitrate" not in command


def test_standalone_recorder_loads_crf_from_config(tmp_path) -> None:
    config_path = tmp_path / "recorder.json"
    config_path.write_text(json.dumps({"crf": 31}), encoding="utf-8")

    config = RecorderConfig.load(config_path)
    command = config.command(tmp_path / "custom-crf.h264")

    assert config.crf == 31
    assert "crf=31" in _value_after(command, "--libav-video-codec-opts").split(";")


def test_standalone_preview_forces_grayscale_camera_output() -> None:
    command = preview_command()
    config = RecorderConfig.load(DEFAULT_PREVIEW_CONFIG_PATH)

    _assert_zero_saturation(command)
    assert _value_after(command, "--shutter") == str(config.exposure_us)
    assert _value_after(command, "--gain") == str(config.analogue_gain)


def test_standalone_recorder_rejects_exposure_at_least_one_frame() -> None:
    with pytest.raises(RecordingError, match="shorter than one frame"):
        RecorderConfig(framerate=30, exposure_us=33_334).validate()


def test_standalone_recorder_rejects_invalid_crf() -> None:
    for crf in (-1, 52, True, 24.5):
        with pytest.raises(RecordingError, match="crf"):
            RecorderConfig(crf=crf).validate()


def test_standalone_recorder_rejects_invalid_gain() -> None:
    for gain in (0, -1, True, float("inf")):
        with pytest.raises(RecordingError, match="analogue_gain"):
            RecorderConfig(analogue_gain=gain).validate()


def test_interrupted_startup_stops_detached_recorder(monkeypatch, tmp_path) -> None:
    stopped: list[bool] = []

    class FakeProcess:
        returncode = None

        @staticmethod
        def poll():
            return None

    class FakeRecording:
        def __init__(self, *_args):
            pass

        def stop(self):
            stopped.append(True)

    config = RecorderConfig(cpu_affinity=(), nice=0, startup_check_seconds=0.1)
    monkeypatch.setattr(recorder_module.RecorderConfig, "load", lambda _path: config)
    monkeypatch.setattr(recorder_module, "_require_executable", lambda _name: None)
    monkeypatch.setattr(
        recorder_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    monkeypatch.setattr(recorder_module, "Recording", FakeRecording)

    def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(recorder_module.time, "sleep", interrupt)

    with pytest.raises(KeyboardInterrupt):
        recorder_module.record(tmp_path / "interrupted.h264")

    assert stopped == [True]


def test_failed_startup_removes_partial_recording(monkeypatch, tmp_path) -> None:
    destination = tmp_path / "failed.h264"

    class FakeProcess:
        returncode = 1

        @staticmethod
        def poll():
            return 1

    class FakeThread:
        @staticmethod
        def join(timeout=None):
            del timeout

    class FakeRecording:
        _stderr_thread = FakeThread()

        def __init__(self, *_args):
            pass

        @staticmethod
        def _error(message):
            return RecordingError(message)

    config = RecorderConfig(cpu_affinity=(), nice=0, startup_check_seconds=0.0)
    monkeypatch.setattr(recorder_module.RecorderConfig, "load", lambda _path: config)
    monkeypatch.setattr(recorder_module, "_require_executable", lambda _name: None)

    def start_process(*_args, **_kwargs):
        destination.write_bytes(b"partial")
        return FakeProcess()

    monkeypatch.setattr(recorder_module.subprocess, "Popen", start_process)
    monkeypatch.setattr(recorder_module, "Recording", FakeRecording)

    with pytest.raises(RecordingError, match="status 1"):
        recorder_module.record(destination)

    assert not destination.exists()
