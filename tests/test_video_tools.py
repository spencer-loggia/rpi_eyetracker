import json

import pytest

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


def test_standalone_recorder_rejects_saturation_override(tmp_path) -> None:
    config = RecorderConfig(extra_args=("--saturation=1",))

    with pytest.raises(RecordingError, match="forced grayscale"):
        config.command(tmp_path / "test.h264")


def test_standalone_recorder_rejects_encoding_and_exposure_overrides(tmp_path) -> None:
    for option in ("--codec=h264", "--libav-video-codec-opts=crf=10", "--shutter=1"):
        config = RecorderConfig(extra_args=(option,))

        with pytest.raises(RecordingError, match="managed option"):
            config.command(tmp_path / "test.h264")


def test_standalone_recorder_rejects_exposure_at_least_one_frame() -> None:
    with pytest.raises(RecordingError, match="shorter than one frame"):
        RecorderConfig(framerate=30, exposure_us=33_334).validate()


def test_standalone_recorder_rejects_invalid_crf() -> None:
    for crf in (-1, 52, True, 24.5):
        with pytest.raises(RecordingError, match="crf"):
            RecorderConfig(crf=crf).validate()
