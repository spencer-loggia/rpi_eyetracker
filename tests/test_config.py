from __future__ import annotations

import json

import numpy as np
import pytest

from macaque_tracker.config import (
    AppConfig,
    CameraConfig,
    ConfigError,
    PreviewConfig,
    RecordingConfig,
    RoiLayout,
    TrackerConfig,
    TransportConfig,
)
from macaque_tracker.models import EyeImageSettings, NormalizedRoi, PixelRoi


def test_normalized_roi_pixel_round_trip() -> None:
    source = PixelRoi(100, 50, 300, 120)
    normalized = NormalizedRoi.from_pixels(
        eye_id=0,
        label="left eye",
        roi=source,
        frame_width=1000,
        frame_height=500,
    )
    assert normalized.to_pixels(1000, 500) == source
    assert normalized.to_pixels(2000, 1000) == PixelRoi(200, 100, 600, 240)


def test_pixel_roi_extract_always_owns_its_memory() -> None:
    image = np.arange(24, dtype=np.uint8).reshape(4, 6)
    crop = PixelRoi(0, 0, 6, 4).extract(image)

    image[:] = 0
    assert crop.sum() > 0


def test_roi_layout_atomic_round_trip(tmp_path) -> None:
    settings = EyeImageSettings(
        gain=1.4,
        brightness=-0.1,
        contrast=1.2,
        sharpness=0.6,
        pupil_size_bias=-0.35,
    )
    layout = RoiLayout(
        rois=(NormalizedRoi(0, "eye_0", 0.1, 0.2, 0.2, 0.3, settings),),
        source_width=1000,
        source_height=500,
    )
    path = layout.save(tmp_path / "rois.json")
    assert RoiLayout.load(path) == layout
    assert not (tmp_path / "rois.json.tmp").exists()

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["rois"][0]["settings"] == settings.as_dict()


def test_roi_layout_loads_legacy_roi_without_settings(tmp_path) -> None:
    path = tmp_path / "legacy-rois.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source_width": 1000,
                "source_height": 500,
                "rois": [
                    {
                        "eye_id": 0,
                        "label": "eye_0",
                        "x": 0.1,
                        "y": 0.2,
                        "width": 0.2,
                        "height": 0.3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert RoiLayout.load(path).rois[0].settings == EyeImageSettings()


def test_roi_layout_migrates_legacy_fixed_threshold_to_neutral_bias(tmp_path) -> None:
    path = tmp_path / "legacy-threshold-rois.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source_width": 1000,
                "source_height": 500,
                "rois": [
                    {
                        "eye_id": 0,
                        "label": "eye_0",
                        "x": 0.1,
                        "y": 0.2,
                        "width": 0.2,
                        "height": 0.3,
                        "settings": {"pupil_threshold": 73},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert RoiLayout.load(path).rois[0].settings.pupil_size_bias == 0.0


def test_eye_image_settings_validate_software_ranges() -> None:
    with pytest.raises(ValueError, match="gain"):
        EyeImageSettings(gain=0.0)
    with pytest.raises(ValueError, match="brightness"):
        EyeImageSettings(brightness=1.1)
    with pytest.raises(ValueError, match="pupil_size_bias"):
        EyeImageSettings(pupil_size_bias=1.01)


def test_roi_layout_requires_contiguous_ids() -> None:
    with pytest.raises(ConfigError, match="contiguous"):
        RoiLayout(
            rois=(NormalizedRoi(1, "eye", 0.0, 0.0, 0.5, 0.5),),
            source_width=10,
            source_height=10,
        )


def test_config_rejects_unknown_fields(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"camera": {"mystery": 1}}), encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown camera fields"):
        AppConfig.load(path)


def test_app_config_atomic_round_trip(tmp_path) -> None:
    config = AppConfig()
    path = config.save(tmp_path / "eye_tracker.json")

    assert AppConfig.load(path) == config
    assert not (tmp_path / "eye_tracker.json.tmp").exists()


def test_default_project_config_loads() -> None:
    config = AppConfig.load("config/eye_tracker.json")
    assert config.camera.analysis_width == config.camera.sensor_width
    assert config.camera.analysis_height == config.camera.sensor_height
    assert config.recording.container == "mkv"
    assert 0 <= config.recording.crf <= 51
    assert config.recording.bitrate > 0
    assert config.transport.uart_baud == 460_800
    assert not config.preview.enabled
    assert config.preview.show_threshold_mask


def test_uart_project_config_loads() -> None:
    config = AppConfig.load("config/eye_tracker.uart.json")
    assert config.transport.backend == "uart"
    assert config.transport.uart_device == "/dev/ttyAMA0"


def test_preview_config_rejects_invalid_display_size() -> None:
    with pytest.raises(ConfigError, match="positive integer"):
        PreviewConfig(max_display_width=0)


def test_tracker_pupil_size_bias_is_symmetric_and_bounded() -> None:
    assert TrackerConfig().pupil_size_bias == 0.0
    assert TrackerConfig(pupil_size_bias=-0.75).pupil_size_bias == -0.75
    with pytest.raises(ConfigError, match="pupil_size_bias"):
        TrackerConfig(pupil_size_bias=-1.01)
    with pytest.raises(ConfigError, match="pupil_size_bias"):
        TrackerConfig(pupil_size_bias=1.01)


def test_tracker_rejects_strongly_elliptical_fits_by_default() -> None:
    assert TrackerConfig().min_axis_ratio == pytest.approx(0.45)


def test_recording_crf_defaults_to_24_and_is_bounded() -> None:
    assert RecordingConfig().crf == 24
    assert RecordingConfig(crf=0).crf == 0
    assert RecordingConfig(crf=51).crf == 51
    with pytest.raises(ConfigError, match="recording.crf"):
        RecordingConfig(crf=-1)
    with pytest.raises(ConfigError, match="recording.crf"):
        RecordingConfig(crf=52)


def test_app_config_loads_recording_crf(tmp_path) -> None:
    path = tmp_path / "custom-crf.json"
    path.write_text(json.dumps({"recording": {"crf": 31}}), encoding="utf-8")

    assert AppConfig.load(path).recording.crf == 31


def test_app_config_migrates_legacy_fixed_threshold_to_neutral_bias(tmp_path) -> None:
    path = tmp_path / "legacy-threshold-config.json"
    path.write_text(
        json.dumps({"tracker": {"pupil_threshold": 80}}),
        encoding="utf-8",
    )

    assert AppConfig.load(path).tracker.pupil_size_bias == 0.0


def test_roi_layout_rejects_mismatched_frame_and_tiny_crop() -> None:
    layout = RoiLayout(
        rois=(NormalizedRoi(0, "eye", 0.1, 0.1, 0.01, 0.2),),
        source_width=1000,
        source_height=500,
    )
    with pytest.raises(ConfigError, match="at least 24"):
        layout.validate_for_frame(1000, 500)
    with pytest.raises(ConfigError, match="aspect ratio"):
        layout.validate_for_frame(1000, 1000)


def test_roi_layout_rejects_impossible_pupil_diameter_limits() -> None:
    layout = RoiLayout(
        rois=(NormalizedRoi(0, "eye", 0.1, 0.1, 0.03, 0.1),),
        source_width=1000,
        source_height=500,
    )

    with pytest.raises(ConfigError, match="too small"):
        layout.validate_for_frame(
            1000,
            500,
            tracker_config=TrackerConfig(
                min_pupil_diameter_px=20,
                max_pupil_diameter_fraction=0.5,
            ),
        )


def test_uart_rejects_ir_led_on_serial_pins() -> None:
    with pytest.raises(ConfigError, match="GPIO14/15"):
        AppConfig(
            camera=CameraConfig(ir_led_pin=14),
            transport=TransportConfig(backend="uart"),
        )
