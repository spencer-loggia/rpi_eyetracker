from __future__ import annotations

import numpy as np

from macaque_tracker import roi_tool
from macaque_tracker.config import CameraConfig, TrackerConfig
from macaque_tracker.roi_tool import (
    RoiEditor,
    _ControlSlider,
    _control_sliders,
    _posthoc_preview_image,
)


def test_camera_control_sliders_use_hardware_and_frame_period_limits() -> None:
    config = CameraConfig(fps=60.0, exposure_us=10_000)
    sliders = {
        slider.config_name: slider
        for slider in _control_sliders(
            config,
            {
                "exposure_us": (100.0, 100_000.0),
                "contrast": (0.0, 32.0),
            },
        )
    }

    assert sliders["exposure_us"].minimum == 100.0
    assert sliders["exposure_us"].maximum == 16_666.0
    assert sliders["exposure_us"].value_for(
        sliders["exposure_us"].position_for(12_345)
    ) == 12_345
    assert sliders["contrast"].minimum == 0.01
    assert sliders["analogue_gain"].minimum == 0.01
    assert sliders["analogue_gain"].maximum == 32.0


def test_offline_controls_include_posthoc_sliders_but_not_exposure() -> None:
    sliders = {slider.config_name for slider in _control_sliders(CameraConfig(), {})}

    assert sliders == {"analogue_gain", "brightness", "contrast", "sharpness"}


def _editor_for_slider_test():
    editor = object.__new__(RoiEditor)
    editor.cv2 = type("FakeCv2", (), {"setTrackbarPos": lambda *args: None})()
    editor.camera_config = CameraConfig()
    editor.applied_camera_config = editor.camera_config
    editor._source_camera_config = editor.camera_config
    editor._initializing_sliders = False
    editor._source_image = np.full((30, 40), 100, dtype=np.uint8)
    editor.image = editor._source_image.copy()
    editor._analysis_image = np.ones((30, 40, 3), dtype=np.uint8)
    editor._detections = {0: object()}
    editor._status = ""
    return editor


def test_non_exposure_slider_applies_and_refreshes_immediately() -> None:
    editor = _editor_for_slider_test()
    slider = _ControlSlider("contrast", "Contrast x100", 0.01, 32.0, 100)

    editor._slider_changed(slider, slider.position_for(1.75))

    assert editor.camera_config.contrast == 1.75
    assert editor.applied_camera_config.contrast == 1.75
    assert np.all(editor._source_image == 100)
    assert np.all(editor.image == 79)
    assert editor._analysis_image is None
    assert editor._detections == {}
    assert editor._status == "Contrast x100 applied to cached preview"

    editor._slider_changed(slider, slider.position_for(1.0))
    assert np.array_equal(editor.image, editor._source_image)


def test_all_posthoc_controls_transform_only_the_cached_image() -> None:
    class FakeCv2:
        @staticmethod
        def GaussianBlur(image, kernel_size, sigma):
            del kernel_size, sigma
            return np.full_like(image, 80)

    source = CameraConfig()
    image = np.array([[60, 90], [120, 150]], dtype=np.uint8)
    targets = (
        CameraConfig(analogue_gain=source.analogue_gain * 1.5),
        CameraConfig(brightness=0.2),
        CameraConfig(contrast=1.5),
        CameraConfig(sharpness=source.sharpness + 1.0),
    )

    for target in targets:
        adjusted = _posthoc_preview_image(image, source, target, FakeCv2)
        assert adjusted.dtype == np.uint8
        assert not np.array_equal(adjusted, image)
    assert np.array_equal(image, np.array([[60, 90], [120, 150]], dtype=np.uint8))


def test_exposure_slider_remains_pending_until_explicit_recapture() -> None:
    editor = _editor_for_slider_test()
    source_image = editor._source_image.copy()
    slider = _ControlSlider("exposure_us", "Exposure us", 100.0, 30_000.0, 1)

    editor._slider_changed(slider, 12_000)

    assert editor.camera_config.exposure_us == 12_000
    assert editor.applied_camera_config.exposure_us != 12_000
    assert np.array_equal(editor._source_image, source_image)
    assert np.array_equal(editor.image, source_image)
    assert "press R" in editor._status


def test_pupil_threshold_slider_applies_to_detector_immediately(monkeypatch) -> None:
    editor = _editor_for_slider_test()
    editor.tracker_config = TrackerConfig()
    editor.applied_tracker_config = editor.tracker_config
    monkeypatch.setattr(
        roi_tool,
        "AdaptivePupilDetector",
        lambda config: ("detector", config),
    )

    editor._threshold_changed(61)

    assert editor.tracker_config.pupil_threshold == 61
    assert editor.applied_tracker_config.pupil_threshold == 61
    assert editor.detector == ("detector", editor.tracker_config)
    assert editor._analysis_image is None
    assert editor._detections == {}
    assert editor._status == "Pupil threshold 61 applied to detector"

    editor._threshold_changed(0)
    assert editor.tracker_config.pupil_threshold is None
    assert "AUTO" in editor._status
