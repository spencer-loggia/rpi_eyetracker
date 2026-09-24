from __future__ import annotations

import numpy as np

from macaque_tracker.config import CameraConfig
from macaque_tracker.roi_tool import RoiEditor, _ControlSlider, _control_sliders


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
    assert "analogue_gain" not in sliders


def _editor_for_slider_test():
    editor = object.__new__(RoiEditor)
    editor.cv2 = type("FakeCv2", (), {"setTrackbarPos": lambda *args: None})()
    editor.camera_config = CameraConfig()
    editor.applied_camera_config = editor.camera_config
    editor._initializing_sliders = False
    editor.image = np.zeros((30, 40), dtype=np.uint8)
    editor._analysis_image = np.ones((30, 40, 3), dtype=np.uint8)
    editor._detections = {0: object()}
    editor._status = ""
    return editor


def test_non_exposure_slider_applies_and_refreshes_immediately() -> None:
    editor = _editor_for_slider_test()
    calls: list[tuple[str, int | float]] = []

    def live_update(name: str, value: int | float) -> np.ndarray:
        calls.append((name, value))
        return np.full((30, 40), 7, dtype=np.uint8)

    editor._live_update = live_update
    slider = _ControlSlider("contrast", "Contrast x100", 0.01, 32.0, 100)

    editor._slider_changed(slider, slider.position_for(1.75))

    assert calls == [("contrast", 1.75)]
    assert editor.camera_config.contrast == 1.75
    assert editor.applied_camera_config.contrast == 1.75
    assert np.all(editor.image == 7)
    assert editor._analysis_image is None
    assert editor._detections == {}
    assert editor._status == "Contrast x100 applied live"


def test_exposure_slider_remains_pending_until_explicit_recapture() -> None:
    editor = _editor_for_slider_test()
    calls: list[tuple[str, int | float]] = []
    editor._live_update = lambda name, value: calls.append((name, value))
    slider = _ControlSlider("exposure_us", "Exposure us", 100.0, 30_000.0, 1)

    editor._slider_changed(slider, 12_000)

    assert calls == []
    assert editor.camera_config.exposure_us == 12_000
    assert editor.applied_camera_config.exposure_us != 12_000
    assert "press R" in editor._status
