from __future__ import annotations

from macaque_tracker.config import CameraConfig
from macaque_tracker.roi_tool import _control_sliders


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
