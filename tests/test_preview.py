from __future__ import annotations

import numpy as np
import pytest

try:
    import cv2
except ImportError:
    cv2 = None

from macaque_tracker.config import PreviewConfig
from macaque_tracker.models import AnalysisFrame, EyeMeasurement, FrameResult
from macaque_tracker.preview import render_preview


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_render_preview_draws_fit_dashboard_and_respects_size_limit() -> None:
    crop = np.full((100, 160), 120, dtype=np.uint8)
    second_crop = np.full((80, 120), 135, dtype=np.uint8)
    original_crop = crop.copy()
    original_second_crop = second_crop.copy()
    frame = AnalysisFrame(5, 100, ((0, crop), (1, second_crop)))
    result = FrameResult(
        frame_sequence=5,
        sensor_timestamp_ns=100,
        produced_timestamp_ns=200,
        eyes=(
            EyeMeasurement(0, 81.0, 52.0, 40.0, 0.85, True, False),
            EyeMeasurement(1, 55.0, 38.0, 0.0, 0.0, False, True),
        ),
        dropped_analysis_frames=3,
        processing_time_us=1400,
        diagnostics={
            "eyes": {
                "0": {
                    "detected": True,
                    "missing_frames": 0,
                    "candidate_x": 80.0,
                    "candidate_y": 51.0,
                    "ellipse_x": 80.0,
                    "ellipse_y": 51.0,
                    "ellipse_width": 45.0,
                    "ellipse_height": 35.0,
                    "ellipse_angle_degrees": 12.0,
                    "equivalent_diameter": 40.0,
                    "confidence": 0.85,
                    "contrast": 42.0,
                    "threshold": 55,
                },
                "1": {"detected": False, "missing_frames": 2},
            }
        },
    )
    rendered = render_preview(
        frame,
        result,
        PreviewConfig(max_display_width=300, max_display_height=200),
        display_fps=29.8,
        now_ns=1_000,
    )

    assert rendered.dtype == np.uint8
    assert rendered.ndim == 3
    assert rendered.shape[2] == 3
    assert rendered.shape[1] <= 300
    assert rendered.shape[0] <= 200
    assert np.any(rendered[:, :, 0] != rendered[:, :, 1])
    assert np.array_equal(crop, original_crop)
    assert np.array_equal(second_crop, original_second_crop)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_render_preview_rejects_mismatched_eye_ids() -> None:
    frame = AnalysisFrame(1, 1, ((0, np.zeros((30, 30), dtype=np.uint8)),))
    result = FrameResult(
        frame_sequence=1,
        sensor_timestamp_ns=1,
        produced_timestamp_ns=1,
        eyes=(EyeMeasurement(1, 0.0, 0.0, 0.0, 0.0, False, False),),
    )
    with pytest.raises(ValueError, match="do not match"):
        render_preview(frame, result, PreviewConfig())
