from __future__ import annotations

import numpy as np
import pytest

try:
    import cv2
except ImportError:
    cv2 = None

import macaque_tracker.preview as preview_module
from macaque_tracker.config import PreviewConfig
from macaque_tracker.models import AnalysisFrame, EyeMeasurement, FrameResult, NormalizedRoi
from macaque_tracker.preview import _tracking_rate_sample, render_preview


def _result(sequence: int, produced_ns: int, dropped: int) -> FrameResult:
    return FrameResult(
        frame_sequence=sequence,
        sensor_timestamp_ns=produced_ns,
        produced_timestamp_ns=produced_ns,
        eyes=(EyeMeasurement(0, 1.0, 1.0, 1.0, 1.0, True, False),),
        dropped_analysis_frames=dropped,
    )


def test_tracking_rate_is_independent_of_preview_and_analysis_drops() -> None:
    previous = _result(100, 1_000_000_000, 3)
    current = _result(112, 1_200_000_000, 3)
    with_analysis_drops = _result(112, 1_200_000_000, 5)

    assert _tracking_rate_sample(previous, current) == pytest.approx(60.0)
    assert _tracking_rate_sample(previous, with_analysis_drops) == pytest.approx(50.0)


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


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_idle_preview_renders_live_crops_without_measurements_or_masks(monkeypatch) -> None:
    crop = np.full((100, 160), 120, dtype=np.uint8)
    original = crop.copy()
    frame = AnalysisFrame(10, 100, ((0, crop),))
    labels = []
    draw_text = preview_module._put_text

    def record_text(image, text, *args, **kwargs):
        labels.append(text)
        return draw_text(image, text, *args, **kwargs)

    def unexpected_mask(*_args):
        raise AssertionError("Idle preview must not reconstruct a pupil mask")

    monkeypatch.setattr(preview_module, "_put_text", record_text)
    monkeypatch.setattr(preview_module, "pupil_mask_for_threshold", unexpected_mask)
    first = render_preview(frame, None, PreviewConfig())
    assert any("TRACKING IDLE" in label for label in labels)
    assert "EYE 0  IDLE" in labels
    assert not any(label.startswith("x ") for label in labels)
    assert np.array_equal(crop, original)
    next_frame = AnalysisFrame(11, 200, ((0, crop + 30),))
    second = render_preview(next_frame, None, PreviewConfig())
    assert not np.array_equal(first, second)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("tracking", [False, True])
@pytest.mark.parametrize("eye_count", [1, 2])
def test_overview_shows_four_views_and_configured_boxes(monkeypatch, tracking, eye_count):
    overview = np.repeat(np.array([[40, 80, 120, 160]], dtype=np.uint8), 320, axis=1)
    overview = np.repeat(overview, 180, axis=0)
    original = overview.copy()
    crops = tuple((i, np.full((100, 160), 11, dtype=np.uint8)) for i in range(eye_count))
    frame = AnalysisFrame(1, 1, crops, overview=overview)
    rois = tuple(
        NormalizedRoi(i, f"eye {i}", 0.13 + i * 0.5, 0.4, 0.07, 0.3)
        for i in range(eye_count)
    )
    result = (
        FrameResult(1, 1, 1, tuple(
            EyeMeasurement(i, 80, 50, 20, 0.9, True, False) for i in range(eye_count)
        )) if tracking else None
    )
    drawn = []
    draw_box = preview_module.draw_roi_box

    def record_box(canvas, roi, eye_id, **kwargs):
        draw_box(canvas, roi, eye_id, **kwargs)
        drawn.append((canvas, roi, eye_id))

    monkeypatch.setattr(preview_module, "draw_roi_box", record_box)
    config = PreviewConfig(max_display_width=2000, max_display_height=2000)
    rendered = render_preview(frame, result, config, rois=rois)
    details = render_preview(AnalysisFrame(1, 1, crops), result, config)
    assert rendered.shape[0] > details.shape[0]
    assert rendered.shape[1] == details.shape[1]
    assert len(drawn) == eye_count
    for canvas, pixel_roi, eye_id in drawn:
        height, width = canvas.shape[:2]
        assert pixel_roi == rois[eye_id].to_pixels(width, height)
        color = ((80, 220, 80), (220, 90, 220))[eye_id]
        assert tuple(canvas[pixel_roi.y + pixel_roi.height // 2, pixel_roi.x]) == color
        assert np.any(np.all(rendered == color, axis=2))
        # All four camera views remain visible, including those without an eye.
        for index, shade in enumerate((40, 80, 120, 160)):
            assert np.all(canvas[height - 5, round((index + 0.5) * width / 4)] == shade)
            assert np.any(np.all(rendered == shade, axis=2))
    assert np.array_equal(overview, original)
    assert all(np.all(crop == 11) for _eye_id, crop in crops)
    limited = render_preview(
        frame, result, PreviewConfig(max_display_width=640, max_display_height=360),
        rois=rois,
    )
    assert limited.shape[1] <= 640 and limited.shape[0] <= 360
