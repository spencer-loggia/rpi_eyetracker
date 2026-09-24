from __future__ import annotations

import numpy as np
import pytest

try:
    import cv2
except ImportError:
    cv2 = None

from macaque_tracker.config import TrackerConfig
from macaque_tracker.models import AnalysisFrame
from macaque_tracker.tracker import AdaptivePupilDetector, MultiEyeTracker


def _synthetic_eye(*, blink: bool = False) -> np.ndarray:
    image = np.full((160, 240), 145, dtype=np.uint8)
    if not blink:
        cv2.ellipse(image, (125, 82), (31, 22), 12, 0, 360, 24, -1)
        cv2.circle(image, (116, 75), 4, 255, -1)
        cv2.circle(image, (136, 87), 3, 245, -1)
    return image


def test_ellipse_residuals_use_the_ellipse_center() -> None:
    points = np.asarray(((15.0, 20.0), (5.0, 20.0), (10.0, 23.0), (10.0, 17.0)))
    residuals = AdaptivePupilDetector._ellipse_residuals(
        points,
        ((10.0, 20.0), (10.0, 6.0), 0.0),
    )
    assert residuals == pytest.approx(np.zeros(4))


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_detects_dark_pupil_with_multiple_glints() -> None:
    tracker = MultiEyeTracker((0,), TrackerConfig(min_confidence=0.35))
    result = tracker.process(AnalysisFrame(1, 100, ((0, _synthetic_eye()),)))
    eye = result.eyes[0]
    assert eye.valid
    assert eye.x == pytest.approx(125, abs=3)
    assert eye.y == pytest.approx(82, abs=3)
    assert eye.pupil_diameter > 30
    diagnostics = result.diagnostics["eyes"]["0"]
    assert diagnostics["detected"]
    assert diagnostics["ellipse_width"] > 0
    assert diagnostics["ellipse_height"] > 0
    assert diagnostics["contrast"] > 0
    assert 1 <= diagnostics["threshold"] <= 254


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_detection_mask_contains_only_the_selected_pupil_contour() -> None:
    image = _synthetic_eye()
    detector = AdaptivePupilDetector(TrackerConfig(min_confidence=0.35))

    candidate, mask = detector.detect_with_mask(image)

    assert candidate is not None
    assert mask.dtype == np.uint8 and mask.shape == image.shape
    assert mask[round(candidate.y), round(candidate.x)] == 255
    assert mask[10, 10] == 0
    selected_area = np.count_nonzero(mask)
    expected_area = np.pi * (candidate.diameter / 2.0) ** 2
    assert selected_area == pytest.approx(expected_area, rel=0.12)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_missing_pupil_reports_zero_then_blink() -> None:
    tracker = MultiEyeTracker((0,), TrackerConfig(min_confidence=0.35))
    tracker.process(AnalysisFrame(1, 100, ((0, _synthetic_eye()),)))
    first = tracker.process(AnalysisFrame(2, 200, ((0, _synthetic_eye(blink=True)),))).eyes[0]
    second = tracker.process(AnalysisFrame(3, 300, ((0, _synthetic_eye(blink=True)),))).eyes[0]
    assert not first.valid and first.pupil_diameter == 0.0
    assert not first.blink
    assert second.blink and second.pupil_diameter == 0.0
