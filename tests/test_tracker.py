from __future__ import annotations

import numpy as np
import pytest

try:
    import cv2
except ImportError:
    cv2 = None

from macaque_tracker.config import TrackerConfig
from macaque_tracker.models import (
    AnalysisFrame,
    EyeImageSettings,
    decoded_monochrome_frame,
)
from macaque_tracker.tracker import (
    AdaptivePupilDetector,
    MultiEyeTracker,
    _adaptive_appearance_scores,
    _pupil_size_bias_adjustment,
    apply_eye_image_settings,
)


def _synthetic_eye(*, blink: bool = False) -> np.ndarray:
    image = np.full((160, 240), 145, dtype=np.uint8)
    if not blink:
        cv2.ellipse(image, (125, 82), (31, 22), 12, 0, 360, 24, -1)
        cv2.circle(image, (116, 75), 4, 255, -1)
        cv2.circle(image, (136, 87), 3, 245, -1)
    return image


def _synthetic_dark_iris(*, offset: int = 0, pupil: bool = True) -> np.ndarray:
    image = np.full((160, 240), np.clip(185 + offset, 0, 255), dtype=np.uint8)
    cv2.ellipse(image, (122, 81), (55, 40), 8, 0, 360, 70 + offset, -1)
    if pupil:
        cv2.ellipse(image, (124, 82), (28, 20), 11, 0, 360, 24 + offset, -1)
        cv2.circle(image, (116, 75), 4, 245, -1)
        cv2.circle(image, (134, 88), 3, 250, -1)
    return image


def test_ellipse_residuals_use_the_ellipse_center() -> None:
    points = np.asarray(((15.0, 20.0), (5.0, 20.0), (10.0, 23.0), (10.0, 17.0)))
    residuals = AdaptivePupilDetector._ellipse_residuals(
        points,
        ((10.0, 20.0), (10.0, 6.0), 0.0),
    )
    assert residuals == pytest.approx(np.zeros(4))


def test_software_image_settings_do_not_modify_source() -> None:
    image = np.array([[60, 90], [120, 150]], dtype=np.uint8)
    original = image.copy()

    adjusted = apply_eye_image_settings(
        image,
        EyeImageSettings(gain=1.2, brightness=0.1, contrast=1.4),
    )

    assert adjusted.dtype == np.uint8
    assert not np.array_equal(adjusted, image)
    assert np.array_equal(image, original)


def test_decoded_monochrome_frame_collapses_only_redundant_channels() -> None:
    expanded = np.repeat(
        np.array([[30, 90], [140, 220]], dtype=np.uint8)[:, :, None],
        3,
        axis=2,
    )

    gray = decoded_monochrome_frame(expanded)

    assert gray.shape == (2, 2)
    assert gray.dtype == np.uint8
    assert np.array_equal(gray, expanded[:, :, 0])


def test_decoded_monochrome_frame_rejects_meaningful_colour() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = (10, 40, 10)

    with pytest.raises(ValueError, match="contains colour"):
        decoded_monochrome_frame(image)


def test_analysis_frame_rejects_multichannel_crop() -> None:
    with pytest.raises(ValueError, match="single-channel"):
        AnalysisFrame(
            1,
            100,
            ((0, np.zeros((30, 30, 3), dtype=np.uint8)),),
        )


def test_adaptive_appearance_scoring_penalizes_an_iris_containing_a_dark_core() -> None:
    adaptive_range = (20.0, 145.0)

    pupil = _adaptive_appearance_scores(25.0, 3.0, 38, adaptive_range)
    iris = _adaptive_appearance_scores(70.0, 46.0, 72, adaptive_range)

    assert pupil[0] > iris[0]
    assert pupil[1] > iris[1]
    assert pupil[2] > iris[2]


def test_pupil_size_bias_adjustment_is_symmetric() -> None:
    small_with_small_bias = _pupil_size_bias_adjustment(0.1, 0.1, 0.7, -1.0)
    large_with_small_bias = _pupil_size_bias_adjustment(0.7, 0.1, 0.7, -1.0)
    small_with_large_bias = _pupil_size_bias_adjustment(0.1, 0.1, 0.7, 1.0)
    large_with_large_bias = _pupil_size_bias_adjustment(0.7, 0.1, 0.7, 1.0)

    assert small_with_small_bias > large_with_small_bias
    assert large_with_large_bias > small_with_large_bias
    assert _pupil_size_bias_adjustment(0.4, 0.1, 0.7, 0.0) == 0.0


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_per_eye_settings_override_global_size_bias_independently() -> None:
    tracker = MultiEyeTracker(
        (0, 1),
        TrackerConfig(pupil_size_bias=0.25),
        {
            0: EyeImageSettings(pupil_size_bias=-0.75),
            1: EyeImageSettings(pupil_size_bias=0.8),
        },
    )

    assert tracker.trackers[0].detector.config.pupil_size_bias == -0.75
    assert tracker.trackers[1].detector.config.pupil_size_bias == 0.8


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
def test_adaptive_detector_prefers_pupil_over_larger_dark_iris() -> None:
    detector = AdaptivePupilDetector(TrackerConfig(min_confidence=0.35))

    candidate = detector.detect(_synthetic_dark_iris())

    assert candidate is not None
    assert candidate.x == pytest.approx(124, abs=3)
    assert candidate.y == pytest.approx(82, abs=3)
    assert candidate.major < 75
    assert candidate.threshold < 60
    assert candidate.median_intensity < 40


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_adaptive_pupil_selection_tracks_frame_brightness_changes() -> None:
    detector = AdaptivePupilDetector(TrackerConfig(min_confidence=0.35))

    candidates = [detector.detect(_synthetic_dark_iris(offset=value)) for value in (-12, 18)]

    assert all(candidate is not None for candidate in candidates)
    assert all(candidate.major < 75 for candidate in candidates if candidate is not None)
    thresholds = [candidate.threshold for candidate in candidates if candidate is not None]
    assert thresholds[0] < thresholds[1]


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_adaptive_tracker_recovers_from_concentric_iris_lock() -> None:
    tracker = MultiEyeTracker(
        (0,),
        TrackerConfig(min_confidence=0.35, max_diameter_change_fraction=0.30),
    )
    iris_result = tracker.process(
        AnalysisFrame(1, 100, ((0, _synthetic_dark_iris(pupil=False)),))
    )
    pupil_result = tracker.process(AnalysisFrame(2, 200, ((0, _synthetic_dark_iris()),)))

    assert iris_result.eyes[0].pupil_diameter > 80
    assert pupil_result.eyes[0].valid
    assert pupil_result.eyes[0].pupil_diameter < 70


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
def test_size_biased_detector_still_selects_threshold_adaptively() -> None:
    detector = AdaptivePupilDetector(
        TrackerConfig(min_confidence=0.35, pupil_size_bias=-0.8)
    )

    candidates = [detector.detect(_synthetic_dark_iris(offset=value)) for value in (-12, 18)]

    assert all(candidate is not None for candidate in candidates)
    thresholds = [candidate.threshold for candidate in candidates if candidate is not None]
    assert thresholds[0] < thresholds[1]


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_missing_pupil_reports_zero_then_blink() -> None:
    tracker = MultiEyeTracker((0,), TrackerConfig(min_confidence=0.35))
    tracker.process(AnalysisFrame(1, 100, ((0, _synthetic_eye()),)))
    first = tracker.process(AnalysisFrame(2, 200, ((0, _synthetic_eye(blink=True)),))).eyes[0]
    second = tracker.process(AnalysisFrame(3, 300, ((0, _synthetic_eye(blink=True)),))).eyes[0]
    assert not first.valid and first.pupil_diameter == 0.0
    assert not first.blink
    assert second.blink and second.pupil_diameter == 0.0
