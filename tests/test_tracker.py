from __future__ import annotations

import threading

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
    EyeMeasurement,
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


def _synthetic_eye_at(center_x: int) -> np.ndarray:
    image = np.full((160, 240), 185, dtype=np.uint8)
    cv2.ellipse(image, (center_x, 80), (52, 39), 8, 0, 360, 75, -1)
    cv2.ellipse(image, (center_x, 80), (28, 22), 11, 0, 360, 28, -1)
    cv2.circle(image, (center_x - 8, 73), 4, 250, -1)
    return image


def _synthetic_noisy_eye(seed: int, offset: int) -> np.ndarray:
    image = np.full((160, 240), 185 + offset, dtype=np.float32)
    cv2.ellipse(image, (124, 82), (55, 40), 8, 0, 360, 72 + offset, -1)
    cv2.ellipse(image, (124, 82), (29, 21), 11, 0, 360, 27 + offset, -1)
    cv2.circle(image, (116, 75), 4, 245, -1)
    noise = np.random.default_rng(seed).normal(0.0, 4.0, image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def _synthetic_glint_eye(
    *,
    radius: int = 0,
    position: tuple[int, int] = (124, 82),
    noise: float = 0.0,
    offset: int = 0,
    glint_intensity: int = 255,
) -> np.ndarray:
    image = np.full((160, 240), 185 + offset, dtype=np.float32)
    cv2.ellipse(image, (124, 82), (55, 40), 8, 0, 360, 70 + offset, -1)
    cv2.ellipse(image, (124, 82), (28, 20), 11, 0, 360, 24 + offset, -1)
    if radius:
        cv2.circle(image, position, radius, glint_intensity + offset, -1)
    image += np.random.default_rng(7).normal(0.0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


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


@pytest.mark.parametrize("channels", [3, 4])
def test_decoded_monochrome_frame_preserves_redundant_channels_exactly(channels) -> None:
    expanded = np.repeat(np.arange(256, dtype=np.uint8).reshape(16, 16, 1), channels, axis=2)
    if channels == 4:
        expanded[:, :, 3] = 0  # Alpha is ignored.

    gray = decoded_monochrome_frame(expanded)

    assert gray.shape == (16, 16)
    assert gray.dtype == np.uint8
    assert np.array_equal(gray, expanded[:, :, 0])


def test_decoded_monochrome_frame_collapses_channel_differences_to_luma() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = (10, 40, 10)

    gray = decoded_monochrome_frame(image)

    assert gray.shape == (2, 2)
    assert gray.dtype == np.uint8
    assert gray[0, 0] == 28
    assert gray[1, 1] == 0


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
    tracker.close()


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_two_eye_tracker_runs_independent_fits_concurrently() -> None:
    tracker = MultiEyeTracker((0, 1), TrackerConfig())
    barrier = threading.Barrier(2)

    class ConcurrentEye:
        def __init__(self, eye_id: int) -> None:
            self.eye_id = eye_id
            self.last_diagnostics: dict[str, object] = {}

        def process(self, _image: np.ndarray) -> EyeMeasurement:
            barrier.wait(timeout=1.0)
            return EyeMeasurement(
                self.eye_id,
                1.0,
                2.0,
                3.0,
                1.0,
                True,
                False,
            )

    tracker.trackers = {0: ConcurrentEye(0), 1: ConcurrentEye(1)}
    frame = AnalysisFrame(
        1,
        1,
        (
            (0, np.zeros((24, 24), dtype=np.uint8)),
            (1, np.zeros((24, 24), dtype=np.uint8)),
        ),
    )
    try:
        result = tracker.process(frame)
    finally:
        tracker.close()

    assert tuple(eye.eye_id for eye in result.eyes) == (0, 1)


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
@pytest.mark.parametrize("offset", [-30, 0, 20])
@pytest.mark.parametrize("glint_intensity", [230, 255])
def test_large_enclosed_glint_preserves_pupil_instead_of_selecting_iris(
    offset: int, glint_intensity: int,
) -> None:
    image = _synthetic_glint_eye(
        radius=12, offset=offset, glint_intensity=glint_intensity, noise=3.0,
    )
    original = image.copy()
    candidate = AdaptivePupilDetector(TrackerConfig()).detect(image)

    assert candidate is not None
    assert candidate.x == pytest.approx(124, abs=1)
    assert candidate.y == pytest.approx(82, abs=1)
    assert candidate.diameter == pytest.approx(np.sqrt(56 * 40), rel=0.08)
    assert 0.20 < candidate.glint_fraction <= 0.40
    assert np.array_equal(image, original)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("angle", np.arange(0, 360, 45))
@pytest.mark.parametrize("noise", [0.0, 3.0, 6.0])
def test_edge_glint_recovery_uses_visible_boundary_without_centroid_bias(
    angle: int, noise: float,
) -> None:
    radians = np.deg2rad(angle)
    position = (round(124 + 26 * np.cos(radians)), round(82 + 20 * np.sin(radians)))
    image = _synthetic_glint_eye(radius=12, position=position, noise=noise)

    candidate = AdaptivePupilDetector(TrackerConfig()).detect(image)

    assert candidate is not None and candidate.boundary_recovered
    assert candidate.x == pytest.approx(124, abs=1)
    assert candidate.y == pytest.approx(82, abs=1)
    assert candidate.diameter == pytest.approx(np.sqrt(56 * 40), rel=0.10)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("radius,position", [(16, (124, 82)), (20, (124, 82)), (20, (147, 82))])
def test_heavy_glint_occlusion_is_rejected_without_an_iris_fallback(
    radius: int, position: tuple[int, int],
) -> None:
    image = _synthetic_glint_eye(radius=radius, position=position)

    assert AdaptivePupilDetector(TrackerConfig()).detect(image) is None


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_glint_recovery_does_not_bypass_configured_confidence() -> None:
    detector = AdaptivePupilDetector(TrackerConfig(min_confidence=0.90))

    assert detector.detect(_synthetic_glint_eye()) is not None
    assert detector.detect(_synthetic_glint_eye(radius=12)) is None


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("shape", ["blank", "eyelid", "elongated", "tendril"])
@pytest.mark.parametrize("noise", [0.0, 4.0])
def test_glints_do_not_turn_nonpupil_shapes_into_valid_recovered_fits(
    shape: str, noise: float,
) -> None:
    image = np.full((160, 240), 185, dtype=np.float32)
    if shape == "eyelid":
        cv2.line(image, (40, 82), (190, 82), 24, 9)
    elif shape == "elongated":
        cv2.ellipse(image, (124, 82), (45, 12), 0, 0, 360, 24, -1)
    elif shape == "tendril":
        cv2.ellipse(image, (124, 82), (28, 20), 0, 0, 360, 24, -1)
        cv2.rectangle(image, (148, 77), (190, 87), 24, -1)
    cv2.circle(image, (105, 82), 12, 255, -1)
    image += np.random.default_rng(4).normal(0, noise, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)

    assert AdaptivePupilDetector(TrackerConfig()).detect(image) is None


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_moving_glint_does_not_move_stationary_pupil_or_require_smoothing() -> None:
    tracker = MultiEyeTracker((0,), TrackerConfig())
    try:
        for index, position in enumerate([(147, 82), (101, 82), (124, 101), (124, 63)]):
            image = _synthetic_glint_eye(radius=12, position=position, noise=3.0)
            result = tracker.process(AnalysisFrame(index, index, ((0, image),)))
            eye = result.eyes[0]
            assert eye.valid
            assert eye.x == pytest.approx(124, abs=1)
            assert eye.y == pytest.approx(82, abs=1)
            assert result.diagnostics["eyes"]["0"]["boundary_recovered"]
    finally:
        tracker.close()


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
def test_size_bias_changes_ranking_without_changing_fit_confidence(monkeypatch) -> None:
    image = _synthetic_dark_iris()
    fits: dict[float, list] = {0.0: [], 1.0: []}
    original_candidate = AdaptivePupilDetector._candidate

    def record_candidate(self, *args, **kwargs):
        candidate = original_candidate(self, *args, **kwargs)
        if candidate is not None:
            fits[self.config.pupil_size_bias].append(candidate)
        return candidate

    monkeypatch.setattr(AdaptivePupilDetector, "_candidate", record_candidate)
    neutral = AdaptivePupilDetector(
        TrackerConfig(min_confidence=0.88, pupil_size_bias=0.0)
    ).detect(image)
    prefer_large = AdaptivePupilDetector(
        TrackerConfig(min_confidence=0.88, pupil_size_bias=1.0)
    ).detect(image)

    assert neutral is not None and prefer_large is not None
    assert min(neutral.confidence, prefer_large.confidence) >= 0.88
    # Size bias can select another threshold/fit; compare confidence on the
    # same candidates, rather than assuming the selected fit never changes.
    assert len(fits[0.0]) == len(fits[1.0]) > 0
    for unbiased, biased in zip(fits[0.0], fits[1.0], strict=True):
        assert (unbiased.x, unbiased.y, unbiased.threshold) == (
            biased.x, biased.y, biased.threshold,
        )
        assert unbiased.confidence == biased.confidence
    assert prefer_large.ranking_score < prefer_large.confidence


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_automatic_threshold_continuity_rejects_noise_driven_size_jumps() -> None:
    tracker = MultiEyeTracker((0,), TrackerConfig(min_confidence=0.35))
    diameters: list[float] = []
    thresholds: list[int] = []
    threshold_fractions: list[float] = []

    for index in range(30):
        offset = round(5.0 * np.sin(index / 5.0))
        result = tracker.process(
            AnalysisFrame(index, index, ((0, _synthetic_noisy_eye(index, offset)),))
        )
        assert result.eyes[0].valid
        diagnostics = result.diagnostics["eyes"]["0"]
        diameters.append(result.eyes[0].pupil_diameter)
        thresholds.append(int(diagnostics["threshold"]))
        threshold_fractions.append(float(diagnostics["threshold_fraction"]))

    assert max(diameters) - min(diameters) < 1.0
    assert max(threshold_fractions) - min(threshold_fractions) < 0.01
    assert max(thresholds) - min(thresholds) >= 5


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_rejects_implausibly_elliptical_dark_region() -> None:
    image = np.full((160, 240), 180, dtype=np.uint8)
    cv2.ellipse(image, (120, 80), (40, 15), 12, 0, 360, 20, -1)

    candidate = AdaptivePupilDetector(TrackerConfig(min_confidence=0.35)).detect(image)

    assert candidate is None


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_rejects_pupil_contour_with_long_dark_tendril() -> None:
    image = np.full((160, 240), 180, dtype=np.uint8)
    cv2.ellipse(image, (100, 80), (28, 22), 0, 0, 360, 25, -1)
    cv2.rectangle(image, (125, 77), (165, 83), 25, -1)

    candidate = AdaptivePupilDetector(TrackerConfig(min_confidence=0.35)).detect(image)

    assert candidate is None


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_minimum_pupil_diameter_applies_to_reported_diameter() -> None:
    image = np.full((160, 240), 180, dtype=np.uint8)
    cv2.ellipse(image, (120, 80), (12, 5), 0, 0, 360, 20, -1)
    detector = AdaptivePupilDetector(
        TrackerConfig(
            min_pupil_diameter_px=20,
            min_axis_ratio=0.2,
            min_confidence=0.35,
        )
    )

    assert detector.detect(image) is None


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_tracker_reacquires_without_stale_prior_after_blink() -> None:
    tracker = MultiEyeTracker((0,), TrackerConfig(min_confidence=0.35))
    blank = np.full((160, 240), 145, dtype=np.uint8)

    first = tracker.process(AnalysisFrame(1, 100, ((0, _synthetic_eye_at(40)),)))
    tracker.process(AnalysisFrame(2, 200, ((0, blank),)))
    blink = tracker.process(AnalysisFrame(3, 300, ((0, blank),)))
    reacquired = tracker.process(
        AnalysisFrame(4, 400, ((0, _synthetic_eye_at(200)),))
    )

    assert first.eyes[0].valid
    assert blink.eyes[0].blink
    assert reacquired.eyes[0].valid
    assert reacquired.eyes[0].x == pytest.approx(200, abs=3)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_missing_pupil_reports_zero_then_blink() -> None:
    tracker = MultiEyeTracker((0,), TrackerConfig(min_confidence=0.35))
    tracker.process(AnalysisFrame(1, 100, ((0, _synthetic_eye()),)))
    first = tracker.process(AnalysisFrame(2, 200, ((0, _synthetic_eye(blink=True)),))).eyes[0]
    second = tracker.process(AnalysisFrame(3, 300, ((0, _synthetic_eye(blink=True)),))).eyes[0]
    assert not first.valid and first.pupil_diameter == 0.0
    assert not first.blink
    assert second.blink and second.pupil_diameter == 0.0
