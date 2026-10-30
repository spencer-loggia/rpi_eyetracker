from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest

try:
    import cv2
except ImportError:
    cv2 = None

from macaque_tracker.config import TrackerConfig
from macaque_tracker.models import (
    AnalysisFrame,
    EyeFitSettings,
    EyeImageSettings,
    EyeMeasurement,
    decoded_monochrome_frame,
)
from macaque_tracker.tracker import (
    AdaptivePupilDetector,
    MultiEyeTracker,
    PupilState,
    TemporalEyeTracker,
    _adaptive_appearance_scores,
    _pupil_size_bias_adjustment,
    apply_eye_image_settings,
    apply_normalized_eye_settings,
    detector_config_for_eye,
    fit_settings_from_config,
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


def test_cached_normalized_crop_is_unchanged_by_software_controls() -> None:
    normalized = np.array([[12.5, 90.0], [180.0, 260.0]], dtype=np.float32)
    original = normalized.copy()
    settings = EyeImageSettings(gain=1.2, brightness=-0.1, contrast=1.3)
    adjusted = apply_normalized_eye_settings(normalized, settings)

    expected = np.clip(original * 1.2 * 1.3 + 128 * (1 - 1.3) - 25.5, 0, 255)
    assert np.array_equal(adjusted, expected.astype(np.uint8))
    assert np.array_equal(normalized, original)
    assert not np.shares_memory(adjusted, normalized)


def test_calibrated_fit_overrides_only_per_eye_limits() -> None:
    shared = TrackerConfig(position_smoothing=0.5, max_position_jump_fraction=0.3)
    fit = EyeFitSettings(12, 0.6, 0.5, 10, 0.7, 0.4, 1.5, 2.0, 0.12, 0.5)
    settings = EyeImageSettings(pupil_size_bias=-0.5, fit=fit)
    effective = detector_config_for_eye(shared, settings)

    for name, value in fit.as_dict().items():
        assert getattr(effective, name) == value
    assert effective.pupil_size_bias == -0.5
    assert effective.position_smoothing == shared.position_smoothing
    assert effective.max_position_jump_fraction == shared.max_position_jump_fraction
    assert effective.threshold_percentiles == shared.threshold_percentiles
    assert shared.min_confidence == 0.48


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_calibrated_percentile_coverage_reaches_a_dilated_pupil():
    yy, xx = np.mgrid[:160, :240]
    radius = ((xx - 120) / 65) ** 2 + ((yy - 80) / 55) ** 2
    image = np.full((160, 240), 210, np.uint8)
    image[radius <= 1] = 35 + 60 * radius[radius <= 1]
    config = TrackerConfig(max_pupil_diameter_fraction=1, threshold_range_fraction=0.12,
                           appearance_weight=0, pupil_size_bias=1)
    original = AdaptivePupilDetector(config).detect(image)
    tuned = AdaptivePupilDetector(replace(config, threshold_percentile_scale=2)).detect(image)
    assert original is not None and original.diameter < 85
    assert tuned is not None and tuned.diameter == pytest.approx(np.sqrt(130 * 110), abs=4)
    assert (tuned.x, tuned.y) == pytest.approx((120, 80), abs=0.2)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_calibrated_ellipse_strictness_rejects_dark_patch_but_accepts_pupil():
    patch = np.full((160, 240), 185, np.uint8)
    cv2.rectangle(patch, (85, 60), (155, 105), 24, -1)
    pupil = np.full_like(patch, 185)
    cv2.ellipse(pupil, (120, 80), (35, 25), 0, 0, 360, 24, -1)
    assert AdaptivePupilDetector(TrackerConfig()).detect(patch) is not None
    tuned = AdaptivePupilDetector(TrackerConfig(max_fit_residual=0.06))
    assert tuned.detect(patch) is None
    detected = tuned.detect(pupil)
    assert detected is not None
    assert detected.diameter == pytest.approx(np.sqrt(70 * 50), abs=4)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_calibrated_fit_is_applied_independently_during_live_tracking() -> None:
    fit = EyeFitSettings(8, 0.75, 0.45, 8, 0.48)
    tracker = MultiEyeTracker(
        (0, 1), TrackerConfig(min_pupil_diameter_px=70),
        {0: EyeImageSettings(fit=fit), 1: EyeImageSettings()},
    )
    try:
        result = tracker.process(AnalysisFrame(1, 100, ((0, _synthetic_eye()), (1, _synthetic_eye()))))
        assert result.eyes[0].valid
        assert not result.eyes[1].valid
    finally:
        tracker.close()


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_calibrated_dark_reference_recovers_small_pupil_inside_bright_iris() -> None:
    image = np.full((160, 240), 210, dtype=np.uint8)
    cv2.ellipse(image, (120, 80), (55, 40), 0, 0, 360, 75, -1)
    cv2.ellipse(image, (120, 80), (12, 10), 0, 0, 360, 25, -1)
    config = TrackerConfig()
    original = AdaptivePupilDetector(config).detect(image)
    fit = EyeFitSettings(8, 0.75, 0.45, 8, 0.48, dark_reference_percentile=0.5)
    calibrated = AdaptivePupilDetector(
        detector_config_for_eye(config, EyeImageSettings(fit=fit)),
    ).detect(image)

    # Less than 2% of this crop is pupil: the old dark anchor was already iris.
    assert original is not None and original.diameter > 80
    assert calibrated is not None
    assert calibrated.diameter == pytest.approx(np.sqrt(24 * 20), abs=1.5)
    assert (calibrated.x, calibrated.y) == pytest.approx((120, 80), abs=0.1)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_calibrated_threshold_range_reaches_large_uneven_pupil_boundary() -> None:
    yy, xx = np.mgrid[:160, :240]
    radius = ((xx - 120) / 65) ** 2 + ((yy - 80) / (65 * 0.85)) ** 2
    image = np.full((160, 240), 210, dtype=np.uint8)
    image[radius <= 1] = 35 + 60 * radius[radius <= 1]
    base_fit = EyeFitSettings(8, 1.0, 0.45, 8, 0.48)
    settings = EyeImageSettings(pupil_size_bias=1.0, fit=base_fit)
    original = TemporalEyeTracker(0, TrackerConfig(), settings).process(image)
    settings = replace(settings, fit=replace(base_fit, threshold_range_fraction=0.65))
    calibrated = TemporalEyeTracker(0, TrackerConfig(), settings).process(image)

    # The camera remains bright as the pupil dilates. Low-percentile thresholds
    # fall inside the pupil; changing only contrast cannot select the outer edge.
    assert original.valid and original.pupil_diameter < 100
    assert calibrated.valid
    assert calibrated.pupil_diameter == pytest.approx(130 * np.sqrt(0.85), abs=3)
    assert (calibrated.x, calibrated.y) == pytest.approx((120, 80), abs=0.2)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_calibrated_large_pupil_limit_is_not_overruled_by_hidden_crop_area_limit() -> None:
    image = np.full((100, 100), 210, dtype=np.uint8)
    cv2.circle(image, (50, 50), 43, 25, -1)
    config = TrackerConfig(max_pupil_diameter_fraction=0.95)
    candidate = AdaptivePupilDetector(config).detect(image)

    assert candidate is not None
    assert candidate.diameter == pytest.approx(86, abs=4)
    assert (candidate.x, candidate.y) == pytest.approx((50, 50), abs=0.1)
    # Raising the explicit size limit must not turn an unstructured frame into
    # a pupil; nor may a pupil outside that configured limit be accepted.
    assert AdaptivePupilDetector(config).detect(np.full_like(image, 25)) is None
    assert AdaptivePupilDetector(replace(config, max_pupil_diameter_fraction=0.7)).detect(
        image,
    ) is None


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_calibrated_dilation_and_blink_reacquisition_report_current_size() -> None:
    fit = EyeFitSettings(8, 0.95, 0.45, 8, 0.48, 0.45, 0.5)
    settings = EyeImageSettings(fit=fit)
    config = TrackerConfig(max_pupil_diameter_fraction=0.5)
    tracker = TemporalEyeTracker(0, config, settings)
    for radius in range(20, 67, 4):
        image = np.full((160, 240), 210, dtype=np.uint8)
        cv2.ellipse(image, (120, 80), (radius, round(radius * 0.85)), 0, 0, 360, 25, -1)
        result = tracker.process(image)
        assert result.valid
        assert result.pupil_diameter == pytest.approx(2 * radius * np.sqrt(0.85), abs=4)
    for _ in range(2):
        assert not tracker.process(np.full_like(image, 210)).valid
    result = tracker.process(image)
    fresh = TemporalEyeTracker(0, config, settings).process(image)
    assert result.valid and not result.blink
    assert (result.x, result.y, result.pupil_diameter) == pytest.approx(
        (fresh.x, fresh.y, fresh.pupil_diameter), abs=0.01,
    )


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_pupil_state_holds_misses_and_applies_explicit_smoothing() -> None:
    candidate = AdaptivePupilDetector(TrackerConfig()).detect(_synthetic_eye())
    assert candidate is not None
    prior = PupilState()
    prior.update(candidate, position_alpha=0.2, diameter_alpha=0.2)
    assert (prior.x, prior.y, prior.diameter) == (candidate.x, candidate.y, candidate.diameter)

    moved = replace(candidate, x=candidate.x + 10, equivalent_diameter=candidate.diameter + 8)
    prior.update(moved, position_alpha=0.25, diameter_alpha=0.5)
    assert prior.x == candidate.x + 2.5
    assert prior.diameter == candidate.diameter + 4
    prior.update(None)
    prior.update(None)
    assert prior.missing_frames == 2 and prior.has_lock
    assert prior.x == candidate.x + 2.5

    prior.update(moved)
    assert (prior.x, prior.y, prior.diameter) == (moved.x, moved.y, moved.diameter)
    assert prior.missing_frames == 0


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("profile", ["default", "manual", "optimized"])
def test_live_and_calibration_agree_after_saccade_near_dark_distractor_and_blink(profile) -> None:
    config = TrackerConfig()
    settings = EyeImageSettings()
    if profile == "manual":
        settings = EyeImageSettings(gain=0.7, brightness=0.04, contrast=1.3)
    elif profile == "optimized":
        settings = EyeImageSettings(fit=fit_settings_from_config(config))
    tracker = TemporalEyeTracker(0, config, settings)
    detector = AdaptivePupilDetector(detector_config_for_eye(config, settings))
    first = np.full((160, 240), 185, np.uint8)
    cv2.ellipse(first, (40, 100), (20, 16), 0, 0, 360, 24, -1)
    moved = np.full_like(first, 185)
    cv2.ellipse(moved, (190, 90), (28, 22), 0, 0, 360, 24, -1)
    cv2.ellipse(moved, (50, 28), (14, 10), 0, 0, 360, 40, -1)
    blink = np.full_like(first, 185)
    cv2.line(blink, (30, 85), (200, 85), 24, 7)

    # A previous position gate rejected the true pupil after this saccade and
    # selected the small dark patch near the old lock instead. Selection must
    # match calibration immediately, including the first reopened frame.
    for raw, center in ((first, (40, 100)), (moved, (190, 90)), (blink, None),
                        (blink, None), (moved, (190, 90)), (first, (40, 100))):
        candidate = detector.detect(apply_eye_image_settings(raw, settings))
        live = tracker.process(raw)
        assert live.valid == (candidate is not None)
        if center is None:
            assert not live.valid
        else:
            assert candidate is not None and not live.blink
            assert (live.x, live.y) == pytest.approx(center, abs=0.1)
            assert (live.x, live.y, live.pupil_diameter, live.confidence) == pytest.approx(
                (candidate.x, candidate.y, candidate.diameter, candidate.confidence), abs=1e-9,
            )


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("cutoff", [0.84, 0.9])
def test_high_calibrated_confidence_accepts_first_saccade_and_reopened_frames(cutoff) -> None:
    settings = EyeImageSettings(fit=EyeFitSettings(8, 0.75, 0.45, 8, cutoff))
    tracker = TemporalEyeTracker(0, TrackerConfig(), settings)
    assert tracker.process(_synthetic_eye_at(40)).valid

    moved = tracker.process(_synthetic_eye_at(130))
    assert moved.valid and moved.x == pytest.approx(130, abs=1)
    assert tracker.state.missing_frames == 0
    for _ in range(2):
        assert not tracker.process(np.full((160, 240), 185, np.uint8)).valid
    reopened = tracker.process(_synthetic_eye_at(40))
    assert reopened.valid and not reopened.blink
    assert reopened.x == pytest.approx(40, abs=1)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
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


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("level", [0, 128, 255])
def test_normalization_preserves_flat_frames_and_does_not_expand_noise(level: int) -> None:
    image = np.full((40, 60), level, dtype=np.uint8)
    image[10:30, 20:40] = max(0, level - 4)

    adjusted = apply_eye_image_settings(image, EyeImageSettings())

    assert np.array_equal(adjusted, image)
    assert not np.shares_memory(adjusted, image)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_normalization_bounds_amplification_of_low_contrast_structure() -> None:
    image = np.full((40, 60), 100, dtype=np.uint8)
    image[10:30, 20:40] = 75

    adjusted = apply_eye_image_settings(image, EyeImageSettings())

    assert 0 < int(adjusted[0, 0]) - int(adjusted[20, 30]) <= 50


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_normalization_handles_strided_crops() -> None:
    image = np.random.default_rng(3).integers(0, 256, (160, 240), dtype=np.uint8)
    crop = image[::2, ::-2]
    settings = EyeImageSettings(gain=1.2, brightness=0.06, contrast=1.1)
    strided = apply_eye_image_settings(crop, settings)
    contiguous = apply_eye_image_settings(crop.copy(), settings)

    assert np.array_equal(strided, contiguous)


def test_eye_normalization_requires_opencv(monkeypatch) -> None:
    monkeypatch.setattr("macaque_tracker.tracker.cv2", None)
    with pytest.raises(RuntimeError, match="OpenCV"):
        apply_eye_image_settings(np.zeros((24, 24), dtype=np.uint8), EyeImageSettings())


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("lighting", ["horizontal", "vertical", "bright_patch"])
def test_spatial_normalization_flattens_lighting_without_moving_pupil(lighting) -> None:
    base = _synthetic_dark_iris()
    y, x = np.mgrid[:base.shape[0], :base.shape[1]].astype(float)
    x /= base.shape[1] - 1
    y /= base.shape[0] - 1
    if lighting == "horizontal":
        illumination = 0.4 + 0.6 * x
    elif lighting == "vertical":
        illumination = 0.4 + 0.6 * y
    else:
        illumination = 1.0 - 0.6 * (0.7 * (2 * x - 1) ** 2 + 0.3 * (2 * y - 1) ** 2)
    uneven = (base * illumination).astype(np.uint8)
    original = uneven.copy()
    settings = EyeImageSettings()
    adjusted = apply_eye_image_settings(uneven, settings)
    uniform = apply_eye_image_settings(base, settings)

    # Compare known, identical background tissue on opposite sides of the eye.
    # Whole-crop percentile scaling alone cannot remove this spatial gradient.
    background = np.ones(base.shape, dtype=np.uint8)
    cv2.ellipse(background, (124, 82), (65, 50), 8, 0, 360, 0, -1)
    visible = background != 0
    assert np.std(adjusted[visible]) < 0.15 * np.std(uneven[visible])
    assert adjusted.shape == uneven.shape
    assert np.array_equal(uneven, original)

    detector = AdaptivePupilDetector(TrackerConfig())
    reference = detector.detect(uniform)
    candidate = detector.detect(adjusted)
    assert reference is not None and candidate is not None
    assert (candidate.x, candidate.y) == pytest.approx((reference.x, reference.y), abs=0.5)
    assert candidate.diameter == pytest.approx(reference.diameter, abs=1.0)
    assert candidate.major == pytest.approx(reference.major, abs=1.0)
    assert candidate.minor == pytest.approx(reference.minor, abs=1.0)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_normalized_tracker_uses_one_setting_and_one_fit_across_lighting_changes(
    monkeypatch,
) -> None:
    settings = EyeImageSettings(gain=1.2, brightness=0.06)
    tracker = MultiEyeTracker((0,), TrackerConfig(), {0: settings})
    detect = tracker.trackers[0].detector.detect
    calls = []

    def count_fit(image):
        calls.append(image)
        return detect(image)

    monkeypatch.setattr(tracker.trackers[0].detector, "detect", count_fit)
    base = _synthetic_dark_iris()
    lighting = [(1.0, 0), (0.65, 0), (0.65, 110), (0.2, 200), (1.0, 0)]
    diameters = []
    try:
        for index, (gain, offset) in enumerate(lighting):
            image = np.clip(base.astype(float) * gain + offset, 0, 255).astype(np.uint8)
            original = image.copy()
            result = tracker.process(AnalysisFrame(index, index, ((0, image),)))
            eye = result.eyes[0]
            # The 0.2x + 200 frame would clip completely with these software
            # settings without normalization; the same pupil must survive it.
            assert eye.valid and not eye.blink
            assert (eye.x, eye.y) == pytest.approx((124, 82), abs=0.75)
            diameters.append(eye.pupil_diameter)
            assert np.array_equal(image, original)
    finally:
        tracker.close()
    assert len(calls) == len(lighting)
    assert max(diameters) - min(diameters) < 1.0


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_normalization_is_independent_for_each_cropped_eye() -> None:
    base = _synthetic_dark_iris()
    dark = (base.astype(float) * 0.65).astype(np.uint8)
    bright = (base.astype(float) * 0.2 + 200).astype(np.uint8)
    settings = EyeImageSettings(gain=1.2, brightness=0.06)
    tracker = MultiEyeTracker((0, 1), TrackerConfig(), {0: settings, 1: settings})
    try:
        result = tracker.process(AnalysisFrame(1, 1, ((0, dark), (1, bright))))
    finally:
        tracker.close()
    assert all(eye.valid for eye in result.eyes)
    for eye in result.eyes:
        assert (eye.x, eye.y) == pytest.approx((124, 82), abs=0.75)
    assert result.eyes[0].pupil_diameter == pytest.approx(
        result.eyes[1].pupil_diameter, abs=1.0,
    )


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


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("pupil_intensity", [24, 120, 180, 220])
def test_confidence_in_clear_grey_pupil_is_independent_of_remote_dark_pixels(
    pupil_intensity: int,
) -> None:
    image = np.full((160, 240), 255, dtype=np.uint8)
    cv2.ellipse(image, (124, 82), (28, 20), 11, 0, 360, pupil_intensity, -1)
    cluttered = image.copy()
    cv2.rectangle(cluttered, (0, 0), (239, 10), 0, -1)
    detector = AdaptivePupilDetector(TrackerConfig(min_confidence=0.88))

    clean = detector.detect(image)
    with_clutter = detector.detect(cluttered)

    assert clean is not None and with_clutter is not None
    assert with_clutter.confidence == pytest.approx(clean.confidence, abs=0.005)
    assert (with_clutter.x, with_clutter.y) == pytest.approx((124, 82), abs=0.5)
    assert with_clutter.diameter == pytest.approx(clean.diameter, abs=0.5)
    # The original whole-crop preference still ranks competing candidates.
    assert with_clutter.ranking_score < clean.ranking_score


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_tuned_grey_pupil_on_clipped_white_background_keeps_high_confidence() -> None:
    image = _synthetic_dark_iris()
    cv2.rectangle(image, (0, 0), (239, 10), 0, -1)
    tracker = MultiEyeTracker(
        (0,), TrackerConfig(min_confidence=0.88),
        {0: EyeImageSettings(gain=3.0, brightness=0.2)},
    )
    try:
        result = tracker.process(AnalysisFrame(1, 1, ((0, image),)))
    finally:
        tracker.close()

    eye = result.eyes[0]
    assert eye.valid and eye.confidence >= 0.88
    assert (eye.x, eye.y) == pytest.approx((124, 82), abs=0.75)
    diagnostics = result.diagnostics["eyes"]["0"]
    assert diagnostics["median_intensity"] > 128
    assert diagnostics["median_intensity"] + diagnostics["contrast"] == pytest.approx(255)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_confidence_excludes_one_pixel_boundary_but_keeps_interior_occlusion_evidence() -> None:
    image = np.full((100, 140), 255, np.uint8)
    cv2.ellipse(image, (70, 50), (9, 7), 12, 0, 360, 25, -1)
    cv2.circle(image, (77, 50), 3, 255, -1)
    detector = AdaptivePupilDetector(TrackerConfig())
    candidate = detector.detect(image)

    assert candidate is not None
    # The ellipse boundary contains clipped white pixels, but its observed
    # interior is uniform. Those pixels still penalize ranking/gates; they
    # must not masquerade as full-range noise in the confidence estimate.
    assert candidate.interior_spread == 230
    assert candidate.confidence_interior_spread == 0
    assert candidate.confidence > 0.88
    assert (candidate.x, candidate.y) == pytest.approx((70, 50), abs=2)

    # A larger occlusion reaches the interior. Excluding one boundary pixel
    # must not erase that evidence or inflate its confidence as well.
    cv2.circle(image, (77, 50), 7, 255, -1)
    occluded = detector.detect(image)
    assert occluded is not None
    assert occluded.confidence_interior_spread == 230
    assert occluded.confidence < 0.8


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_low_contrast_and_noise_still_reduce_confidence() -> None:
    detector = AdaptivePupilDetector(TrackerConfig())
    candidates = []
    for pupil, background, noise in [(180, 255, 0), (245, 255, 0), (190, 200, 2)]:
        image = np.full((160, 240), background, dtype=np.float32)
        cv2.ellipse(image, (124, 82), (28, 20), 11, 0, 360, pupil, -1)
        image += np.random.default_rng(4).normal(0, noise, image.shape)
        candidate = detector.detect(np.clip(image, 0, 255).astype(np.uint8))
        assert candidate is not None
        candidates.append(candidate)

    strong, weak, noisy = candidates
    assert strong.confidence > 0.88
    assert noisy.confidence < weak.confidence < strong.confidence - 0.15
    assert noisy.local_separation < strong.local_separation
    assert noisy.local_uniformity < strong.local_uniformity


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("level", [50, 150, 250])
@pytest.mark.parametrize("occlusion", ["flat", "noise", "eyelid", "saturated"])
def test_grey_pupil_confidence_change_does_not_create_fits_during_occlusion(
    level: int, occlusion: str,
) -> None:
    image = np.full((160, 240), level, dtype=np.float32)
    if occlusion == "noise":
        image += np.random.default_rng(4).normal(0, 12, image.shape)
    elif occlusion == "eyelid":
        cv2.line(image, (20, 82), (220, 82), 20, 9)
    elif occlusion == "saturated":
        image[:] = 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    tracker = MultiEyeTracker(
        (0,), TrackerConfig(), {0: EyeImageSettings(gain=3.0, brightness=0.2)},
    )
    try:
        locked = tracker.process(AnalysisFrame(1, 1, ((0, _synthetic_dark_iris()),)))
        assert locked.eyes[0].valid
        for index in (2, 3):
            lost = tracker.process(AnalysisFrame(index, index, ((0, image),))).eyes[0]
            assert not lost.valid and lost.confidence == 0.0
            assert lost.pupil_diameter == 0.0
        assert lost.blink
    finally:
        tracker.close()


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_grey_pupil_confidence_still_respects_configured_minimum() -> None:
    image = np.full((160, 240), 255, dtype=np.uint8)
    cv2.ellipse(image, (124, 82), (28, 20), 11, 0, 360, 180, -1)
    cv2.rectangle(image, (0, 0), (239, 10), 0, -1)

    assert AdaptivePupilDetector(TrackerConfig(min_confidence=0.88)).detect(image) is not None
    assert AdaptivePupilDetector(TrackerConfig(min_confidence=0.99)).detect(image) is None


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
    assert 0 <= diagnostics["local_separation"] <= 1
    assert 0 <= diagnostics["local_uniformity"] <= 1
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
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("angle", np.arange(0, 360, 45))
@pytest.mark.parametrize("noise", [0.0, 3.0, 6.0])
def test_edge_glint_recovery_uses_visible_boundary_without_centroid_bias(
    angle: int, noise: float, normalize: bool,
) -> None:
    radians = np.deg2rad(angle)
    position = (round(124 + 26 * np.cos(radians)), round(82 + 20 * np.sin(radians)))
    image = _synthetic_glint_eye(radius=12, position=position, noise=noise)
    if normalize:
        image = apply_eye_image_settings(image, EyeImageSettings())

    candidate = AdaptivePupilDetector(TrackerConfig()).detect(image)

    assert candidate is not None and candidate.boundary_recovered
    assert candidate.x == pytest.approx(124, abs=1)
    assert candidate.y == pytest.approx(82, abs=1)
    assert candidate.diameter == pytest.approx(np.sqrt(56 * 40), rel=0.10)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("radius,position", [(16, (124, 82)), (20, (124, 82)), (20, (147, 82))])
def test_heavy_glint_occlusion_is_rejected_without_an_iris_fallback(
    radius: int, position: tuple[int, int], normalize: bool,
) -> None:
    image = _synthetic_glint_eye(radius=radius, position=position)
    if normalize:
        image = apply_eye_image_settings(image, EyeImageSettings())

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
@pytest.mark.parametrize("position", [(150, 82), (98, 82), (124, 102), (124, 62)])
def test_calibration_mask_scores_recovered_pupil_geometry_through_edge_glints(position) -> None:
    image = _synthetic_glint_eye(radius=12, position=position)
    truth = np.zeros_like(image)
    cv2.ellipse(truth, (124, 82), (28, 20), 11, 0, 360, 255, -1)
    candidate, mask = AdaptivePupilDetector(TrackerConfig()).detect_with_mask(image)

    assert candidate is not None and candidate.boundary_recovered
    overlap = np.count_nonzero((mask != 0) & (truth != 0))
    union = np.count_nonzero((mask != 0) | (truth != 0))
    assert overlap / union > 0.85
    assert np.count_nonzero(mask) == pytest.approx(np.pi * candidate.diameter**2 / 4, rel=0.06)


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
def test_stateless_threshold_selection_remains_stable_across_noise_and_brightness() -> None:
    tracker = MultiEyeTracker((0,), TrackerConfig(min_confidence=0.35))
    diameters: list[float] = []
    detector = tracker.trackers[0].detector

    for index in range(30):
        offset = round(5.0 * np.sin(index / 5.0))
        image = _synthetic_noisy_eye(index, offset)
        result = tracker.process(AnalysisFrame(index, index, ((0, image),)))
        eye = result.eyes[0]
        assert eye.valid
        independent = detector.detect(apply_eye_image_settings(image, EyeImageSettings()))
        assert independent is not None
        assert (eye.x, eye.y) == pytest.approx((124, 82), abs=0.6)
        assert eye.pupil_diameter == independent.diameter
        diameters.append(eye.pupil_diameter)

    # Independent noisy images can select slightly different edges; do not
    # hide that uncertainty with a threshold inherited from previous frames.
    expected_diameter = np.sqrt(58 * 42)
    assert np.mean(diameters) == pytest.approx(expected_diameter, rel=0.05)
    assert diameters == pytest.approx([expected_diameter] * 30, rel=0.12)


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
@pytest.mark.parametrize("missing_frames", [2, 12])
@pytest.mark.parametrize("axes", [(18, 14), (40, 28)])
@pytest.mark.parametrize("calibrated", [False, True])
def test_blink_reacquires_changed_size_and_position_on_first_visible_frame(
    missing_frames: int, axes: tuple[int, int], calibrated: bool,
) -> None:
    fit = EyeFitSettings(8.0, 0.75, 0.45, 8.0, 0.48) if calibrated else None
    settings = EyeImageSettings(fit=fit)
    # Even deliberately enabled smoothing must be released after a blink.
    config = TrackerConfig(position_smoothing=0.2, diameter_smoothing=0.2)
    tracker = TemporalEyeTracker(0, config, settings)
    assert tracker.process(_synthetic_eye_at(40)).valid
    for _ in range(missing_frames):
        assert not tracker.process(np.full((160, 240), 185, dtype=np.uint8)).valid
    image = np.full((160, 240), 185, dtype=np.uint8)
    cv2.ellipse(image, (195, 50), axes, 25, 0, 360, 24, -1)

    # Neither a prior pupil scale nor its location may delay reacquisition.
    fresh = TemporalEyeTracker(0, config, settings).process(image)
    reacquired = tracker.process(image)
    assert reacquired.valid and fresh.valid and not reacquired.blink
    assert (reacquired.x, reacquired.y, reacquired.pupil_diameter) == pytest.approx(
        (fresh.x, fresh.y, fresh.pupil_diameter), abs=0.01,
    )
    assert tracker.state.missing_frames == 0


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("missing_frames", [0, 2, 12])
def test_gaze_jump_and_foreshortening_preserve_current_frame_geometry(missing_frames: int) -> None:
    tracker = TemporalEyeTracker(0, TrackerConfig())
    assert tracker.process(_synthetic_dark_iris()).valid
    for _ in range(missing_frames):
        tracker.process(np.full((160, 240), 185, dtype=np.uint8))
    image = np.full((160, 240), 185, dtype=np.uint8)
    cv2.ellipse(image, (175, 50), (28, 14), 25, 0, 360, 24, -1)

    fresh = TemporalEyeTracker(0, TrackerConfig()).process(image)
    moved = tracker.process(image)
    assert moved.valid and fresh.valid
    assert (moved.x, moved.y, moved.pupil_diameter) == pytest.approx(
        (fresh.x, fresh.y, fresh.pupil_diameter), abs=0.25,
    )


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_missing_pupil_reports_zero_then_blink() -> None:
    tracker = MultiEyeTracker((0,), TrackerConfig(min_confidence=0.35))
    tracker.process(AnalysisFrame(1, 100, ((0, _synthetic_eye()),)))
    first = tracker.process(AnalysisFrame(2, 200, ((0, _synthetic_eye(blink=True)),))).eyes[0]
    second = tracker.process(AnalysisFrame(3, 300, ((0, _synthetic_eye(blink=True)),))).eyes[0]
    assert not first.valid and first.pupil_diameter == 0.0
    assert not first.blink
    assert second.blink and second.pupil_diameter == 0.0
