from __future__ import annotations

import json
import stat

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from macaque_tracker import calibration as calibration_module
from macaque_tracker.calibration import (
    CalibrationCancelled,
    ControlSample,
    load_control_samples,
    optimize_calibration,
    save_control_samples,
)
from macaque_tracker.config import TrackerConfig
from macaque_tracker.models import EyeImageSettings
from macaque_tracker.tracker import (
    AdaptivePupilDetector,
    apply_eye_image_settings,
    detector_config_for_eye,
)


def _control(*, blink=False, x=120):
    image = np.full((160, 240), 185, np.uint8)
    mask = np.zeros_like(image)
    if blink:
        # The eyelid supplies enough dark pixels to establish a dark reference;
        # the separate round lash patch then looks like a plausible small pupil.
        cv2.line(image, (30, 85), (200, 85), 24, 7)
        cv2.circle(image, (180, 38), 10, 24, -1)
    else:
        cv2.ellipse(image, (x, 80), (53, 39), 11, 0, 360, 72, -1)
        cv2.ellipse(image, (x, 80), (28, 20), 11, 0, 360, 24, -1)
        cv2.ellipse(mask, (x, 80), (28, 20), 11, 0, 360, 255, -1)
    return ControlSample(image, mask)


def _detect(sample, settings):
    detector = AdaptivePupilDetector(detector_config_for_eye(TrackerConfig(), settings))
    return detector.detect(apply_eye_image_settings(sample.image, settings))


def _variable_pupil_controls():
    # Bright IR illumination persists when the pupil dilates. Its outer pixels
    # are lighter than its centre, so aggressive gain loses the true boundary.
    small = np.full((160, 240), 210, np.uint8)
    small_mask = np.zeros_like(small)
    cv2.ellipse(small, (90, 80), (55, 40), 0, 0, 360, 75, -1)
    cv2.ellipse(small, (90, 80), (12, 10), 0, 0, 360, 25, -1)
    cv2.ellipse(small_mask, (90, 80), (12, 10), 0, 0, 360, 255, -1)

    yy, xx = np.mgrid[:160, :240]
    radius = ((xx - 145) / 65) ** 2 + ((yy - 80) / (65 * 0.85)) ** 2
    large = np.full_like(small, 210)
    large[radius <= 1] = 35 + 60 * radius[radius <= 1]
    large_mask = np.uint8(radius <= 1) * 255

    blink = np.full_like(small, 210)
    cv2.line(blink, (30, 85), (200, 85), 24, 7)
    cv2.circle(blink, (180, 38), 9, 24, -1)
    return (
        ControlSample(small, small_mask),
        ControlSample(large, large_mask),
        ControlSample(blink, np.zeros_like(blink)),
    )


def test_control_sample_owns_unchanged_read_only_raw_pixels_and_mask():
    sample = _control()
    raw, painted = sample.image.copy(), sample.mask.copy()
    owned = ControlSample(raw[:, ::-1], painted[:, ::-1])
    expected_raw, expected_mask = raw[:, ::-1].copy(), painted[:, ::-1].copy()
    raw[:] = 0
    painted[:] = 0

    assert np.array_equal(owned.image, expected_raw)
    assert np.array_equal(owned.mask, expected_mask)
    for pixels in (owned.image, owned.mask):
        assert pixels.flags.c_contiguous and pixels.flags.owndata
        assert not pixels.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            pixels[0, 0] = 3


def test_editing_mask_shares_immutable_raw_pixels_and_owns_new_paint():
    sample = _control()
    raw, original_mask = sample.image.copy(), sample.mask.copy()
    paint = np.zeros_like(sample.mask)
    cv2.circle(paint, (80, 70), 15, 255, -1)
    expected_paint = paint.copy()

    edited = sample.with_mask(paint)
    paint[:] = 0

    assert edited.image is sample.image
    assert not edited.image.flags.writeable
    assert edited.mask.flags.owndata and edited.mask.flags.c_contiguous
    assert not edited.mask.flags.writeable
    assert np.array_equal(edited.mask, expected_paint)
    assert np.array_equal(sample.image, raw)
    assert np.array_equal(sample.mask, original_mask)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "gray"])
def test_editing_mask_rejects_invalid_paint_without_changing_sample(invalid):
    sample = _control()
    before = sample.mask.copy()
    paint = np.zeros_like(sample.mask)
    if invalid == "shape":
        paint = paint[:, :-1]
    elif invalid == "dtype":
        paint = paint.astype(bool)
    else:
        paint[0, 0] = 1
    with pytest.raises(ValueError):
        sample.with_mask(paint)
    assert np.array_equal(sample.mask, before)


@pytest.mark.parametrize("invalid", ["small", "image_type", "channels", "mask_type", "shape", "gray"])
def test_control_sample_rejects_invalid_pixels_and_masks(invalid):
    image = np.zeros((24, 24), np.uint8)
    mask = np.zeros_like(image)
    if invalid == "small":
        image, mask = image[:23], mask[:23]
    elif invalid == "image_type":
        image = image.astype(np.float32)
    elif invalid == "channels":
        image = image[:, :, None]
    elif invalid == "mask_type":
        mask = mask.astype(bool)
    elif invalid == "shape":
        mask = mask[:23]
    else:
        mask[3, 3] = 1
    with pytest.raises(ValueError):
        ControlSample(image, mask)


def test_optimization_recovers_saturated_pupils_and_rejects_round_blink_patch():
    samples = (_control(x=80), _control(x=160), _control(blink=True))
    initial = EyeImageSettings(gain=8, brightness=1)
    snapshots = tuple((sample.image.copy(), sample.mask.copy()) for sample in samples)
    assert _detect(samples[0], initial) is None
    assert _detect(samples[-1], EyeImageSettings()) is not None

    result = optimize_calibration(samples, initial, TrackerConfig())

    assert result.before_score == 0
    assert result.after_score > 0.85
    assert result.sample_scores[0] > 0.85 and result.sample_scores[1] > 0.85
    assert result.sample_scores[2] == 1
    # Each check starts with no temporal prior. Recovery and rejection therefore
    # come from the saved profile, never a wait for earlier frames to expire.
    for sample, x in zip(samples[:2], (80, 160), strict=True):
        candidate = _detect(sample, result.settings)
        assert candidate is not None
        assert candidate.x == pytest.approx(x, abs=1)
        assert candidate.y == pytest.approx(80, abs=1)
        assert 42 < candidate.diameter < 52
    assert _detect(samples[-1], result.settings) is None
    for sample, (image, mask) in zip(samples, snapshots, strict=True):
        assert np.array_equal(sample.image, image)
        assert np.array_equal(sample.mask, mask)
        assert not sample.image.flags.writeable and not sample.mask.flags.writeable


def test_joint_calibration_recovers_small_and_dilated_pupils_from_high_gain():
    samples = _variable_pupil_controls()
    initial = EyeImageSettings(gain=8, pupil_size_bias=0.2)
    clipped_large = _detect(samples[1], initial)
    assert clipped_large is not None and clipped_large.diameter < 40

    result = optimize_calibration(samples, initial, TrackerConfig())

    assert result.before_score < 0.4
    assert result.after_score > 0.85
    assert min(result.sample_scores[:2]) > 0.85
    assert result.sample_scores[2] == 1
    assert result.evaluations <= 160
    # Every frame is independently detectable, including a large pupil after a
    # blink. Calibration must not rely on temporal confirmation or smoothing.
    small = _detect(samples[0], result.settings)
    large = _detect(samples[1], result.settings)
    assert small is not None and large is not None
    assert small.x == pytest.approx(90, abs=1)
    assert large.x == pytest.approx(145, abs=1)
    assert 20 < small.diameter < 25
    assert 110 < large.diameter < 125
    assert _detect(samples[2], result.settings) is None


@pytest.mark.parametrize("count", [2, 6, 12])
def test_search_cannot_win_by_rejecting_everything_when_blinks_outnumber_pupils(count):
    samples = (_control(),) + (_control(blink=True),) * (count - 1)
    result = optimize_calibration(samples, EyeImageSettings(brightness=1), TrackerConfig())
    assert result.before_score == 0  # All-reject gets no credit for the blink majority.
    assert result.after_score > 0.85
    assert result.sample_scores[0] > 0.85
    assert result.sample_scores[1:] == (1.0,) * (count - 1)
    assert result.evaluations <= 160


def test_optimization_is_deterministic_bounded_and_normalizes_each_raw_crop_once(monkeypatch):
    samples = (_control(x=80), _control(x=160), _control(blink=True))
    normalizations = []
    normalize = calibration_module._normalized_eye_image

    def counted(image):
        normalizations.append(image)
        return normalize(image)

    monkeypatch.setattr(calibration_module, "_normalized_eye_image", counted)
    first = optimize_calibration(samples, EyeImageSettings(), TrackerConfig())
    assert len(normalizations) == len(samples)
    assert all(image is sample.image for image, sample in zip(normalizations, samples, strict=True))
    second = optimize_calibration(samples, EyeImageSettings(), TrackerConfig())
    assert first == second
    assert 1 <= first.evaluations <= 160
    assert len(normalizations) == 2 * len(samples)


@pytest.mark.parametrize("count", [0, 1])
def test_optimization_requires_at_least_two_samples(count):
    with pytest.raises(ValueError, match="at least 2"):
        optimize_calibration((_control(),) * count, EyeImageSettings(), TrackerConfig())


def test_optimization_requires_at_least_one_filled_positive_annotation():
    blink = _control(blink=True)
    with pytest.raises(ValueError, match="at least one"):
        optimize_calibration((blink, blink), EyeImageSettings(), TrackerConfig())
    point = blink.mask.copy()
    point[80, 120] = 255
    with pytest.raises(ValueError, match="filled pupil area"):
        optimize_calibration(
            (ControlSample(blink.image, point), blink), EyeImageSettings(), TrackerConfig(),
        )


def test_optimization_rejects_mixed_crop_geometry_and_non_samples():
    small = ControlSample(np.zeros((24, 24), np.uint8), np.zeros((24, 24), np.uint8))
    with pytest.raises(ValueError, match="same crop dimensions"):
        optimize_calibration((_control(), small), EyeImageSettings(), TrackerConfig())
    with pytest.raises(TypeError, match="ControlSample"):
        optimize_calibration((_control(), object()), EyeImageSettings(), TrackerConfig())


def test_cancellation_before_work_does_not_normalize_images(monkeypatch):
    def unexpected(_image):
        pytest.fail("Cancelled search should not normalize crops")

    monkeypatch.setattr(calibration_module, "_normalized_eye_image", unexpected)
    with pytest.raises(CalibrationCancelled):
        optimize_calibration(
            (_control(), _control(blink=True)), EyeImageSettings(), TrackerConfig(),
            cancelled=lambda: True,
        )


def test_cancellation_stops_during_search_without_applying_settings(monkeypatch):
    detections = 0
    detect = AdaptivePupilDetector.detect_with_mask

    def counted(self, image, *args, **kwargs):
        nonlocal detections
        detections += 1
        return detect(self, image, *args, **kwargs)

    monkeypatch.setattr(AdaptivePupilDetector, "detect_with_mask", counted)
    initial = EyeImageSettings(brightness=1)
    with pytest.raises(CalibrationCancelled):
        optimize_calibration(
            (_control(), _control(blink=True)), initial, TrackerConfig(),
            cancelled=lambda: detections >= 5,
        )
    assert detections == 5
    assert initial == EyeImageSettings(brightness=1)


@pytest.mark.parametrize("shared", [False, True])
def test_control_sidecar_round_trip_preserves_raw_images_and_empty_eye(tmp_path, shared):
    path = tmp_path / "calibration.json"
    context = {"camera_rotations": [0, 90, 0, 0], "rois": [[0.5, 0.1, 0.2, 0.7]]}
    bank = ((_control(), _control(blink=True)), ())
    saved = save_control_samples(path, bank, context=context, shared=shared)
    assert saved == tmp_path / "calibration.controls.npz"
    restored = load_control_samples(path, context=context)
    assert len(restored) == 2 and restored[1] == ()
    for original, sample in zip(bank[0], restored[0], strict=True):
        assert np.array_equal(sample.image, original.image)
        assert np.array_equal(sample.mask, original.mask)
        assert not sample.image.flags.writeable and not sample.mask.flags.writeable
    if shared:
        assert stat.S_IMODE(saved.stat().st_mode) == 0o666


def test_control_sidecar_preserves_more_than_six_examples_per_eye(tmp_path):
    path = tmp_path / "calibration.json"
    samples = (_control(x=80), _control(x=160)) + (_control(blink=True),) * 10
    save_control_samples(path, (samples,), context={})

    restored, = load_control_samples(path, context={})

    assert len(restored) == 12
    for original, sample in zip(samples, restored, strict=True):
        assert np.array_equal(sample.image, original.image)
        assert np.array_equal(sample.mask, original.mask)


def test_control_memory_budget_applies_to_optimization_save_and_load(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    samples = (_control(), _control(blink=True))
    saved = save_control_samples(path, (samples,), context={})
    before = saved.read_bytes()
    monkeypatch.setattr(
        calibration_module, "MAX_CONTROL_BYTES",
        sum(sample.image.nbytes + sample.mask.nbytes for sample in samples) - 1,
    )

    with pytest.raises(ValueError, match="memory budget"):
        optimize_calibration(samples, EyeImageSettings(), TrackerConfig())
    with pytest.raises(ValueError, match="memory budget"):
        save_control_samples(path, (samples,), context={})
    assert saved.read_bytes() == before
    with pytest.raises(ValueError, match="memory budget"):
        load_control_samples(path, context={})


def test_control_memory_budget_is_per_eye(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    sample = _control()
    monkeypatch.setattr(
        calibration_module, "MAX_CONTROL_BYTES", sample.image.nbytes + sample.mask.nbytes,
    )
    save_control_samples(path, ((sample,), (sample,)), context={})
    bank = load_control_samples(path, context={})
    assert tuple(len(samples) for samples in bank) == (1, 1)


def test_control_sidecar_missing_or_stale_geometry_returns_no_samples(tmp_path):
    path = tmp_path / "calibration.json"
    context = {"size": [240, 160], "rotation": 0}
    assert load_control_samples(path, context=context) == ()
    save_control_samples(path, ((_control(),),), context=context)
    assert load_control_samples(path, context={**context, "rotation": 90}) == ()
    assert load_control_samples(path, context={**context, "size": [160, 240]}) == ()


@pytest.mark.parametrize("corruption", ["eye_count", "image_shape"])
def test_control_sidecar_rejects_samples_inconsistent_with_matching_context(tmp_path, corruption):
    path = tmp_path / "calibration.json"
    context = {"source_size": [960, 160], "boxes": [[0, 0, 240, 160]],
               "rotations": [0, 0, 0, 0]}
    bank = ((_control(),), ()) if corruption == "eye_count" else (
        (ControlSample(np.zeros((24, 24), np.uint8), np.zeros((24, 24), np.uint8)),),
    )
    save_control_samples(path, bank, context=context)
    with pytest.raises(ValueError, match="does not match|do not match"):
        load_control_samples(path, context=context)


def test_control_sidecar_accepts_rotated_crop_dimensions(tmp_path):
    path = tmp_path / "calibration.json"
    context = {"source_size": [960, 160], "boxes": [[0, 0, 240, 160]],
               "rotations": [90, 0, 0, 0]}
    sample = _control()
    rotated = ControlSample(np.rot90(sample.image), np.rot90(sample.mask))
    save_control_samples(path, ((rotated,),), context=context)
    restored = load_control_samples(path, context=context)
    assert np.array_equal(restored[0][0].mask, rotated.mask)


def _write_archive(path, *, counts=(1,), image=None, mask=None, metadata=None):
    sample = _control()
    content = {"version": 1, "context": {}, "counts": list(counts)}
    np.savez_compressed(
        path.with_suffix(".controls.npz"),
        metadata=np.asarray(json.dumps(content)) if metadata is None else metadata,
        image_0_0=sample.image if image is None else image,
        mask_0_0=sample.mask if mask is None else mask,
    )


@pytest.mark.parametrize("counts", [(), (7,), (-1,), (True,), ("1",), (0, 0, 0), (2,)])
def test_control_sidecar_rejects_invalid_counts_and_missing_arrays(tmp_path, counts):
    path = tmp_path / "calibration.json"
    _write_archive(path, counts=counts)
    with pytest.raises(ValueError, match="Cannot load control images"):
        load_control_samples(path, context={})


@pytest.mark.parametrize("kind", ["gray_mask", "mask_shape", "pickle_image", "pickle_metadata"])
def test_control_sidecar_rejects_malformed_masks_and_pickle(tmp_path, kind):
    path = tmp_path / "calibration.json"
    kwargs = {}
    if kind == "gray_mask":
        kwargs["mask"] = np.ones((160, 240), np.uint8)
    elif kind == "mask_shape":
        kwargs["mask"] = np.zeros((80, 120), np.uint8)
    elif kind == "pickle_image":
        kwargs["image"] = np.asarray([{"unsafe": True}], dtype=object)
    else:
        kwargs["metadata"] = np.asarray({"version": 1}, dtype=object)
    _write_archive(path, **kwargs)
    with pytest.raises(ValueError, match="Cannot load control images"):
        load_control_samples(path, context={})


def test_control_sidecar_rejects_a_truncated_zip_archive(tmp_path):
    path = tmp_path / "calibration.json"
    saved = save_control_samples(path, ((_control(),),), context={})
    saved.write_bytes(saved.read_bytes()[:-12])
    with pytest.raises(ValueError, match="Cannot load control images"):
        load_control_samples(path, context={})


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_control_sidecar_failed_save_preserves_previous_samples(tmp_path, monkeypatch, failure):
    path = tmp_path / "calibration.json"
    saved = save_control_samples(path, ((_control(x=80),),), context={})
    original = saved.read_bytes()

    def fail_write(stream, **_arrays):
        stream.write(b"partial archive")
        raise OSError("disk write failed")

    def fail_replace(_source, _destination):
        raise OSError("rename failed")

    if failure == "write":
        monkeypatch.setattr(np, "savez_compressed", fail_write)
    else:
        monkeypatch.setattr(type(saved), "replace", fail_replace)
    with pytest.raises(OSError, match="failed"):
        save_control_samples(path, ((_control(x=160),),), context={})
    assert saved.read_bytes() == original
    assert list(tmp_path.glob(".controls-*")) == []
    assert np.array_equal(load_control_samples(path, context={})[0][0].image, _control(x=80).image)
