"""Small, offline searches against human-painted raw eye crops.

Nothing here runs in the live tracking path. Each trial uses the same stateless
detector as tracking, so a good score cannot depend on a stale temporal lock.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np

from .config import TrackerConfig
from .file_access import SHARED_FILE_MODE
from .models import EyeImageSettings, GrayImage
from .tracker import (
    AdaptivePupilDetector,
    _normalized_eye_image,
    apply_normalized_eye_settings,
    detector_config_for_eye,
    fit_settings_from_config,
)

MAX_CONTROL_BYTES = 64 * 1024 * 1024  # Per eye, raw images plus masks; no six-image cap.
MAX_EVALUATIONS = 640


class CalibrationCancelled(RuntimeError):
    """An offline search was cancelled before its result was applied."""


@dataclass(frozen=True, eq=False)
class ControlSample:
    """Owned raw pixels and a filled pupil mask; an empty mask means no pupil."""

    image: GrayImage
    mask: GrayImage

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        if (image.dtype != np.uint8 or image.ndim != 2 or min(image.shape) < 24
                or image.size > 4_000_000):
            raise ValueError("Control image must be a 24px-or-larger uint8 eye crop")
        owned = image.copy(order="C")
        owned.flags.writeable = False
        object.__setattr__(self, "image", owned)
        object.__setattr__(self, "mask", self._owned_mask(self.mask))

    def _owned_mask(self, mask: GrayImage) -> GrayImage:
        mask = np.asarray(mask)
        if mask.dtype != np.uint8 or mask.shape != self.image.shape:
            raise ValueError("Pupil mask must be uint8 and match its control image")
        if np.any((mask != 0) & (mask != 255)):
            raise ValueError("Pupil mask must contain only 0 and 255")
        owned = mask.copy(order="C")
        owned.flags.writeable = False
        return owned

    def with_mask(self, mask: GrayImage) -> ControlSample:
        """Replace paint while sharing these already-owned immutable raw pixels."""
        edited = object.__new__(ControlSample)
        object.__setattr__(edited, "image", self.image)
        object.__setattr__(edited, "mask", self._owned_mask(mask))
        return edited


def validate_control_samples(samples: Sequence[ControlSample]) -> None:
    """Allow extra examples while keeping calibration storage bounded per eye."""
    if not all(isinstance(sample, ControlSample) for sample in samples):
        raise TypeError("Calibration requires ControlSample instances")
    if sum(sample.image.nbytes + sample.mask.nbytes for sample in samples) > MAX_CONTROL_BYTES:
        raise ValueError("Control images exceed the 64 MiB per-eye memory budget; delete unused images")


@dataclass(frozen=True)
class CalibrationResult:
    settings: EyeImageSettings
    before_score: float
    after_score: float
    sample_scores: tuple[float, ...]
    evaluations: int


def _agreement(scores: Sequence[float], positive: Sequence[bool]) -> float:
    pupils = [value for value, present in zip(scores, positive, strict=True) if present]
    absent = [value for value, present in zip(scores, positive, strict=True) if not present]
    # Balance blink examples independently of their count. Missing every pupil
    # must never beat detecting the pupils just because most examples are blinks.
    visible_score = 0.7 * float(np.mean(pupils)) + 0.3 * min(pupils)
    return visible_score * (0.5 + 0.5 * float(np.mean(absent))) if absent else visible_score


def optimize_calibration(
    samples: Sequence[ControlSample],
    initial: EyeImageSettings,
    tracker_config: TrackerConfig,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> CalibrationResult:
    """Fit one saved profile to two or more examples, with a bounded search.

    Positive examples use mask overlap (IoU); blink examples require no fit.
    Equal scores retain earlier settings, including the original calibration.
    Reported scores describe these examples, not held-out accuracy.
    """
    import cv2

    samples = tuple(samples)
    if len(samples) < 2:
        raise ValueError("Use at least 2 control images per eye; 2–6 are usually enough")
    validate_control_samples(samples)
    positive = tuple(bool(np.any(sample.mask)) for sample in samples)
    if not any(positive):
        raise ValueError("Paint the pupil on at least one control image")

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise CalibrationCancelled("Calibration optimization cancelled")

    diameters, major_fractions, axis_ratios = [], [], []
    for sample, present in zip(samples, positive, strict=True):
        if not present:
            continue
        area = int(np.count_nonzero(sample.mask))
        contours, _ = cv2.findContours(sample.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)
        if area < 16 or len(contour) < 5:
            raise ValueError("Paint the filled pupil area, not just a point or line")
        _, axes, _ = cv2.fitEllipse(contour)
        major, minor = sorted(axes, reverse=True)
        if minor <= 1 or not math.isfinite(major):
            raise ValueError("Paint a visible, filled pupil on positive control images")
        diameters.append(2 * math.sqrt(area / math.pi))
        major_fractions.append(major / min(sample.image.shape))
        axis_ratios.append(minor / major)

    check_cancelled()
    normalized = tuple(_normalized_eye_image(sample.image) for sample in samples)

    @lru_cache(maxsize=2)
    def processed(gain: float, brightness: float, contrast: float) -> tuple[GrayImage, ...]:
        settings = EyeImageSettings(gain=gain, brightness=brightness, contrast=contrast)
        return tuple(apply_normalized_eye_settings(image, settings) for image in normalized)

    evaluated: dict[EyeImageSettings, tuple[float, tuple[float, ...]]] = {}

    def evaluate(settings: EyeImageSettings) -> tuple[float, tuple[float, ...]]:
        check_cancelled()
        if settings in evaluated:
            return evaluated[settings]
        detector = AdaptivePupilDetector(detector_config_for_eye(tracker_config, settings))
        images = processed(settings.gain, settings.brightness, settings.contrast)
        scores = []
        for image, sample, present in zip(images, samples, positive, strict=True):
            check_cancelled()
            candidate, mask = detector.detect_with_mask(image)
            if not present:
                scores.append(float(candidate is None))
            else:
                # Normalize each image independently before averaging: a
                # dilated pupil must not outweigh a small pupil by pixel count.
                overlap = np.count_nonzero((mask != 0) & (sample.mask != 0))
                union = np.count_nonzero((mask != 0) | (sample.mask != 0))
                scores.append(float(overlap / union))
        result = (_agreement(scores, positive), tuple(scores))
        evaluated[settings] = result
        return result

    best = initial
    before, best_scores = evaluate(initial)
    best_score = before

    def consider(settings: EyeImageSettings) -> float | None:
        nonlocal best, best_score, best_scores
        check_cancelled()
        if settings not in evaluated and len(evaluated) >= MAX_EVALUATIONS:
            return None
        score, scores = evaluate(settings)
        if score > best_score + 1e-4:
            best, best_score, best_scores = settings, score, scores
        return score

    baseline_fit = fit_settings_from_config(detector_config_for_eye(tracker_config, initial))
    defaults = fit_settings_from_config(tracker_config)
    # Restart from shared defaults too: a previously learned strict gate must
    # not keep every new trial from seeing a newly dilated or oblique pupil.
    learned = replace(
        defaults,
        min_pupil_diameter_px=max(2.0, min(diameters) * 0.65),
        max_pupil_diameter_fraction=max(
            defaults.max_pupil_diameter_fraction, min(1.0, max(major_fractions) * 1.35),
        ),
        min_axis_ratio=max(0.1, min(defaults.min_axis_ratio, min(axis_ratios) * 0.8)),
    )
    for fit in (baseline_fit, learned):
        consider(replace(initial, fit=fit))
        consider(replace(initial, gain=1.0, brightness=0.0, contrast=1.0, fit=fit))

    # Preserve a useful joint intensity/threshold sweep before exploring all
    # parameters. Unclipped affine controls alone often leave segmentation
    # unchanged, while clipping can hide the edge until thresholds also move.
    transforms = dict.fromkeys(((initial.gain, initial.brightness, initial.contrast),
                                (0.5, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)))
    dark_references = dict.fromkeys((baseline_fit.dark_reference_percentile, 0.5, 1.5))
    threshold_ranges = dict.fromkeys((baseline_fit.threshold_range_fraction, 0.12, 0.3, 0.5, 0.8))
    for gain, brightness, contrast in transforms:
        for dark_reference in dark_references:
            for threshold_range in threshold_ranges:
                fit = replace(learned, dark_reference_percentile=dark_reference,
                              threshold_range_fraction=threshold_range)
                consider(replace(initial, gain=gain, brightness=brightness, contrast=contrast,
                                 fit=fit))

    choices = {
        "gain": (0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0),
        "brightness": (-0.2, -0.1, 0.0, 0.1, 0.2),
        "pupil_size_bias": (-1.0, -0.5, 0.0, 0.5, 1.0),
        "dark_reference_percentile": (0.1, 0.5, 1.0, 1.5, 2.0, 4.0),
        "threshold_range_fraction": (0.12, 0.18, 0.3, 0.45, 0.65, 0.85),
        "threshold_percentile_scale": (0.5, 1.0, 1.5, 2.0, 3.0),
        "max_fit_residual": (0.06, 0.12, 0.24, 0.38),
        "appearance_weight": (0.0, 0.5, 1.0, 1.5, 2.0),
        "min_pupil_diameter_px": (defaults.min_pupil_diameter_px,
                                   *(min(diameters) * f for f in (0.45, 0.65, 0.85))),
        "max_pupil_diameter_fraction": (defaults.max_pupil_diameter_fraction,
            *(min(1.0, max(major_fractions) * f) for f in (1.15, 1.35, 1.6))),
        "min_axis_ratio": (defaults.min_axis_ratio, 0.3, min(axis_ratios) * 0.8),
        "min_contrast": (defaults.min_contrast, 4.0, 8.0, 16.0, 32.0),
        "min_confidence": (defaults.min_confidence, 0.35, 0.48, 0.6, 0.72, 0.84, 0.92),
    }
    image_names = {"gain", "brightness", "pupil_size_bias"}

    def changed(settings: EyeImageSettings, values: dict[str, float]) -> EyeImageSettings:
        image_values = {name: value for name, value in values.items() if name in image_names}
        fit_values = {name: value for name, value in values.items() if name not in image_names}
        fit = replace(settings.fit or baseline_fit, **fit_values)
        return replace(settings, fit=fit, **image_values)

    # Stratified, reproducible joint exploration crosses flat score regions
    # without requiring each intermediate one-parameter move to improve.
    rng = np.random.default_rng(0)
    exploration_count = 96
    columns = {
        name: [values[index * len(values) // exploration_count]
               for index in rng.permutation(exploration_count)]
        for name, values in choices.items()
    }
    for index in range(exploration_count):
        consider(changed(replace(initial, contrast=1.0, fit=learned),
                         {name: values[index] for name, values in columns.items()}))

    # Refine three promising but distinct threshold/ranking regions, rather
    # than abandoning every runner-up once a single coarse winner is found.
    def promising_starts() -> list[EyeImageSettings]:
        starts, regions = [], set()
        for settings in sorted(evaluated, key=lambda value: evaluated[value][0], reverse=True):
            fit = settings.fit or baseline_fit
            region = (round(fit.threshold_range_fraction * 4),
                      round(fit.dark_reference_percentile),
                      round(fit.threshold_percentile_scale), round(fit.appearance_weight * 2))
            if region not in regions:
                starts.append(settings)
                regions.add(region)
            if len(starts) == 3:
                break
        return starts

    # Paired moves around good seeds bridge narrow valleys that a broad random
    # restart may miss. Keep the seeds fixed during this exploration batch.
    seeds = promising_starts()
    names = tuple(choices)
    for index in range(64):
        seed = seeds[index % len(seeds)]
        fit = seed.fit or baseline_fit
        values = {}
        for name in rng.choice(names, size=2, replace=False):
            current = getattr(seed if name in image_names else fit, name)
            candidates = list(choices[name])
            if name == "gain":
                candidates += [max(0.01, current * 0.8), min(8.0, current * 1.2)]
            elif name == "threshold_range_fraction":
                candidates += [max(0.01, current - 0.08), min(1.0, current + 0.08)]
            nearby = sorted({value for value in candidates if value != current},
                            key=lambda value: abs(value - current))[:3]
            values[name] = nearby[int(rng.integers(len(nearby)))]
        consider(changed(seed, values))
    starts = promising_starts()
    for start in starts:
        local, local_score = start, evaluated[start][0]
        # Reserve comparable work for each basin, even if the first keeps improving.
        limit = min(MAX_EVALUATIONS, len(evaluated) + 120)
        for _ in range(2):
            previous = local
            for name, values in choices.items():
                for value in values:
                    if len(evaluated) >= limit:
                        break
                    trial = changed(local, {name: value})
                    score = consider(trial)
                    if score is not None and score > local_score + 1e-4:
                        local, local_score = trial, score
            if local == previous or len(evaluated) >= limit:
                break

    # Small continuous moves around the best discrete solution remove grid
    # quantization; search paired edge/appearance moves as well as single ones.
    for step in (0.10, 0.04):
        anchor = best
        fit = anchor.fit or baseline_fit
        for edge_delta in (-step, 0.0, step):
            for appearance_delta in (-0.5, 0.0, 0.5):
                consider(changed(anchor, {
                    "threshold_range_fraction": float(np.clip(
                        fit.threshold_range_fraction + edge_delta, 0.01, 1.0,
                    )),
                    "appearance_weight": float(np.clip(
                        fit.appearance_weight + appearance_delta, 0.0, 2.0,
                    )),
                }))
        for gain_factor in (0.8, 1.0, 1.2):
            for offset in (-step, 0.0, step):
                consider(replace(best, gain=float(np.clip(best.gain * gain_factor, 0.01, 8.0)),
                                 brightness=float(np.clip(best.brightness + offset, -1.0, 1.0))))
    check_cancelled()
    if best.fit is None:
        # Even a perfect starting profile is now calibrated. Materialize its
        # unchanged limits so live tracking uses the evaluated stateless model.
        best = replace(best, fit=fit_settings_from_config(detector_config_for_eye(tracker_config, best)))
    return CalibrationResult(best, before, best_score, best_scores, len(evaluated))


def save_control_samples(
    path: str | Path,
    samples_by_eye: Sequence[Sequence[ControlSample]],
    *,
    context: dict | None = None,
    shared: bool = False,
) -> Path:
    """Save independent raw crops and masks; legacy context is no longer a filter."""
    bank = tuple(tuple(samples) for samples in samples_by_eye)
    if not 1 <= len(bank) <= 2:
        raise ValueError("Save control samples for one or two eyes")
    for samples in bank:
        validate_control_samples(samples)
    arrays = {"metadata": np.asarray(json.dumps({
        "version": 2, "counts": [len(samples) for samples in bank],
    }, sort_keys=True))}
    for eye, samples in enumerate(bank):
        for index, sample in enumerate(samples):
            arrays[f"image_{eye}_{index}"] = sample.image
            arrays[f"mask_{eye}_{index}"] = sample.mask
    path = Path(path).with_suffix(".controls.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".controls-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            if shared:
                os.fchmod(stream.fileno(), SHARED_FILE_MODE)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def load_control_samples(
    path: str | Path, *, context: dict | None = None,
) -> tuple[tuple[ControlSample, ...], ...]:
    """Load original crops across ROI changes, including version-1 archives."""
    path = Path(path).with_suffix(".controls.npz")
    if not path.exists():
        return ()
    try:
        # Check uncompressed sizes before loading arrays from an optional archive.
        with ZipFile(path) as zipped:
            if any(entry.file_size > 4_001_024 for entry in zipped.infolist()):
                raise ValueError("Control-image archive contains an oversized array")
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata["version"] not in (1, 2):
                raise ValueError("Unsupported control-image version")
            counts = metadata["counts"]
            if (not isinstance(counts, list) or not 1 <= len(counts) <= 2
                    or any(type(count) is not int or count < 0 for count in counts)
                    or sum(counts) * 2 + 1 != len(archive.files)):
                raise ValueError("Invalid control-image counts")
            eyes = []
            for eye, count in enumerate(counts):
                samples = []
                used_bytes = 0
                for index in range(count):
                    sample = ControlSample(archive[f"image_{eye}_{index}"],
                                           archive[f"mask_{eye}_{index}"])
                    used_bytes += sample.image.nbytes + sample.mask.nbytes
                    if used_bytes > MAX_CONTROL_BYTES:
                        raise ValueError("Control images exceed the 64 MiB per-eye memory budget")
                    samples.append(sample)
                eyes.append(tuple(samples))
            return tuple(eyes)
    except (KeyError, IndexError, TypeError, ValueError, OSError, BadZipFile) as exc:
        raise ValueError(f"Cannot load control images: {exc}") from exc
