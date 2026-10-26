from __future__ import annotations

import math
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from functools import lru_cache

import numpy as np

from .config import TrackerConfig
from .models import (
    AnalysisFrame,
    EyeFitSettings,
    EyeImageSettings,
    EyeMeasurement,
    FrameResult,
    GrayImage,
)

try:  # Kept optional so protocol/config tools work on non-camera development hosts.
    import cv2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised on hosts without the vision extra
    cv2 = None


def fit_settings_from_config(config: TrackerConfig) -> EyeFitSettings:
    """Materialize per-eye defaults without losing fields as calibration grows."""
    return EyeFitSettings(**{
        field.name: getattr(config, field.name) for field in fields(EyeFitSettings)
    })


def detector_config_for_eye(config: TrackerConfig, settings: EyeImageSettings) -> TrackerConfig:
    """Combine shared tracking behavior with one eye's calibrated fit limits."""

    overrides = {} if settings.fit is None else settings.fit.as_dict()
    return replace(config, pupil_size_bias=settings.pupil_size_bias, **overrides)


def _eye_intensity_anchors(image: np.ndarray, bins: int = 256) -> np.ndarray:
    histogram = cv2.calcHist([image], [0], None, [bins], [0, bins]).reshape(-1)
    return np.searchsorted(
        np.cumsum(histogram), np.asarray((0.02, 0.70, 0.98)) * image.size,
    )


@lru_cache(maxsize=16)
def _illumination_basis(height: int, width: int) -> np.ndarray:
    """Six smooth surface terms, cached for the small illumination grid."""

    y, x = np.mgrid[:height, :width].astype(np.float32)
    x = 2.0 * x / max(width - 1, 1) - 1.0
    y = 2.0 * y / max(height - 1, 1) - 1.0
    return np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=-1).reshape(-1, 6)


def _normalized_eye_image(gray: GrayImage) -> np.ndarray:
    """Correct broad spatial illumination without resampling the actual eye.

    Estimate a quadratic lighting surface on a grid at most 64 pixels across.
    Closing identifies dark structures to exclude from that estimate; neither
    the closed image nor the reduced image is passed to pupil detection.
    """

    if gray.size == 0:
        return gray.astype(np.float32)
    dark, light, ceiling = _eye_intensity_anchors(gray)
    if light - dark < 24:
        # Do not stretch a nearly flat frame or sensor noise.
        return gray.astype(np.float32)

    height, width = gray.shape
    reduction = min(1.0, 64.0 / max(height, width))
    small = cv2.resize(
        gray,
        (max(1, round(width * reduction)), max(1, round(height * reduction))),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    kernel_size = max(3, round(0.8 * min(small.shape)) | 1)
    background = cv2.morphologyEx(
        np.minimum(small, float(ceiling)),
        cv2.MORPH_CLOSE,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
        borderType=cv2.BORDER_REPLICATE,
    )
    # Exclude pupil/eyelid shadows and small bright glints. Fit observed image
    # values, not the closing itself, whose border effects can imitate shading.
    visible = ((small >= 0.85 * background) & (small <= ceiling)).reshape(-1)
    maximum_gain = 1.0
    adjusted = None
    if np.count_nonzero(visible) >= max(64, 0.25 * small.size):
        basis = _illumination_basis(*small.shape)
        coefficients, _, rank, _ = np.linalg.lstsq(
            basis[visible], small.reshape(-1)[visible], rcond=None,
        )
        illumination = (basis @ coefficients).reshape(small.shape)
        reference = float(np.median(illumination))
        if (
            rank == 6
            and np.all(np.isfinite(illumination))
            and float(np.min(illumination)) > 0.0
            and reference >= 16.0
            and float(np.ptp(illumination)) > 0.02 * reference
        ):
            gain = np.clip(reference / np.maximum(illumination, 16.0), 0.5, 2.0)
            maximum_gain = float(np.max(gain))
            gain = cv2.resize(gain, (width, height), interpolation=cv2.INTER_LINEAR)
            adjusted = cv2.multiply(gray, gain, dtype=cv2.CV_32F)
            # Values remain floating point (0..510); no intermediate clipping
            # or quantization before normalization and the user's controls.
            # Bound float-histogram work independently of camera resolution.
            # Sample actual corrected pixels, not area-averaged pixels that
            # would suppress noise and shift the detector's dark reference.
            step = max(1, (max(height, width) + 127) // 128)
            dark, light, _ = _eye_intensity_anchors(adjusted[::step, ::step], bins=512)

    if adjusted is None:
        adjusted = gray.astype(np.float32)
    if light - dark >= 24:
        # Limit the *combined* illumination/range gain, and leave headroom for
        # glints above the 70th-percentile light reference.
        scale = min(2.0 / maximum_gain, 144.0 / float(light - dark))
        adjusted *= scale
        adjusted += 32.0 - float(dark) * scale
    return adjusted


def apply_eye_image_settings(
    image: GrayImage,
    settings: EyeImageSettings,
) -> GrayImage:
    """Normalize one eye crop, then apply its saved software image controls.

    Calibration, tracking, and preview share this stateless path. Lighting is
    estimated at low resolution, but only per-pixel intensity is changed on the
    original crop. Quantize once, after applying the user's controls.
    """

    gray = np.asarray(image)
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("Eye image adjustments require a two-dimensional uint8 image")
    if cv2 is None:
        raise RuntimeError("OpenCV is required for eye intensity normalization")
    return apply_normalized_eye_settings(_normalized_eye_image(gray), settings, copy=False)


def apply_normalized_eye_settings(
    normalized: np.ndarray,
    settings: EyeImageSettings,
    *,
    copy: bool = True,
) -> GrayImage:
    """Apply controls to a normalized crop; offline search can reuse its cached input.

    The live path owns its float32 working image and uses ``copy=False`` to
    avoid an extra allocation. Other callers keep their input unchanged.
    """

    adjusted = np.array(normalized, dtype=np.float32, copy=copy)
    adjusted *= settings.gain * settings.contrast
    adjusted += 128.0 * (1.0 - settings.contrast) + 255.0 * settings.brightness
    np.clip(adjusted, 0, 255, out=adjusted)
    return adjusted.astype(np.uint8)


def _segmentation_inputs(gray: GrayImage) -> tuple[GrayImage, np.ndarray]:
    height, width = gray.shape
    blur_size = max(3, round(min(width, height) * 0.015) | 1)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    morphology_size = max(3, round(min(width, height) * 0.012) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morphology_size, morphology_size),
    )
    return blurred, kernel


def _mask_at_threshold(
    blurred: GrayImage,
    threshold: int,
    kernel: np.ndarray,
) -> GrayImage:
    _value, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _glint_mask(
    gray: GrayImage,
    adaptive_range: tuple[float, float],
    kernel: np.ndarray,
) -> GrayImage | None:
    """Find distinctly bright reflections once per crop, including their halos.

    Do not cap the cutoff at 255: a clipped bright background is not evidence
    of a localized reflection. Ambiguous highlights retain the normal checks.
    """

    dark_floor, light_reference = adaptive_range
    cutoff = light_reference + max(16.0, 0.25 * (light_reference - dark_floor))
    _value, mask = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(mask) == 0:
        return None
    blur_size = max(3, round(min(gray.shape) * 0.015) | 1)
    radius = blur_size // 2 + kernel.shape[0] // 2 + 1
    halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.dilate(mask, halo)


def pupil_mask_for_threshold(image: GrayImage, threshold: int) -> GrayImage:
    """Rebuild one detector mask for the separate diagnostic display process."""

    if cv2 is None:
        raise RuntimeError("OpenCV is required to build a pupil mask")
    gray = np.asarray(image)
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("Pupil mask expects a two-dimensional uint8 image")
    if not 1 <= threshold <= 254:
        raise ValueError("Pupil threshold must be in [1, 254]")
    blurred, kernel = _segmentation_inputs(gray)
    return _mask_at_threshold(blurred, threshold, kernel)


def _adaptive_appearance_scores(
    median_intensity: float,
    interior_spread: float,
    threshold: int,
    adaptive_range: tuple[float, float],
) -> tuple[float, float, float]:
    """Whole-crop appearance priors for candidate ranking, not confidence."""

    dark_floor, light_reference = adaptive_range
    intensity_span = max(light_reference - dark_floor, 12.0)
    darkness = float(
        np.clip(1.0 - (median_intensity - dark_floor) / intensity_span, 0.0, 1.0)
    )
    uniformity = float(np.exp(-4.0 * interior_spread / intensity_span))
    threshold_selectivity = float(
        np.clip(1.0 - (threshold - dark_floor) / intensity_span, 0.0, 1.0)
    )
    return darkness, uniformity, threshold_selectivity


def _local_appearance_scores(
    contrast: float,
    interior_spread: float,
) -> tuple[float, float]:
    """Score observed pupil/ring separation independently of absolute grey level.

    Contrast is the surrounding ring's 35th percentile minus the pupil median.
    Interior spread also exposes iris-sized fits containing a darker pupil.
    The fixed floors keep tiny intensity differences from earning full credit;
    no pixels are stretched and no new image statistics are required.
    """

    separation = contrast / max(contrast + 2.0 * interior_spread, 24.0)
    uniformity = math.exp(-4.0 * interior_spread / max(contrast, 12.0))
    return separation, uniformity


def _pupil_size_bias_adjustment(
    diameter_fraction: float,
    minimum_fraction: float,
    maximum_fraction: float,
    bias: float,
) -> float:
    """Return a bounded ranking adjustment for a relative pupil-size prior.

    A negative bias favors smaller candidates, a positive bias favors larger
    candidates, and zero leaves the automatic detector's ranking unchanged.
    """

    span = max(maximum_fraction - minimum_fraction, 1e-6)
    size_position = float(
        np.clip((diameter_fraction - minimum_fraction) / span, 0.0, 1.0)
    )
    centered_size = 2.0 * size_position - 1.0
    return 0.12 * float(np.clip(bias, -1.0, 1.0)) * centered_size


@dataclass(frozen=True)
class PupilCandidate:
    x: float
    y: float
    ellipse_x: float
    ellipse_y: float
    ellipse_width: float
    ellipse_height: float
    major: float
    minor: float
    angle_degrees: float
    equivalent_diameter: float
    confidence: float
    ranking_score: float
    contrast: float
    threshold: int
    threshold_fraction: float
    median_intensity: float
    interior_spread: float
    glint_fraction: float = 0.0
    boundary_recovered: bool = False
    local_separation: float = 0.0
    local_uniformity: float = 0.0

    @property
    def diameter(self) -> float:
        """Area-equivalent diameter; supported glint recovery uses the ellipse."""
        return self.equivalent_diameter


@dataclass
class PupilPrior:
    """Last accepted pupil state used to validate and rank current-frame fits."""

    x: float = 0.0
    y: float = 0.0
    diameter: float = 0.0
    threshold_fraction: float = 0.0
    has_lock: bool = False
    missing_frames: int = 0

    def update(
        self,
        candidate: PupilCandidate | None,
        *,
        position_alpha: float = 1.0,
        diameter_alpha: float = 1.0,
    ) -> None:
        """Advance the accepted state without smoothing unless explicitly enabled."""

        if candidate is None:
            self.missing_frames += 1
            return
        if not self.has_lock:
            position_alpha = diameter_alpha = 1.0
        self.x = position_alpha * candidate.x + (1.0 - position_alpha) * self.x
        self.y = position_alpha * candidate.y + (1.0 - position_alpha) * self.y
        self.diameter = (
            diameter_alpha * candidate.diameter + (1.0 - diameter_alpha) * self.diameter
        )
        self.threshold_fraction = candidate.threshold_fraction
        self.has_lock = True
        self.missing_frames = 0


class AdaptivePupilDetector:
    """Fast dark-pupil ellipse detector for a fixed macaque-eye crop.

    Candidate thresholds are always derived from each frame, not from an
    absolute brightness constant. Bright corneal reflections are excluded from
    appearance and boundary evidence, with strict visibility checks. Contours
    are ranked using pupil/iris contrast, ellipse residual, filled area, border
    distance, an optional relative size bias, and a temporal prior.
    """

    def __init__(self, config: TrackerConfig) -> None:
        if cv2 is None:
            raise RuntimeError(
                "OpenCV is required for pupil tracking. Install the vision extra "
                "or Raspberry Pi OS package python3-opencv."
            )
        self.config = config

    @staticmethod
    def _ellipse_residuals(points: np.ndarray, ellipse: tuple) -> np.ndarray:
        (ellipse_cx, ellipse_cy), axes, angle = ellipse
        width, height = float(axes[0]), float(axes[1])
        if width <= 0.0 or height <= 0.0:
            return np.full(len(points), np.inf)
        theta = math.radians(float(angle))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        dx = points[:, 0] - float(ellipse_cx)
        dy = points[:, 1] - float(ellipse_cy)
        xr = cos_t * dx + sin_t * dy
        yr = -sin_t * dx + cos_t * dy
        radius = np.sqrt((xr / (0.5 * width)) ** 2 + (yr / (0.5 * height)) ** 2)
        return np.abs(radius - 1.0)

    def _refined_ellipse(self, contour: np.ndarray) -> tuple | None:
        if len(contour) < 5:
            return None
        try:
            ellipse = cv2.fitEllipse(contour)
        except cv2.error:
            return None
        points = contour[:, 0, :].astype(np.float64, copy=False)
        residuals = self._ellipse_residuals(points, ellipse)
        finite = residuals[np.isfinite(residuals)]
        if len(finite) < 5:
            return None
        # A single robust refit removes eyelashes and small threshold tendrils
        # without the cost and frame-to-frame jitter of random RANSAC sampling.
        cutoff = min(0.35, max(0.08, float(np.quantile(finite, 0.82))))
        inliers = points[residuals <= cutoff]
        if len(inliers) >= 8:
            try:
                ellipse = cv2.fitEllipse(inliers.astype(np.float32).reshape(-1, 1, 2))
            except cv2.error:
                pass
        return ellipse

    @staticmethod
    def _supported_ellipse_boundary(
        points: np.ndarray,
        ellipse: tuple,
        residuals: np.ndarray,
    ) -> bool:
        """Require accurate, widely distributed observed arcs, without a search."""

        if len(points) < 12 or not np.all(np.isfinite(residuals)):
            return False
        if float(np.quantile(residuals, 0.90)) > 0.12:
            return False
        theta = math.radians(float(ellipse[2]))
        dx = points[:, 0] - ellipse[0][0]
        dy = points[:, 1] - ellipse[0][1]
        xr = (math.cos(theta) * dx + math.sin(theta) * dy) / ellipse[1][0]
        yr = (-math.sin(theta) * dx + math.cos(theta) * dy) / ellipse[1][1]
        angles = np.arctan2(yr, xr)
        sectors = np.floor((angles + math.pi) * (16.0 / (2.0 * math.pi))).astype(int) % 16
        occupied = np.flatnonzero(np.bincount(sectors[residuals <= 0.08], minlength=16))
        # At least 11/16 sectors, with no unsupported gap exceeding five sectors.
        return bool(
            len(occupied) >= 11
            and np.max(np.diff(np.r_[occupied, occupied[0] + 16])) <= 6
        )

    def _contrast_and_fill(
        self,
        gray: GrayImage,
        mask: GrayImage,
        ellipse: tuple,
        glints: GrayImage | None,
    ) -> tuple[float, float, float, float, float]:
        # All photometric evidence is local to the ellipse and its outer ring.
        # Keeping these operations off the rest of the crop pays for masking.
        major = max(float(ellipse[1][0]), float(ellipse[1][1]))
        ring_width = max(3, round(0.12 * major))
        extent = 0.5 * major + ring_width + 2
        cx, cy = ellipse[0]
        x0, y0 = max(0, math.floor(cx - extent)), max(0, math.floor(cy - extent))
        x1 = min(gray.shape[1], math.ceil(cx + extent) + 1)
        y1 = min(gray.shape[0], math.ceil(cy + extent) + 1)
        gray = gray[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]
        if glints is not None:
            glints = glints[y0:y1, x0:x1]
        ellipse = ((cx - x0, cy - y0), ellipse[1], ellipse[2])
        pupil_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.ellipse(pupil_mask, ellipse, 255, -1)
        interior = pupil_mask != 0
        pupil_area = int(np.count_nonzero(interior))
        visible = interior if glints is None else interior & (glints == 0)
        visible_area = int(np.count_nonzero(visible))
        glint_fraction = 1.0 - visible_area / max(pupil_area, 1)
        if visible_area < 8 or glint_fraction > 0.40:
            return 0.0, 0.0, 255.0, 255.0, glint_fraction
        pupil_pixels = gray[visible]

        kernel_size = 2 * ring_width + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded = cv2.dilate(pupil_mask, kernel, iterations=1)
        ring = (expanded != 0) & ~interior
        visible_ring = ring if glints is None else ring & (glints == 0)
        ring_pixels = gray[visible_ring]
        if ring_pixels.size < max(8, 0.65 * np.count_nonzero(ring)):
            return 0.0, 0.0, 255.0, 255.0, glint_fraction

        # Once glints hide part of the pupil, a smaller dark core must still
        # penalize iris-sized fits. Never substitute invented pixels for evidence.
        lower_percentile = 5.0 if glint_fraction > 0.02 else 10.0
        lower, median, upper_quartile = np.percentile(
            pupil_pixels, (lower_percentile, 50.0, 75.0)
        )
        # Require most of the fitted boundary to be lighter than the interior.
        # This rejects rounded eyelash/eyelid endpoints that can otherwise look
        # circular even though they remain dark on one side.
        contrast = float(np.percentile(ring_pixels, 35.0) - median)
        covered = int(np.count_nonzero((mask != 0) & visible))
        fill = covered / visible_area
        # A pupil is normally a compact dark basin. An iris-sized ellipse often
        # contains both the much darker pupil and lighter iris, producing a
        # large robust interior spread even when its outer edge is high-contrast.
        interior_spread = float(upper_quartile - lower)
        return contrast, float(fill), float(median), interior_spread, glint_fraction

    def _candidate(
        self,
        gray: GrayImage,
        mask: GrayImage,
        contour: np.ndarray,
        threshold: int,
        prior: PupilPrior | None,
        adaptive_range: tuple[float, float],
        glints: GrayImage | None,
    ) -> PupilCandidate | None:
        height, width = gray.shape
        area = float(cv2.contourArea(contour))
        diameter = 2.0 * math.sqrt(area / math.pi)
        maximum_major = self.config.max_pupil_diameter_fraction * min(width, height)
        maximum_area = math.pi * maximum_major * maximum_major / 4.0
        if (
            diameter < self.config.min_pupil_diameter_px
            or area > 1.10 * maximum_area
        ):
            return None

        fit_contour = contour
        boundary_recovered = False
        if glints is not None:
            points = contour[:, 0, :]
            visible = glints[points[:, 1], points[:, 0]] == 0
            if not np.all(visible):
                # A majority of the original contour must remain real evidence.
                if np.count_nonzero(visible) < max(12, 0.60 * len(contour)):
                    return None
                fit_contour = contour[visible]
                boundary_recovered = True
        ellipse = self._refined_ellipse(fit_contour)
        if ellipse is None:
            return None
        (ellipse_cx, ellipse_cy), axes, angle = ellipse
        ellipse_width, ellipse_height = float(axes[0]), float(axes[1])
        major, minor = sorted((ellipse_width, ellipse_height), reverse=True)
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            return None
        # Retain contour moments for intact boundaries. A reflection cutting
        # into the contour requires a separately validated ellipse measurement.
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        if not all(np.isfinite((cx, cy, ellipse_cx, ellipse_cy, major, minor, angle))):
            return None
        if minor <= 1.0:
            return None
        if major > maximum_major:
            return None
        axis_ratio = minor / major
        if axis_ratio < self.config.min_axis_ratio:
            return None
        points = fit_contour[:, 0, :].astype(np.float64, copy=False)
        residuals = self._ellipse_residuals(points, ellipse)
        if boundary_recovered:
            if not self._supported_ellipse_boundary(points, ellipse, residuals):
                return None
            ellipse_area = math.pi * major * minor / 4.0
            if not 0.65 <= area / ellipse_area <= 1.10:
                return None
            cx, cy = float(ellipse_cx), float(ellipse_cy)
            diameter = math.sqrt(major * minor)
            if diameter < self.config.min_pupil_diameter_px:
                return None
        elif math.hypot(cx - ellipse_cx, cy - ellipse_cy) > max(2.0, 0.15 * minor):
            return None
        if not (0.0 <= cx < width and 0.0 <= cy < height):
            return None

        residual = float(np.median(residuals))
        if not math.isfinite(residual) or residual > self.config.max_fit_residual:
            return None

        contrast, fill, median_intensity, interior_spread, glint_fraction = self._contrast_and_fill(
            gray,
            mask,
            ellipse,
            glints,
        )
        if contrast < self.config.min_contrast or not 0.28 <= fill <= 1.20:
            return None

        diagonal = math.hypot(width, height)
        temporal_score = 0.72
        continuity_adjustment = 0.0
        dark_floor, light_reference = adaptive_range
        intensity_span = max(light_reference - dark_floor, 12.0)
        # Recovery adds stricter evidence requirements, never relaxed shape
        # or confidence limits. Reject mixed pupil/iris interiors as well.
        if (glint_fraction > 0.02 or boundary_recovered) and (
            fill < 0.80 or interior_spread > max(12.0, 0.20 * intensity_span)
        ):
            return None
        threshold_fraction = float(
            np.clip((threshold - dark_floor) / intensity_span, 0.0, 1.0)
        )
        if (
            prior is not None
            and prior.has_lock
            and prior.missing_frames < self.config.blink_after_missing_frames
        ):
            distance = math.hypot(cx - prior.x, cy - prior.y)
            allowed_jump = self.config.max_position_jump_fraction * diagonal
            # After several missing frames the prior is intentionally loosened.
            allowed_jump *= min(2.0, 1.0 + 0.18 * prior.missing_frames)
            if distance > allowed_jump:
                return None
            relative_diameter_change = abs(diameter - prior.diameter) / max(prior.diameter, 1.0)
            allowed_change = self.config.max_diameter_change_fraction * min(
                2.0, 1.0 + 0.20 * prior.missing_frames
            )
            # Permit a smaller concentric candidate to correct a prior
            # iris-sized lock. The reverse transition remains constrained, so
            # a stable pupil cannot suddenly expand to the surrounding iris.
            nested_recovery = (
                diameter < prior.diameter
                and distance < 0.25 * prior.diameter
            )
            if relative_diameter_change > allowed_change and not nested_recovery:
                return None
            temporal_score = math.exp(-distance / max(0.12 * diagonal, 1.0))
            # Prefer the same adaptive cutoff and pupil scale as the previous
            # frame. This provides threshold hysteresis without freezing an
            # absolute brightness value or smoothing genuine gaze motion.
            position_similarity = math.exp(
                -distance / max(0.35 * prior.diameter, 2.0)
            )
            diameter_similarity = math.exp(-relative_diameter_change / 0.18)
            threshold_similarity = math.exp(
                -abs(threshold_fraction - prior.threshold_fraction) / 0.10
            )
            continuity_adjustment = 0.12 * (
                0.20 * position_similarity
                + 0.50 * diameter_similarity
                + 0.30 * threshold_similarity
            )

        contrast_score = float(
            np.clip(
                (contrast - self.config.min_contrast)
                / max(24.0 - self.config.min_contrast, 1.0),
                0.0,
                1.0,
            )
        )
        residual_score = float(np.clip(1.0 - residual / 0.38, 0.0, 1.0))
        fill_score = float(np.clip(1.0 - abs(fill - 0.82) / 0.62, 0.0, 1.0))

        border_distance = min(cx, cy, width - 1.0 - cx, height - 1.0 - cy)
        border_score = float(np.clip(border_distance / max(0.35 * major, 1.0), 0.0, 1.0))
        darkness_score, uniformity_score, threshold_score = _adaptive_appearance_scores(
            median_intensity,
            interior_spread,
            threshold,
            adaptive_range,
        )
        diameter_fraction = diameter / min(width, height)
        minimum_fraction = self.config.min_pupil_diameter_px / min(width, height)
        preferred_maximum = min(self.config.max_pupil_diameter_fraction, 0.68)
        size_score = float(
            np.clip(
                1.0
                - max(0.0, diameter_fraction - 0.32)
                / max(preferred_maximum - 0.32, 0.10),
                0.0,
                1.0,
            )
        )
        size_bias_adjustment = _pupil_size_bias_adjustment(
            diameter_fraction,
            minimum_fraction,
            preferred_maximum,
            self.config.pupil_size_bias,
        )
        fit_score = (
            0.14 * contrast_score
            + 0.18 * residual_score
            + 0.12 * fill_score
            + 0.18 * temporal_score
            + 0.04 * border_score
            + 0.05 * size_score
            - 0.10 * glint_fraction
        )
        local_separation, local_uniformity = _local_appearance_scores(
            contrast, interior_spread,
        )
        # Acceptance must agree with stateless calibration: a valid saccade
        # must not lose confidence because the previous pupil was elsewhere.
        # Keep motion evidence in ranking, using the no-prior contribution
        # for reported image confidence and its calibrated acceptance cutoff.
        confidence = (
            fit_score + 0.18 * (0.72 - temporal_score)
            + 0.19 * local_separation + 0.10 * local_uniformity
        )
        # Keep the existing selection preferences, including the penalty for
        # an iris-sized fit with a darker core. An unrelated eyelash or an
        # absolute threshold must not lower confidence in a clear grey pupil.
        ranking_score = (
            fit_score
            + self.config.appearance_weight * (
                0.14 * darkness_score + 0.10 * uniformity_score + 0.05 * threshold_score
            )
            + size_bias_adjustment
            + continuity_adjustment
        )
        return PupilCandidate(
            x=float(cx),
            y=float(cy),
            ellipse_x=float(ellipse_cx),
            ellipse_y=float(ellipse_cy),
            ellipse_width=ellipse_width,
            ellipse_height=ellipse_height,
            major=major,
            minor=minor,
            angle_degrees=float(angle),
            equivalent_diameter=diameter,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            ranking_score=ranking_score,
            contrast=contrast,
            threshold=threshold,
            threshold_fraction=threshold_fraction,
            median_intensity=median_intensity,
            interior_spread=interior_spread,
            glint_fraction=glint_fraction,
            boundary_recovered=boundary_recovered,
            local_separation=local_separation,
            local_uniformity=local_uniformity,
        )

    def _detect_best(
        self,
        image: GrayImage,
        prior: PupilPrior | None = None,
    ) -> tuple[PupilCandidate, np.ndarray] | None:
        gray = np.asarray(image)
        if gray.dtype != np.uint8 or gray.ndim != 2:
            raise ValueError("Pupil detector expects a two-dimensional uint8 image")
        height, width = gray.shape
        if height < 24 or width < 24:
            raise ValueError("Eye crop must be at least 24x24 pixels")

        blurred, kernel = _segmentation_inputs(gray)
        intensity_references = np.percentile(
            blurred,
            (self.config.dark_reference_percentile, 70.0,
             *(min(49.0, p * self.config.threshold_percentile_scale)
               for p in self.config.threshold_percentiles)),
        )
        dark_floor, light_reference = intensity_references[:2]
        adaptive_span = max(light_reference - dark_floor, 1.0)
        threshold_span = max(light_reference - dark_floor, 12.0)
        percentile_values = intensity_references[2:]
        # A large pupil can occupy every low percentile. Keep the number of
        # candidate cutoffs fixed while letting calibration reach its lighter
        # boundary instead of fitting only the darkest central patch. Preserve
        # the low cutoffs so that the same profile still handles small pupils.
        adaptive = [
            dark_floor + min(fraction, self.config.threshold_range_fraction) * adaptive_span
            for fraction in (0.07, 0.12, self.config.threshold_range_fraction)
        ]
        if (
            prior is not None
            and prior.has_lock
            and prior.missing_frames < self.config.blink_after_missing_frames
        ):
            adaptive.append(dark_floor + prior.threshold_fraction * threshold_span)
        thresholds = sorted(
            {
                int(np.clip(round(value), 1, 254))
                for value in (*percentile_values.tolist(), *adaptive)
            }
        )
        adaptive_range = (float(dark_floor), float(light_reference))
        glints = _glint_mask(gray, adaptive_range, kernel)

        best_candidate: PupilCandidate | None = None
        best_contour: np.ndarray | None = None
        for threshold in thresholds:
            mask = _mask_at_threshold(blurred, threshold, kernel)
            contours, _hierarchy = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            for contour in contours:
                candidate = self._candidate(
                    gray,
                    mask,
                    contour,
                    threshold,
                    prior,
                    adaptive_range,
                    glints,
                )
                if (
                    candidate is not None
                    and candidate.confidence >= self.config.min_confidence
                    and (
                        best_candidate is None
                        or candidate.ranking_score > best_candidate.ranking_score
                    )
                ):
                    best_candidate = candidate
                    best_contour = contour

        if best_candidate is None or best_contour is None:
            return None
        return best_candidate, best_contour

    def detect(
        self,
        image: GrayImage,
        prior: PupilPrior | None = None,
    ) -> PupilCandidate | None:
        selected = self._detect_best(image, prior)
        return None if selected is None else selected[0]

    def detect_with_mask(
        self,
        image: GrayImage,
        prior: PupilPrior | None = None,
    ) -> tuple[PupilCandidate | None, GrayImage]:
        """Return the chosen candidate and only the pixels belonging to its contour."""

        gray = np.asarray(image)
        selected = self._detect_best(gray, prior)
        mask = np.zeros_like(gray, dtype=np.uint8)
        if selected is None:
            return None, mask
        candidate, contour = selected
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        return candidate, mask


class TemporalEyeTracker:
    """Adds smoothing, hold-last-position, and explicit blink output."""

    def __init__(
        self,
        eye_id: int,
        config: TrackerConfig,
        image_settings: EyeImageSettings | None = None,
    ) -> None:
        if eye_id not in (0, 1):
            raise ValueError("eye_id must be 0 or 1")
        self.eye_id = eye_id
        self.config = config
        self.image_settings = (
            EyeImageSettings(pupil_size_bias=config.pupil_size_bias)
            if image_settings is None
            else image_settings
        )
        self.detector = AdaptivePupilDetector(detector_config_for_eye(config, self.image_settings))
        self.state = PupilPrior()
        self.last_diagnostics: dict[str, object] = {
            "detected": False,
            "missing_frames": 0,
        }

    def reset(self) -> None:
        self.state = PupilPrior()
        self.last_diagnostics = {"detected": False, "missing_frames": 0}

    def process(self, image: GrayImage) -> EyeMeasurement:
        processed = apply_eye_image_settings(image, self.image_settings)
        candidate = self.detector.detect(processed, self.state)
        if candidate is None:
            self.state.update(None)
            self.last_diagnostics = {
                "detected": False,
                "missing_frames": self.state.missing_frames,
            }
            return EyeMeasurement(
                eye_id=self.eye_id,
                x=self.state.x if self.state.has_lock else 0.0,
                y=self.state.y if self.state.has_lock else 0.0,
                pupil_diameter=0.0,
                confidence=0.0,
                valid=False,
                blink=self.state.missing_frames >= self.config.blink_after_missing_frames,
            )

        short_prior = (
            self.state.has_lock
            and self.state.missing_frames < self.config.blink_after_missing_frames
        )
        self.state.update(
            candidate,
            position_alpha=self.config.position_smoothing if short_prior else 1.0,
            diameter_alpha=self.config.diameter_smoothing if short_prior else 1.0,
        )
        self.last_diagnostics = {
            "detected": True,
            "missing_frames": 0,
            "candidate_x": candidate.x,
            "candidate_y": candidate.y,
            "ellipse_x": candidate.ellipse_x,
            "ellipse_y": candidate.ellipse_y,
            "ellipse_width": candidate.ellipse_width,
            "ellipse_height": candidate.ellipse_height,
            "ellipse_angle_degrees": candidate.angle_degrees,
            "equivalent_diameter": candidate.equivalent_diameter,
            "confidence": candidate.confidence,
            "ranking_score": candidate.ranking_score,
            "contrast": candidate.contrast,
            "threshold": candidate.threshold,
            "threshold_fraction": candidate.threshold_fraction,
            "pupil_size_bias": self.image_settings.pupil_size_bias,
            "median_intensity": candidate.median_intensity,
            "interior_spread": candidate.interior_spread,
            "local_separation": candidate.local_separation,
            "local_uniformity": candidate.local_uniformity,
            "glint_fraction": candidate.glint_fraction,
            "boundary_recovered": candidate.boundary_recovered,
        }
        return EyeMeasurement(
            eye_id=self.eye_id,
            x=self.state.x,
            y=self.state.y,
            pupil_diameter=self.state.diameter,
            confidence=candidate.confidence,
            valid=True,
            blink=False,
        )


class MultiEyeTracker:
    def __init__(
        self,
        eye_ids: tuple[int, ...],
        config: TrackerConfig,
        image_settings: Mapping[int, EyeImageSettings] | None = None,
    ) -> None:
        if not 1 <= len(eye_ids) <= 2:
            raise ValueError("One or two eye IDs are required")
        if tuple(sorted(set(eye_ids))) != eye_ids:
            raise ValueError("eye_ids must be unique and sorted")
        configured = {} if image_settings is None else dict(image_settings)
        if set(configured) - set(eye_ids):
            raise ValueError("Image settings contain an unconfigured eye ID")
        self.trackers = {
            eye_id: TemporalEyeTracker(eye_id, config, configured.get(eye_id))
            for eye_id in eye_ids
        }
        self._executor: ThreadPoolExecutor | None = None
        if len(eye_ids) == 2:
            # Each eye has independent state. Application-level parallelism is
            # cheaper than copying crops into worker processes, and limiting
            # OpenCV's own pool prevents nested oversubscription on a four-core Pi.
            cv2.setNumThreads(1)
            self._executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="eye-fit",
            )

    def reset(self) -> None:
        for tracker in self.trackers.values():
            tracker.reset()

    def close(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def process(self, frame: AnalysisFrame, dropped_frames: int = 0) -> FrameResult:
        started_ns = time.monotonic_ns()
        supplied = {eye_id: image for eye_id, image in frame.crops}
        if set(supplied) != set(self.trackers):
            raise ValueError("Analysis frame eye IDs do not match the configured tracker")
        eye_ids = tuple(sorted(self.trackers))
        if self._executor is None:
            eyes = tuple(
                self.trackers[eye_id].process(supplied[eye_id]) for eye_id in eye_ids
            )
        else:
            futures = tuple(
                self._executor.submit(
                    self.trackers[eye_id].process,
                    supplied[eye_id],
                )
                for eye_id in eye_ids
            )
            eyes = tuple(future.result() for future in futures)
        produced_ns = time.monotonic_ns()
        return FrameResult(
            frame_sequence=frame.frame_sequence,
            camera_time_ns=frame.camera_time_ns,
            processed_monotonic_ns=produced_ns,
            eyes=eyes,
            dropped_analysis_frames=dropped_frames,
            processing_time_us=max(0, (produced_ns - started_ns) // 1_000),
            diagnostics={
                "eyes": {
                    str(eye_id): dict(self.trackers[eye_id].last_diagnostics)
                    for eye_id in sorted(self.trackers)
                }
            },
        )
