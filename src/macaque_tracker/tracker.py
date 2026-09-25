from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from .config import TrackerConfig
from .models import AnalysisFrame, EyeImageSettings, EyeMeasurement, FrameResult, GrayImage

try:  # Kept optional so protocol/config tools work on non-camera development hosts.
    import cv2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised on hosts without the vision extra
    cv2 = None


def apply_eye_image_settings(
    image: GrayImage,
    settings: EyeImageSettings,
) -> GrayImage:
    """Apply one eye's software image controls without modifying the source."""

    gray = np.asarray(image)
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("Eye image adjustments require a two-dimensional uint8 image")
    adjusted = gray.astype(np.float32)
    adjusted *= settings.gain
    adjusted = 128.0 + (adjusted - 128.0) * settings.contrast
    adjusted += 255.0 * settings.brightness
    if settings.sharpness > 0.0:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for software sharpness")
        working = np.clip(adjusted, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(working, (0, 0), 1.2).astype(np.float32)
        adjusted += min(settings.sharpness, 16.0) * 0.35 * (adjusted - blurred)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _segmentation_inputs(gray: GrayImage) -> tuple[GrayImage, np.ndarray]:
    height, width = gray.shape
    blur_size = max(3, round(min(width, height) * 0.015) | 1)
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

    # Divide by a broad local background estimate before thresholding. Camera
    # shading is primarily multiplicative, so this retains a dark pupil edge
    # while flattening slow illumination changes across the eye crop. A box
    # filter is intentionally used here: its cost is independent of the window
    # size and is small enough for the live path on the Pi.
    background_size = max(9, round(min(width, height) * 0.50) | 1)
    background = cv2.blur(
        blurred,
        (background_size, background_size),
        borderType=cv2.BORDER_REPLICATE,
    )
    normalized = cv2.divide(blurred, np.maximum(background, 8), scale=128.0)

    morphology_size = max(3, round(min(width, height) * 0.012) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morphology_size, morphology_size),
    )
    return normalized, kernel


def _mask_at_threshold(
    segmentation: GrayImage,
    threshold: int,
    kernel: np.ndarray,
) -> GrayImage:
    _value, mask = cv2.threshold(segmentation, threshold, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def pupil_mask_for_threshold(image: GrayImage, threshold: int) -> GrayImage:
    """Rebuild one detector mask for the separate diagnostic display process."""

    if cv2 is None:
        raise RuntimeError("OpenCV is required to build a pupil mask")
    gray = np.asarray(image)
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("Pupil mask expects a two-dimensional uint8 image")
    if not 1 <= threshold <= 254:
        raise ValueError("Pupil threshold must be in [1, 254]")
    segmentation, kernel = _segmentation_inputs(gray)
    return _mask_at_threshold(segmentation, threshold, kernel)


def _adaptive_appearance_scores(
    median_intensity: float,
    interior_spread: float,
    threshold: int,
    adaptive_range: tuple[float, float],
) -> tuple[float, float, float]:
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
    median_intensity: float
    interior_spread: float

    @property
    def diameter(self) -> float:
        """Area-equivalent external-contour diameter in pixels."""
        return self.equivalent_diameter


@dataclass
class _EyeState:
    x: float = 0.0
    y: float = 0.0
    diameter: float = 0.0
    has_lock: bool = False
    missing_frames: int = 0


class AdaptivePupilDetector:
    """Fast dark-pupil ellipse detector for a fixed macaque-eye crop.

    Candidate thresholds are always derived from each frame, not from an
    absolute brightness constant. Bright corneal reflections therefore become
    holes in the dark mask instead of assumed landmarks. Contours are ranked
    using pupil/iris contrast, ellipse residual, filled area, border distance,
    an optional relative size bias, and a temporal prior.
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

    def _contrast_and_fill(
        self,
        segmentation: GrayImage,
        mask: GrayImage,
        ellipse: tuple,
    ) -> tuple[float, float, float, float]:
        pupil_mask = np.zeros_like(segmentation, dtype=np.uint8)
        cv2.ellipse(pupil_mask, ellipse, 255, -1)
        pupil_pixels = segmentation[pupil_mask != 0]
        if pupil_pixels.size < 8:
            return 0.0, 0.0, 255.0, 255.0

        major = max(float(ellipse[1][0]), float(ellipse[1][1]))
        ring_width = max(3, round(0.12 * major))
        kernel_size = 2 * ring_width + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded = cv2.dilate(pupil_mask, kernel, iterations=1)
        ring_pixels = segmentation[(expanded != 0) & (pupil_mask == 0)]
        if ring_pixels.size < 8:
            return 0.0, 0.0, 255.0, 255.0

        # Medians make multiple saturated glints inside the pupil cheap and harmless.
        tenth, median, upper_quartile = np.percentile(pupil_pixels, (10.0, 50.0, 75.0))
        # Require most of the fitted boundary to be lighter than the interior.
        # This rejects rounded eyelash/eyelid endpoints that can otherwise look
        # circular even though they remain dark on one side.
        contrast = float(np.percentile(ring_pixels, 35.0) - median)
        pupil_area = int(np.count_nonzero(pupil_mask))
        covered = int(np.count_nonzero((mask != 0) & (pupil_mask != 0)))
        fill = 0.0 if pupil_area == 0 else covered / pupil_area
        # A pupil is normally a compact dark basin. An iris-sized ellipse often
        # contains both the much darker pupil and lighter iris, producing a
        # large robust interior spread even when its outer edge is high-contrast.
        interior_spread = float(upper_quartile - tenth)
        return contrast, float(fill), float(median), interior_spread

    def _candidate(
        self,
        segmentation: GrayImage,
        mask: GrayImage,
        contour: np.ndarray,
        threshold: int,
        prior: _EyeState | None,
        adaptive_range: tuple[float, float],
    ) -> PupilCandidate | None:
        height, width = segmentation.shape
        area = float(cv2.contourArea(contour))
        diameter = 2.0 * math.sqrt(area / math.pi)
        if (
            diameter < self.config.min_pupil_diameter_px
            or area > 0.42 * width * height
        ):
            return None

        ellipse = self._refined_ellipse(contour)
        if ellipse is None:
            return None
        (ellipse_cx, ellipse_cy), axes, angle = ellipse
        ellipse_width, ellipse_height = float(axes[0]), float(axes[1])
        major, minor = sorted((ellipse_width, ellipse_height), reverse=True)
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            return None
        # Head-fixed NHP trackers have found contour moments quieter than an
        # ellipse centre. The ellipse remains valuable as a shape validation.
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        if not all(np.isfinite((cx, cy, ellipse_cx, ellipse_cy, major, minor, angle))):
            return None
        if minor <= 1.0:
            return None
        if major > self.config.max_pupil_diameter_fraction * min(width, height):
            return None
        axis_ratio = minor / major
        if axis_ratio < self.config.min_axis_ratio:
            return None
        if math.hypot(cx - ellipse_cx, cy - ellipse_cy) > max(2.0, 0.15 * minor):
            return None
        if not (0.0 <= cx < width and 0.0 <= cy < height):
            return None

        points = contour[:, 0, :].astype(np.float64, copy=False)
        residual = float(np.median(self._ellipse_residuals(points, ellipse)))
        if not math.isfinite(residual) or residual > 0.38:
            return None

        contrast, fill, median_intensity, interior_spread = self._contrast_and_fill(
            segmentation,
            mask,
            ellipse,
        )
        if contrast < self.config.min_contrast or not 0.28 <= fill <= 1.20:
            return None

        diagonal = math.hypot(width, height)
        temporal_score = 0.72
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
        confidence = (
            0.14 * contrast_score
            + 0.18 * residual_score
            + 0.12 * fill_score
            + 0.18 * temporal_score
            + 0.04 * border_score
            + 0.14 * darkness_score
            + 0.10 * uniformity_score
            + 0.05 * threshold_score
            + 0.05 * size_score
        )
        ranking_score = confidence + size_bias_adjustment
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
            median_intensity=median_intensity,
            interior_spread=interior_spread,
        )

    def _detect_best(
        self,
        image: GrayImage,
        prior: _EyeState | None = None,
    ) -> tuple[PupilCandidate, np.ndarray] | None:
        gray = np.asarray(image)
        if gray.dtype != np.uint8 or gray.ndim != 2:
            raise ValueError("Pupil detector expects a two-dimensional uint8 image")
        height, width = gray.shape
        if height < 24 or width < 24:
            raise ValueError("Eye crop must be at least 24x24 pixels")

        segmentation, kernel = _segmentation_inputs(gray)
        dark_floor, light_reference = np.percentile(segmentation, (2.0, 70.0))
        percentile_values = np.percentile(segmentation, self.config.threshold_percentiles)
        adaptive = [
            dark_floor + fraction * max(light_reference - dark_floor, 1.0)
            for fraction in (0.07, 0.12, 0.18)
        ]
        thresholds = sorted(
            {
                int(np.clip(round(value), 1, 254))
                for value in (*percentile_values.tolist(), *adaptive)
            }
        )
        adaptive_range = (float(dark_floor), float(light_reference))

        best_candidate: PupilCandidate | None = None
        best_contour: np.ndarray | None = None
        for threshold in thresholds:
            mask = _mask_at_threshold(segmentation, threshold, kernel)
            contours, _hierarchy = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            for contour in contours:
                candidate = self._candidate(
                    segmentation,
                    mask,
                    contour,
                    threshold,
                    prior,
                    adaptive_range,
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

    def detect(self, image: GrayImage, prior: _EyeState | None = None) -> PupilCandidate | None:
        selected = self._detect_best(image, prior)
        return None if selected is None else selected[0]

    def detect_with_mask(
        self,
        image: GrayImage,
        prior: _EyeState | None = None,
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
        detector_config = replace(
            config,
            pupil_size_bias=self.image_settings.pupil_size_bias,
        )
        self.detector = AdaptivePupilDetector(detector_config)
        self.state = _EyeState()
        self.last_diagnostics: dict[str, object] = {
            "detected": False,
            "missing_frames": 0,
        }

    def reset(self) -> None:
        self.state = _EyeState()
        self.last_diagnostics = {"detected": False, "missing_frames": 0}

    def process(self, image: GrayImage) -> EyeMeasurement:
        processed = apply_eye_image_settings(image, self.image_settings)
        candidate = self.detector.detect(processed, self.state)
        if candidate is None:
            self.state.missing_frames += 1
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

        if (
            self.state.has_lock
            and self.state.missing_frames < self.config.blink_after_missing_frames
        ):
            position_alpha = self.config.position_smoothing
            diameter_alpha = self.config.diameter_smoothing
            self.state.x = position_alpha * candidate.x + (1.0 - position_alpha) * self.state.x
            self.state.y = position_alpha * candidate.y + (1.0 - position_alpha) * self.state.y
            self.state.diameter = (
                diameter_alpha * candidate.diameter
                + (1.0 - diameter_alpha) * self.state.diameter
            )
        else:
            self.state.x = candidate.x
            self.state.y = candidate.y
            self.state.diameter = candidate.diameter
            self.state.has_lock = True
        self.state.missing_frames = 0
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
            "pupil_size_bias": self.image_settings.pupil_size_bias,
            "median_intensity": candidate.median_intensity,
            "interior_spread": candidate.interior_spread,
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

    def reset(self) -> None:
        for tracker in self.trackers.values():
            tracker.reset()

    def process(self, frame: AnalysisFrame, dropped_frames: int = 0) -> FrameResult:
        started_ns = time.monotonic_ns()
        supplied = {eye_id: image for eye_id, image in frame.crops}
        if set(supplied) != set(self.trackers):
            raise ValueError("Analysis frame eye IDs do not match the configured tracker")
        eyes = tuple(
            self.trackers[eye_id].process(supplied[eye_id])
            for eye_id in sorted(self.trackers)
        )
        produced_ns = time.monotonic_ns()
        return FrameResult(
            frame_sequence=frame.frame_sequence,
            sensor_timestamp_ns=frame.sensor_timestamp_ns,
            produced_timestamp_ns=produced_ns,
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
