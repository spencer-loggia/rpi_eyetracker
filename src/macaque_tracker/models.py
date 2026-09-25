from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

GrayImage = NDArray[np.uint8]


@dataclass(frozen=True)
class EyeImageSettings:
    """Software processing and fit preference applied to one eye crop."""

    gain: float = 1.0
    brightness: float = 0.0
    contrast: float = 1.0
    sharpness: float = 0.0
    pupil_size_bias: float = 0.0

    def __post_init__(self) -> None:
        for name, minimum, maximum, inclusive in (
            ("gain", 0.0, 8.0, False),
            ("contrast", 0.0, 8.0, False),
            ("sharpness", 0.0, 8.0, True),
        ):
            value = getattr(self, name)
            valid = np.isfinite(value) and (
                value >= minimum if inclusive else value > minimum
            ) and value <= maximum
            if not valid:
                interval = "[0, 8]" if inclusive else "(0, 8]"
                raise ValueError(f"Eye image {name} must be finite and in {interval}")
        if not np.isfinite(self.brightness) or not -1.0 <= self.brightness <= 1.0:
            raise ValueError("Eye image brightness must be finite and in [-1, 1]")
        if (
            isinstance(self.pupil_size_bias, bool)
            or not np.isfinite(self.pupil_size_bias)
            or not -1.0 <= self.pupil_size_bias <= 1.0
        ):
            raise ValueError("Eye pupil_size_bias must be finite and in [-1, 1]")

    def as_dict(self) -> dict[str, Any]:
        return {
            "gain": self.gain,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "sharpness": self.sharpness,
            "pupil_size_bias": self.pupil_size_bias,
        }


@dataclass(frozen=True)
class PixelRoi:
    """A half-open rectangle in one image: ``[x:x+width, y:y+height]``."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("ROI x and y must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def extract(self, image: GrayImage) -> GrayImage:
        if image.ndim != 2:
            raise ValueError("ROI extraction expects a two-dimensional grayscale image")
        height, width = image.shape
        if self.x2 > width or self.y2 > height:
            raise ValueError(f"ROI {self} lies outside image dimensions {width}x{height}")
        # This must own its memory: the source can be a Picamera2 MappedArray
        # whose buffer is released as soon as the camera request context exits.
        return image[self.y : self.y2, self.x : self.x2].copy(order="C")


@dataclass(frozen=True)
class NormalizedRoi:
    """Resolution-independent crop coordinates in the closed interval [0, 1]."""

    eye_id: int
    label: str
    x: float
    y: float
    width: float
    height: float
    settings: EyeImageSettings = field(default_factory=EyeImageSettings)

    def __post_init__(self) -> None:
        if self.eye_id not in (0, 1):
            raise ValueError("eye_id must be 0 or 1")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("ROI label must not be empty")
        if not isinstance(self.settings, EyeImageSettings):
            raise ValueError("ROI settings must be EyeImageSettings")
        values = (self.x, self.y, self.width, self.height)
        if not all(np.isfinite(values)):
            raise ValueError("ROI coordinates must be finite")
        if self.x < 0.0 or self.y < 0.0 or self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("ROI coordinates and size must be positive and in frame")
        if self.x + self.width > 1.0 + 1e-9 or self.y + self.height > 1.0 + 1e-9:
            raise ValueError("ROI must fit inside the normalized frame")

    @classmethod
    def from_pixels(
        cls,
        *,
        eye_id: int,
        label: str,
        roi: PixelRoi,
        frame_width: int,
        frame_height: int,
        settings: EyeImageSettings | None = None,
    ) -> NormalizedRoi:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive")
        if roi.x2 > frame_width or roi.y2 > frame_height:
            raise ValueError("ROI lies outside the source frame")
        return cls(
            eye_id=eye_id,
            label=label,
            x=roi.x / frame_width,
            y=roi.y / frame_height,
            width=roi.width / frame_width,
            height=roi.height / frame_height,
            settings=EyeImageSettings() if settings is None else settings,
        )

    def to_pixels(self, frame_width: int, frame_height: int) -> PixelRoi:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive")
        x1 = round(self.x * frame_width)
        y1 = round(self.y * frame_height)
        x2 = round((self.x + self.width) * frame_width)
        y2 = round((self.y + self.height) * frame_height)
        x1 = min(max(x1, 0), frame_width - 1)
        y1 = min(max(y1, 0), frame_height - 1)
        x2 = min(max(x2, x1 + 1), frame_width)
        y2 = min(max(y2, y1 + 1), frame_height)
        return PixelRoi(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "eye_id": self.eye_id,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "settings": self.settings.as_dict(),
        }


@dataclass(frozen=True)
class EyeMeasurement:
    """One eye result in crop-local pixels.

    During a blink or failed detection, ``pupil_diameter`` is exactly zero and
    x/y retain the last reliable position (or zero before the first lock).
    """

    eye_id: int
    x: float
    y: float
    pupil_diameter: float
    confidence: float
    valid: bool
    blink: bool

    def __post_init__(self) -> None:
        if self.eye_id not in (0, 1):
            raise ValueError("eye_id must be 0 or 1")
        if not all(np.isfinite((self.x, self.y, self.pupil_diameter, self.confidence))):
            raise ValueError("Eye measurement values must be finite")
        if self.pupil_diameter < 0.0:
            raise ValueError("pupil_diameter must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if (self.blink or not self.valid) and self.pupil_diameter != 0.0:
            raise ValueError("Invalid/blink measurements must report pupil_diameter=0")
        if self.valid and self.pupil_diameter <= 0.0:
            raise ValueError("Valid measurements must report a positive pupil_diameter")


@dataclass(frozen=True)
class FrameResult:
    frame_sequence: int
    sensor_timestamp_ns: int
    produced_timestamp_ns: int
    eyes: tuple[EyeMeasurement, ...]
    dropped_analysis_frames: int = 0
    processing_time_us: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.frame_sequence < 0:
            raise ValueError("frame_sequence must be non-negative")
        if self.sensor_timestamp_ns < 0 or self.produced_timestamp_ns < 0:
            raise ValueError("timestamps must be non-negative")
        if not 1 <= len(self.eyes) <= 2:
            raise ValueError("A frame result must contain one or two eyes")
        ids = [eye.eye_id for eye in self.eyes]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("Eye measurements must be unique and ordered by eye_id")
        if self.dropped_analysis_frames < 0 or self.processing_time_us < 0:
            raise ValueError("Frame counters and processing time must be non-negative")


@dataclass(frozen=True)
class AnalysisFrame:
    frame_sequence: int
    sensor_timestamp_ns: int
    crops: tuple[tuple[int, GrayImage], ...]
