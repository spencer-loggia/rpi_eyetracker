"""Camera-frame graphical sync edges; no image retention, UI, or I/O."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from numbers import Real

from .models import GrayImage, PixelRoi


def _validate_threshold(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 < value <= 510:
        raise ValueError("Sync threshold must be a finite number in (0, 510]")


@dataclass(frozen=True)
class SyncBox:
    """Raw normalized sync patch."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
               for v in values):
            raise ValueError("Sync ROI coordinates must be finite numbers")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Sync ROI coordinates and size must be positive and in frame")
        if self.x + self.width > 1 + 1e-9 or self.y + self.height > 1 + 1e-9:
            raise ValueError("Sync ROI must fit inside the normalized frame")

    @classmethod
    def from_pixels(
        cls, roi: PixelRoi, frame_width: int, frame_height: int,
    ) -> SyncBox:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive")
        if roi.x2 > frame_width or roi.y2 > frame_height:
            raise ValueError("Sync ROI lies outside the source frame")
        return cls(roi.x / frame_width, roi.y / frame_height,
                   roi.width / frame_width, roi.height / frame_height)

    def to_pixels(self, frame_width: int, frame_height: int) -> PixelRoi:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive")
        x = min(max(round(self.x * frame_width), 0), frame_width - 1)
        y = min(max(round(self.y * frame_height), 0), frame_height - 1)
        x2 = min(max(round((self.x + self.width) * frame_width), x + 1), frame_width)
        y2 = min(max(round((self.y + self.height) * frame_height), y + 1), frame_height)
        return PixelRoi(x, y, x2 - x, y2 - y)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SyncRoi:
    """Two independently placed, opposite-phase patches and their edge threshold."""

    boxes: tuple[SyncBox, SyncBox]
    threshold: float = 20.0

    def __post_init__(self) -> None:
        if (not isinstance(self.boxes, tuple) or len(self.boxes) != 2
                or any(not isinstance(box, SyncBox) for box in self.boxes)):
            raise ValueError("Sync configuration requires exactly two SyncBox regions")
        _validate_threshold(self.threshold)

    @classmethod
    def from_pixels(
        cls, boxes: tuple[PixelRoi, PixelRoi], frame_width: int, frame_height: int,
        threshold: float = 20.0,
    ) -> SyncRoi:
        return cls(tuple(SyncBox.from_pixels(box, frame_width, frame_height) for box in boxes),
                   threshold)

    def to_pixels(self, frame_width: int, frame_height: int) -> tuple[PixelRoi, PixelRoi]:
        return tuple(box.to_pixels(frame_width, frame_height) for box in self.boxes)

    def as_dict(self) -> dict:
        return {"boxes": [box.as_dict() for box in self.boxes], "threshold": self.threshold}

    @classmethod
    def from_dict(cls, value: dict, frame_width: int, frame_height: int) -> SyncRoi:
        """Migrate old split boxes once, at their saved source pixel resolution."""
        if not isinstance(value, dict):
            raise TypeError("Sync configuration must be a JSON object")
        values = dict(value)
        if "boxes" in values:
            values["boxes"] = tuple(SyncBox(**box) for box in values["boxes"])
            return cls(**values)
        threshold = values.pop("threshold", 20.0)
        old = SyncBox(**values).to_pixels(frame_width, frame_height)
        if old.width < 2:
            raise ValueError("Legacy split sync ROI must be at least two pixels wide")
        left_width = old.width // 2
        return cls.from_pixels((
            PixelRoi(old.x, old.y, left_width, old.height),
            PixelRoi(old.x + left_width, old.y, old.width - left_width, old.height),
        ), frame_width, frame_height, threshold)


def sync_signal(gray: GrayImage, boxes: tuple[PixelRoi, PixelRoi]) -> float:
    """Mean box 0 minus mean box 1, without copying or retaining either crop.

    Each box is averaged separately so unequal-sized regions still reject a
    uniform change in ambient illumination.
    """
    first, second = boxes
    return (float(gray[first.y:first.y2, first.x:first.x2].mean())
            - float(gray[second.y:second.y2, second.x:second.x2].mean()))


@dataclass(frozen=True)
class SyncTransition:
    frame_sequence: int
    camera_time_ns: int
    previous_camera_time_ns: int
    previous_signal: float
    signal: float
    delta: float
    direction: str


class SyncDetector:
    """Emit abrupt box 0/1 swaps at the first frame with the opposite polarity.

    One threshold is measured in box-0-minus-box-1 luma units per camera frame.
    Both a polarity reversal and an above-threshold change are required.
    Opposite edges are detected immediately, even on adjacent frames. A frame
    gap establishes a fresh baseline rather than accumulating ambient drift.
    Sensor time zero remains unknown, as it does for untimed video playback.
    """

    def __init__(self, threshold: float, fps: float) -> None:
        _validate_threshold(threshold)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Sync frame rate must be positive and finite")
        self.threshold = threshold
        self._maximum_interval_ns = math.ceil(1.5e9 / fps)
        self.reset()

    def reset(self) -> None:
        self._previous: tuple[float, int, int] | None = None
        self._polarity = 0

    def process(
        self, signal: float, frame_sequence: int, camera_time_ns: int,
    ) -> SyncTransition | None:
        previous = self._previous
        self._previous = (signal, frame_sequence, camera_time_ns)
        polarity = 1 if signal > 0 else -1 if signal < 0 else 0
        previous_polarity = self._polarity
        if polarity:
            self._polarity = polarity
        if previous is None:
            return None
        before, sequence, timestamp = previous
        elapsed = camera_time_ns - timestamp
        if (
            (frame_sequence - sequence) & 0xFFFFFFFF != 1
            or bool(camera_time_ns) != bool(timestamp)
            or (camera_time_ns and not 0 < elapsed <= self._maximum_interval_ns)
        ):
            self._polarity = polarity
            return None
        delta = signal - before
        if (not polarity or not previous_polarity or polarity == previous_polarity
                or abs(delta) < self.threshold):
            return None
        return SyncTransition(frame_sequence, camera_time_ns, timestamp, before, signal,
                              delta, "rising" if delta > 0 else "falling")
