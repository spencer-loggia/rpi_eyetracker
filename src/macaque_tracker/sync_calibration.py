"""Bounded graphical-sync calibration samples, independent of UI toolkits."""

from collections import deque
from threading import Lock

from .graphical_sync import SyncDetector, sync_signal


class SyncCalibration:
    """Sample every captured frame while retaining only one crop and scalar state."""

    def __init__(self, box, threshold, fps, *, validated=False):
        self.box = box
        self.threshold = threshold
        self.fps = fps
        self._lock = Lock()
        self._validated_threshold = threshold if validated else None
        self._history = deque(maxlen=256)
        self.detector = SyncDetector(threshold, fps)
        self.crop = None
        self.intensity = 0.0
        self.delta = 0.0
        self.last_event = None
        self.rising = self.falling = self.samples = 0
        self._previous = None

    def reset(self, *, fps=None, threshold=None):
        with self._lock:
            if fps is not None:
                self.fps = fps
            if threshold is not None:
                self.threshold = threshold
            else:
                self._history.clear()
            self.detector = SyncDetector(self.threshold, self.fps)
            self.rising = self.falling = self.samples = 0
            self.last_event = self._previous = None
            self.delta = 0.0
            for sample in self._history:
                self._process(*sample)

    def _process(self, intensity, sequence, timestamp):
        event = self.detector.process(intensity, sequence, timestamp)
        self.intensity = intensity
        self.delta = 0.0 if self._previous is None else intensity - self._previous
        self._previous = intensity
        self.samples += 1
        if event is not None:
            self.last_event = event
            self.rising += event.direction == "rising"
            self.falling += event.direction == "falling"

    def observe(self, image, sequence, timestamp):
        intensity = sync_signal(image, self.box)
        crop = self.box.extract(image)
        with self._lock:
            self.crop = crop
            self._history.append((intensity, sequence, timestamp))
            self._process(intensity, sequence, timestamp)

    def snapshot(self):
        with self._lock:
            return (self.crop, self.intensity, self.delta, self.last_event,
                    self.rising, self.falling, self.samples)

    def validate(self):
        with self._lock:
            if self.threshold != self._validated_threshold and not (self.rising and self.falling):
                raise ValueError("Use live/video pulses to detect both left/right swaps before saving")
