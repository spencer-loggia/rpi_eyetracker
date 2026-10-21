"""Cheap transmitter-side rates; a successful write is not a receiver receipt."""

from __future__ import annotations

import threading
import time


class TelemetryRates:
    """Count distinct fresh valid eye positions actually written to the transport.

    Two-second wall-time windows include idle time and failed detection periods.
    Repeated polls for the same camera result never increase the valid rate.
    """

    def __init__(self, *, clock=time.monotonic, window_s=2.0):
        self._clock = clock
        self._window = window_s
        self._started = clock()
        self._last = None
        self._counts = [0, 0]
        self._rates = (0.0, 0.0)
        self._eyes = 0
        self._lock = threading.Lock()

    def _advance(self, now):
        elapsed = now - self._started
        if elapsed >= self._window:
            self._rates = tuple(count / elapsed for count in self._counts)
            self._counts = [0, 0]
            self._started = now

    def sent(self, snapshot):
        """Record only after the complete status packet was successfully written."""
        with self._lock:
            self._advance(self._clock())
            self._eyes = snapshot.configured_eye_count
            result = snapshot.result
            if result is None:
                return
            key = (snapshot.session_id, result.frame_sequence, result.camera_time_ns)
            if key == self._last:
                return
            self._last = key
            if not snapshot.tracking or snapshot.stale or result.camera_time_ns == 0:
                return
            for eye in result.eyes:
                if eye.valid and not eye.blink:
                    self._counts[eye.eye_id] += 1

    def snapshot(self):
        """Return per-eye valid transmit Hz, aging to zero when transmission stops."""
        with self._lock:
            self._advance(self._clock())
            return self._rates[:self._eyes]
