"""Exact sensor timing for saved video, independent of tracking and container PTS."""

from __future__ import annotations

import csv
import os
import threading
from collections import deque
from fractions import Fraction
from pathlib import Path

from .file_access import SHARED_FILE_MODE

_HEADER = ("video_frame_index", "camera_time_ns")


class SensorTimingError(RuntimeError):
    """A camera frame cannot be assigned trustworthy sensor timing."""


def camera_time_ns(metadata: dict, key="SensorTimestamp") -> int:
    """Return SensorTimestamp unchanged; never substitute a software clock."""
    value = metadata.get(key)
    if type(value) is not int or not 0 < value <= 0xFFFFFFFFFFFFFFFF:
        raise SensorTimingError("Missing or invalid SensorTimestamp; camera_time_ns is unavailable")
    return value


def request_camera_time_ns(request, sensor_control=None) -> int:
    """Read the raw libcamera control without translating all metadata on the Pi."""
    if sensor_control is not None:
        return camera_time_ns(request.request.metadata, sensor_control)
    return camera_time_ns(request.get_metadata())


def video_index_path(video: Path) -> Path:
    """Use one compact CSV beside each video, sharing its filename stem."""
    return video.with_suffix(".frames.csv")


class VideoFrameIndex:
    """Match submitted frames to successful output writes with bounded, constant-cost state.

    The codec uses relative microseconds internally; the CSV retains the original
    nanoseconds. Zero B-frames make packet order equal decoded frame order. Rows
    are committed by the output's post-write callback, including codec flush.
    """

    def __init__(self, video: Path, *, shared: bool = False):
        self.path = video_index_path(video)
        self._pending: deque[tuple[int, int]] = deque()
        self._lock = threading.Lock()
        self._first_us: int | None = None
        self._last_us = -1
        self._rows = 0
        self.error: Exception | None = None
        self._stream = self.path.open("x", encoding="ascii", newline="", buffering=64 * 1024)
        try:
            if shared:
                os.fchmod(self._stream.fileno(), SHARED_FILE_MODE)
            self._stream.write(",".join(_HEADER) + "\n")
            self._stream.flush()
        except BaseException:
            self._stream.close()
            self.path.unlink(missing_ok=True)
            raise

    def submit(self, value: int) -> int:
        """Register one selected sensor frame before encoding; return its codec PTS."""
        if self.error is not None:
            raise self.error
        stamp_us = value // 1000
        with self._lock:
            if stamp_us <= self._last_us:
                raise SensorTimingError("Video camera_time_ns must increase on every saved frame")
            if len(self._pending) >= 256:
                raise SensorTimingError("Video output is not retiring frame timestamps")
            self._last_us = stamp_us
            if self._first_us is None:
                self._first_us = stamp_us
            pts = stamp_us - self._first_us
            self._pending.append((pts, value))
        return pts

    def written(self, pts: int) -> None:
        """Append only after successful mux/write; reject missing/reordered output frames."""
        if self.error is not None:
            raise self.error
        try:
            with self._lock:
                if not self._pending or self._pending[0][0] != pts:
                    raise SensorTimingError("Video output does not match its sensor frame index")
                _, value = self._pending.popleft()
            # Only the encoder/output worker writes the CSV. Never hold the
            # submission lock during disk I/O or flush on the tracking thread.
            self._stream.write(f"{self._rows},{value}\n")
            self._rows += 1
            if self._rows % 60 == 0:
                self._stream.flush()
        except Exception as exc:
            self.error = exc
            raise

    def close(self) -> None:
        """Flush after the encoder/output closes and surface any missing output frames."""
        if self._stream.closed:
            return
        try:
            self._stream.close()
        finally:
            if self.error is None and self._pending:
                self.error = SensorTimingError("Video ended with sensor frames missing from its output")
        if self.error is not None:
            raise self.error


def indexed_encoder_type(
    base_encoder: type, index: VideoFrameIndex, *, sensor_control=None,
    recording_fps: float | None = None,
    camera_fps: float | None = None,
) -> type:
    """Select by sensor time before packing, using one metadata read per request.

    A fixed request stride undershoots the recording rate when capture runs
    below its nominal rate (55 Hz / 2 gives 27.5 fps). Absolute sensor deadlines
    retain 30 real frames per second whenever acquisition supplies enough frames.
    Integer arithmetic avoids accumulated rounding drift at fractional rates.
    """
    rate = None if recording_fps is None else Fraction(str(recording_fps))
    if rate is not None and rate <= 0:
        raise ValueError("Recording frame rate must be positive")
    period = None if rate is None else 1_000_000_000 * rate.denominator
    # Accept the frame nearest each deadline. Half an acquisition period keeps
    # small sensor jitter from alternating 50/16.7 ms gaps in a 60 -> 30 stream.
    allowance = (0 if rate is None or camera_fps is None else
                 round(500_000_000 * rate.numerator / camera_fps))

    class IndexedEncoder(base_encoder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.recording_error = None
            self._next_frame_tick = None
            self._last_sensor_time_ns = 0
            self._selected_pts = 0

        def _timestamp(self, _request):
            return self._selected_pts

        def _encode(self, stream, request):
            if self.recording_error is not None:
                return
            try:
                stamp = request_camera_time_ns(request, sensor_control)
                if stamp <= self._last_sensor_time_ns:
                    raise SensorTimingError("Video camera_time_ns must increase on every request")
                self._last_sensor_time_ns = stamp
                if rate is not None:
                    tick = stamp * rate.numerator
                    if self._next_frame_tick is None:
                        self._next_frame_tick = tick - allowance
                    # Deadlines are quantized only for comparison with the
                    # integer-nanosecond clock; retain the fractional remainder.
                    if stamp < self._next_frame_tick // rate.numerator:
                        return
                    # After a capture gap, skip expired deadlines instead of
                    # inventing duplicates or emitting a catch-up burst.
                    self._next_frame_tick += (
                        max(0, tick - self._next_frame_tick) // period + 1
                    ) * period
                self._selected_pts = index.submit(stamp)
                super()._encode(stream, request)
            except Exception as exc:  # noqa: BLE001 - expose camera callback errors through health
                self.recording_error = exc

    return IndexedEncoder


def attach_index(output, index: VideoFrameIndex, report_error) -> None:
    """Use Picamera2's post-write timestamp hook, not encoder submission counts.

    PyavOutput reports mux failures via error_callback before outputtimestamp;
    FileOutput reports failed writes via connectiondead. Neither failure may
    produce a CSV row. The hook also excludes packets skipped before a keyframe.
    """
    def failed(error):
        index.error = index.error or error
        report_error(error)

    def written(pts):
        try:
            index.written(pts)
        except Exception as exc:
            failed(exc)
            raise

    output.error_callback = failed
    if hasattr(output, "connectiondead"):
        output.connectiondead = failed
    output.outputtimestamp = written
    if hasattr(output, "_container"):
        original_stop = output.stop

        def stop():
            """Surface PyAV trailer/close failures that Picamera2 otherwise suppresses."""
            try:
                if output._container is not None:
                    output._container.close()
            except Exception as exc:
                failed(exc)
                raise
            finally:
                original_stop()  # Idempotent close plus normal output-state cleanup.

        output.stop = stop


class VideoFrameTimes:
    """Stream a recording's authoritative index during playback; absent timing is zero."""

    def __init__(self, video: Path):
        path = video_index_path(video)
        self._stream = path.open(encoding="ascii", newline="") if path.exists() else None
        try:
            self.rewind()
        except BaseException:
            self.close()
            raise

    def rewind(self) -> None:
        """Restart alongside the video decoder, never synthesize time on loops."""
        self._position, self._last = 0, 0
        if self._stream is not None:
            self._stream.seek(0)
            self._reader = csv.reader(self._stream)
            if tuple(next(self._reader, ())) != _HEADER:
                raise SensorTimingError("Invalid video frame index header")

    def next(self) -> int:
        """Consume one row for every decoded or deliberately skipped video frame."""
        if self._stream is None:
            return 0
        try:
            position, stamp = next(self._reader)
            position, stamp = int(position), int(stamp)
        except (StopIteration, ValueError) as exc:
            raise SensorTimingError("Video frame index is truncated or invalid") from exc
        if position != self._position or not self._last < stamp <= 0xFFFFFFFFFFFFFFFF:
            raise SensorTimingError("Video frame index is out of order or has invalid camera_time_ns")
        self._position += 1
        self._last = stamp
        return stamp

    def seek(self, index: int) -> None:
        """Align the next row with an offline decoder seek without inventing time."""
        if self._stream is None:
            return
        if index < self._position:
            self.rewind()
        while self._position < index:
            self.next()

    def end(self) -> None:
        """Detect a truncated video rather than silently ignoring extra index rows."""
        if self._stream is not None and next(self._reader, None) is not None:
            raise SensorTimingError("Video ended before its frame index")

    def close(self) -> None:
        """Release the playback index file if present."""
        if self._stream is not None:
            self._stream.close()
