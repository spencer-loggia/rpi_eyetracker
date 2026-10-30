from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from .config import RecordingConfig, RoiLayout
from .models import GrayImage, NormalizedRoi, PixelRoi
from .orientation import rotate_image, rotated_size


@dataclass(frozen=True)
class VideoRegion:
    """A raw sensor rectangle and its physically oriented location in the video."""

    label: str
    source: PixelRoi
    destination: PixelRoi
    rotation: int = 0

    def __post_init__(self) -> None:
        if type(self.rotation) is not int or self.rotation not in (0, 90, 180, 270):
            raise ValueError("Recording rotation must be 0, 90, 180, or 270 degrees")


@dataclass(frozen=True)
class RecordingLayout:
    """Lossless pixel mapping between the raw CamArray and the encoded montage."""

    source_width: int
    source_height: int
    width: int
    height: int
    regions: tuple[VideoRegion, ...]

    def __post_init__(self) -> None:
        for value in (self.source_width, self.source_height, self.width, self.height):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("Recording layout dimensions must be positive integers")
        if self.width % 2 or self.height % 2 or not self.regions:
            raise ValueError("Recording layout requires even dimensions and regions")
        for region in self.regions:
            source, target = region.source, region.destination
            if (
                source.x2 > self.source_width or source.y2 > self.source_height
                or target.x2 > self.width or target.y2 > self.height
                or rotated_size(source.width, source.height, region.rotation)
                != (target.width, target.height)
            ):
                raise ValueError("Recording region is outside its frame or changes pixel size")

    def pack(self, gray: GrayImage) -> GrayImage:
        """Own only the selected luma pixels; synthesize neutral YUV420 chroma."""
        if gray.shape != (self.source_height, self.source_width):
            raise ValueError("Recording source dimensions do not match the layout")
        yuv = np.empty((self.height * 3 // 2, self.width), dtype=np.uint8)
        yuv[:self.height] = 16
        yuv[self.height:] = 128
        for region in self.regions:
            source, target = region.source, region.destination
            yuv[target.y:target.y2, target.x:target.x2] = rotate_image(
                gray[source.y:source.y2, source.x:source.x2], region.rotation,
            )
        return yuv

    def as_dict(self) -> dict[str, Any]:
        """Return the complete reversible mapping for the recording sidecar."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RecordingLayout:
        """Read current mappings and older sidecars that have no rotation field."""
        values = dict(value)
        values["regions"] = tuple(
            VideoRegion(
                item["label"], PixelRoi(**item["source"]), PixelRoi(**item["destination"]),
                item.get("rotation", 0),
            )
            for item in value["regions"]
        )
        return cls(**values)

    def unpack(
        self, gray: GrayImage, *, size: tuple[int, int] | None = None,
        eye_rois: tuple[NormalizedRoi, ...] = (),
    ) -> GrayImage:
        """Restore each recorded region separately; never resize missing black pixels.

        Original normalized eye boxes disambiguate rounding between recording and
        analysis resolutions. Newly selected playback boxes do not relocate pixels.
        """
        if gray.shape != (self.height, self.width):
            raise ValueError("Packed video dimensions do not match its metadata")
        width, height = (self.source_width, self.source_height) if size is None else size
        result = np.zeros((height, width), dtype=np.uint8)
        eyes = {f"eye_{roi.eye_id}": roi for roi in eye_rois}
        for region in self.regions:
            source, target = region.source, region.destination
            pixels = rotate_image(
                gray[target.y:target.y2, target.x:target.x2], (-region.rotation) % 360,
            )
            if (width, height) != (self.source_width, self.source_height):
                import cv2

                original = eyes.get(region.label) or NormalizedRoi.from_pixels(
                    eye_id=0, label=region.label, roi=source,
                    frame_width=self.source_width, frame_height=self.source_height,
                )
                source = original.to_pixels(width, height)
                pixels = cv2.resize(pixels, (source.width, source.height), interpolation=cv2.INTER_AREA)
            result[source.y:source.y2, source.x:source.x2] = pixels
        return result


def load_recording_header(video: Path) -> dict[str, Any]:
    """Read recording metadata once, without scanning per-frame tracking records."""
    sidecar = video.with_suffix(".jsonl")
    if not sidecar.is_file():
        return {}
    try:
        with sidecar.open(encoding="utf-8") as stream:
            header = json.loads(stream.readline(1024 * 1024))
        if not isinstance(header, dict):
            raise TypeError("Recording header must be an object")
        return header
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid recording layout in {sidecar}: {exc}") from exc


def load_recording_layout(video: Path) -> RecordingLayout | None:
    """Restore a packed recording's reversible pixel mapping when present."""
    layout = load_recording_header(video).get("video_layout")
    return None if layout is None else RecordingLayout.from_dict(layout)


def _pack_regions(regions: tuple[tuple[str, PixelRoi, int], ...]) -> tuple[VideoRegion, ...]:
    """Minimize montage area over horizontal/vertical rectangle subdivisions.

    At most six rectangles are planned once when recording starts. Pareto
    pruning discards dimensions that cannot improve a larger layout; no packing
    search, resizing, or intermediate rotated-image allocation occurs per frame.
    """
    sizes = tuple(rotated_size(source.width, source.height, angle)
                  for _, source, angle in regions)

    @cache
    def candidates(mask):
        """Return non-dominated bounding boxes and placements for one region subset."""
        if mask.bit_count() == 1:
            index = mask.bit_length() - 1
            return {sizes[index]: ((index, 0, 0),)}
        layouts = {}
        subset = (mask - 1) & mask
        while subset:
            other = mask ^ subset
            # Mirroring the two groups does not change their bounding box.
            if other and subset & (mask & -mask):
                for (aw, ah), a in candidates(subset).items():
                    for (bw, bh), b in candidates(other).items():
                        for size, offset in (((aw + bw, max(ah, bh)), (aw, 0)),
                                             ((max(aw, bw), ah + bh), (0, ah))):
                            dx, dy = offset
                            placement = tuple(sorted(a + tuple(
                                (index, x + dx, y + dy) for index, x, y in b
                            )))
                            previous = layouts.get(size)
                            if previous is None or placement < previous:
                                layouts[size] = placement
            subset = (subset - 1) & mask
        frontier = {}
        lowest_height = float("inf")
        for (width, height), placement in sorted(layouts.items()):
            if height < lowest_height:
                frontier[width, height] = placement
                lowest_height = height
        return frontier

    _, placement = min(
        candidates((1 << len(regions)) - 1).items(),
        key=lambda item: ((item[0][0] + item[0][0] % 2)
                          * (item[0][1] + item[0][1] % 2), item[0][1], item[1]),
    )
    return tuple(
        VideoRegion(label, source, PixelRoi(x, y, *sizes[index]), angle)
        for (index, x, y), (label, source, angle) in zip(placement, regions, strict=True)
    )


def selected_recording_layout(config: RecordingConfig, rois: RoiLayout) -> RecordingLayout:
    """Pack oriented full pose camera views and unscaled oriented eye crops."""
    return _camera_recording_layout(config, rois, full_frame=False)


def recording_layout(config: RecordingConfig, rois: RoiLayout | None) -> RecordingLayout | None:
    """Keep raw full-frame recording fast; describe every packed/oriented output."""
    if config.layout == "hands_eyes":
        if rois is None:
            raise ValueError("Selected-view recording requires configured eye ROIs")
        return selected_recording_layout(config, rois)
    if rois is not None and any(rois.camera_rotations):
        return _camera_recording_layout(config, rois, full_frame=True)
    if config.height % 4:
        # Picamera2's unwrapped zero-copy YUV path has the same height rounding
        # limitation. Keep full-frame geometry but use our checked encoder.
        frame = PixelRoi(0, 0, config.width, config.height)
        return RecordingLayout(config.width, config.height, config.width, config.height,
                               (VideoRegion("full_frame", frame, frame),))
    return None


def _camera_recording_layout(
    config: RecordingConfig, rois: RoiLayout, *, full_frame: bool,
) -> RecordingLayout:
    """Build the pixel-preserving montage for selected views or all four cameras."""
    width, height = config.width, config.height
    if width % 4:
        raise ValueError("Oriented/selected recording requires four equal-width camera views")
    camera_width = width // 4
    cameras = range(4) if full_frame else config.pose_cameras
    sources = [(f"camera_{i}", PixelRoi(i * camera_width, 0, camera_width, height),
                rois.camera_rotations[i]) for i in cameras]
    if not full_frame:
        sources.extend(
            (f"eye_{roi.eye_id}", roi.to_pixels(width, height), rois.rotation_for_roi(roi))
            for roi in rois.rois
        )
    regions = _pack_regions(tuple(sources))
    packed_width = max(region.destination.x2 for region in regions)
    packed_height = max(region.destination.y2 for region in regions)
    # YUV420 requires even frame dimensions; no ROI pixels are trimmed or scaled.
    return RecordingLayout(
        width, height, packed_width + packed_width % 2,
        packed_height + packed_height % 2, regions,
    )


def frame_rate_encoder_type(base_encoder: type, camera_fps: float, recording_fps: float) -> type:
    """Select fractional-rate frames before copying pixels or invoking the codec.

    The integer accumulator keeps the configured proportion of camera requests
    without cumulative rounding error. Kept frames retain their real sensor
    timestamps; e.g. 60 Hz to 45 Hz keeps three of every four frames.
    """
    ratio = Fraction(str(recording_fps)) / Fraction(str(camera_fps))
    if not 0 < ratio <= 1:
        raise ValueError("Recording frame rate must be positive and not exceed camera rate")
    step, period = ratio.numerator, ratio.denominator

    class FrameRateEncoder(base_encoder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._frame_rate_phase = period - step

        def _start(self):
            self._frame_rate_phase = period - step
            super()._start()

        def _encode(self, stream, request):
            self._frame_rate_phase += step
            if self._frame_rate_phase >= period:
                self._frame_rate_phase -= period
                super()._encode(stream, request)

    return FrameRateEncoder


def selected_encoder_type(base_encoder: type, mapped_array: Any) -> type:
    """Isolate libx264/mux work from Picamera2's camera-request event loop."""

    class SelectedEncoder(base_encoder):
        def __init__(self, *args, recording_layout: RecordingLayout, **kwargs):
            super().__init__(*args, **kwargs)
            self.recording_layout = recording_layout
            self.recording_error: Exception | None = None
            self._pending: queue.Queue[tuple[GrayImage, int] | None] = queue.Queue(maxsize=8)
            self._worker: threading.Thread | None = None

        def _start(self):
            # Picamera2 sets these to the source stream dimensions just before
            # start. The codec and output must instead describe the packed image.
            self.width = self.recording_layout.width
            self.height = self.recording_layout.height
            super()._start()
            self._worker = threading.Thread(
                target=self._encode_worker, name="eye-video-encode", daemon=True,
            )
            self._worker.start()

        def _encode(self, stream, request):
            if self.recording_error is not None:
                return
            try:
                timestamp = self._timestamp(request)
                layout = self.recording_layout
                with mapped_array(request, stream) as mapped:
                    pixels = layout.pack(mapped.array[:layout.source_height, :layout.source_width])
                # No retained camera request: neither encoding nor slow storage
                # can exhaust libcamera's buffers. Bound memory and fail visibly
                # on overload rather than silently dropping scientific video.
                self._pending.put_nowait((pixels, timestamp))
            except queue.Full:
                self.recording_error = RuntimeError(
                    "Selected video encoder cannot keep up; recording queue is full. "
                    "Reduce recording.fps to 30 or reduce recording source dimensions."
                )
            except Exception as exc:  # noqa: BLE001 - report through camera health
                self.recording_error = exc

        def _encode_worker(self):
            import av

            while True:
                item = self._pending.get()
                if item is None:
                    return
                pixels, timestamp = item
                try:
                    # PyAV's zero-copy YUV420 constructor rounds height down to
                    # a multiple of four. For e.g. 540x2006 montages that makes
                    # libav rescale 2004 rows into 2006, moving pupil geometry
                    # and mixing a preceding camera tile into the eye border.
                    frame = (
                        av.VideoFrame.from_ndarray(pixels, format="yuv420p")
                        if self.height % 4 else av.VideoFrame.from_numpy_buffer(
                            pixels, format="yuv420p", width=self.width,
                        )
                    )
                    if (frame.width, frame.height) != (self.width, self.height):
                        raise RuntimeError("Video frame geometry changed before encoding")
                    frame.pts = timestamp
                    frame.time_base = self._stream.codec_context.time_base
                    for packet in self._stream.encode(frame):
                        self._lasttimestamp = (time.monotonic_ns(), packet.pts)
                        self.outputframe(
                            bytes(packet), packet.is_keyframe, timestamp=packet.pts, packet=packet,
                        )
                except Exception as exc:  # noqa: BLE001 - drain queue and report failure
                    if self.recording_error is None:
                        self.recording_error = exc

        def _stop(self):
            if self._worker is not None:
                # Called after Picamera2 disables new encoder submissions. Drain
                # accepted frames before flushing codec and closing the output.
                self._pending.put(None)
                self._worker.join()
                self._worker = None
            try:
                super()._stop()
            except Exception as exc:  # noqa: BLE001 - allow Encoder.stop to close outputs
                if self.recording_error is None:
                    self.recording_error = exc
                try:
                    self._container.close()
                except Exception:  # noqa: BLE001, S110 - preserve the original codec error
                    pass

    return SelectedEncoder
