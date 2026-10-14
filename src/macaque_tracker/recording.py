from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import RecordingConfig, RoiLayout
from .models import GrayImage, PixelRoi


@dataclass(frozen=True)
class VideoRegion:
    label: str
    source: PixelRoi
    destination: PixelRoi


@dataclass(frozen=True)
class RecordingLayout:
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
                or (source.width, source.height) != (target.width, target.height)
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
            yuv[target.y:target.y2, target.x:target.x2] = gray[
                source.y:source.y2, source.x:source.x2
            ]
        return yuv

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RecordingLayout:
        values = dict(value)
        values["regions"] = tuple(
            VideoRegion(item["label"], PixelRoi(**item["source"]), PixelRoi(**item["destination"]))
            for item in value["regions"]
        )
        return cls(**values)

    def unpack(self, gray: GrayImage) -> GrayImage:
        """Restore recorded regions for playback; omitted camera pixels are black."""
        if gray.shape != (self.height, self.width):
            raise ValueError("Packed video dimensions do not match its metadata")
        result = np.zeros((self.source_height, self.source_width), dtype=np.uint8)
        for region in self.regions:
            source, target = region.source, region.destination
            result[source.y:source.y2, source.x:source.x2] = gray[
                target.y:target.y2, target.x:target.x2
            ]
        return result


def load_recording_layout(video: Path) -> RecordingLayout | None:
    sidecar = video.with_suffix(".jsonl")
    if not sidecar.is_file():
        return None
    try:
        with sidecar.open(encoding="utf-8") as stream:
            header = json.loads(stream.readline(1024 * 1024))
        layout = header.get("video_layout")
        return None if layout is None else RecordingLayout.from_dict(layout)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid recording layout in {sidecar}: {exc}") from exc


def selected_recording_layout(config: RecordingConfig, rois: RoiLayout) -> RecordingLayout:
    """Keep the first two horizontal camera views and each unscaled eye crop."""
    width, height = config.width, config.height
    if width % 4:
        raise ValueError("Selected recording requires four equal-width camera views")
    camera_width = width // 4
    hands_width = camera_width * 2
    crops = tuple(roi.to_pixels(width, height) for roi in rois.rois)
    candidates = []
    # Try a column beside the hands and a row below them; use less padding.
    for beside, stacked in ((True, True), (True, False), (False, False)):
        x, y = (hands_width, 0) if beside else (0, height)
        regions = [
            VideoRegion(f"camera_{i}", PixelRoi(i * camera_width, 0, camera_width, height),
                        PixelRoi(i * camera_width, 0, camera_width, height))
            for i in (0, 1)
        ]
        for roi, crop in zip(rois.rois, crops, strict=True):
            target = PixelRoi(x, y, crop.width, crop.height)
            regions.append(VideoRegion(f"eye_{roi.eye_id}", crop, target))
            if stacked:
                y += crop.height
            else:
                x += crop.width
        packed_width = max(region.destination.x2 for region in regions)
        packed_height = max(region.destination.y2 for region in regions)
        # Pad to even dimensions without changing or trimming any ROI pixels.
        packed_width += packed_width % 2
        packed_height += packed_height % 2
        candidates.append(RecordingLayout(
            width, height, packed_width, packed_height, tuple(regions),
        ))
    return min(candidates, key=lambda layout: layout.width * layout.height)


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
                    frame = av.VideoFrame.from_numpy_buffer(
                        pixels, format="yuv420p", width=self.width,
                    )
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
