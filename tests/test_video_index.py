from __future__ import annotations

import csv
from contextlib import contextmanager
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from macaque_tracker.camera import _crf_h264_encoder_type
from macaque_tracker.config import RecordingConfig, RoiLayout
from macaque_tracker.models import NormalizedRoi
from macaque_tracker.recording import (
    frame_rate_encoder_type,
    recording_layout,
    selected_encoder_type,
)
from macaque_tracker.video_index import (
    SensorTimingError,
    VideoFrameIndex,
    VideoFrameTimes,
    attach_index,
    indexed_encoder_type,
    request_camera_time_ns,
    video_index_path,
)


def _rows(path):
    with video_index_path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_pi_timing_reads_only_the_raw_sensor_control_without_metadata_conversion():
    key = object()
    stamp = 9_123_456_789_123
    request = SimpleNamespace(request=SimpleNamespace(metadata={key: stamp}),
                              get_metadata=lambda: pytest.fail("unnecessary metadata translation"))
    assert request_camera_time_ns(request, key) == stamp


@pytest.mark.parametrize("mode", ["full", "selected", "rotated"])
@pytest.mark.parametrize("fps", [30, 45, 60])
@pytest.mark.parametrize("suffix", [".mkv", ".mp4", ".h264"])
def test_real_encoded_frame_index_matches_decoded_video_exactly(tmp_path, mode, fps, suffix):
    """Exercise selection, async packing, real H.264/muxing, and final codec drain."""
    av = pytest.importorskip("av")
    path = tmp_path / f"recording{suffix}"
    config = RecordingConfig(width=80, height=16, layout="full" if mode == "full" else "hands_eyes")
    rois = RoiLayout((NormalizedRoi(0, "eye", .55, .25, .1, .5),), 80, 16,
                    camera_rotations=(90, 0, 270, 0) if mode == "rotated" else (0, 0, 0, 0))
    layout = recording_layout(config, rois)
    index = VideoFrameIndex(path)
    output = SimpleNamespace(_container=None)

    def stop_output():
        if output._container is not None:
            output._container.close()
            output._container = None

    output.stop = stop_output
    errors = []
    attach_index(output, index, errors.append)

    class LibavBase:
        def __init__(self, *, framerate, bitrate, **kwargs):
            self.framerate = framerate
            self.width, self.height = 80, 16

        def _start(self):
            self._container = av.open(str(path), "w")
            output._container = self._container
            self._stream = self._container.add_stream("libx264", rate=self.framerate)
            self._stream.width, self._stream.height = self.width, self.height
            self._stream.pix_fmt = "yuv420p"
            self._stream.codec_context.time_base = Fraction(1, 1_000_000)
            self._stream.codec_context.thread_count = 2
            self._stream.codec_context.thread_type = "FRAME"
            self._stream.codec_context.options = {"tune": "zerolatency"}

        def _encode(self, stream, request):
            frame = av.VideoFrame.from_numpy_buffer(request.pixels, format="yuv420p", width=self.width)
            frame.pts = self._timestamp(request)
            frame.time_base = self._stream.codec_context.time_base
            for packet in self._stream.encode(frame):
                self.outputframe(bytes(packet), packet.is_keyframe, timestamp=packet.pts, packet=packet)

        def outputframe(self, _data, _keyframe, *, timestamp, packet):
            self._container.mux(packet)
            output.outputtimestamp(timestamp)  # Same post-write hook as Picamera2 outputs.

        def _stop(self):
            for packet in self._stream.encode():
                self.outputframe(bytes(packet), packet.is_keyframe, timestamp=packet.pts, packet=packet)
            output.stop()

    @contextmanager
    def mapped(request, stream):
        yield SimpleNamespace(array=request.pixels)

    encoder_type = _crf_h264_encoder_type(LibavBase)
    options = {}
    if layout is not None:
        encoder_type = selected_encoder_type(encoder_type, mapped)
        options["recording_layout"] = layout
    if fps == 45:
        encoder_type = frame_rate_encoder_type(encoder_type, 60, fps)
    encoder_type = indexed_encoder_type(encoder_type, index)
    encoder = encoder_type(framerate=fps, bitrate=None, crf=0, maximum_bitrate=50_000_000,
                           preset="ultrafast", **options)
    selected = {60: [0, 1, 2, 3, 4, 5], 45: [0, 2, 3, 4], 30: [0, 2, 4]}[fps]
    # Deliberately non-microsecond-aligned sensor values; no interpolation/rebasing in CSV.
    stamps = [4_123_456_789_123 + i * 16_666_667 for i in range(6)]
    expected = []
    encoder._start()
    for i, stamp in enumerate(stamps):
        pixels = np.full((24, 80), 128, dtype=np.uint8)
        pixels[:16] = 25 + i * 30
        if i in selected:
            expected.append(layout.pack(pixels[:16]) if layout is not None else pixels.copy())
        if fps != 30 or i % 2 == 0:  # Picamera2's native pre-encode fixed stride.
            encoder._encode("main", SimpleNamespace(pixels=pixels,
                                                     get_metadata=lambda s=stamp: {"SensorTimestamp": s}))
    encoder._stop()
    index.close()
    assert encoder.recording_error is None
    assert not errors
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    rows = _rows(path)
    assert len(rows) == len(frames) == len(selected)
    assert [int(row["video_frame_index"]) for row in rows] == list(range(len(frames)))
    assert [int(row["camera_time_ns"]) for row in rows] == [stamps[i] for i in selected]
    for frame, pixels in zip(frames, expected):
        assert np.array_equal(frame.to_ndarray(format="yuv420p"), pixels)


def test_index_records_only_successful_output_and_rejects_missing_frames(tmp_path):
    path = tmp_path / "video.mkv"
    index = VideoFrameIndex(path)
    first = index.submit(1_000_000_123)
    second = index.submit(1_016_666_790)
    index.written(first)
    output, errors = SimpleNamespace(), []
    attach_index(output, index, errors.append)
    output.error_callback(OSError("mux disk full"))
    with pytest.raises(OSError, match="mux disk full"):
        output.outputtimestamp(second)
    with pytest.raises(OSError, match="mux disk full"):
        index.close()
    assert _rows(path) == [{"video_frame_index": "0", "camera_time_ns": "1000000123"}]
    assert errors


def test_missing_and_reordered_output_cannot_claim_a_complete_index(tmp_path):
    index = VideoFrameIndex(tmp_path / "missing.mkv")
    index.submit(1_000_000_123)
    later = index.submit(1_016_666_790)
    with pytest.raises(SensorTimingError, match="does not match"):
        index.written(later)
    with pytest.raises(SensorTimingError):
        index.close()
    missing = VideoFrameIndex(tmp_path / "unflushed.mkv")
    missing.submit(1_000_000_123)
    with pytest.raises(SensorTimingError, match="missing from its output"):
        missing.close()


@pytest.mark.parametrize("value", [None, 0, -1])
def test_missing_sensor_time_never_reaches_encoder(tmp_path, value):
    index = VideoFrameIndex(tmp_path / "missing.mkv")

    class Encoder:
        def _encode(self, stream, request):
            self._timestamp(request)
            pytest.fail("frame with missing timing must not be encoded")

    encoder = indexed_encoder_type(Encoder, index)()
    encoder._encode("main", SimpleNamespace(get_metadata=lambda: {"SensorTimestamp": value}))
    assert isinstance(encoder.recording_error, SensorTimingError)
    index.close()


@pytest.mark.parametrize("rows", ["0,1\n2,3\n", "0,2\n1,1\n", "0,0\n", "0,1\n"])
def test_bad_or_short_playback_index_fails_explicitly(tmp_path, rows):
    path = tmp_path / "video.mkv"
    video_index_path(path).write_text("video_frame_index,camera_time_ns\n" + rows)
    reader = VideoFrameTimes(path)
    try:
        with pytest.raises(SensorTimingError):
            reader.next()
            reader.next()
    finally:
        reader.close()


def test_frame_index_state_is_bounded_when_encoder_does_not_output(tmp_path):
    index = VideoFrameIndex(tmp_path / "stuck.mkv")
    for i in range(256):
        index.submit(1_000_000_123 + i * 1_000_000)
    with pytest.raises(SensorTimingError, match="not retiring"):
        index.submit(2_000_000_123)
    with pytest.raises(SensorTimingError):
        index.close()


def test_container_close_failure_is_not_silently_accepted_as_complete_video(tmp_path):
    class Output:
        def __init__(self):
            self._container = SimpleNamespace(close=self.fail_close)
            self.stopped = False

        def fail_close(self):
            raise OSError("container trailer failed")

        def stop(self):
            # Picamera2 PyavOutput.stop suppresses container-close errors.
            try:
                self._container.close()
            except OSError:
                pass
            self._container = None
            self.stopped = True

    index = VideoFrameIndex(tmp_path / "trailer.mkv")
    output, errors = Output(), []
    attach_index(output, index, errors.append)
    with pytest.raises(OSError, match="container trailer failed"):
        output.stop()
    assert output.stopped
    assert len(errors) == 1
    with pytest.raises(OSError, match="container trailer failed"):
        index.close()
