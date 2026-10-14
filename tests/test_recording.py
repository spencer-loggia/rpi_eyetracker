from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from macaque_tracker.camera import _crf_h264_encoder_type
from macaque_tracker.config import RecordingConfig, RoiLayout
from macaque_tracker.models import NormalizedRoi
from macaque_tracker.recording import (
    RecordingLayout,
    load_recording_layout,
    selected_encoder_type,
    selected_recording_layout,
)


def _layout(eyes=2):
    rois = RoiLayout(
        tuple(NormalizedRoi(i, f"eye {i}", .55 + .25 * i, .25, .075, .25)
              for i in range(eyes)),
        source_width=80, source_height=12,
    )
    return selected_recording_layout(RecordingConfig(width=80, height=12), rois)


@pytest.mark.parametrize("eyes", [1, 2])
def test_selected_layout_preserves_two_left_views_and_exact_eye_pixels(eyes):
    layout = _layout(eyes)
    source = np.arange(80 * 12, dtype=np.uint16).reshape(12, 80).astype(np.uint8)
    packed = layout.pack(source)
    assert (layout.width, layout.height) == (46, 12)
    assert packed.flags.owndata and packed.flags.c_contiguous
    assert np.all(packed[layout.height:] == 128)
    assert [region.label for region in layout.regions] == [
        "camera_0", "camera_1", *[f"eye_{i}" for i in range(eyes)],
    ]
    assert np.array_equal(packed[:12, :40], source[:, :40])
    expected = np.zeros_like(source)
    for region in layout.regions:
        s, d = region.source, region.destination
        assert np.array_equal(packed[d.y:d.y2, d.x:d.x2], source[s.y:s.y2, s.x:s.x2])
        expected[s.y:s.y2, s.x:s.x2] = source[s.y:s.y2, s.x:s.x2]
    assert np.array_equal(layout.unpack(packed[:12]), expected)
    assert RecordingLayout.from_dict(layout.as_dict()) == layout
    source[:] = 0
    assert packed[:12].sum() > 0


def test_large_eyes_use_a_row_when_it_encodes_fewer_padding_pixels():
    config = RecordingConfig(width=80, height=12)
    rois = RoiLayout((NormalizedRoi(0, "eye", .5, .1, .45, .4),), 80, 12)
    layout = selected_recording_layout(config, rois)
    eye = layout.regions[-1]
    assert eye.destination.x == 0 and eye.destination.y == 12
    assert layout.width == 40 and layout.height == 18
    assert eye.source.width == eye.destination.width == 36


def test_recording_layout_loads_only_header_and_checks_region_bounds(tmp_path):
    path = tmp_path / "recording.mkv"
    assert load_recording_layout(path) is None
    layout = _layout()
    header = {"video_layout": layout.as_dict()}
    sidecar = path.with_suffix(".jsonl")
    sidecar.write_text(json.dumps(header) + "\nnot a header\n")
    assert load_recording_layout(path) == layout
    header["video_layout"]["regions"][0]["destination"]["x"] = 10000
    sidecar.write_text(json.dumps(header))
    with pytest.raises(ValueError, match="outside its frame"):
        load_recording_layout(path)


@contextmanager
def _mapped(request, _stream):
    yield SimpleNamespace(array=request.pixels)


def test_encoder_does_not_wait_for_compression_or_retain_camera_memory():
    pytest.importorskip("av")
    entered, release, submitted = threading.Event(), threading.Event(), threading.Event()
    encoded = []

    class Base:
        def __init__(self):
            self._lasttimestamp = None

        def _start(self):
            def encode(frame):
                entered.set()
                assert release.wait(2)
                encoded.append(frame.to_ndarray(format="yuv420p").copy())
                return []
            self._stream = SimpleNamespace(
                encode=encode, codec_context=SimpleNamespace(time_base=Fraction(1, 1_000_000)),
            )

        def _timestamp(self, request):
            return request.timestamp

        def _stop(self):
            pass

    layout = _layout()
    encoder = selected_encoder_type(Base, _mapped)(recording_layout=layout)
    encoder._start()
    source = np.full((12, 80), 80, dtype=np.uint8)
    request = SimpleNamespace(pixels=source, timestamp=0)
    producer = None
    try:
        encoder._encode("main", request)
        assert entered.wait(1)
        source[:] = 150

        def submit_backlog():
            for _ in range(9):
                encoder._encode("main", request)
            submitted.set()

        producer = threading.Thread(target=submit_backlog, daemon=True)
        producer.start()
        assert submitted.wait(1), "The camera callback blocked behind compression"
        assert encoder.recording_error is not None
        assert "queue is full" in str(encoder.recording_error)
        assert encoder._pending.qsize() == 8
    finally:
        release.set()
        if producer is not None:
            producer.join(1)
        encoder._stop()
    assert len(encoded) == 9  # Every accepted frame was drained before closing.
    assert np.all(encoded[0][:12, :40] == 80)
    assert np.all(encoded[1][:12, :40] == 150)
    assert encoder._worker is None


def test_selected_encoder_real_h264_round_trip_preserves_frames_and_timestamps(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "selected.mkv"

    class LibavBase:
        def __init__(self, *, bitrate, framerate, **kwargs):
            self.bitrate, self.framerate = bitrate, framerate

        def _start(self):
            self._container = av.open(str(path), "w")
            self._stream = self._container.add_stream("libx264", rate=self.framerate)
            self._stream.width, self._stream.height = self.width, self.height
            self._stream.pix_fmt = "yuv420p"
            self._stream.codec_context.time_base = Fraction(1, 1_000_000)
            self._stream.codec_context.options = {"tune": "zerolatency"}

        def _timestamp(self, request):
            return request.timestamp

        def outputframe(self, _data, _keyframe, *, timestamp, packet):
            self._container.mux(packet)

        def _stop(self):
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()

    layout = _layout()
    encoder_type = selected_encoder_type(_crf_h264_encoder_type(LibavBase), _mapped)
    encoder = encoder_type(
        recording_layout=layout, bitrate=None, framerate=60, crf=0,
        maximum_bitrate=50_000_000, preset="ultrafast",
    )
    # Picamera2 supplies the source dimensions before start; the output must be packed.
    encoder.width, encoder.height = 80, 12
    encoder._start()
    expected = []
    try:
        for index in range(3):
            source = np.full((12, 80), 40 + index * 30, dtype=np.uint8)
            source[:, 20:40] += 10
            expected.append(layout.pack(source))
            encoder._encode("main", SimpleNamespace(pixels=source, timestamp=index * 16_667))
            source[:] = 0
    finally:
        encoder._stop()
    assert encoder.recording_error is None
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == 3
    for index, frame in enumerate(frames):
        assert (frame.width, frame.height) == (layout.width, layout.height)
        assert np.array_equal(frame.to_ndarray(format="yuv420p"), expected[index])
        assert float(frame.pts * frame.time_base) == pytest.approx(index / 60, abs=.001)


@pytest.mark.parametrize("fail_during_flush", [False, True])
def test_worker_and_flush_failures_remain_visible_after_shutdown(fail_during_flush):
    pytest.importorskip("av")
    closed = []

    class Base:
        def _start(self):
            def encode(_frame):
                if not fail_during_flush:
                    raise OSError("codec worker failed")
                return []

            self._stream = SimpleNamespace(
                encode=encode, codec_context=SimpleNamespace(time_base=Fraction(1, 1_000_000)),
            )
            self._container = SimpleNamespace(close=lambda: closed.append(True))

        def _timestamp(self, _request):
            return 0

        def _stop(self):
            if fail_during_flush:
                raise OSError("codec flush failed")
            self._container.close()

    encoder = selected_encoder_type(Base, _mapped)(recording_layout=_layout())
    encoder._start()
    encoder._encode("main", SimpleNamespace(pixels=np.zeros((12, 80), dtype=np.uint8)))
    encoder._stop()
    assert closed == [True]
    assert encoder._worker is None
    assert isinstance(encoder.recording_error, OSError)
    assert str(encoder.recording_error) == (
        "codec flush failed" if fail_during_flush else "codec worker failed"
    )
