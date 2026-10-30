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
from macaque_tracker.models import NormalizedRoi, PixelRoi
from macaque_tracker.recording import (
    RecordingLayout,
    VideoRegion,
    frame_rate_encoder_type,
    load_recording_layout,
    recording_layout,
    selected_encoder_type,
    selected_recording_layout,
)


def _layout(eyes=2, rotations=(0, 0, 0, 0)):
    rois = RoiLayout(
        tuple(NormalizedRoi(i, f"eye {i}", .55 + .25 * i, .25, .075, .25)
              for i in range(eyes)),
        source_width=80, source_height=12, camera_rotations=rotations,
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


@pytest.mark.parametrize("cameras", [(1, 3), (), (0, 1, 2, 3), (3, 1)])
def test_selected_layout_preserves_configured_camera_and_eye_pixels(cameras):
    config = RecordingConfig(width=80, height=12, pose_cameras=cameras)
    rois = RoiLayout(
        (NormalizedRoi(0, "eye 0", .55, .25, .075, .25),
         NormalizedRoi(1, "eye 1", .8, .25, .075, .25)),
        source_width=80, source_height=12,
    )
    layout = selected_recording_layout(config, rois)
    source = np.arange(80 * 12, dtype=np.uint16).reshape(12, 80).astype(np.uint8)
    packed = layout.pack(source)

    assert [region.label for region in layout.regions] == [
        *[f"camera_{index}" for index in cameras], "eye_0", "eye_1",
    ]
    occupied = np.zeros((layout.height, layout.width), dtype=bool)
    restored = np.zeros_like(source)
    for region in layout.regions:
        s, d = region.source, region.destination
        assert not occupied[d.y:d.y2, d.x:d.x2].any()
        occupied[d.y:d.y2, d.x:d.x2] = True
        assert np.array_equal(packed[d.y:d.y2, d.x:d.x2], source[s.y:s.y2, s.x:s.x2])
        restored[s.y:s.y2, s.x:s.x2] = source[s.y:s.y2, s.x:s.x2]
    assert np.array_equal(layout.unpack(packed[:layout.height]), restored)
    assert np.all(packed[:layout.height][~occupied] == 16)
    if not cameras:
        assert (layout.width, layout.height) == (6, 6)
    else:
        for index, region in zip(cameras, layout.regions):
            assert (region.source.x, region.source.width, region.source.height) == (
                index * 20, 20, 12,
            )


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


def test_resized_packed_eye_uses_original_roi_rounding_without_black_borders():
    """The real 85x120 recording crop must fill its 114x160 analysis box."""
    cv2 = pytest.importorskip("cv2")
    roi = NormalizedRoi(0, "eye", .903515625, .6847222222222222, .022265625, .2222222222222222)
    layout = RoiLayout((roi,), 5120, 720, camera_rotations=(90, 90, 90, 270))
    mapping = selected_recording_layout(RecordingConfig(width=3840, height=540), layout)
    raw = np.full((540, 3840), 180, np.uint8)
    recorded = roi.to_pixels(3840, 540)
    analysis = roi.to_pixels(5120, 720)
    assert (recorded.width, recorded.height) == (85, 120)
    assert (analysis.width, analysis.height) == (114, 160)
    eye = raw[recorded.y:recorded.y2, recorded.x:recorded.x2]
    eye[:] = np.arange(85, dtype=np.uint8)[None, :] + 80
    packed = mapping.pack(raw)[:mapping.height]
    restored = mapping.unpack(packed, size=(5120, 720), eye_rois=(roi,))
    crop = restored[analysis.y:analysis.y2, analysis.x:analysis.x2]
    assert np.array_equal(crop, cv2.resize(eye, (114, 160), interpolation=cv2.INTER_AREA))
    assert np.all(crop >= 80)
    # Missing camera content remains blank, without contaminating edge pixels.
    assert restored[analysis.y - 1, analysis.x] == 0


def test_full_frame_with_non_multiple_four_height_uses_checked_encoder_mapping():
    config = RecordingConfig(width=80, height=14, layout="full")
    mapping = recording_layout(config, None)
    raw = np.arange(80 * 14, dtype=np.uint16).reshape(14, 80).astype(np.uint8)
    assert mapping is not None
    assert (mapping.width, mapping.height) == (80, 14)
    assert np.array_equal(mapping.pack(raw)[:14], raw)
    assert np.array_equal(mapping.unpack(raw), raw)


@pytest.mark.parametrize("full_frame", [False, True])
@pytest.mark.parametrize("rotations", [(90, 180, 270, 90), (270, 90, 0, 180)])
def test_oriented_recordings_preserve_every_selected_pixel_and_inverse_mapping(
    full_frame, rotations,
):
    config = RecordingConfig(
        width=80, height=12, layout="full" if full_frame else "hands_eyes",
    )
    rois = RoiLayout(
        (NormalizedRoi(0, "eye 0", .55, .25, .075, .25),
         NormalizedRoi(1, "eye 1", .8, .25, .075, .25)),
        80, 12, camera_rotations=rotations,
    )
    layout = recording_layout(config, rois)
    assert layout is not None
    source = np.arange(80 * 12, dtype=np.uint16).reshape(12, 80).astype(np.uint8)
    packed = layout.pack(source)
    occupied = np.zeros((layout.height, layout.width), dtype=bool)
    expected_raw = np.zeros_like(source)
    for region in layout.regions:
        raw, target = region.source, region.destination
        crop = source[raw.y:raw.y2, raw.x:raw.x2]
        assert np.array_equal(
            packed[target.y:target.y2, target.x:target.x2],
            np.rot90(crop, -(region.rotation // 90)),
        )
        assert not occupied[target.y:target.y2, target.x:target.x2].any()
        occupied[target.y:target.y2, target.x:target.x2] = True
        expected_raw[raw.y:raw.y2, raw.x:raw.x2] = crop
    if full_frame:
        assert [region.label for region in layout.regions] == [f"camera_{i}" for i in range(4)]
        assert np.array_equal(expected_raw, source)
    assert np.array_equal(layout.unpack(packed[:layout.height]), expected_raw)
    assert np.all(packed[:layout.height][~occupied] == 16)
    assert RecordingLayout.from_dict(layout.as_dict()) == layout
    # Planning must use no more area than laying all the rotated regions in a row.
    row_width = sum(region.destination.width for region in layout.regions)
    row_height = max(region.destination.height for region in layout.regions)
    assert layout.width * layout.height <= (row_width + row_width % 2) * (
        row_height + row_height % 2
    )


def test_unrotated_full_recording_preserves_direct_encoder_fast_path():
    config = RecordingConfig(layout="full")
    rois = RoiLayout((NormalizedRoi(0, "eye", .55, .25, .075, .25),), 80, 12)
    assert recording_layout(config, rois) is None
    assert recording_layout(config, None) is None


def test_legacy_recording_metadata_defaults_to_zero_rotation():
    layout = _layout()
    payload = layout.as_dict()
    for region in payload["regions"]:
        del region["rotation"]
    assert RecordingLayout.from_dict(payload) == layout


@pytest.mark.parametrize("rotation", [True, 45, -90, 360, "90"])
def test_recording_metadata_rejects_invalid_rotation(rotation):
    with pytest.raises(ValueError, match="rotation"):
        VideoRegion("camera_0", PixelRoi(0, 0, 4, 2), PixelRoi(0, 0, 4, 2), rotation)


@pytest.mark.parametrize("camera_fps,video_fps,expected", [(60, 45, 162_000), (60, 40, 144_000)])
def test_fractional_encoder_selects_exact_rate_for_an_hour_without_drift(
    camera_fps, video_fps, expected,
):
    selected = []

    class Base:
        def _start(self):
            pass

        def _encode(self, stream, request):
            assert stream == "main"
            selected.append(request)

    encoder = frame_rate_encoder_type(Base, camera_fps, video_fps)()
    encoder._start()
    # Requests stand in for capture indices. The wrapper must pass the original
    # object through, so sensor timestamps cannot be rewritten or interpolated.
    for frame in range(camera_fps * 3600):
        encoder._encode("main", frame)
    assert len(selected) == expected
    assert selected[0] == 0
    assert set(np.diff(selected)) == {1, 2}
    assert np.array_equal(
        np.bincount(np.asarray(selected) // camera_fps), np.full(3600, video_fps),
    )
    encoder._start()
    encoder._encode("main", 0)
    assert selected[-1] == 0


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


@pytest.mark.parametrize("recording_fps,indices", [(60, [0, 1, 2, 3]), (45, [0, 2, 3])])
@pytest.mark.parametrize("rotations", [(0, 0, 0, 0), (90, 180, 270, 90)])
@pytest.mark.parametrize("camera_size", [(80, 12), (80, 14), (3840, 540)])
def test_selected_encoder_real_h264_round_trip_preserves_frames_and_timestamps(
    tmp_path, recording_fps, indices, rotations, camera_size,
):
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

    width, height = camera_size
    eye = (NormalizedRoi(0, "eye", .903515625, .6847222222222222, .022265625, .2222222222222222)
           if width == 3840 else NormalizedRoi(0, "eye", .6, .25, .1, .5))
    if width == 3840:
        rotations = (90, 90, 90, 270)
    layout = selected_recording_layout(
        RecordingConfig(width=width, height=height),
        RoiLayout((eye,), width, height, camera_rotations=rotations),
    )
    if width == 3840:
        assert (layout.width, layout.height) == (540, 2006)
    mapped_frames = []

    @contextmanager
    def counted_mapping(request, stream):
        mapped_frames.append(request.timestamp)
        with _mapped(request, stream) as mapped:
            yield mapped

    encoder_type = selected_encoder_type(_crf_h264_encoder_type(LibavBase), counted_mapping)
    if recording_fps != 60:
        encoder_type = frame_rate_encoder_type(encoder_type, 60, recording_fps)
    encoder = encoder_type(
        recording_layout=layout, bitrate=None, framerate=recording_fps, crf=0,
        maximum_bitrate=50_000_000, preset="ultrafast",
    )
    # Picamera2 supplies the source dimensions before start; the output must be packed.
    encoder.width, encoder.height = width, height
    encoder._start()
    expected = []
    try:
        for index in range(4):
            source = ((np.arange(width * height).reshape(height, width) * 7 + index * 30)
                      % 256).astype(np.uint8)
            if index in indices:
                expected.append(layout.pack(source))
            encoder._encode("main", SimpleNamespace(pixels=source, timestamp=index * 16_667))
            source[:] = 0
    finally:
        encoder._stop()
    assert encoder.recording_error is None
    assert mapped_frames == [index * 16_667 for index in indices]
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == len(indices)
    for position, frame in enumerate(frames):
        assert (frame.width, frame.height) == (layout.width, layout.height)
        assert np.array_equal(frame.to_ndarray(format="yuv420p"), expected[position])
        assert float(frame.pts * frame.time_base) == pytest.approx(indices[position] / 60, abs=.001)


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
