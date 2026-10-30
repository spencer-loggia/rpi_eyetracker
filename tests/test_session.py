from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from macaque_tracker.config import AppConfig, RecordingConfig, RoiLayout
from macaque_tracker.experiments import ExperimentStore
from macaque_tracker.models import EyeMeasurement, FrameResult, NormalizedRoi
from macaque_tracker.recording import RecordingLayout
from macaque_tracker.session import RecordingSession


class _SessionCamera:
    def __init__(self) -> None:
        self.recording = False
        self.stop_calls = 0

    def start_recording(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.touch()
        self.recording = True
        return path

    def stop_recording(self) -> None:
        self.stop_calls += 1
        self.recording = False


def _block_session(tmp_path):
    config = AppConfig(recording=RecordingConfig(directory=str(tmp_path)))
    config.save(tmp_path / "config.json")
    layout = RoiLayout((NormalizedRoi(0, "eye", .6, .3, .05, .25),), 5120, 720)
    layout.save(tmp_path / "rois.json")
    store = ExperimentStore(tmp_path / "data", tmp_path / "config.json")
    experiment = store.create("A", 1_792_000_000_000_000_000)
    block, saved = store.create_block(experiment, config, layout)
    return block, RecordingSession(_SessionCamera(), saved, layout, session_id=123)


@pytest.mark.parametrize("failure", ["start", "stop", "metadata", "none"])
def test_block_end_times_are_saved_after_writers_close_even_on_failure(tmp_path, monkeypatch, failure):
    started = 1_792_000_000_123_456_789
    ended = started + 2_345_678_901
    monkeypatch.setattr("macaque_tracker.experiments.time.time_ns", lambda: started)
    block, session = _block_session(tmp_path)
    path = block.directory / "block.json"
    manifest = json.loads(path.read_text())
    assert manifest["started_timestamp_ns"] == started
    assert manifest["started_utc"] == "2026-10-14T17:46:40+00:00"
    assert manifest["ended_timestamp_ns"] is manifest["ended_utc"] is None

    def fail(*args, **kwargs):
        raise OSError("recording test failure")

    monkeypatch.setattr("macaque_tracker.experiments.time.time_ns", lambda: ended)
    if failure == "start":
        monkeypatch.setattr(session.camera, "start_recording", fail)
        with pytest.raises(OSError, match="recording test failure"):
            session.start()
    else:
        session.start()
        if failure == "stop":
            monkeypatch.setattr(session.camera, "stop_recording", fail)
        elif failure == "metadata":
            monkeypatch.setattr(session, "_write", fail)
        if failure == "none":
            session.stop()
        else:
            with pytest.raises(OSError, match="recording test failure"):
                session.stop()
    manifest = json.loads(path.read_text())
    assert manifest["ended_timestamp_ns"] == ended
    assert manifest["ended_utc"] == "2026-10-14T17:46:42+00:00"
    assert manifest["status"] == ("closed" if failure == "none" else "error")
    assert session._stream is None
    monkeypatch.setattr("macaque_tracker.experiments.time.time_ns", lambda: ended + 10_000_000_000)
    session.stop()
    assert json.loads(path.read_text()) == manifest


def test_low_disk_video_stop_keeps_block_open_until_eye_trace_closes(tmp_path):
    block, session = _block_session(tmp_path)
    session.start()
    with pytest.raises(RuntimeError, match="already active"):
        session.start()
    assert json.loads((block.directory / "block.json").read_text())["ended_timestamp_ns"] is None
    session.stop_video(reason="low_disk_space")
    assert json.loads((block.directory / "block.json").read_text())["ended_timestamp_ns"] is None
    session.write_result(FrameResult(1, 1_234_567_890_123, 55,
                         (EyeMeasurement(0, 10, 20, 5, .9, True, False),)))
    session.stop()
    assert json.loads((block.directory / "block.json").read_text())["ended_timestamp_ns"] is not None
def test_sidecar_header_failure_stops_encoder_and_removes_partial_files(
    tmp_path, monkeypatch
) -> None:
    config = replace(
        AppConfig(),
        recording=RecordingConfig(directory=str(tmp_path), crf=27, record_on_tracking=True),
    )
    layout = RoiLayout(
        rois=(NormalizedRoi(0, "eye", 0.1, 0.1, 0.2, 0.2),),
        source_width=config.camera.analysis_width,
        source_height=config.camera.analysis_height,
    )
    camera = _SessionCamera()
    session = RecordingSession(camera, config, layout, session_id=123)

    def fail_header(_value) -> None:
        raise OSError("metadata disk failure")

    monkeypatch.setattr(session, "_write", fail_header)
    with pytest.raises(OSError, match="metadata disk failure"):
        session.start()

    assert not camera.recording
    assert camera.stop_calls == 1
    assert session.video_path is None
    assert session.metadata_path is None
    assert list(tmp_path.iterdir()) == []


def test_sidecar_declares_full_stitched_video_scope(tmp_path) -> None:
    config = replace(
        AppConfig(),
        recording=RecordingConfig(directory=str(tmp_path), crf=27, record_on_tracking=True),
    )
    layout = RoiLayout(
        rois=(NormalizedRoi(0, "eye", 0.1, 0.1, 0.2, 0.2),),
        source_width=config.camera.analysis_width,
        source_height=config.camera.analysis_height,
    )
    camera = _SessionCamera()
    session = RecordingSession(camera, config, layout, session_id=456)

    session.start()
    _video_path, metadata_path = session.stop()
    assert metadata_path is not None
    header = json.loads(metadata_path.read_text(encoding="utf-8").splitlines()[0])
    assert header["video_scope"] == (
        "full stitched recording stream; all cameras; not ROI crops"
    )
    assert header["video_size"] == [
        config.recording.width,
        config.recording.height,
    ]
    assert header["camera_fps"] == config.camera.fps
    assert header["video_fps"] == config.recording.fps
    assert header["encoder_frame_skip_count"] == 1
    assert header["pixel_model"] == "grayscale luma"
    assert header["encoded_pixel_format"] == "YUV420 with neutral chroma"
    assert header["video_codec"] == "H.264/libx264"
    assert header["encoder_preset"] == "ultrafast"
    assert header["encoder_crf"] == 27
    assert header["encoder_maximum_bitrate"] == config.recording.bitrate
    assert header["schema_version"] == 4
    assert header["video_layout"] is None
    assert header["camera_rotations"] == [0, 0, 0, 0]
    assert header["eye_coordinate_systems"] == [{
        "eye_id": 0,
        "rotation_degrees": 0,
        "analysis_crop_size": [512, 72],
        "recording_crop_size": [768, 108],
    }]


def test_selected_sidecar_maps_original_eye_coordinates_into_packed_video(tmp_path):
    config = AppConfig.load("config/eye_tracker.json")
    config = replace(config, recording=replace(config.recording, fps=45, directory=str(tmp_path)))
    layout = RoiLayout(
        (NormalizedRoi(0, "eye", .6, .3, .05, .25),), 5120, 720,
    )
    session = RecordingSession(_SessionCamera(), config, layout, session_id=123)
    session.start()
    _, metadata = session.stop()
    header = json.loads(metadata.read_text().splitlines()[0])
    assert header["schema_version"] == 4
    assert header["video_fps"] == 45
    assert header["encoder_frame_skip_count"] is None
    assert header["encoder_frame_selection"] == "phase_accumulator"
    packed = RecordingLayout.from_dict(header["video_layout"])
    assert header["video_size"] == [packed.width, packed.height]
    assert header["video_scope"].startswith("pose cameras 0, 1")
    assert header["pose_cameras"] == [0, 1]
    assert packed.regions[0].source.width == packed.regions[1].source.width == 960
    assert packed.regions[0].source.height == packed.regions[1].source.height == 540
    assert packed.regions[2].source == layout.rois[0].to_pixels(3840, 540)


@pytest.mark.parametrize("cameras", [(1, 3), ()])
def test_selected_sidecar_names_configured_pose_cameras(tmp_path, cameras):
    config = AppConfig(recording=RecordingConfig(
        directory=str(tmp_path), layout="hands_eyes", pose_cameras=cameras,
    ))
    layout = RoiLayout((NormalizedRoi(0, "eye", .6, .3, .05, .25),), 5120, 720)
    session = RecordingSession(_SessionCamera(), config, layout, session_id=123)
    session.start()
    _, metadata = session.stop()
    header = json.loads(metadata.read_text().splitlines()[0])

    assert header["pose_cameras"] == list(cameras)
    assert header["video_scope"] == (
        "pose cameras 1, 3 (indices from left to right) plus unscaled eye ROI crops"
        if cameras else "unscaled eye ROI crops only"
    )
    assert [region["label"] for region in header["video_layout"]["regions"]] == [
        *[f"camera_{index}" for index in cameras], "eye_0",
    ]


@pytest.mark.parametrize("video_layout", ["full", "hands_eyes"])
def test_oriented_sidecar_describes_crop_axes_and_saved_views(tmp_path, video_layout):
    """Metadata must disambiguate raw ROIs, oriented measurements, and encoded pixels."""
    config = AppConfig(recording=RecordingConfig(
        directory=str(tmp_path), layout=video_layout,
    ))
    layout = RoiLayout(
        (
            NormalizedRoi(0, "left", .55, .25, .10, .50),
            NormalizedRoi(1, "right", .80, .30, .10, .40),
        ),
        5120, 720, camera_rotations=(0, 180, 90, 270),
    )
    session = RecordingSession(_SessionCamera(), config, layout, session_id=123)
    session.start()
    result = FrameResult(1, 100, 110, (
        EyeMeasurement(0, 71.5, 32.25, 25, .95, True, False),
        EyeMeasurement(1, 31.25, 56.5, 26, .94, True, False),
    ))
    session.write_result(result)
    _, metadata = session.stop()
    header, frame, _end = [json.loads(line) for line in metadata.read_text().splitlines()]

    assert header["schema_version"] == 4
    assert frame["camera_time_ns"] == 100
    assert "sensor_timestamp_ns" not in frame
    assert "produced_timestamp_ns" not in frame
    assert "processed_monotonic_ns" not in frame
    assert "started_utc" not in header and "ended_utc" not in _end
    assert header["camera_rotations"] == [0, 180, 90, 270]
    assert header["coordinate_space"] == (
        "crop-local analysis-stream pixels after calibration rotation; "
        "origin top-left; +x right; +y down; not gaze degrees"
    )
    assert header["rotation_convention"].startswith("clockwise degrees")
    assert header["eye_coordinate_systems"] == [
        {
            "eye_id": 0, "rotation_degrees": 90,
            "analysis_crop_size": [180, 256], "recording_crop_size": [270, 384],
        },
        {
            "eye_id": 1, "rotation_degrees": 270,
            "analysis_crop_size": [144, 256], "recording_crop_size": [216, 384],
        },
    ]
    assert header["rois"] == [roi.as_dict() for roi in layout.rois]
    assert [(eye["x"], eye["y"]) for eye in frame["eyes"]] == [(71.5, 32.25), (31.25, 56.5)]
    packed = RecordingLayout.from_dict(header["video_layout"])
    assert header["video_size"] == [packed.width, packed.height]
    assert [region.rotation for region in packed.regions] == [0, 180, 90, 270]
    if video_layout == "full":
        assert header["video_scope"] == (
            "all four camera views with calibration rotations; not ROI crops"
        )
        assert [region.label for region in packed.regions] == [
            "camera_0", "camera_1", "camera_2", "camera_3",
        ]
        assert all(
            (region.source.width, region.source.height) == (960, 540)
            for region in packed.regions
        )
    else:
        assert header["video_scope"].startswith("pose cameras 0, 1")
        assert [region.label for region in packed.regions] == [
            "camera_0", "camera_1", "eye_0", "eye_1",
        ]
        assert [(region.destination.width, region.destination.height)
                for region in packed.regions[2:]] == [(270, 384), (216, 384)]


@pytest.mark.parametrize("shared", [False, True])
def test_recording_permissions_follow_explicit_setting_even_with_restrictive_umask(tmp_path, shared):
    import os

    config = AppConfig(recording=RecordingConfig(
        directory=str(tmp_path / "recordings"), shared_permissions=shared,
    ))
    layout = RoiLayout((NormalizedRoi(0, "eye", .6, .3, .05, .25),), 5120, 720)
    session = RecordingSession(_SessionCamera(), config, layout, session_id=123)
    previous = os.umask(0o077)
    try:
        video = session.start()
        assert video.stat().st_mode & 0o777 == (0o666 if shared else 0o600)
        assert session.metadata_path.stat().st_mode & 0o777 == (0o666 if shared else 0o600)
        assert video.parent.stat().st_mode & 0o777 == (0o777 if shared else 0o700)
        session.stop()
        assert video.stat().st_mode & 0o777 == (0o666 if shared else 0o600)
    finally:
        os.umask(previous)


def test_slow_metadata_writer_does_not_delay_live_results_and_stop_drains(tmp_path, monkeypatch):
    """Disk stalls stay on the writer thread; accepted tracking rows retain order."""
    _block, session = _block_session(tmp_path)
    session.start(record_video=False)
    entered, release, submitted = threading.Event(), threading.Event(), threading.Event()
    original = session._persist_result

    def slow_disk(result):
        entered.set()
        assert release.wait(2)
        original(result)

    monkeypatch.setattr(session, "_persist_result", slow_disk)
    first = FrameResult(1, 100, 10, (EyeMeasurement(0, 10, 20, 5, .9, True, False),))
    second = replace(first, frame_sequence=2, camera_time_ns=200)
    session.write_result(first)
    assert entered.wait(1)

    def submit():
        session.write_result(second)
        submitted.set()

    producer = threading.Thread(target=submit)
    producer.start()
    try:
        assert submitted.wait(.5), "Tracking waited for the blocked metadata disk write"
    finally:
        release.set()
        producer.join(2)
        _, metadata = session.stop()
    rows = [json.loads(line) for line in metadata.read_text().splitlines()]
    assert [row["frame_sequence"] for row in rows if row["type"] == "frame"] == [1, 2]
    assert rows[-1]["tracked_rows"] == 2
    assert session._writer is None and session._stream is None


def test_metadata_backlog_fails_visibly_without_discarding_accepted_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(RecordingSession, "_RESULT_QUEUE_SIZE", 2)
    _block, session = _block_session(tmp_path)
    session.start(record_video=False)
    entered, release = threading.Event(), threading.Event()
    original = session._persist_result

    def slow_disk(result):
        entered.set()
        assert release.wait(2)
        original(result)

    monkeypatch.setattr(session, "_persist_result", slow_disk)
    result = FrameResult(1, 100, 10, (EyeMeasurement(0, 10, 20, 5, .9, True, False),))
    try:
        session.write_result(result)
        assert entered.wait(1)
        session.write_result(replace(result, frame_sequence=2))
        session.write_result(replace(result, frame_sequence=3))
        with pytest.raises(RuntimeError, match="result queue is full"):
            session.write_result(replace(result, frame_sequence=4))
    finally:
        release.set()
        with pytest.raises(RuntimeError, match="result queue is full"):
            session.stop()
    rows = [json.loads(line) for line in session.metadata_path.read_text().splitlines()]
    assert [row["frame_sequence"] for row in rows if row["type"] == "frame"] == [1, 2, 3]
    assert "result queue is full" in rows[-1]["metadata_write_error"]
    assert rows[-1]["encoder_stop_error"] is None
    assert session._writer is None and session._stream is None


def test_background_metadata_failure_reaches_stop_and_block_manifest(tmp_path, monkeypatch):
    block, session = _block_session(tmp_path)
    session.start(record_video=False)

    def disk_failure(_result):
        raise OSError("metadata drive disconnected")

    monkeypatch.setattr(session, "_persist_result", disk_failure)
    session.write_result(FrameResult(1, 100, 10, (EyeMeasurement(0, 10, 20, 5, .9, True, False),)))
    with pytest.raises(OSError, match="metadata drive disconnected"):
        session.stop()
    with pytest.raises(RuntimeError, match="metadata drive disconnected"):
        session.write_result(FrameResult(2, 200, 10, (EyeMeasurement(0, 10, 20, 5, .9, True, False),)))
    assert json.loads((block.directory / "block.json").read_text())["status"] == "error"
    assert session._writer is None and session._stream is None


def test_metadata_stop_timeout_retains_writer_ownership_until_disk_returns(tmp_path, monkeypatch):
    block, session = _block_session(tmp_path)
    monkeypatch.setattr(session, "_WRITER_STOP_TIMEOUT_S", .02)
    entered, release = threading.Event(), threading.Event()
    original = session._write

    def stalled_disk(record):
        if record["type"] == "frame":
            entered.set()
            assert release.wait(2)
        original(record)

    monkeypatch.setattr(session, "_write", stalled_disk)
    session.start()
    session.write_result(FrameResult(1, 100, 10, (EyeMeasurement(0, 10, 20, 5, .9, True, False),)))
    writer = session._writer
    try:
        assert entered.wait(1)
        with pytest.raises(RuntimeError, match="metadata writer did not finish before stop"):
            session.stop()
        assert writer.is_alive() and session._stream is not None
        assert not session._stream.closed
        assert json.loads((block.directory / "block.json").read_text())["ended_timestamp_ns"] is None
    finally:
        release.set()
        writer.join(2)
    assert not writer.is_alive() and session._stream is None
    rows = [json.loads(line) for line in session.metadata_path.read_text().splitlines()]
    assert [row["frame_sequence"] for row in rows if row["type"] == "frame"] == [1]
    session.stop()
    assert json.loads((block.directory / "block.json").read_text())["status"] == "closed"
