from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from macaque_tracker.config import AppConfig, RecordingConfig, RoiLayout
from macaque_tracker.models import NormalizedRoi
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


def test_sidecar_header_failure_stops_encoder_and_removes_partial_files(
    tmp_path, monkeypatch
) -> None:
    config = replace(
        AppConfig(),
        recording=RecordingConfig(directory=str(tmp_path), record_on_tracking=True),
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
        recording=RecordingConfig(directory=str(tmp_path), record_on_tracking=True),
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
    assert header["video_scope"] == "full stitched main stream; all cameras; not ROI crops"
    assert header["video_size"] == [
        config.camera.video_width,
        config.camera.video_height,
    ]
