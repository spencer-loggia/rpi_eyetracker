import os
import pty
import select
import signal
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from macaque_tracker import cli, preview, transport
from macaque_tracker.cli import _parser
from macaque_tracker.config import AppConfig, CameraConfig, RecordingConfig, RoiLayout
from macaque_tracker.models import NormalizedRoi
from macaque_tracker.protocol import (
    TRANSACTION_SIZE,
    CommandCode,
    CommandPacket,
    StatusFlag,
    decode_status,
    encode_command,
)


def test_preview_accepts_prerecorded_video_source() -> None:
    args = _parser().parse_args(["preview", "--video", "session.mkv"])

    assert args.video == Path("session.mkv")


def test_run_accepts_prerecorded_video_source() -> None:
    args = _parser().parse_args(["run", "--video", "session.mkv", "--start-tracking"])

    assert args.video == Path("session.mkv")
    assert args.start_tracking


def test_configure_accepts_video_but_not_video_and_image_together() -> None:
    parser = _parser()
    args = parser.parse_args(["configure", "--video", "session.mkv"])

    assert args.video == Path("session.mkv")
    assert args.image is None
    with pytest.raises(SystemExit):
        parser.parse_args(["configure", "--video", "session.mkv", "--image", "still.png"])


def test_configure_is_live_by_default_and_static_is_explicit() -> None:
    parser = _parser()

    assert not parser.parse_args(["configure"]).static
    assert parser.parse_args(["configure", "--static"]).static


@pytest.fixture
def video_preview(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "eyes.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 80))
    assert writer.isOpened()
    try:
        for offset in range(4):
            image = np.full((80, 160, 3), 180, dtype=np.uint8)
            cv2.circle(image, (78 + offset, 40), 9, (20, 20, 20), -1)
            writer.write(image)
    finally:
        writer.release()
    config = AppConfig(
        camera=CameraConfig(
            sensor_width=160, sensor_height=80, analysis_width=160, analysis_height=80
        ),
        recording=RecordingConfig(width=160, height=80, directory=str(tmp_path / "recordings")),
    )
    layout = RoiLayout(
        rois=(NormalizedRoi(0, "eye", 0.25, 0.25, 0.5, 0.5),),
        source_width=160,
        source_height=80,
    )
    state = SimpleNamespace(
        config=config, layout=layout, video=video, closed=False, stopped=False, signals={},
        preview_frames=[], preview_rois=None,
    )

    class FakePreview:
        error_message = None

        def __init__(self, _config, *, image_settings, rois):
            state.preview_rois = rois

        @property
        def closed(self):
            return state.closed

        def start(self):
            pass

        def publish(self, frame, result):
            state.preview_frames.append((frame, result))

        def stop(self):
            state.stopped = True

    monkeypatch.setattr(preview, "LivePreview", FakePreview)
    monkeypatch.setattr(cli, "_load", lambda _path: (state.config, layout))
    monkeypatch.setattr(
        cli.signal, "signal", lambda sig, handler: state.signals.update({sig: handler})
    )
    return state


@pytest.mark.parametrize("command", ["preview", "run", "run-commanded"])
def test_video_tracking_results_reach_uart(monkeypatch, video_preview, command) -> None:
    state = video_preview
    master, slave = pty.openpty()
    state.config = replace(
        state.config,
        transport=replace(
            state.config.transport, uart_device=os.ttyname(slave), uart_baud=115_200
        ),
    )
    ready = threading.Event()
    configure_uart = transport.configure_raw_uart

    def configure(*args, **kwargs):
        configure_uart(*args, **kwargs)
        ready.set()

    monkeypatch.setattr(transport, "configure_raw_uart", configure)
    errors = []
    results = []

    def transact(code, sequence):
        os.write(master, encode_command(CommandPacket(code, sequence)))
        packet = bytearray()
        deadline = time.monotonic() + 3.0
        while len(packet) < TRANSACTION_SIZE:
            remaining = deadline - time.monotonic()
            assert remaining > 0 and select.select([master], [], [], remaining)[0]
            packet.extend(os.read(master, TRANSACTION_SIZE - len(packet)))
        result = decode_status(bytes(packet))
        assert result.request_sequence == sequence
        assert not result.flags & StatusFlag.ERROR
        return result

    def downstream():
        try:
            assert ready.wait(3.0)
            if command == "run-commanded":
                deadline = time.monotonic() + 1.0
                while not state.preview_frames and time.monotonic() < deadline:
                    time.sleep(0.005)
                assert state.preview_frames and state.preview_frames[-1][1] is None
                idle = transact(CommandCode.POLL, 99)
                assert not idle.flags & StatusFlag.TRACKING
                assert idle.sensor_timestamp_ns == 0
                transact(CommandCode.START_TRACKING, 1)
            deadline = time.monotonic() + 3.0
            sequence = 2
            while time.monotonic() < deadline:
                result = transact(CommandCode.POLL, sequence)
                sequence += 1
                if result.frame_sequence and (
                    not results or result.frame_sequence > results[-1].frame_sequence
                ):
                    results.append(result)
                if len(results) >= 2 and results[-1].frame_sequence > 8:
                    break
                time.sleep(0.03)
            assert len(results) >= 2
            assert results[-1].frame_sequence > 8  # Results continue across file loops.
            if command in ("preview", "run-commanded"):
                stopped = transact(CommandCode.STOP_TRACKING, 1000)
                assert not stopped.flags & (StatusFlag.TRACKING | StatusFlag.RECORDING)
                assert stopped.sensor_timestamp_ns == 0
                last_sequence = state.preview_frames[-1][0].frame_sequence
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    frame, result = state.preview_frames[-1]
                    if frame.frame_sequence > last_sequence + 1 and result is None:
                        break
                    time.sleep(0.005)
                else:
                    raise AssertionError("Preview did not remain live after UART STOP")
                assert not state.stopped and not state.closed
                restarted = transact(CommandCode.START_TRACKING, 1001)
                assert restarted.flags & StatusFlag.TRACKING
        except BaseException as exc:  # noqa: BLE001 - surface worker assertions in test thread
            errors.append(exc)
        finally:
            if command == "preview":
                state.closed = True
            else:
                state.signals[signal.SIGINT]()

    thread = threading.Thread(target=downstream, daemon=True)
    thread.start()
    try:
        args = ["preview" if command == "preview" else "run", "--video", str(state.video)]
        if command == "run":
            args.append("--start-tracking")
        if command == "run-commanded":
            args.append("--preview")
        assert cli.main(args) == 0
    finally:
        thread.join(timeout=4.0)
        os.close(slave)
        os.close(master)
    assert not thread.is_alive()
    assert not errors
    assert results[1].sensor_timestamp_ns > results[0].sensor_timestamp_ns
    assert all(result.flags & StatusFlag.TRACKING for result in results)
    assert all(not result.flags & StatusFlag.RECORDING for result in results)
    assert all(result.configured_eye_count == 1 for result in results)
    assert not Path(state.config.recording.directory).exists()
    if command in ("preview", "run-commanded"):
        assert state.preview_rois == state.layout.rois
        assert state.preview_frames
        assert all(frame.overview is not None for frame, _result in state.preview_frames)
    if command == "preview":
        assert state.stopped


def test_video_preview_reports_uart_failure_and_stops_preview(video_preview, capsys) -> None:
    state = video_preview
    state.config = replace(
        state.config,
        transport=replace(
            state.config.transport, uart_device=str(state.video.parent / "missing-tty")
        ),
    )

    assert cli.main(["preview", "--video", str(state.video)]) == 1
    assert "Could not open UART" in capsys.readouterr().err
    assert state.stopped


def test_video_preview_can_disable_transport(video_preview) -> None:
    state = video_preview
    state.config = replace(
        state.config, transport=replace(state.config.transport, backend="none")
    )
    state.closed = True

    assert cli.main(["preview", "--video", str(state.video)]) == 0
    assert state.stopped
