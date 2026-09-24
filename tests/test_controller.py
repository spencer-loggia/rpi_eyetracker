from __future__ import annotations

import os
import pty
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from macaque_tracker.controller import (
    AcknowledgementTimeout,
    CommandRejected,
    EyeTrackerController,
    ResponseTimeout,
    SerialPacketChannel,
    UnixPacketChannel,
    _extract_status_packet,
    _read_status_packet,
    _recv_exact,
    main,
)
from macaque_tracker.models import EyeMeasurement, FrameResult
from macaque_tracker.protocol import (
    TRANSACTION_SIZE,
    CommandCode,
    ProtocolError,
    StatusCode,
    StatusSnapshot,
    decode_command,
    decode_status,
    encode_status,
)


def _status(
    *,
    request_sequence: int = 0,
    frame: int = 0,
    session: int = 1,
    code: StatusCode = StatusCode.OK,
) -> bytes:
    result = None
    if frame:
        result = FrameResult(
            frame_sequence=frame,
            sensor_timestamp_ns=frame * 1_000,
            produced_timestamp_ns=1,
            eyes=(EyeMeasurement(0, 4.0, 5.0, 6.0, 0.9, True, False),),
            dropped_analysis_frames=0,
            processing_time_us=100,
        )
    return encode_status(
        StatusSnapshot(
            tracking=frame > 0,
            recording=False,
            configured=True,
            result=result,
            session_id=session,
            configured_eye_count=1,
            request_sequence=request_sequence,
            status=code,
            error=code is not StatusCode.OK,
        ),
        now_ns=frame * 1_000 if frame else 0,
    )


class ScriptedChannel:
    def __init__(self, responder: Callable[[bytes, int], bytes]) -> None:
        self.responder = responder
        self.requests: list[bytes] = []
        self.closed = False

    def exchange(self, packet: bytes, timeout_s: float) -> bytes:
        assert timeout_s > 0.0
        self.requests.append(packet)
        return self.responder(packet, len(self.requests) - 1)

    def close(self) -> None:
        self.closed = True


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.now += duration


def test_state_command_uses_nonzero_sequence_and_returns_ack() -> None:
    def respond(packet: bytes, _call: int) -> bytes:
        command = decode_command(packet)
        return _status(request_sequence=command.sequence)

    channel = ScriptedChannel(respond)
    client = EyeTrackerController(channel, initial_sequence=41)
    status = client.send_state_command(CommandCode.START_TRACKING)

    command = decode_command(channel.requests[0])
    assert command.code is CommandCode.START_TRACKING
    assert command.sequence == 41
    assert command.controller_timestamp_ns > 0
    assert status.request_sequence == 41


def test_missing_ack_retries_identical_state_packet_without_polling() -> None:
    clock = FakeClock()

    def respond(_packet: bytes, call: int) -> bytes:
        return _status(request_sequence=73 if call >= 1 else 0)

    channel = ScriptedChannel(respond)
    client = EyeTrackerController(
        channel,
        initial_sequence=73,
        retries=1,
        retry_delay_s=0.01,
        clock=clock,
        sleep=clock.sleep,
    )
    status = client.send_state_command(CommandCode.START_RECORDING)

    assert status.request_sequence == 73
    assert [decode_command(packet).code for packet in channel.requests] == [
        CommandCode.START_RECORDING,
        CommandCode.START_RECORDING,
    ]
    assert channel.requests[0] == channel.requests[1]


def test_matching_error_ack_is_rejected_without_retry() -> None:
    channel = ScriptedChannel(
        lambda packet, _call: _status(
            request_sequence=decode_command(packet).sequence,
            code=StatusCode.INVALID_STATE,
        )
    )
    client = EyeTrackerController(channel, initial_sequence=9)

    with pytest.raises(CommandRejected) as raised:
        client.send_state_command(CommandCode.STOP_RECORDING)
    assert raised.value.sequence == 9
    assert raised.value.status is StatusCode.INVALID_STATE
    assert len(channel.requests) == 1


def test_missing_ack_times_out_after_configured_attempts() -> None:
    clock = FakeClock()
    channel = ScriptedChannel(lambda _packet, _call: _status(request_sequence=0))
    client = EyeTrackerController(
        channel,
        initial_sequence=11,
        retries=1,
        retry_delay_s=0.01,
        clock=clock,
        sleep=clock.sleep,
    )

    with pytest.raises(AcknowledgementTimeout) as raised:
        client.send_state_command(CommandCode.STOP_TRACKING)
    assert raised.value.sequence == 11
    state_packets = [
        packet
        for packet in channel.requests
        if decode_command(packet).code is CommandCode.STOP_TRACKING
    ]
    assert len(state_packets) == 2
    assert state_packets[0] == state_packets[1]


def test_poll_new_deduplicates_frames_but_not_new_sessions() -> None:
    responses = iter(
        (
            (8, 2),
            (8, 2),
            (8, 3),
        )
    )

    def respond(packet: bytes, _call: int) -> bytes:
        frame, session = next(responses)
        return _status(
            request_sequence=decode_command(packet).sequence,
            frame=frame,
            session=session,
        )

    channel = ScriptedChannel(respond)
    client = EyeTrackerController(channel, initial_sequence=1)

    assert client.poll_new() is not None
    assert client.poll_new() is None
    restarted = client.poll_new()
    assert restarted is not None
    assert restarted.session_id == 3
    assert all(decode_command(packet).code is CommandCode.POLL for packet in channel.requests)
    assert [decode_command(packet).sequence for packet in channel.requests] == [1, 2, 3]


def test_poll_new_ignores_status_before_first_sensor_result() -> None:
    channel = ScriptedChannel(
        lambda packet, _call: _status(
            request_sequence=decode_command(packet).sequence,
        )
    )
    client = EyeTrackerController(channel, initial_sequence=1)

    assert client.poll_new() is None


def test_uart_status_parser_recovers_from_noise_and_corrupt_record() -> None:
    corrupt = bytearray(_status(frame=3))
    corrupt[30] ^= 0x40
    expected = _status(frame=4)
    pending = bytearray(b"noiseE" + bytes(corrupt) + expected[:23])

    with pytest.raises(ProtocolError, match="corrupt UART"):
        _extract_status_packet(pending)
    pending.extend(expected[23:])

    assert _extract_status_packet(pending) == expected
    assert pending == b""


def test_monitor_recovers_after_transient_response_timeout() -> None:
    clock = FakeClock()

    def respond(packet: bytes, call: int) -> bytes:
        if call == 0:
            raise ResponseTimeout("temporary")
        return _status(
            request_sequence=decode_command(packet).sequence,
            frame=12,
        )

    errors: list[tuple[str, int]] = []
    client = EyeTrackerController(
        ScriptedChannel(respond),
        initial_sequence=1,
        clock=clock,
        sleep=clock.sleep,
    )
    samples = client.iter_new_samples(
        poll_rate_hz=100.0,
        on_transient_error=lambda error, count: errors.append((str(error), count)),
    )

    assert next(samples).frame_sequence == 12
    assert errors == [("temporary", 1)]


def test_partial_uart_status_uses_short_inter_byte_timeout() -> None:
    read_fd, write_fd = os.pipe()
    os.write(write_fd, _status(frame=7)[:-1])
    started = time.monotonic()
    try:
        with pytest.raises(ResponseTimeout):
            _read_status_packet(
                read_fd,
                bytearray(),
                time.monotonic() + 1.0,
                inter_byte_timeout_s=0.01,
            )
    finally:
        os.close(write_fd)
        os.close(read_fd)
    assert time.monotonic() - started < 0.2


def test_socket_reader_assembles_fragmented_status_exactly() -> None:
    controller_socket, tracker_socket = socket.socketpair()
    expected = _status(frame=17)

    def serve_once() -> None:
        with tracker_socket:
            tracker_socket.sendall(expected[:7])
            tracker_socket.sendall(expected[7:31])
            tracker_socket.sendall(expected[31:])

    thread = threading.Thread(target=serve_once)
    thread.start()
    try:
        with controller_socket:
            packet = _recv_exact(controller_socket, TRANSACTION_SIZE)
        assert packet == expected
        assert decode_status(packet).frame_sequence == 17
    finally:
        thread.join(timeout=1.0)


def test_unix_channel_drains_late_duplicate_before_new_poll() -> None:
    controller_socket, tracker_socket = socket.socketpair()
    # Build the channel around a real connected stream socket without binding a
    # filesystem AF_UNIX path, which restricted test sandboxes may prohibit.
    channel = UnixPacketChannel.__new__(UnixPacketChannel)
    channel.path = Path("socketpair")
    channel._socket = controller_socket
    channel._lock = threading.Lock()
    channel._closed = False
    errors: list[BaseException] = []

    def serve_once() -> None:
        try:
            with tracker_socket:
                # This is the response to an earlier timed-out request. It is
                # already queued when the next poll's matching response arrives.
                tracker_socket.sendall(_status(request_sequence=50, frame=6))
                request = _recv_exact(tracker_socket, TRANSACTION_SIZE)
                sequence = decode_command(request).sequence
                tracker_socket.sendall(_status(request_sequence=sequence, frame=7))
        except BaseException as exc:  # noqa: BLE001 - report worker failures in test thread
            errors.append(exc)

    thread = threading.Thread(target=serve_once)
    thread.start()
    try:
        controller = EyeTrackerController(channel, initial_sequence=51)
        status = controller.poll()
        assert status.request_sequence == 51
        assert status.frame_sequence == 7
    finally:
        channel.close()
        thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert errors == []


def test_uart_channel_drains_late_duplicate_before_new_poll() -> None:
    master_fd, original_slave_fd = pty.openpty()
    path = os.ttyname(original_slave_fd)
    os.close(original_slave_fd)
    channel = SerialPacketChannel(path, baud=115_200)
    errors: list[BaseException] = []

    def serve_once() -> None:
        try:
            os.write(master_fd, _status(request_sequence=70, frame=8))
            request = bytearray()
            while len(request) < TRANSACTION_SIZE:
                request.extend(os.read(master_fd, TRANSACTION_SIZE - len(request)))
            sequence = decode_command(bytes(request)).sequence
            os.write(master_fd, _status(request_sequence=sequence, frame=9))
        except BaseException as exc:  # noqa: BLE001 - report worker failures in test thread
            errors.append(exc)

    thread = threading.Thread(target=serve_once)
    thread.start()
    try:
        controller = EyeTrackerController(channel, initial_sequence=71)
        status = controller.poll()
        assert status.request_sequence == 71
        assert status.frame_sequence == 9
    finally:
        channel.close()
        os.close(master_fd)
        thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert errors == []


def test_cli_rejects_poll_rate_above_uart_wire_capacity(capsys) -> None:
    assert main(["--baud", "115200", "monitor", "--rate", "100"]) == 1
    assert "cannot fit at 115200 baud" in capsys.readouterr().err
