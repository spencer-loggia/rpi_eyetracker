from __future__ import annotations

import threading

import macaque_tracker.transport as transport_module
from macaque_tracker.protocol import (
    CommandCode,
    CommandPacket,
    StatusSnapshot,
    decode_status,
    encode_command,
)
from macaque_tracker.transport import ProtocolEngine, UartServer, UnixSocketServer


class StubService:
    def snapshot(self) -> StatusSnapshot:
        return StatusSnapshot(False, False, True, None)

    def execute(self, _command: CommandPacket) -> StatusSnapshot:
        return self.snapshot()


def test_protocol_engine_echoes_every_valid_request_sequence() -> None:
    packet = encode_command(CommandPacket(CommandCode.POLL, 0x10203040))

    response = ProtocolEngine(StubService()).handle(packet)

    assert decode_status(response).request_sequence == 0x10203040


def test_uart_parser_handles_fragmentation_and_leading_noise() -> None:
    first = encode_command(CommandPacket(CommandCode.POLL, 1))
    second = encode_command(CommandPacket(CommandCode.START_TRACKING, 9))
    pending = bytearray(b"noise" + first[:17])

    assert UartServer._next_commands(pending) == []
    pending.extend(first[17:] + second)

    assert UartServer._next_commands(pending) == [first, second]
    assert pending == b""


def test_uart_parser_preserves_split_magic_prefix() -> None:
    packet = encode_command(CommandPacket(CommandCode.POLL, 1))
    pending = bytearray(b"garbageE")
    assert UartServer._next_commands(pending) == []
    assert pending == b"E"

    pending.extend(packet[1:])
    assert UartServer._next_commands(pending) == [packet]


def test_unix_server_survives_client_reset_during_response(monkeypatch, tmp_path) -> None:
    stop = threading.Event()
    packet = encode_command(CommandPacket(CommandCode.POLL, 1))

    class ResettingConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def settimeout(_timeout):
            pass

        @staticmethod
        def recv(_length):
            return packet

        @staticmethod
        def sendall(_response):
            stop.set()
            raise BrokenPipeError("client reset")

    class FakeServerSocket:
        def bind(self, _path):
            pass

        def listen(self, _backlog):
            pass

        def settimeout(self, _timeout):
            pass

        def accept(self):
            return ResettingConnection(), None

        def close(self):
            pass

    monkeypatch.setattr(transport_module.socket, "socket", lambda *_args: FakeServerSocket())
    server = UnixSocketServer(tmp_path / "tracker.sock", ProtocolEngine(StubService()))

    server.serve(stop)

    assert stop.is_set()


def test_uart_parser_recovers_next_start_after_dropped_inserted_and_corrupt_bytes():
    packet = encode_command(CommandPacket(CommandCode.POLL, 1))
    start = encode_command(CommandPacket(CommandCode.START_TRACKING, 2))
    damaged = [packet[:12] + packet[13:], packet[:20] + b"ET!" + packet[20:],
               packet[:-1] + bytes([packet[-1] ^ 0x80]), b"ET" + bytes(10)]
    for bad in damaged:
        pending = bytearray()
        received = []
        for value in bad + start:
            pending.append(value)
            received.extend(UartServer._next_commands(pending))
            assert len(pending) < 64
        assert received == [start]
        assert pending == b""


def test_uart_write_is_bounded_when_reader_stops_consuming():
    import os
    import time

    import pytest

    read_fd, write_fd = os.pipe()
    try:
        os.set_blocking(write_fd, False)
        while True:
            try:
                os.write(write_fd, bytes(4096))
            except BlockingIOError:
                break
        started = time.monotonic()
        with pytest.raises(transport_module.TransportError, match="timed out"):
            UartServer._write_all(write_fd, bytes(64))
        assert time.monotonic() - started < 1
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_uart_listener_reopens_after_failed_ack_and_accepts_next_start(monkeypatch):
    import errno
    import os
    import pty
    import select

    master, slave = pty.openpty()
    stop = threading.Event()
    ready = threading.Event()
    opened = []
    failures = []
    configure = transport_module.configure_raw_uart

    def configured(*args, **kwargs):
        configure(*args, **kwargs)
        opened.append(True)
        ready.set()

    monkeypatch.setattr(transport_module, "configure_raw_uart", configured)
    server = UartServer(os.ttyname(slave), 115200, ProtocolEngine(StubService()))
    write_all = server._write_all
    writes = []

    def dropped_ack(fd, data):
        writes.append(data)
        if len(writes) == 1:
            ready.clear()
            raise OSError(errno.EIO, "temporary UART failure")
        write_all(fd, data)

    monkeypatch.setattr(server, "_write_all", dropped_ack)

    def serve():
        try:
            server.serve(stop)
        except Exception as exc:  # noqa: BLE001 - surface failures in the test thread
            failures.append(exc)

    worker = threading.Thread(target=serve)
    worker.start()
    try:
        assert ready.wait(2)
        ready.clear()
        os.write(master, encode_command(CommandPacket(CommandCode.POLL, 1)))
        assert ready.wait(2), "Listener failed to reopen the same UART"
        assert len(opened) == 2
        # A corrupt/truncated command immediately before START must not swallow it.
        os.write(master, b"ET" + bytes(20)
                 + encode_command(CommandPacket(CommandCode.START_TRACKING, 2)))
        assert select.select([master], [], [], 2)[0]
        response = bytearray()
        while len(response) < 64:
            assert select.select([master], [], [], 1)[0]
            response.extend(os.read(master, 64 - len(response)))
        assert decode_status(response).request_sequence == 2
        assert worker.is_alive()
    finally:
        stop.set()
        worker.join(2)
        os.close(master)
        os.close(slave)
    assert not worker.is_alive()
    assert failures == []


def test_uart_retries_termios_errors_that_are_not_oserrors(monkeypatch):
    import termios

    stop = threading.Event()
    server = UartServer("unused", 115200, ProtocolEngine(StubService()))
    attempts = []

    def connection(_stop):
        attempts.append(True)
        if len(attempts) == 1:
            raise termios.error(5, "temporary tty I/O error")
        stop.set()

    monkeypatch.setattr(server, "_serve_connection", connection)
    server.serve(stop)
    assert len(attempts) == 2
