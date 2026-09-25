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
