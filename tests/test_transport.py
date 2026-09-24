from __future__ import annotations

from macaque_tracker.protocol import (
    CommandCode,
    CommandPacket,
    StatusSnapshot,
    decode_status,
    encode_command,
)
from macaque_tracker.transport import ProtocolEngine, UartServer


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
