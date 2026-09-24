from __future__ import annotations

import struct

import pytest

from macaque_tracker.models import EyeMeasurement, FrameResult
from macaque_tracker.protocol import (
    TRANSACTION_SIZE,
    CommandCode,
    CommandPacket,
    ProtocolError,
    StatusFlag,
    StatusSnapshot,
    crc32c,
    decode_command,
    decode_status,
    encode_command,
    encode_status,
)


def test_crc32c_standard_check() -> None:
    assert crc32c(b"123456789") == 0xE3069283


def test_command_round_trip_and_size() -> None:
    command = CommandPacket(
        code=CommandCode.START_TRACKING,
        sequence=0x11223344,
        controller_timestamp_ns=987654321,
        flags=3,
    )
    packet = encode_command(command)
    assert len(packet) == TRANSACTION_SIZE
    assert decode_command(packet) == command


def test_command_crc_rejects_corruption() -> None:
    packet = bytearray(encode_command(CommandPacket(CommandCode.POLL, 1)))
    packet[20] ^= 0x40
    with pytest.raises(ProtocolError, match="BAD_CRC"):
        decode_command(bytes(packet))


def test_command_rejects_nonzero_reserved_payload_with_valid_crc() -> None:
    packet = bytearray(encode_command(CommandPacket(CommandCode.POLL, 1)))
    packet[24] = 1
    struct.pack_into("<I", packet, 60, crc32c(packet[:60]))
    with pytest.raises(ProtocolError, match="BAD_RESERVED"):
        decode_command(bytes(packet))


def test_command_sequence_must_be_nonzero() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        encode_command(CommandPacket(CommandCode.POLL, 0))

    packet = bytearray(encode_command(CommandPacket(CommandCode.POLL, 1)))
    struct.pack_into("<I", packet, 8, 0)
    struct.pack_into("<I", packet, 60, crc32c(packet[:60]))
    with pytest.raises(ProtocolError, match="BAD_SEQUENCE"):
        decode_command(bytes(packet))


def test_status_round_trip_two_eyes_and_blink_zero_size() -> None:
    result = FrameResult(
        frame_sequence=42,
        sensor_timestamp_ns=123456789,
        produced_timestamp_ns=1000,
        eyes=(
            EyeMeasurement(0, 123.25, 45.5, 32.25, 0.98, True, False),
            EyeMeasurement(1, 88.0, 90.0, 0.0, 0.0, False, True),
        ),
        dropped_analysis_frames=7,
        processing_time_us=3210,
    )
    packet = encode_status(
        StatusSnapshot(
            tracking=True,
            recording=True,
            configured=True,
            result=result,
            session_id=0xAABBCCDD,
            configured_eye_count=2,
            request_sequence=9,
        ),
        now_ns=2_001_000,
    )
    assert len(packet) == TRANSACTION_SIZE
    decoded = decode_status(packet)
    assert decoded.flags & StatusFlag.TRACKING
    assert decoded.flags & StatusFlag.RECORDING
    assert decoded.frame_sequence == 42
    assert decoded.request_sequence == 9
    assert decoded.session_id == 0xAABBCCDD
    assert decoded.eyes[0].x == pytest.approx(123.25, abs=1 / 65536)
    assert decoded.eyes[0].pupil_diameter == pytest.approx(32.25, abs=1 / 16)
    assert decoded.eyes[1].blink
    assert decoded.eyes[1].pupil_diameter == 0.0
    assert decoded.dropped_analysis_frames == 7
    assert decoded.processing_time_us == 3210
    assert decoded.result_age_ms == 2


def test_crc_is_little_endian_tail() -> None:
    packet = encode_command(CommandPacket(CommandCode.STOP_TRACKING, 2))
    assert struct.unpack_from("<I", packet, 60)[0] == crc32c(packet[:60])


def test_status_rejects_invalid_eye_flags_and_reserved_byte() -> None:
    packet = bytearray(
        encode_status(
            StatusSnapshot(
                tracking=False,
                recording=False,
                configured=True,
                result=None,
                configured_eye_count=1,
            )
        )
    )
    packet[35] = 0
    struct.pack_into("<I", packet, 60, crc32c(packet[:60]))
    with pytest.raises(ProtocolError, match="exactly one"):
        decode_status(bytes(packet))

    packet = bytearray(encode_status(StatusSnapshot(False, False, True, None)))
    packet[59] = 1
    struct.pack_into("<I", packet, 60, crc32c(packet[:60]))
    with pytest.raises(ProtocolError, match="reserved"):
        decode_status(bytes(packet))
