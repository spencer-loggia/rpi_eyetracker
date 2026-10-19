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
    validate_subject_name,
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


@pytest.mark.parametrize("number", [1, 12, 0xFFFFFFFF])
@pytest.mark.parametrize("code", [CommandCode.START_EXPERIMENT, CommandCode.START_TRACKING,
                                  CommandCode.START_RECORDING])
def test_numbered_starts_use_uint32_header_without_changing_packet_size(code, number):
    fields = ({"subject": "A" * 32, "controller_timestamp_ns": 1,
               "experiment_number": number} if code is CommandCode.START_EXPERIMENT
              else {"block_number": number})
    command = CommandPacket(code, 5, **fields)
    packet = encode_command(command)
    assert len(packet) == 64
    assert struct.unpack_from("<I", packet, 12)[0] == number
    assert decode_command(packet) == command


@pytest.mark.parametrize("number", [0, -1, True, 1.5, 1 << 32])
def test_invalid_block_and_experiment_numbers_are_rejected(number):
    with pytest.raises(ValueError, match="number"):
        encode_command(CommandPacket(CommandCode.START_TRACKING, 1, block_number=number))
    with pytest.raises(ValueError, match="number"):
        encode_command(CommandPacket(CommandCode.START_EXPERIMENT, 1, 1,
                                     subject="A", experiment_number=number))


def test_numbers_are_reserved_for_start_commands():
    with pytest.raises(ValueError, match="Only START"):
        encode_command(CommandPacket(CommandCode.POLL, 1, block_number=3))
    with pytest.raises(ValueError, match="Only START"):
        encode_command(CommandPacket(CommandCode.START_TRACKING, 1, experiment_number=3))
    packet = bytearray(encode_command(CommandPacket(CommandCode.STOP_TRACKING, 1)))
    struct.pack_into("<I", packet, 12, 3)
    struct.pack_into("<I", packet, 60, crc32c(packet[:60]))
    with pytest.raises(ProtocolError, match="BAD_RESERVED"):
        decode_command(bytes(packet))


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


@pytest.mark.parametrize("subject", ["M", "animal_1.test-2", "A" * 32])
def test_start_experiment_payload_round_trip(subject: str) -> None:
    command = CommandPacket(
        CommandCode.START_EXPERIMENT,
        12,
        controller_timestamp_ns=1_792_000_000_123_456_789,
        subject=subject,
    )
    packet = encode_command(command)
    assert len(packet) == TRANSACTION_SIZE
    assert struct.unpack_from("<Q", packet, 16)[0] == command.controller_timestamp_ns
    assert packet[24] == len(subject)
    assert packet[25 : 25 + len(subject)] == subject.encode("ascii")
    assert packet[25 + len(subject) : 60] == bytes(35 - len(subject))
    assert decode_command(packet) == command


@pytest.mark.parametrize(
    "subject", [None, "", ".", "..", "../x", "a/b", "_a", "a b", "café", "A" * 33]
)
def test_invalid_subject_names_rejected(subject) -> None:
    with pytest.raises(ValueError, match="Subject"):
        validate_subject_name(subject)
    with pytest.raises(ValueError, match="Subject"):
        encode_command(CommandPacket(CommandCode.START_EXPERIMENT, 1, 1, subject=subject))


@pytest.mark.parametrize("timestamp", [0, -1, 1 << 64])
def test_start_experiment_requires_nonzero_uint64_timestamp(timestamp: int) -> None:
    with pytest.raises(ValueError):
        encode_command(CommandPacket(CommandCode.START_EXPERIMENT, 1, timestamp, subject="A"))


@pytest.mark.parametrize(
    ("offset", "value", "error"),
    [
        (24, 0, "BAD_SUBJECT"),
        (24, 33, "BAD_SUBJECT"),
        (25, ord("/"), "BAD_SUBJECT"),
        (25, 255, "BAD_SUBJECT"),
        (26, ord("B"), "BAD_RESERVED"),
        (57, 1, "BAD_RESERVED"),
        (59, 1, "BAD_RESERVED"),
        (16, 0, "BAD_TIMESTAMP"),
    ],
)
def test_start_experiment_rejects_malformed_payload(
    offset: int, value: int, error: str
) -> None:
    packet = bytearray(
        encode_command(CommandPacket(CommandCode.START_EXPERIMENT, 1, 1, subject="A"))
    )
    packet[offset] = value
    struct.pack_into("<I", packet, 60, crc32c(packet[:60]))
    with pytest.raises(ProtocolError, match=error):
        decode_command(bytes(packet))


@pytest.mark.parametrize(
    "code", [c for c in CommandCode if c is not CommandCode.START_EXPERIMENT]
)
def test_other_commands_keep_payload_reserved(code: CommandCode) -> None:
    packet = encode_command(CommandPacket(code, 9))
    assert packet[24:60] == bytes(36)
    assert decode_command(packet).subject is None
    with pytest.raises(ValueError, match="Only START_EXPERIMENT"):
        encode_command(CommandPacket(code, 9, subject="A"))
    malformed = bytearray(packet)
    malformed[24] = 1
    struct.pack_into("<I", malformed, 60, crc32c(malformed[:60]))
    with pytest.raises(ProtocolError, match="BAD_RESERVED"):
        decode_command(bytes(malformed))


def test_experiment_status_flags_preserve_frame_layout() -> None:
    packet = encode_status(
        StatusSnapshot(False, False, True, None, experiment_active=True, calibrating=True)
    )
    decoded = decode_status(packet)
    assert decoded.flags == (
        StatusFlag.CONFIGURED | StatusFlag.EXPERIMENT_ACTIVE | StatusFlag.CALIBRATING
    )
    assert decoded.sensor_timestamp_ns == 0
    assert not any(eye.valid for eye in decoded.eyes)
    assert len(packet) == TRANSACTION_SIZE


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
