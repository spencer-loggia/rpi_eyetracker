from __future__ import annotations

import enum
import math
import struct
import time
from dataclasses import dataclass

from .models import EyeMeasurement, FrameResult

MAGIC = b"ET"
PROTOCOL_VERSION = 1
TRANSACTION_SIZE = 64
_HEADER = struct.Struct("<2sBBHHIIQ")
_EYE = struct.Struct("<iiHBB")
_TAIL = struct.Struct("<IHHHBB")
_CRC = struct.Struct("<I")
assert _HEADER.size + 2 * _EYE.size + _TAIL.size + _CRC.size == TRANSACTION_SIZE


class ProtocolError(ValueError):
    pass


class PacketType(enum.IntEnum):
    COMMAND = 1
    STATUS = 2


class CommandCode(enum.IntEnum):
    POLL = 0
    START_TRACKING = 1
    STOP_TRACKING = 2
    START_RECORDING = 3
    STOP_RECORDING = 4
    CLEAR_ERROR = 5


class StatusCode(enum.IntEnum):
    OK = 0
    BAD_MAGIC = 1
    BAD_VERSION = 2
    BAD_PACKET_TYPE = 3
    BAD_CRC = 4
    UNKNOWN_COMMAND = 5
    INVALID_STATE = 6
    INTERNAL_ERROR = 7
    SEQUENCE_CONFLICT = 8


class StatusFlag(enum.IntFlag):
    TRACKING = 1 << 0
    RECORDING = 1 << 1
    CONFIGURED = 1 << 2
    ERROR = 1 << 3
    RESULT_STALE = 1 << 4


class EyeFlag(enum.IntFlag):
    VALID = 1 << 0
    BLINK = 1 << 1
    LOST = 1 << 2


@dataclass(frozen=True)
class CommandPacket:
    code: CommandCode
    sequence: int
    controller_timestamp_ns: int = 0
    flags: int = 0


@dataclass(frozen=True)
class StatusSnapshot:
    tracking: bool
    recording: bool
    configured: bool
    result: FrameResult | None
    session_id: int = 0
    configured_eye_count: int = 0
    request_sequence: int = 0
    status: StatusCode = StatusCode.OK
    error: bool = False
    stale: bool = False


@dataclass(frozen=True)
class DecodedStatus:
    status: StatusCode
    flags: StatusFlag
    frame_sequence: int
    sensor_timestamp_ns: int
    request_sequence: int
    session_id: int
    configured_eye_count: int
    eyes: tuple[EyeMeasurement, ...]
    dropped_analysis_frames: int
    processing_time_us: int
    result_age_ms: int


def crc32c(data: bytes, initial: int = 0xFFFFFFFF) -> int:
    """CRC-32C/Castagnoli with the conventional initial/final XOR."""
    crc = initial & 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


def _checked_u32(value: int, name: str) -> int:
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must fit in uint32")
    return value


def encode_command(command: CommandPacket) -> bytes:
    sequence = _checked_u32(command.sequence, "command sequence")
    if sequence == 0:
        raise ValueError("command sequence must be nonzero")
    if not 0 <= command.flags <= 0xFFFF:
        raise ValueError("command flags must fit in uint16")
    if not 0 <= command.controller_timestamp_ns <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("controller_timestamp_ns must fit in uint64")
    packet = bytearray(TRANSACTION_SIZE)
    _HEADER.pack_into(
        packet,
        0,
        MAGIC,
        PROTOCOL_VERSION,
        PacketType.COMMAND,
        command.flags,
        int(command.code),
        sequence,
        0,
        command.controller_timestamp_ns,
    )
    _CRC.pack_into(packet, TRANSACTION_SIZE - _CRC.size, crc32c(packet[:-4]))
    return bytes(packet)


def decode_command(packet: bytes) -> CommandPacket:
    if len(packet) != TRANSACTION_SIZE:
        raise ProtocolError(
            f"Expected a {TRANSACTION_SIZE}-byte transaction, received {len(packet)}"
        )
    expected_crc = _CRC.unpack_from(packet, TRANSACTION_SIZE - 4)[0]
    if crc32c(packet[:-4]) != expected_crc:
        raise ProtocolError("BAD_CRC")
    magic, version, packet_type, flags, code, sequence, related, timestamp_ns = (
        _HEADER.unpack_from(packet)
    )
    if magic != MAGIC:
        raise ProtocolError("BAD_MAGIC")
    if version != PROTOCOL_VERSION:
        raise ProtocolError("BAD_VERSION")
    if packet_type != PacketType.COMMAND:
        raise ProtocolError("BAD_PACKET_TYPE")
    if sequence == 0:
        raise ProtocolError("BAD_SEQUENCE")
    if related != 0 or any(packet[_HEADER.size : TRANSACTION_SIZE - _CRC.size]):
        raise ProtocolError("BAD_RESERVED")
    try:
        command_code = CommandCode(code)
    except ValueError as exc:
        raise ProtocolError("UNKNOWN_COMMAND") from exc
    return CommandPacket(
        code=command_code,
        sequence=sequence,
        controller_timestamp_ns=timestamp_ns,
        flags=flags,
    )


def _empty_eye(eye_id: int) -> EyeMeasurement:
    return EyeMeasurement(
        eye_id=eye_id,
        x=0.0,
        y=0.0,
        pupil_diameter=0.0,
        confidence=0.0,
        valid=False,
        blink=False,
    )


def encode_status(snapshot: StatusSnapshot, *, now_ns: int | None = None) -> bytes:
    request_sequence = _checked_u32(snapshot.request_sequence, "request sequence")
    flags = StatusFlag(0)
    if snapshot.tracking:
        flags |= StatusFlag.TRACKING
    if snapshot.recording:
        flags |= StatusFlag.RECORDING
    if snapshot.configured:
        flags |= StatusFlag.CONFIGURED
    if snapshot.error or snapshot.status is not StatusCode.OK:
        flags |= StatusFlag.ERROR
    if snapshot.stale:
        flags |= StatusFlag.RESULT_STALE

    result = snapshot.result
    eyes_by_id = {} if result is None else {eye.eye_id: eye for eye in result.eyes}
    eyes = tuple(eyes_by_id.get(eye_id, _empty_eye(eye_id)) for eye_id in (0, 1))
    frame_sequence = 0 if result is None else result.frame_sequence & 0xFFFFFFFF
    timestamp_ns = 0 if result is None else result.sensor_timestamp_ns
    dropped = 0 if result is None else min(result.dropped_analysis_frames, 0xFFFF)
    processing_us = 0 if result is None else min(result.processing_time_us, 0xFFFF)
    if result is None:
        age_ms = 0xFFFF
    else:
        current = time.monotonic_ns() if now_ns is None else now_ns
        age_ms = min(max(0, (current - result.produced_timestamp_ns) // 1_000_000), 0xFFFF)

    packet = bytearray(TRANSACTION_SIZE)
    _HEADER.pack_into(
        packet,
        0,
        MAGIC,
        PROTOCOL_VERSION,
        PacketType.STATUS,
        int(flags),
        int(snapshot.status),
        frame_sequence,
        request_sequence,
        timestamp_ns,
    )
    for index, eye in enumerate(eyes):
        eye_flags = EyeFlag(0)
        if eye.valid:
            eye_flags |= EyeFlag.VALID
        elif eye.blink:
            eye_flags |= EyeFlag.BLINK
        else:
            eye_flags |= EyeFlag.LOST
        _EYE.pack_into(
            packet,
            _HEADER.size + index * _EYE.size,
            _encode_q16_16(eye.x),
            _encode_q16_16(eye.y),
            _encode_q12_4(eye.pupil_diameter),
            round(min(max(eye.confidence, 0.0), 1.0) * 255.0),
            int(eye_flags),
        )
    configured_eye_count = snapshot.configured_eye_count
    if configured_eye_count == 0 and result is not None:
        configured_eye_count = len(result.eyes)
    if not 0 <= configured_eye_count <= 2:
        raise ValueError("configured_eye_count must be 0, 1, or 2")
    _TAIL.pack_into(
        packet,
        _HEADER.size + 2 * _EYE.size,
        snapshot.session_id & 0xFFFFFFFF,
        dropped,
        processing_us,
        age_ms,
        configured_eye_count,
        0,
    )
    _CRC.pack_into(packet, TRANSACTION_SIZE - 4, crc32c(packet[:-4]))
    return bytes(packet)


def decode_status(packet: bytes) -> DecodedStatus:
    if len(packet) != TRANSACTION_SIZE:
        raise ProtocolError(
            f"Expected a {TRANSACTION_SIZE}-byte transaction, received {len(packet)}"
        )
    expected_crc = _CRC.unpack_from(packet, TRANSACTION_SIZE - 4)[0]
    if crc32c(packet[:-4]) != expected_crc:
        raise ProtocolError("BAD_CRC")
    magic, version, packet_type, raw_flags, code, sequence, related, timestamp_ns = (
        _HEADER.unpack_from(packet)
    )
    if magic != MAGIC:
        raise ProtocolError("BAD_MAGIC")
    if version != PROTOCOL_VERSION:
        raise ProtocolError("BAD_VERSION")
    if packet_type != PacketType.STATUS:
        raise ProtocolError("BAD_PACKET_TYPE")
    try:
        status = StatusCode(code)
    except ValueError as exc:
        raise ProtocolError(f"Unknown status code: {code}") from exc
    known_status_flags = int(
        StatusFlag.TRACKING
        | StatusFlag.RECORDING
        | StatusFlag.CONFIGURED
        | StatusFlag.ERROR
        | StatusFlag.RESULT_STALE
    )
    if raw_flags & ~known_status_flags:
        raise ProtocolError("Status contains unknown flag bits")
    flags = StatusFlag(raw_flags)
    eyes: list[EyeMeasurement] = []
    for eye_id in (0, 1):
        x_raw, y_raw, diameter_raw, confidence_raw, eye_flags_raw = _EYE.unpack_from(
            packet, _HEADER.size + eye_id * _EYE.size
        )
        eye_flags = EyeFlag(eye_flags_raw)
        known_eye_flags = int(EyeFlag.VALID | EyeFlag.BLINK | EyeFlag.LOST)
        if eye_flags_raw & ~known_eye_flags:
            raise ProtocolError(f"Eye {eye_id} contains unknown flag bits")
        valid = bool(eye_flags & EyeFlag.VALID)
        blink = bool(eye_flags & EyeFlag.BLINK)
        lost = bool(eye_flags & EyeFlag.LOST)
        if sum((valid, blink, lost)) != 1:
            raise ProtocolError("Eye record must set exactly one state flag")
        if not valid and diameter_raw != 0:
            raise ProtocolError("Blink/lost eye record must have zero diameter")
        if valid and diameter_raw == 0:
            raise ProtocolError("Valid eye record must have a positive diameter")
        eyes.append(
            EyeMeasurement(
                eye_id=eye_id,
                x=x_raw / 65536.0,
                y=y_raw / 65536.0,
                pupil_diameter=diameter_raw / 16.0 if valid and not blink else 0.0,
                confidence=confidence_raw / 255.0,
                valid=valid,
                blink=blink,
            )
        )
    session_id, dropped, processing_us, age_ms, eye_count, _reserved = _TAIL.unpack_from(
        packet, _HEADER.size + 2 * _EYE.size
    )
    if eye_count > 2:
        raise ProtocolError("Configured eye count must be 0, 1, or 2")
    if _reserved != 0:
        raise ProtocolError("Status reserved byte must be zero")
    return DecodedStatus(
        status=status,
        flags=flags,
        frame_sequence=sequence,
        sensor_timestamp_ns=timestamp_ns,
        request_sequence=related,
        session_id=session_id,
        configured_eye_count=eye_count,
        eyes=tuple(eyes),
        dropped_analysis_frames=dropped,
        processing_time_us=processing_us,
        result_age_ms=age_ms,
    )


def _encode_q16_16(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("Coordinate must be finite")
    scaled = round(value * 65536.0)
    return min(max(scaled, -0x80000000), 0x7FFFFFFF)


def _encode_q12_4(value: float) -> int:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Pupil diameter must be finite and non-negative")
    return min(round(value * 16.0), 0xFFFF)


def protocol_error_status(error: ProtocolError) -> StatusCode:
    name = str(error)
    return {
        "BAD_MAGIC": StatusCode.BAD_MAGIC,
        "BAD_VERSION": StatusCode.BAD_VERSION,
        "BAD_PACKET_TYPE": StatusCode.BAD_PACKET_TYPE,
        "BAD_CRC": StatusCode.BAD_CRC,
        "UNKNOWN_COMMAND": StatusCode.UNKNOWN_COMMAND,
        "BAD_SEQUENCE": StatusCode.BAD_PACKET_TYPE,
        "BAD_RESERVED": StatusCode.BAD_PACKET_TYPE,
    }.get(name, StatusCode.BAD_PACKET_TYPE)
