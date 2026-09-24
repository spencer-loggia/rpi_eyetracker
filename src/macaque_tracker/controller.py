from __future__ import annotations

import argparse
import json
import os
import secrets
import select
import socket
import sys
import termios
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from .protocol import (
    TRANSACTION_SIZE,
    CommandCode,
    CommandPacket,
    DecodedStatus,
    ProtocolError,
    StatusCode,
    decode_command,
    decode_status,
    encode_command,
)
from .uart import configure_raw_uart, open_exclusive_uart

DEFAULT_BAUD = 460_800
DEFAULT_RESPONSE_TIMEOUT_S = 0.25
DEFAULT_STATE_TIMEOUT_S = 10.0
DEFAULT_RETRY_DELAY_S = 0.02
DEFAULT_INTER_BYTE_TIMEOUT_S = 0.10
DEFAULT_POLL_RATE_HZ = 100.0


class ControllerError(RuntimeError):
    """Base class for controller-side transport and command errors."""


class ResponseTimeout(ControllerError):
    """The eye-tracker did not return one complete status packet in time."""


class AcknowledgementTimeout(ControllerError):
    def __init__(self, sequence: int, attempts: int) -> None:
        super().__init__(
            f"No acknowledgement for command sequence {sequence} after {attempts} attempt(s)"
        )
        self.sequence = sequence
        self.attempts = attempts


class CommandRejected(ControllerError):
    def __init__(self, sequence: int, status: StatusCode) -> None:
        super().__init__(f"Command sequence {sequence} was rejected: {status.name}")
        self.sequence = sequence
        self.status = status


class PacketChannel(Protocol):
    """Injectable request/response channel for one 64-byte transaction."""

    def exchange(self, packet: bytes, timeout_s: float) -> bytes: ...

    def close(self) -> None: ...


class UnixPacketChannel:
    """Controller connection to the development Unix-domain socket server."""

    def __init__(self, path: str | Path, *, connect_timeout_s: float = 2.0) -> None:
        self.path = Path(path)
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(connect_timeout_s)
        try:
            self._socket.connect(str(self.path))
        except Exception:
            self._socket.close()
            raise
        self._lock = threading.Lock()
        self._closed = False

    def exchange(self, packet: bytes, timeout_s: float) -> bytes:
        _validate_request(packet, timeout_s)
        expected_sequence = decode_command(packet).sequence
        with self._lock:
            if self._closed:
                raise ControllerError("Unix packet channel is closed")
            deadline = time.monotonic() + timeout_s
            try:
                self._socket.settimeout(timeout_s)
                self._socket.sendall(packet)
                while True:
                    response = _recv_exact(
                        self._socket,
                        TRANSACTION_SIZE,
                        deadline=deadline,
                    )
                    if _response_is_current_or_protocol_error(response, expected_sequence):
                        return response
            except TimeoutError as exc:
                raise ResponseTimeout("Timed out waiting for eye-tracker status") from exc
            except OSError as exc:
                raise ControllerError(f"Unix socket transaction failed: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._socket.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


class SerialPacketChannel:
    """Raw 8-N-1 UART channel implemented with the Python standard library."""

    def __init__(self, path: str | Path, *, baud: int = DEFAULT_BAUD) -> None:
        self.path = Path(path)
        self.baud = baud
        self._lock = threading.Lock()
        self._closed = False
        try:
            self._fd = open_exclusive_uart(self.path)
        except OSError as exc:
            raise ControllerError(f"Could not open UART {self.path}: {exc}") from exc
        try:
            self._original_attributes = termios.tcgetattr(self._fd)
            configure_raw_uart(self._fd, baud, read_timeout_deciseconds=0)
        except Exception:
            os.close(self._fd)
            raise
        self._pending = bytearray()

    def exchange(self, packet: bytes, timeout_s: float) -> bytes:
        _validate_request(packet, timeout_s)
        expected_sequence = decode_command(packet).sequence
        with self._lock:
            if self._closed:
                raise ControllerError("UART packet channel is closed")
            deadline = time.monotonic() + timeout_s
            try:
                _write_fd_exact(self._fd, packet, deadline)
                while True:
                    response = _read_status_packet(self._fd, self._pending, deadline)
                    if _response_is_current_or_protocol_error(response, expected_sequence):
                        return response
            except OSError as exc:
                raise ControllerError(f"UART transaction failed: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                termios.tcsetattr(self._fd, termios.TCSANOW, self._original_attributes)
            finally:
                os.close(self._fd)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


class EyeTrackerController:
    """Reliable command and telemetry client for the eye-tracker service."""

    def __init__(
        self,
        channel: PacketChannel,
        *,
        initial_sequence: int | None = None,
        response_timeout_s: float = DEFAULT_RESPONSE_TIMEOUT_S,
        state_timeout_s: float = DEFAULT_STATE_TIMEOUT_S,
        retries: int = 3,
        retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if initial_sequence is None:
            initial_sequence = secrets.randbelow(0xFFFFFFFF) + 1
        if not 1 <= initial_sequence <= 0xFFFFFFFF:
            raise ValueError("initial_sequence must be in [1, 2^32-1]")
        if response_timeout_s <= 0.0:
            raise ValueError("response_timeout_s must be positive")
        if state_timeout_s <= 0.0:
            raise ValueError("state_timeout_s must be positive")
        if retries < 0:
            raise ValueError("retries cannot be negative")
        if retry_delay_s < 0.0:
            raise ValueError("retry_delay_s cannot be negative")
        self.channel = channel
        self.response_timeout_s = response_timeout_s
        self.state_timeout_s = state_timeout_s
        self.retries = retries
        self.retry_delay_s = retry_delay_s
        self._next_sequence = initial_sequence
        self._clock = clock
        self._sleep = sleep
        self._last_sample_key: tuple[int, int, int] | None = None

    def poll(self, *, timeout_s: float | None = None) -> DecodedStatus:
        packet = encode_command(CommandPacket(CommandCode.POLL, self._take_sequence()))
        return self._exchange(packet, timeout_s)

    def poll_new(self, *, timeout_s: float | None = None) -> DecodedStatus | None:
        """Poll once, returning only a sensor frame not returned previously."""
        status = self.poll(timeout_s=timeout_s)
        if status.sensor_timestamp_ns == 0:
            return None
        key = (status.session_id, status.frame_sequence, status.sensor_timestamp_ns)
        if key == self._last_sample_key:
            return None
        self._last_sample_key = key
        return status

    def send_state_command(self, code: CommandCode) -> DecodedStatus:
        if code is CommandCode.POLL:
            raise ValueError("Use poll() for POLL commands")
        sequence = self._take_sequence()
        command_packet = encode_command(
            CommandPacket(
                code=code,
                sequence=sequence,
                controller_timestamp_ns=time.monotonic_ns(),
            )
        )
        attempts = self.retries + 1
        for attempt in range(attempts):
            try:
                status = self._exchange(command_packet, self.state_timeout_s)
            except (ProtocolError, ResponseTimeout):
                status = None
            acknowledged = self._matching_ack(sequence, status)
            if acknowledged is not None:
                return acknowledged
            # A state action is synchronous. If its response was lost or corrupt,
            # repeat the exact command: the service's sequence cache makes this
            # idempotent and preserves a rejection outcome. A POLL is never used
            # as proof that a state command succeeded.
            if attempt + 1 < attempts and self.retry_delay_s:
                self._sleep(self.retry_delay_s)
        raise AcknowledgementTimeout(sequence, attempts)

    def iter_new_samples(
        self,
        *,
        poll_rate_hz: float = DEFAULT_POLL_RATE_HZ,
        on_transient_error: Callable[[Exception, int], None] | None = None,
    ):
        if poll_rate_hz <= 0.0:
            raise ValueError("poll_rate_hz must be positive")
        period = 1.0 / poll_rate_hz
        next_poll = self._clock()
        consecutive_errors = 0
        while True:
            try:
                status = self.poll_new()
            except (ProtocolError, ResponseTimeout) as exc:
                consecutive_errors += 1
                if on_transient_error is not None:
                    on_transient_error(exc, consecutive_errors)
                status = None
            else:
                consecutive_errors = 0
            if status is not None:
                yield status
            next_poll += period
            delay = next_poll - self._clock()
            if delay > 0.0:
                self._sleep(delay)
            else:
                next_poll = self._clock()

    def _exchange(self, packet: bytes, timeout_s: float | None = None) -> DecodedStatus:
        timeout = self.response_timeout_s if timeout_s is None else timeout_s
        response = self.channel.exchange(packet, timeout)
        if len(response) != TRANSACTION_SIZE:
            raise ProtocolError(
                f"Expected a {TRANSACTION_SIZE}-byte status, received {len(response)}"
            )
        status = decode_status(response)
        expected_sequence = decode_command(packet).sequence
        if status.request_sequence != expected_sequence:
            raise ProtocolError(
                "Status echoed request sequence "
                f"{status.request_sequence}, expected {expected_sequence}"
            )
        return status

    @staticmethod
    def _matching_ack(sequence: int, status: DecodedStatus | None) -> DecodedStatus | None:
        if status is None or status.request_sequence != sequence:
            return None
        if status.status is not StatusCode.OK:
            raise CommandRejected(sequence, status.status)
        return status

    def _take_sequence(self) -> int:
        sequence = self._next_sequence
        self._next_sequence = (sequence + 1) & 0xFFFFFFFF
        if self._next_sequence == 0:
            self._next_sequence = 1
        return sequence


def _validate_request(packet: bytes, timeout_s: float) -> None:
    if len(packet) != TRANSACTION_SIZE:
        raise ValueError(f"Transactions must be exactly {TRANSACTION_SIZE} bytes")
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")


def _response_is_current_or_protocol_error(packet: bytes, expected_sequence: int) -> bool:
    status = decode_status(packet)
    return status.request_sequence == expected_sequence or (
        status.request_sequence == 0 and status.status is not StatusCode.OK
    )


def _recv_exact(
    connection: socket.socket,
    size: int,
    *,
    deadline: float | None = None,
) -> bytes:
    received = bytearray()
    while len(received) < size:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise ResponseTimeout("Timed out waiting for eye-tracker status")
            connection.settimeout(remaining)
        chunk = connection.recv(size - len(received))
        if not chunk:
            raise ControllerError("Eye-tracker closed the connection mid-transaction")
        received.extend(chunk)
    return bytes(received)


def _wait_fd(fd: int, *, readable: bool, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0.0:
        raise ResponseTimeout("Timed out waiting for eye-tracker status")
    readers = [fd] if readable else []
    writers = [] if readable else [fd]
    ready_read, ready_write, _exceptional = select.select(readers, writers, [], remaining)
    if not ready_read and not ready_write:
        raise ResponseTimeout("Timed out waiting for eye-tracker status")


def _write_fd_exact(fd: int, packet: bytes, deadline: float) -> None:
    remaining = memoryview(packet)
    while remaining:
        _wait_fd(fd, readable=False, deadline=deadline)
        try:
            written = os.write(fd, remaining)
        except BlockingIOError:
            continue
        if written <= 0:
            raise ControllerError("UART write made no progress")
        remaining = remaining[written:]


def _extract_status_packet(pending: bytearray) -> bytes | None:
    """Return one validated status record, discarding corrupt stream bytes."""
    discarded_complete_record = False
    while True:
        start = pending.find(b"ET")
        if start < 0:
            pending[:] = b"E" if pending.endswith(b"E") else b""
            if discarded_complete_record:
                raise ProtocolError("Discarded a corrupt UART status record")
            return None
        if start:
            del pending[:start]
        if len(pending) < TRANSACTION_SIZE:
            if discarded_complete_record:
                raise ProtocolError("Discarded a corrupt UART status record")
            return None
        candidate = bytes(pending[:TRANSACTION_SIZE])
        try:
            decode_status(candidate)
        except ProtocolError:
            # The magic may have occurred in noise or a damaged record. Advance
            # one byte, then search again so the next intact response can frame.
            del pending[0]
            discarded_complete_record = True
            continue
        del pending[:TRANSACTION_SIZE]
        return candidate


def _read_status_packet(
    fd: int,
    pending: bytearray,
    deadline: float,
    *,
    inter_byte_timeout_s: float = DEFAULT_INTER_BYTE_TIMEOUT_S,
) -> bytes:
    if inter_byte_timeout_s <= 0.0:
        raise ValueError("inter_byte_timeout_s must be positive")
    read_deadline = (
        min(deadline, time.monotonic() + inter_byte_timeout_s) if pending else deadline
    )
    while True:
        packet = _extract_status_packet(pending)
        if packet is not None:
            return packet
        _wait_fd(fd, readable=True, deadline=read_deadline)
        try:
            chunk = os.read(fd, TRANSACTION_SIZE * 4)
        except BlockingIOError:
            continue
        if not chunk:
            continue
        pending.extend(chunk)
        # At 460800 baud a complete record takes about 1.4 ms. Once a
        # response begins, a much shorter idle deadline distinguishes a lost
        # byte from a camera action that legitimately takes several seconds.
        read_deadline = min(deadline, time.monotonic() + inter_byte_timeout_s)


def _status_dict(status: DecodedStatus) -> dict[str, object]:
    return {
        "status": status.status.name,
        "flags": [flag.name for flag in type(status.flags) if flag & status.flags],
        "frame_sequence": status.frame_sequence,
        "sensor_timestamp_ns": status.sensor_timestamp_ns,
        "request_sequence": status.request_sequence,
        "session_id": status.session_id,
        "configured_eye_count": status.configured_eye_count,
        "dropped_analysis_frames": status.dropped_analysis_frames,
        "processing_time_us": status.processing_time_us,
        "result_age_ms": status.result_age_ms,
        "eyes": [
            {
                "eye_id": eye.eye_id,
                "x": eye.x,
                "y": eye.y,
                "pupil_diameter": eye.pupil_diameter,
                "confidence": eye.confidence,
                "valid": eye.valid,
                "blink": eye.blink,
            }
            for eye in status.eyes[: status.configured_eye_count]
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eye-tracker-controller",
        description="Control and read the eye-tracker over direct Pi-to-Pi UART",
    )
    endpoint = parser.add_mutually_exclusive_group()
    endpoint.add_argument("--device", type=Path, default=Path("/dev/ttyAMA0"))
    endpoint.add_argument("--unix", type=Path, help="development Unix socket")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--response-timeout", type=float, default=DEFAULT_RESPONSE_TIMEOUT_S)
    parser.add_argument("--state-timeout", type=float, default=DEFAULT_STATE_TIMEOUT_S)
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY_S)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "command",
        choices=(
            "status",
            "monitor",
            "start-tracking",
            "stop-tracking",
            "start-recording",
            "stop-recording",
            "clear-error",
        ),
    )
    parser.add_argument("--rate", type=float, default=DEFAULT_POLL_RATE_HZ)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    channel: PacketChannel
    try:
        if args.command == "monitor":
            if args.rate <= 0.0:
                raise ValueError("--rate must be positive")
            if args.unix is None:
                maximum_wire_rate = args.baud / (2 * TRANSACTION_SIZE * 10)
                if args.rate >= maximum_wire_rate:
                    raise ValueError(
                        f"--rate {args.rate:g} Hz cannot fit at {args.baud} baud; "
                        f"use a rate below {maximum_wire_rate:.1f} Hz or a faster baud"
                    )
        if args.unix is not None:
            channel = UnixPacketChannel(args.unix)
        else:
            channel = SerialPacketChannel(args.device, baud=args.baud)
        try:
            controller = EyeTrackerController(
                channel,
                response_timeout_s=args.response_timeout,
                state_timeout_s=args.state_timeout,
                retries=args.retries,
                retry_delay_s=args.retry_delay,
            )
            if args.command == "status":
                print(json.dumps(_status_dict(controller.poll()), separators=(",", ":")))
                return 0
            if args.command == "monitor":

                def report_link_error(error: Exception, count: int) -> None:
                    if count == 1 or count % 10 == 0:
                        print(
                            f"eye-tracker-controller: transient UART error "
                            f"({count} consecutive): {error}",
                            file=sys.stderr,
                            flush=True,
                        )

                for status in controller.iter_new_samples(
                    poll_rate_hz=args.rate,
                    on_transient_error=report_link_error,
                ):
                    print(json.dumps(_status_dict(status), separators=(",", ":")), flush=True)
                return 0
            commands = {
                "start-tracking": CommandCode.START_TRACKING,
                "stop-tracking": CommandCode.STOP_TRACKING,
                "start-recording": CommandCode.START_RECORDING,
                "stop-recording": CommandCode.STOP_RECORDING,
                "clear-error": CommandCode.CLEAR_ERROR,
            }
            status = controller.send_state_command(commands[args.command])
            print(json.dumps(_status_dict(status), separators=(",", ":")))
            return 0
        finally:
            channel.close()
    except KeyboardInterrupt:
        return 130
    except (ControllerError, ProtocolError, OSError, ValueError) as exc:
        print(f"eye-tracker-controller: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
