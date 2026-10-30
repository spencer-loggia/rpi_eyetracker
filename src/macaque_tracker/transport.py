from __future__ import annotations

import logging
import os
import select
import socket
import stat
import termios
import threading
import time
from dataclasses import replace
from pathlib import Path

from .protocol import (
    TRANSACTION_SIZE,
    ProtocolError,
    crc32c,
    decode_command,
    encode_status,
    protocol_error_status,
)
from .service import EyeTrackingService
from .telemetry import TelemetryRates
from .uart import close_exclusive_uart, configure_raw_uart, open_exclusive_uart


class TransportError(RuntimeError):
    pass


class ProtocolEngine:
    def __init__(self, service: EyeTrackingService) -> None:
        self.service = service
        self.rates = TelemetryRates()
        self._response_snapshot = None
        bind = getattr(service, "bind_telemetry_rates", None)
        if bind is not None:
            bind(self.rates)

    def handle(self, packet: bytes) -> bytes:
        self._response_snapshot = None
        try:
            command = decode_command(packet)
        except ProtocolError as exc:
            snapshot = self.service.snapshot()
            snapshot = replace(
                snapshot,
                request_sequence=0,
                status=protocol_error_status(exc),
                error=True,
            )
            return encode_status(snapshot)
        snapshot = replace(self.service.execute(command), request_sequence=command.sequence)
        self._response_snapshot = snapshot
        return encode_status(snapshot)

    def response_sent(self) -> None:
        """Count fresh valid samples only after the full packet write succeeds."""
        if self._response_snapshot is not None:
            self.rates.sent(self._response_snapshot)
            self._response_snapshot = None

    @property
    def response_action_pending(self) -> bool:
        return getattr(self.service, "response_action_pending", False)

    def after_response(self) -> None:
        if self.response_action_pending:
            self.service.after_response()


def _read_exact(sock: socket.socket, length: int, pending: bytearray) -> bytes | None:
    """Retain partial requests across socket timeouts on the same connection."""
    while len(pending) < length:
        chunk = sock.recv(length - len(pending))
        if not chunk:
            return None
        pending.extend(chunk)
    packet = bytes(pending[:length])
    del pending[:length]
    return packet


class UnixSocketServer:
    """Development transport with the same request/response records as UART."""

    def __init__(self, path: str | Path, engine: ProtocolEngine) -> None:
        self.path = Path(path)
        self.engine = engine
        self._created = False

    def serve(self, stop: threading.Event) -> None:
        if self.path.exists():
            raise TransportError(
                f"Unix socket path already exists: {self.path}. Remove it only if stale."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self.path))
            self._created = True
            server.listen(1)
            server.settimeout(0.25)
            while not stop.is_set():
                try:
                    connection, _address = server.accept()
                except TimeoutError:
                    continue
                with connection:
                    connection.settimeout(0.5)
                    pending = bytearray()
                    while not stop.is_set():
                        try:
                            packet = _read_exact(connection, TRANSACTION_SIZE, pending)
                        except TimeoutError:
                            continue
                        except (ConnectionError, OSError):
                            break
                        if packet is None:
                            break
                        try:
                            connection.sendall(self.engine.handle(packet))
                            self.engine.response_sent()
                        except (ConnectionError, OSError):
                            break
                        finally:
                            self.engine.after_response()
        finally:
            server.close()
            if self._created:
                try:
                    mode = self.path.lstat().st_mode
                    if stat.S_ISSOCK(mode):
                        self.path.unlink()
                except FileNotFoundError:
                    pass


class UartServer:
    """Direct 3.3 V UART request/response link between two Raspberry Pis."""

    def __init__(self, path: str | Path, baud: int, engine: ProtocolEngine) -> None:
        self.path = Path(path)
        self.baud = baud
        self.engine = engine

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)
        deadline = time.monotonic() + 0.25
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
                raise TransportError("UART response write timed out")
            try:
                written = os.write(fd, view)
            except BlockingIOError:
                continue
            if written <= 0:
                raise TransportError("UART write returned no progress")
            view = view[written:]

    @staticmethod
    def _next_commands(pending: bytearray) -> list[bytes]:
        """Resynchronise on a complete, checksum-valid record after damaged bytes."""
        commands: list[bytes] = []
        while True:
            start = pending.find(b"ET")
            if start < 0:
                pending[:] = b"E" if pending.endswith(b"E") else b""
                return commands
            if start:
                del pending[:start]
            if len(pending) < TRANSACTION_SIZE:
                return commands
            packet = bytes(pending[:TRANSACTION_SIZE])
            if crc32c(packet[:-4]) != int.from_bytes(packet[-4:], "little"):
                del pending[0]
                continue
            commands.append(packet)
            del pending[:TRANSACTION_SIZE]

    def serve(self, stop: threading.Event) -> None:
        """Keep listening across disconnects; command retries retain engine state."""
        last_error = None
        while not stop.is_set():
            try:
                self._serve_connection(stop)
                return
            except (OSError, termios.error, TransportError) as exc:
                if stop.is_set():
                    raise
                if str(exc) != last_error:
                    logging.getLogger(__name__).warning("UART interrupted; retrying: %s", exc)
                    last_error = str(exc)
                stop.wait(0.5)

    def _serve_connection(self, stop: threading.Event) -> None:
        try:
            fd = open_exclusive_uart(self.path)
        except OSError as exc:
            raise TransportError(f"Could not open UART {self.path}: {exc}") from exc
        pending = bytearray()
        try:
            try:
                configure_raw_uart(
                    fd, self.baud, read_timeout_deciseconds=1, discard_input=False,
                )
            except (OSError, termios.error, ValueError) as exc:
                raise TransportError(f"Could not configure UART {self.path}: {exc}") from exc
            while not stop.is_set():
                if not select.select([fd], [], [], 0.1)[0]:
                    continue
                try:
                    chunk = os.read(fd, TRANSACTION_SIZE)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise TransportError("UART disconnected")
                pending.extend(chunk)
                for command in self._next_commands(pending):
                    response = self.engine.handle(command)
                    try:
                        self._write_all(fd, response)
                        self.engine.response_sent()
                        if self.engine.response_action_pending:
                            termios.tcdrain(fd)
                    finally:
                        # A disconnected reader must not cancel an accepted shutdown.
                        self.engine.after_response()
        finally:
            close_exclusive_uart(fd)
