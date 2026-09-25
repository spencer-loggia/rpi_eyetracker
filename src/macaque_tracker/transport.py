from __future__ import annotations

import os
import socket
import stat
import threading
from dataclasses import replace
from pathlib import Path

from .protocol import (
    TRANSACTION_SIZE,
    ProtocolError,
    decode_command,
    encode_status,
    protocol_error_status,
)
from .service import EyeTrackingService
from .uart import configure_raw_uart, open_exclusive_uart


class TransportError(RuntimeError):
    pass


class ProtocolEngine:
    def __init__(self, service: EyeTrackingService) -> None:
        self.service = service

    def handle(self, packet: bytes) -> bytes:
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
        return encode_status(snapshot)


def _read_exact(sock: socket.socket, length: int) -> bytes | None:
    data = bytearray()
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk:
            return None if not data else bytes(data)
        data.extend(chunk)
    return bytes(data)


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
                    while not stop.is_set():
                        try:
                            packet = _read_exact(connection, TRANSACTION_SIZE)
                        except TimeoutError:
                            continue
                        except (ConnectionError, OSError):
                            break
                        if packet is None:
                            break
                        if len(packet) != TRANSACTION_SIZE:
                            break
                        try:
                            connection.sendall(self.engine.handle(packet))
                        except (ConnectionError, OSError):
                            break
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
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise TransportError("UART write returned no progress")
            view = view[written:]

    @staticmethod
    def _next_commands(pending: bytearray) -> list[bytes]:
        """Extract complete records and resynchronise on the ``ET`` magic."""
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
            commands.append(bytes(pending[:TRANSACTION_SIZE]))
            del pending[:TRANSACTION_SIZE]

    def serve(self, stop: threading.Event) -> None:
        try:
            fd = open_exclusive_uart(self.path)
        except OSError as exc:
            raise TransportError(f"Could not open UART {self.path}: {exc}") from exc
        pending = bytearray()
        try:
            try:
                configure_raw_uart(fd, self.baud, read_timeout_deciseconds=1)
            except (OSError, ValueError) as exc:
                raise TransportError(f"Could not configure UART {self.path}: {exc}") from exc
            os.set_blocking(fd, True)
            while not stop.is_set():
                chunk = os.read(fd, TRANSACTION_SIZE)
                if not chunk:
                    continue
                pending.extend(chunk)
                for command in self._next_commands(pending):
                    self._write_all(fd, self.engine.handle(command))
        finally:
            os.close(fd)
