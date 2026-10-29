"""UART acknowledgement recovery and terminal ownership regressions."""
import os
import pty
import select
import termios
import threading
import time
import unittest
from unittest.mock import patch

from macaque_tracker.controller import (
    AcknowledgementTimeout, EyeTrackerController, ResponseTimeout,
    SerialPacketChannel, _read_status_packet,
)
from macaque_tracker.protocol import StatusCode, StatusSnapshot, decode_command, encode_status


def status_bytes(*, sequence=42, code=StatusCode.OK):
    return encode_status(StatusSnapshot(
        tracking=False, recording=False, configured=False, result=None,
        request_sequence=sequence, status=code,
    ))


class UARTAcknowledgementTests(unittest.TestCase):
    def test_close_preserves_binary_mode_releases_claim_and_does_not_echo_late_ack(self):
        master, slave = pty.openpty()
        try:
            attributes = termios.tcgetattr(slave)
            attributes[3] |= termios.ECHO | termios.ICANON
            termios.tcsetattr(slave, termios.TCSANOW, attributes)
            channel = SerialPacketChannel(os.ttyname(slave), baud=9600)
            channel.close()
            channel.close()
            self.assertFalse(termios.tcgetattr(slave)[3] & (termios.ECHO | termios.ICANON))
            os.write(master, status_bytes())
            self.assertFalse(select.select([master], [], [], 0.05)[0], "late ACK echoed back to tracker")
            # The original slave descriptor is intentionally still open.
            with SerialPacketChannel(os.ttyname(slave), baud=9600):
                pass
        finally:
            os.close(slave)
            os.close(master)


    def test_failed_configuration_releases_exclusive_claim(self):
        master, slave = pty.openpty()
        try:
            with patch("macaque_tracker.controller.configure_raw_uart", side_effect=ValueError("bad baud")):
                with self.assertRaisesRegex(ValueError, "bad baud"):
                    SerialPacketChannel(os.ttyname(slave), baud=9600)
            with SerialPacketChannel(os.ttyname(slave), baud=9600):
                pass
        finally:
            os.close(slave)
            os.close(master)


    def test_noise_and_damaged_replies_do_not_end_ack_wait(self):
        corrupt = bytearray(status_bytes())
        corrupt[28] ^= 1
        for prefix, suffix in (
            (b"noiseE", status_bytes()),
            (bytes(corrupt), status_bytes()),
            (status_bytes()[:19], status_bytes()[19:]),
            (status_bytes()[:-1], status_bytes()),
            (status_bytes(sequence=0, code=StatusCode.BAD_PACKET_TYPE), status_bytes()),
            (status_bytes(sequence=41), status_bytes()),
        ):
            with self.subTest(prefix=prefix.hex()):
                master, slave = pty.openpty()
                errors = []
                def peer():
                    try:
                        request = bytearray()
                        while len(request) < 64:
                            if not select.select([master], [], [], 1)[0]:
                                raise TimeoutError("no request")
                            request.extend(os.read(master, 64 - len(request)))
                        self.assertEqual(decode_command(bytes(request)).sequence, 42)
                        os.write(master, prefix)
                        time.sleep(0.15)  # Exceeds the old inter-byte timeout.
                        os.write(master, suffix)
                    except BaseException as exc:
                        errors.append(exc)
                try:
                    with SerialPacketChannel(os.ttyname(slave), baud=9600) as channel:
                        worker = threading.Thread(target=peer, daemon=True)
                        worker.start()
                        try:
                            controller = EyeTrackerController(
                                channel, initial_sequence=42, retries=0, state_timeout_s=0.8,
                            )
                            self.assertEqual(controller.start_experiment("Yuri").request_sequence, 42)
                        finally:
                            worker.join(1)
                        self.assertFalse(worker.is_alive())
                        self.assertFalse(errors, errors)
                finally:
                    os.close(slave)
                    os.close(master)


    def test_unmatched_or_corrupt_replies_never_count_as_ack_and_wait_is_bounded(self):
        master, slave = pty.openpty()
        try:
            with SerialPacketChannel(os.ttyname(slave), baud=9600) as channel:
                os.write(master, status_bytes(sequence=0, code=StatusCode.BAD_CRC) + status_bytes(sequence=41))
                started = time.monotonic()
                controller = EyeTrackerController(channel, initial_sequence=42, retries=0, state_timeout_s=0.06)
                with self.assertRaises(AcknowledgementTimeout):
                    controller.start_tracking()
                self.assertLess(time.monotonic() - started, 0.5)
        finally:
            os.close(slave)
            os.close(master)


    def test_incomplete_reply_uses_transaction_deadline(self):
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, status_bytes()[:-1])
            started = time.monotonic()
            with self.assertRaises(ResponseTimeout):
                _read_status_packet(read_fd, bytearray(), started + 0.15)
            self.assertGreaterEqual(time.monotonic() - started, 0.14)
            self.assertLess(time.monotonic() - started, 0.5)
        finally:
            os.close(read_fd)
            os.close(write_fd)


