from dataclasses import replace

import pytest

from macaque_tracker.models import EyeMeasurement, FrameResult
from macaque_tracker.protocol import CommandCode, CommandPacket, StatusSnapshot, encode_command
from macaque_tracker.telemetry import TelemetryRates
from macaque_tracker.transport import ProtocolEngine


def sample(frame, *, valid=True, stale=False, session=1):
    eyes = (EyeMeasurement(0, 10, 20, 12 if valid else 0, 0.9, valid, False),
            EyeMeasurement(1, 10, 20, 0, 0, False, True))
    return StatusSnapshot(True, True, True, FrameResult(frame, frame * 20_000_000, 1, eyes),
                          session_id=session, configured_eye_count=2, stale=stale)


def test_transmit_rate_counts_fifty_unique_valid_samples_not_duplicate_polls():
    now = [0.0]
    rates = TelemetryRates(clock=lambda: now[0])
    for i in range(100):
        now[0] = i / 50
        status = sample(i + 1)
        rates.sent(status)
        rates.sent(status)
    now[0] = 2.0
    assert rates.snapshot() == pytest.approx((50, 0))
    now[0] = 4.0
    assert rates.snapshot() == (0, 0)


def test_invalid_stale_and_idle_positions_do_not_count():
    now = [0.0]
    rates = TelemetryRates(clock=lambda: now[0])
    rates.sent(sample(1, valid=False))
    rates.sent(sample(2, stale=True))
    rates.sent(replace(sample(3), tracking=False))
    rates.sent(sample(4))
    rates.sent(sample(4, session=2))  # A restarted service is a distinct sample.
    now[0] = 2.0
    assert rates.snapshot() == (1, 0)


def test_preparing_or_corrupting_a_reply_cannot_increment_transmit_rate():
    class Service:
        snapshot = lambda self: sample(1)
        execute = lambda self, command: sample(1)

    now = [0.0]
    engine = ProtocolEngine(Service())
    engine.rates = TelemetryRates(clock=lambda: now[0])
    packet = encode_command(CommandPacket(CommandCode.POLL, 1))
    engine.handle(packet)  # A later transport write can still fail.
    now[0] = 2.0
    assert engine.rates.snapshot() == ()  # No completed reply yet.
    engine.response_sent()
    engine.response_sent()  # Count one full write only once.
    broken = bytearray(packet)
    broken[-1] ^= 1
    engine.handle(bytes(broken))
    engine.response_sent()
    now[0] = 4.0
    assert engine.rates.snapshot() == (0.5, 0)
