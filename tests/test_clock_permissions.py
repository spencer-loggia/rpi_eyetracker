from __future__ import annotations

import errno

import pytest

from macaque_tracker import pi_runtime


@pytest.fixture(autouse=True)
def isolate_clock(monkeypatch):
    # Never let this suite call the real host clock setter, even on a root CI runner.
    def unexpected_call(*_args):
        pytest.fail("Clock setter must be stubbed explicitly")

    monkeypatch.setattr(pi_runtime.time, "clock_settime_ns", unexpected_call, raising=False)
    monkeypatch.setattr(pi_runtime.time, "CLOCK_REALTIME", 123, raising=False)


def test_clock_set_preserves_exact_integer_nanoseconds(monkeypatch):
    calls = []
    monkeypatch.setattr(pi_runtime.time, "clock_settime_ns", lambda *args: calls.append(args))
    timestamp = 1_792_000_000_123_456_789
    pi_runtime.set_system_time(timestamp)
    assert calls == [(123, timestamp)]


def test_missing_clock_api_does_not_claim_missing_capability(monkeypatch):
    monkeypatch.delattr(pi_runtime.time, "clock_settime_ns")
    with pytest.raises(RuntimeError, match="Python/platform does not provide") as error:
        pi_runtime.set_system_time(1_792_000_000_000_000_000)
    assert "CAP_SYS_TIME" not in str(error.value)
    assert isinstance(error.value.__cause__, AttributeError)


def test_clock_permission_error_retains_errno_and_running_process_diagnostics(monkeypatch):
    def denied(*_args):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(pi_runtime.time, "clock_settime_ns", denied)
    monkeypatch.setattr(
        pi_runtime, "_clock_permission_details",
        lambda: "pid=456, uid=1000, CAP_SYS_TIME effective=yes, Seccomp=2",
    )
    with pytest.raises(RuntimeError) as error:
        pi_runtime.set_system_time(1_792_000_000_000_000_000)
    message = str(error.value)
    assert "PermissionError: [Errno 1] Operation not permitted" in message
    assert "pid=456, uid=1000, CAP_SYS_TIME effective=yes" in message
    assert "ProtectClock=no" in message
    assert isinstance(error.value.__cause__, PermissionError)


def test_invalid_clock_value_is_not_reported_as_permission_problem(monkeypatch):
    def invalid(*_args):
        raise OSError(errno.EINVAL, "Invalid argument")

    monkeypatch.setattr(pi_runtime.time, "clock_settime_ns", invalid)
    with pytest.raises(RuntimeError) as error:
        pi_runtime.set_system_time(123)
    assert "[Errno 22] Invalid argument" in str(error.value)
    assert "timestamp_ns=123" in str(error.value)
    assert "CAP_SYS_TIME" not in str(error.value)


@pytest.mark.parametrize("effective,present", [("0000000002000000", "yes"), ("0", "no")])
def test_capability_diagnostic_reads_effective_not_only_ambient_set(monkeypatch, effective, present):
    monkeypatch.setattr(
        pi_runtime.Path, "read_text",
        lambda *_args, **_kwargs: f"CapEff:\t{effective}\nCapAmb:\t02000000\nSeccomp:\t0\n",
    )
    details = pi_runtime._clock_permission_details()
    assert f"CAP_SYS_TIME effective={present}" in details
    assert "Seccomp=0" in details
    assert "pid=" in details and "uid=" in details


def test_unavailable_proc_status_does_not_hide_original_clock_failure(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise FileNotFoundError("proc unavailable")

    monkeypatch.setattr(pi_runtime.Path, "read_text", unavailable)
    assert "effective capabilities unavailable" in pi_runtime._clock_permission_details()
