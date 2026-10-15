from __future__ import annotations

import subprocess

import pytest

from macaque_tracker import system_control


def test_poweroff_requests_normal_systemd_shutdown_without_force(monkeypatch):
    calls = []
    monkeypatch.setattr(system_control.sys, "platform", "linux")
    monkeypatch.setattr(system_control.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))
    system_control.power_off_system()
    assert calls == [(([
        "/usr/bin/systemctl", "--no-ask-password", "--no-block", "poweroff"
    ],), {"check": True, "capture_output": True, "text": True, "timeout": 5})]


@pytest.mark.parametrize("failure", [
    subprocess.CalledProcessError(1, "systemctl", stderr="Access denied"),
    FileNotFoundError("systemctl missing"),
    subprocess.TimeoutExpired("systemctl", 5),
])
def test_poweroff_errors_are_reported(monkeypatch, failure):
    monkeypatch.setattr(system_control.sys, "platform", "linux")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(system_control.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="power-off|poweroff"):
        system_control.power_off_system()


def test_poweroff_rejects_non_linux_without_running_command(monkeypatch):
    monkeypatch.setattr(system_control.sys, "platform", "darwin")
    monkeypatch.setattr(system_control.subprocess, "run", lambda *a, **kw: pytest.fail("spawned"))
    with pytest.raises(RuntimeError, match="Linux"):
        system_control.power_off_system()
