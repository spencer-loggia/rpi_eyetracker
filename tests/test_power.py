from __future__ import annotations

import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from macaque_tracker import power


class Process:
    def __init__(self, reply=b'{"ready": true, "warnings": []}\n'):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(reply)
        self.pid = 123456
        self.waited = False

    def wait(self, timeout):
        assert timeout == 3
        self.waited = True
        return 0

    def poll(self):
        return None


def setup_helper(monkeypatch, process):
    commands = []
    monkeypatch.setattr(power.sys, "platform", "linux")
    monkeypatch.setattr(power.shutil, "which", lambda _: "/usr/bin/systemd-inhibit")
    monkeypatch.setattr(power.select, "select", lambda readers, *_: (readers, [], []))

    def spawn(command, **kwargs):
        assert kwargs["stdin"] is subprocess.PIPE
        assert kwargs["start_new_session"]
        commands.append(command)
        return process

    monkeypatch.setattr(power.subprocess, "Popen", spawn)
    return commands


def test_temporary_sleep_and_idle_lease_released_on_normal_or_exception_exit(monkeypatch):
    process = Process()
    commands = setup_helper(monkeypatch, process)
    warnings = []
    with pytest.raises(ValueError), power.KeepAwake(warnings.append):
        assert "--what=idle:sleep" in commands[0]
        assert "--mode=block" in commands[0]
        assert all("shutdown" not in argument for argument in commands[0])
        assert not process.stdin.closed
        raise ValueError("application exits")
    assert process.stdin.closed and process.stdout.closed and process.waited
    assert not warnings


def test_power_permission_failure_is_visible_and_releases_child(monkeypatch):
    process = Process(b"Failed to inhibit: Access denied\n")
    setup_helper(monkeypatch, process)
    warnings = []
    with power.KeepAwake(warnings.append):
        pass
    assert len(warnings) == 1 and "Access denied" in warnings[0]
    assert process.stdin.closed and process.waited


def test_inhibitor_timeout_does_not_hang_application_start(monkeypatch):
    process = Process()
    setup_helper(monkeypatch, process)
    monkeypatch.setattr(power.select, "select", lambda *_: ([], [], []))
    warnings = []
    with power.KeepAwake(warnings.append):
        pass
    assert "startup" in warnings[0]
    assert process.waited


def test_missing_systemd_still_attempts_x11_protection(monkeypatch):
    process = Process()
    commands = setup_helper(monkeypatch, process)
    monkeypatch.setattr(power.shutil, "which", lambda _: None)
    warnings = []
    with power.KeepAwake(warnings.append):
        assert commands[0][0] == power.sys.executable
    assert "systemd-inhibit is missing" in warnings[0]


def test_helper_holds_x11_until_parent_input_closes(monkeypatch, capsys):
    released = []
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(power, "_hold_x11", lambda: lambda: released.append(True))
    monkeypatch.setattr(power.sys, "stdin", SimpleNamespace(buffer=io.BytesIO()))
    power._hold()
    assert json.loads(capsys.readouterr().out)["ready"]
    assert released == [True]


def test_wayland_uses_logind_lease_without_touching_x11(monkeypatch, capsys):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr(power, "_hold_x11", lambda: pytest.fail("X11 called in Wayland"))
    monkeypatch.setattr(power.sys, "stdin", SimpleNamespace(buffer=io.BytesIO()))
    power._hold()
    assert json.loads(capsys.readouterr().out) == {"ready": True, "warnings": []}


def test_non_linux_never_changes_host_power_settings(monkeypatch):
    monkeypatch.setattr(power.sys, "platform", "darwin")
    monkeypatch.setattr(power.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("spawned"))
    with power.KeepAwake(lambda _: pytest.fail("warning")):
        pass
