"""Temporary Linux idle/sleep inhibition, isolated from camera and GUI workers.

logind's idle inhibitor is respected by Raspberry Pi OS labwc/swayidle. X11
also needs a connection-scoped XScreenSaver/DPMS suspension. Both leases are
released on process exit, without editing desktop or system power settings.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path


class KeepAwake:
    """Hold power leases for the application lifetime; report failures visibly."""

    def __init__(self, warn):
        self._warn = warn
        self._process = None
        self._stop = threading.Event()

    def __enter__(self):
        if sys.platform != "linux":
            return self
        helper = [sys.executable, "-u", str(Path(__file__).resolve()), "--hold"]
        inhibit = shutil.which("systemd-inhibit")
        command = ([inhibit, "--what=idle:sleep", "--mode=block", "--who=Pi Tracker",
                    "--why=Camera tracking and experiment display are active", *helper]
                   if inhibit else helper)
        if not inhibit:
            self._warn("Sleep prevention unavailable: systemd-inhibit is missing.")
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            if not select.select([self._process.stdout], [], [], 3)[0]:
                raise RuntimeError("power inhibitor did not acknowledge startup")
            line = self._process.stdout.readline()
            try:
                ready = json.loads(line)
            except ValueError as exc:
                raise RuntimeError(line.decode(errors="replace").strip() or "helper exited") from exc
            if not ready.get("ready"):
                raise RuntimeError("power inhibitor did not become ready")
            for warning in ready["warnings"]:
                self._warn(warning)
            threading.Thread(target=self._watch, name="power-inhibitor-watch", daemon=True).start()
        except (OSError, RuntimeError) as exc:
            self.close()
            self._warn(f"Could not prevent sleep/screen blanking: {exc}")
        return self

    def _watch(self):
        """Notice a lost lease without any work in the capture/detection threads."""
        while not self._stop.wait(10):
            process = self._process
            if process is None or process.poll() is not None:
                self._warn("Sleep/screen protection stopped unexpectedly; check the desktop session.")
                return

    def close(self):
        """Close the helper's input pipe; leases release even if the parent crashes."""
        self._stop.set()
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.stdin.close()
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)
        finally:
            process.stdout.close()

    def __exit__(self, *_args):
        self.close()


def _hold_x11():
    """Suspend X11 screensaver and DPMS on a private helper-process connection."""
    x11_name, xss_name = ctypes.util.find_library("X11"), ctypes.util.find_library("Xss")
    if not x11_name or not xss_name:
        raise RuntimeError("X11 screen protection needs libX11 and libXss (libxss1)")
    x11, xss = ctypes.CDLL(x11_name), ctypes.CDLL(xss_name)
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XFlush.argtypes = [ctypes.c_void_p]
    xss.XScreenSaverQueryVersion.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    xss.XScreenSaverSuspend.argtypes = [ctypes.c_void_p, ctypes.c_int]
    display = x11.XOpenDisplay(None)
    if not display:
        raise RuntimeError("Cannot connect to X11; preserve DISPLAY and XAUTHORITY when using sudo")
    major, minor = ctypes.c_int(), ctypes.c_int()
    if not xss.XScreenSaverQueryVersion(display, ctypes.byref(major), ctypes.byref(minor)) or (
        major.value, minor.value
    ) < (1, 1):
        x11.XCloseDisplay(display)
        raise RuntimeError("X11 server lacks screensaver suspension support")
    xss.XScreenSaverSuspend(display, 1)
    x11.XFlush(display)
    return lambda: x11.XCloseDisplay(display)  # Disconnect releases only our suspension.


def _hold():
    """Handshake after leases are acquired, then sleep until the parent pipe closes."""
    release = None
    warnings = []
    if os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        try:
            release = _hold_x11()
        except (OSError, RuntimeError) as exc:
            warnings.append(f"Could not prevent X11 screen blanking: {exc}")
    print(json.dumps({"ready": True, "warnings": warnings}), flush=True)
    try:
        sys.stdin.buffer.read()
    finally:
        if release is not None:
            release()


if __name__ == "__main__" and sys.argv[1:] == ["--hold"]:
    _hold()
