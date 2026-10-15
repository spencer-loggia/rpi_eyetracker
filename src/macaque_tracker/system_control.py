"""Host power control, invoked only after experiment cleanup and a UART reply."""

from __future__ import annotations

import subprocess
import sys


def power_off_system() -> None:
    """Ask systemd for normal shutdown; never bypass service or filesystem cleanup."""
    if sys.platform != "linux":
        raise RuntimeError("System power-off is supported only on Raspberry Pi OS/Linux")
    try:
        subprocess.run(
            ["/usr/bin/systemctl", "--no-ask-password", "--no-block", "poweroff"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"systemctl poweroff: {detail}. Run Pi Tracker as root.") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not request system power-off: {exc}") from exc
