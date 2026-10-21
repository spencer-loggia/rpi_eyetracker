"""Launch-time system provenance for saved experiments."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path


def _source_git_commit() -> str | None:
    """Read the source checkout's full HEAD, or return None when unavailable."""
    root = Path(__file__).resolve().parents[2]
    try:
        # Require our own checkout, including worktrees with a .git file. Never
        # accidentally report a parent repository for an installed package.
        if not (root / ".git").exists():
            return None
        result = subprocess.run(
            # The launcher can run as root from a checkout owned by another user.
            # Trust only this source path, for this read-only Git invocation.
            ["git", "-c", f"safe.directory={root}", "-C", str(root),
             "rev-parse", "--verify", "HEAD"],
            env={key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
            check=True, capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) else None


def capture_system_state() -> dict:
    """Capture launcher provenance once, outside camera and tracking work."""
    return {
        "version": 1,
        "git_commit": _source_git_commit(),
        "hostname": platform.node(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python_version": platform.python_version(),
    }
