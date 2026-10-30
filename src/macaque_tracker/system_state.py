"""Launch-time system provenance for saved experiments."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path


def _source_git_output(*arguments: str) -> str | None:
    """Read this checkout's Git metadata without trusting inherited Git overrides."""
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
             *arguments],
            env={key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
            check=True, capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _source_git_commit() -> str | None:
    """Read the source checkout's full HEAD, or return None when unavailable."""
    commit = _source_git_output("rev-parse", "--verify", "HEAD")
    if commit is None:
        return None
    return commit if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit) else None


def capture_system_state() -> dict:
    """Capture launcher provenance once, outside camera and tracking work."""
    commit = _source_git_commit()
    changes = (None if commit is None else
               _source_git_output("status", "--porcelain", "--untracked-files=normal"))
    return {
        "version": 2,
        "git_commit": commit,
        # Never store filenames or diff contents; mark an uncommitted build so
        # its HEAD cannot be mistaken for a complete record of the running code.
        "git_dirty": None if changes is None else bool(changes),
        "hostname": platform.node(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python_version": platform.python_version(),
    }
