"""Recording filesystem checks, outside the per-frame acquisition path."""

from __future__ import annotations

import shutil
from pathlib import Path

MINIMUM_FREE_BYTES = 10_000_000_000  # 10 GB; a higher configured reserve still applies.
DISK_CHECK_INTERVAL_S = 10.0


class LowDiskSpace(RuntimeError):
    """Video must remain off while tracking can continue."""


def recording_space_warning(directory: Path, minimum_free_gib: float) -> str | None:
    """Check the output filesystem, preserving a hard floor of 10 decimal GB."""
    reserve = max(MINIMUM_FREE_BYTES, int(minimum_free_gib * 1024**3))
    try:
        free = shutil.disk_usage(directory).free
    except OSError as exc:
        return f"Video saving disabled: cannot check free space at {directory}: {exc}"
    if free < reserve:
        return (
            f"Video saving disabled: only {free / 1e9:.2f} GB free at {directory}; "
            f"reserve is {reserve / 1e9:.2f} GB. Pupil tracking and UART continue. "
            "Free space, then explicitly restart video recording."
        )
    return None
