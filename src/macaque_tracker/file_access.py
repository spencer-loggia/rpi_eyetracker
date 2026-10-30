"""Permissions for experiment outputs shared by all local users."""
from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

SHARED_FILE_MODE = 0o666
SHARED_DIRECTORY_MODE = 0o777


def share_output(path: Path, *, directory: bool = False) -> None:
    """Set one output's mode without following a final symlink or walking parents."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if directory:
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        expected = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected(info.st_mode) or (not directory and info.st_nlink != 1):
            raise ValueError(f"Expected a directory or a regular output with one hard link: {path}")
        mode = SHARED_DIRECTORY_MODE if directory else SHARED_FILE_MODE
        # A non-root user can reuse an already-shared directory owned by root.
        if stat.S_IMODE(info.st_mode) != mode:
            os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def create_shared_file(path: Path) -> None:
    """Reserve a new video path before an encoder opens it, including lazy muxers."""
    with path.open("xb") as stream:
        try:
            os.fchmod(stream.fileno(), SHARED_FILE_MODE)
        except BaseException:
            path.unlink(missing_ok=True)
            raise


def sync_directory(path: Path) -> None:
    """Make published filenames durable before their source data may be removed."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_shared_json(path: Path, content: dict) -> Path:
    """Durably replace JSON with mode 0666, independent of the process umask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".shared-", delete=False,
    ) as output:
        temporary = Path(output.name)
        try:
            json.dump(content, output, indent=2)
            output.write("\n")
            output.flush()
            os.fchmod(output.fileno(), SHARED_FILE_MODE)
            os.fsync(output.fileno())
            temporary.replace(path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
    return path
