from __future__ import annotations

import os
import pty
import termios

import pytest

from macaque_tracker.uart import open_exclusive_uart


@pytest.mark.skipif(not hasattr(termios, "TIOCEXCL"), reason="TIOCEXCL is unavailable")
def test_uart_open_prevents_a_second_owner() -> None:
    master_fd, original_slave_fd = pty.openpty()
    path = os.ttyname(original_slave_fd)
    owner_fd = open_exclusive_uart(path)
    try:
        with pytest.raises(OSError):
            open_exclusive_uart(path)
    finally:
        os.close(owner_fd)
        os.close(original_slave_fd)
        os.close(master_fd)
