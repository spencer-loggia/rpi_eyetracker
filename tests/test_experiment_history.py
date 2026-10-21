from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from macaque_tracker.experiment_history import (
    HISTORY_COLUMNS,
    HISTORY_FILENAME,
    read_history,
    update_history,
)


def _row(number: int) -> dict:
    row = dict.fromkeys(HISTORY_COLUMNS, "")
    row.update(
        experiment_name=f"exp_A_20261021_{number}_120000", subject="A",
        experiment_number=str(number), date_utc="20261021",
        started_timestamp_ns="1792584000000000000", status="active", block_count="0",
    )
    return row


def test_concurrent_history_updates_keep_all_rows_and_quote_paths(tmp_path):
    path = tmp_path / HISTORY_FILENAME
    rows = [_row(number) for number in range(1, 13)]
    rows[0]["calibration_source"] = '/path with\ttab/and\nnewline/"calibration".json'
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda row: update_history(path, [row]), rows))
    saved = {row["experiment_name"]: row for row in read_history(path)}
    assert saved == {row["experiment_name"]: row for row in rows}
    update_history(path, [{**rows[0], "status": "closed", "block_count": "2"}])
    assert len(read_history(path)) == len(rows)
    assert path.stat().st_mode & 0o777 == 0o666


def test_failed_atomic_replacement_preserves_complete_history(tmp_path, monkeypatch):
    path = tmp_path / HISTORY_FILENAME
    update_history(path, [_row(1)])
    before = path.read_bytes()

    def fail_replace(self, target):
        raise OSError("full disk")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="full disk"):
        update_history(path, [_row(2)])
    assert path.read_bytes() == before
    assert not list(tmp_path.glob(".experiment-history-*"))


@pytest.mark.parametrize("damage", ["header", "truncated", "number", "duplicate", "quoting"])
def test_malformed_log_is_preserved_instead_of_resetting_numbering(tmp_path, damage):
    path = tmp_path / HISTORY_FILENAME
    update_history(path, [_row(1)])
    original = path.read_text()
    if damage == "header":
        path.write_text("wrong\theader\n")
    elif damage == "truncated":
        path.write_text(original.splitlines()[0] + "\nexp_A\tA\n")
    elif damage == "number":
        path.write_text(original.replace("\tA\t1\t", "\tA\tbad\t"))
    elif damage == "quoting":
        path.write_text(original + '"unterminated quoted row')
    else:
        path.write_text(original + original.splitlines()[1] + "\n")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Invalid experiment history"):
        update_history(path, [_row(2)])
    assert path.read_bytes() == before


@pytest.mark.parametrize("filename", [HISTORY_FILENAME, ".experiment_history.lock"])
def test_history_and_lock_symlinks_cannot_redirect_writes(tmp_path, filename):
    outside = tmp_path / "outside.txt"
    outside.write_text("keep")
    (tmp_path / filename).symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        update_history(tmp_path / HISTORY_FILENAME, [_row(1)])
    assert outside.read_text() == "keep"
