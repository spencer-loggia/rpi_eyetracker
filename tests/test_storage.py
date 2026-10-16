from types import SimpleNamespace

import pytest

from macaque_tracker import storage


@pytest.mark.parametrize("free,expected", [(9_999_999_999, True), (10_000_000_000, False)])
def test_ten_gb_floor_cannot_be_lowered_by_an_old_config(monkeypatch, tmp_path, free, expected):
    checked = []

    def usage(path):
        checked.append(path)
        return SimpleNamespace(free=free)

    monkeypatch.setattr(storage.shutil, "disk_usage", usage)
    assert bool(storage.recording_space_warning(tmp_path, 4)) is expected
    assert checked == [tmp_path]


def test_higher_configured_reserve_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: SimpleNamespace(free=15_000_000_000))
    assert storage.recording_space_warning(tmp_path, 20)
    assert storage.DISK_CHECK_INTERVAL_S == 10


def test_unavailable_output_filesystem_disables_video(monkeypatch, tmp_path):
    def unavailable(_path):
        raise OSError("drive disconnected")

    monkeypatch.setattr(storage.shutil, "disk_usage", unavailable)
    assert "drive disconnected" in storage.recording_space_warning(tmp_path, 4)
