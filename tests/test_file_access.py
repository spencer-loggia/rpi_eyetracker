from __future__ import annotations

import os
import stat

import pytest

from macaque_tracker.file_access import create_shared_file, save_shared_json, share_output


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_shared_outputs_override_restrictive_umask_without_changing_it(tmp_path):
    previous = os.umask(0o077)
    try:
        directory = tmp_path / "experiment"
        directory.mkdir()
        share_output(directory, directory=True)
        video = directory / "video.mkv"
        create_shared_file(video)
        config = save_shared_json(directory / "config.json", {"version": 1})
        save_shared_json(config, {"version": 2})
        assert mode(directory) == 0o777
        assert mode(video) == mode(config) == 0o666
        private = tmp_path / "unrelated.txt"
        private.write_text("private")
        assert mode(private) == 0o600
    finally:
        os.umask(previous)


def test_shared_video_reservation_never_overwrites_existing_file(tmp_path):
    video = tmp_path / "video.mkv"
    video.write_bytes(b"existing recording")
    with pytest.raises(FileExistsError):
        create_shared_file(video)
    assert video.read_bytes() == b"existing recording"


def test_shared_permissions_do_not_follow_symlinks_or_hardlinks(tmp_path):
    target = tmp_path / "private.txt"
    target.write_text("private")
    target.chmod(0o600)
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target)
    with pytest.raises(OSError):
        share_output(symlink)
    hardlink = tmp_path / "hardlink"
    os.link(target, hardlink)
    with pytest.raises(ValueError):
        share_output(hardlink)
    assert mode(target) == 0o600


def test_reusing_shared_directory_does_not_require_chmod_permission(tmp_path, monkeypatch):
    tmp_path.chmod(0o777)

    def denied(*_args):
        raise PermissionError("not the directory owner")

    monkeypatch.setattr(os, "fchmod", denied)
    share_output(tmp_path, directory=True)


def test_failed_shared_json_save_preserves_previous_file_and_removes_temporary(tmp_path):
    config = save_shared_json(tmp_path / "config.json", {"version": 1})
    before = config.read_bytes()
    with pytest.raises(TypeError):
        save_shared_json(config, {"invalid": object()})
    assert config.read_bytes() == before
    assert list(tmp_path.iterdir()) == [config]
    assert mode(config) == 0o666


def test_pyav_muxer_preserves_reserved_video_permissions(tmp_path):
    av = pytest.importorskip("av")
    import numpy as np

    video = tmp_path / "shared.mkv"
    previous = os.umask(0o077)
    try:
        create_shared_file(video)
        with av.open(str(video), "w") as output:
            stream = output.add_stream("libx264", rate=30)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32), dtype=np.uint8), format="gray")
            for packet in stream.encode(frame):
                output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        assert mode(video) == 0o666
        with av.open(str(video)) as recording:
            assert len(list(recording.decode(video=0))) == 1
    finally:
        os.umask(previous)
