from pathlib import Path

import pytest

from macaque_tracker.cli import _parser


def test_preview_accepts_prerecorded_video_source() -> None:
    args = _parser().parse_args(["preview", "--video", "session.mkv"])

    assert args.video == Path("session.mkv")


def test_configure_accepts_video_but_not_video_and_image_together() -> None:
    parser = _parser()
    args = parser.parse_args(["configure", "--video", "session.mkv"])

    assert args.video == Path("session.mkv")
    assert args.image is None
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["configure", "--video", "session.mkv", "--image", "still.png"]
        )


def test_configure_is_live_by_default_and_static_is_explicit() -> None:
    parser = _parser()

    assert not parser.parse_args(["configure"]).static
    assert parser.parse_args(["configure", "--static"]).static
