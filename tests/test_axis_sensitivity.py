"""Check image-space axis response without assumptions about animal behavior."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from macaque_tracker.config import TrackerConfig
from macaque_tracker.models import FrameResult
from macaque_tracker.protocol import StatusSnapshot, decode_status, encode_status
from macaque_tracker.tracker import TemporalEyeTracker


def _eye_at(x: float, y: float, *, noise: float, glint: bool) -> np.ndarray:
    """Render a translated elliptical pupil with reproducible sensor noise."""
    yy, xx = np.mgrid[:200, :280]
    image = np.full(xx.shape, 185.0)
    image[((xx - x) / 53) ** 2 + ((yy - y) / 40) ** 2 <= 1] = 70
    image[((xx - x) / 28) ** 2 + ((yy - y) / 21) ** 2 <= 1] = 24
    if glint:
        # An edge reflection exercises the stricter ellipse-recovery path.
        image[(xx - x - 24) ** 2 + (yy - y) ** 2 <= 10**2] = 255
    image += np.random.default_rng(17).normal(0.0, noise, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("noise", [0.0, 3.0])
@pytest.mark.parametrize("glint", [False, True])
def test_equal_horizontal_and_vertical_motion_survives_tracking_and_uart(
    noise: float, glint: bool,
) -> None:
    """Both default tracking axes must retain one output pixel per input pixel."""
    trackers = [TemporalEyeTracker(eye_id, TrackerConfig()) for eye_id in (0, 1)]
    offsets = np.linspace(-16, 16, 17)
    measured = []
    for frame_number, offset in enumerate(offsets, start=1):
        eyes = (
            trackers[0].process(_eye_at(140 + offset, 100, noise=noise, glint=glint)),
            trackers[1].process(_eye_at(140, 100 + offset, noise=noise, glint=glint)),
        )
        assert all(eye.valid for eye in eyes)
        result = FrameResult(frame_number, frame_number, frame_number, eyes)
        snapshot = StatusSnapshot(tracking=True, recording=False, configured=True, result=result)
        received = decode_status(encode_status(snapshot, now_ns=frame_number)).eyes
        for sent, decoded in zip(eyes, received, strict=True):
            assert (decoded.x, decoded.y) == pytest.approx(
                (sent.x, sent.y), abs=0.5 / 65536.0,
            )
        measured.append((received[0].x - 140, received[1].y - 100))

    measured = np.asarray(measured)
    gains = np.polyfit(offsets, measured, 1)[0]
    assert gains == pytest.approx((1.0, 1.0), abs=0.025)
    assert gains[0] == pytest.approx(gains[1], abs=0.025)
    assert np.max(np.abs(measured - offsets[:, None])) < 1.5


@pytest.mark.parametrize("glint", [False, True])
def test_swapping_image_axes_preserves_detection(glint: bool) -> None:
    """Rectangular crops and glint recovery must not favor the horizontal axis."""
    image = _eye_at(146, 94, noise=3.0, glint=glint)
    original = TemporalEyeTracker(0, TrackerConfig()).process(image)
    transposed = TemporalEyeTracker(0, TrackerConfig()).process(image.T.copy())

    assert original.valid and transposed.valid
    assert (transposed.x, transposed.y) == pytest.approx(
        (original.y, original.x), abs=0.25,
    )
    assert transposed.pupil_diameter == pytest.approx(original.pupil_diameter, abs=0.25)
