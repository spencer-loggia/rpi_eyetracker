"""Exercise calibrated rotation from raw capture pixels through pupil UART output."""

from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from macaque_tracker import benchmark
from macaque_tracker.camera import _analysis_crops
from macaque_tracker.config import (
    AppConfig,
    CameraConfig,
    RecordingConfig,
    RoiLayout,
    TrackerConfig,
)
from macaque_tracker.models import AnalysisFrame, NormalizedRoi, PixelRoi
from macaque_tracker.protocol import StatusSnapshot, decode_status, encode_status
from macaque_tracker.recording import recording_layout
from macaque_tracker.tracker import MultiEyeTracker


def _layout(angle: int) -> RoiLayout:
    """Place a nonsquare eye box off-center in the third physical camera."""
    roi = NormalizedRoi.from_pixels(
        eye_id=0, label="eye", roi=PixelRoi(640, 40, 220, 140),
        frame_width=1200, frame_height=200,
    )
    return RoiLayout((roi,), 1200, 200, camera_rotations=(0, 0, angle, 0))


def _frame(x: int, y: int) -> np.ndarray:
    """Draw an asymmetric pupil location measured in the raw eye crop."""
    image = np.full((200, 1200), 185, dtype=np.uint8)
    center = (640 + x, 40 + y)
    cv2.ellipse(image, center, (45, 32), 13, 0, 360, 70, -1)
    cv2.ellipse(image, center, (21, 15), 13, 0, 360, 24, -1)
    return image


@pytest.mark.parametrize("angle,shape,positions", [
    (0, (140, 220), [(80, 65), (92, 65), (80, 74)]),
    (90, (220, 140), [(74, 80), (74, 92), (65, 80)]),
    (180, (140, 220), [(139, 74), (127, 74), (139, 65)]),
    (270, (220, 140), [(65, 139), (65, 127), (74, 139)]),
])
def test_calibrated_rotation_preserves_pupil_displacement_and_uart_axes(angle, shape, positions):
    """Quarter turns swap/negate motion components without gain or scaling changes."""
    layout = _layout(angle)
    tracker = MultiEyeTracker((0,), TrackerConfig())
    measured = []
    diameters = []
    try:
        for index, center in enumerate(((80, 65), (92, 65), (80, 74)), start=1):
            raw = _frame(*center)
            original = raw.copy()
            crops = _analysis_crops(raw, layout)
            assert crops[0][1].shape == shape
            assert crops[0][1].flags.c_contiguous
            assert not np.shares_memory(crops[0][1], raw)
            assert np.array_equal(raw, original)
            result = tracker.process(AnalysisFrame(index, index, crops))
            snapshot = StatusSnapshot(True, False, True, result)
            received = decode_status(encode_status(snapshot, now_ns=result.processed_monotonic_ns))
            eye = received.eyes[0]
            assert eye.valid and not eye.blink
            assert (eye.x, eye.y) == pytest.approx(positions[index - 1], abs=0.75)
            measured.append((eye.x, eye.y))
            diameters.append(eye.pupil_diameter)
    finally:
        tracker.close()

    measured = np.asarray(measured)
    expected = np.asarray(positions)
    assert measured[1:] - measured[0] == pytest.approx(expected[1:] - expected[0], abs=0.5)
    assert np.ptp(diameters) < 0.5


@pytest.mark.parametrize("packed", [False, True])
def test_offline_benchmark_uses_live_rotation_once(tmp_path, monkeypatch, packed):
    """Recorded oriented montages must first unpack, then use the live crop transform."""
    config = AppConfig(
        camera=CameraConfig(
            sensor_width=1200, sensor_height=200, analysis_width=1200, analysis_height=200,
        ),
        recording=RecordingConfig(width=1200, height=200),
    )
    layout = _layout(90)
    image = _frame(80, 65)
    path = tmp_path / "example.mkv"
    if packed:
        mapping = recording_layout(config.recording, layout)
        image = mapping.pack(image)[:mapping.height]
        path.with_suffix(".jsonl").write_text(json.dumps({"video_layout": mapping.as_dict()}))

    class Capture:
        """Supply exactly one decoded frame without relying on host video codecs."""

        def isOpened(self):
            return True

        def read(self):
            return True, image

        def release(self):
            pass

    results = []

    class InspectingTracker(MultiEyeTracker):
        def process(self, frame, dropped_frames=0):
            assert frame.crops[0][1].shape == (220, 140)
            result = super().process(frame, dropped_frames)
            results.append(result)
            return result

    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: Capture())
    monkeypatch.setattr(benchmark, "MultiEyeTracker", InspectingTracker)
    report = benchmark.benchmark_video(path, config, layout, maximum_frames=1)

    assert report["valid_fraction"] == {"0": 1.0}
    assert (results[0].eyes[0].x, results[0].eyes[0].y) == pytest.approx((74, 80), abs=0.75)
