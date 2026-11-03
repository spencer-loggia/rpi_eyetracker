from types import SimpleNamespace

import numpy as np
import pytest

from macaque_tracker.calibration_video import CalibrationVideo
from macaque_tracker.models import PixelRoi
from macaque_tracker.pi_runtime import CameraFeed
from macaque_tracker.sync_calibration import SyncCalibration

BOXES = (PixelRoi(1, 1, 4, 5), PixelRoi(10, 2, 6, 2))


def pattern(left, right):
    image = np.full((8, 16), right, np.uint8)
    image[:, :8] = left
    return image


def test_sync_threshold_replays_bounded_scalar_history_and_preserves_both_raw_crops():
    sampling = SyncCalibration(BOXES, 500, 60)
    images = [pattern(30, 170), pattern(170, 30), pattern(30, 170)]
    for sequence, image in enumerate(images):
        sampling.observe(image, sequence, (sequence + 1) * 16_666_667)
    assert sampling.snapshot()[4:6] == (0, 0)
    sampling.reset(threshold=200)
    crops, signal, delta, event, rising, falling, samples = sampling.snapshot()
    assert (signal, delta, event.direction, rising, falling, samples) == (
        -140, -280, "falling", 1, 1, 3,
    )
    for crop, box in zip(crops, BOXES, strict=True):
        assert np.array_equal(crop, box.extract(images[-1]))
    images[-1][:] = 0
    assert all(np.any(crop) for crop in crops)
    for sequence in range(3, 400):
        sampling.observe(pattern(30, 170), sequence, (sequence + 1) * 16_666_667)
    assert len(sampling._history) == 256
    sampling.reset(threshold=210)
    assert sampling.snapshot()[4:6] == (0, 0)  # Pulses left the bounded replay window.


def test_sampler_requires_both_boxes():
    with pytest.raises(ValueError, match="two boxes"):
        SyncCalibration(BOXES[:1], 20, 60)


def test_video_observer_sees_intermediate_frames_with_recording_timestamps(monkeypatch):
    frames = [pattern(30, 170), pattern(170, 30), pattern(30, 170), pattern(170, 30)]

    class Camera:
        def __init__(self, *_args, **_kwargs):
            self.frame_index = -1
            self.frame_count = len(frames)
            self.video_fps = 30
            self.closed = False

        def start(self):
            pass

        def capture_preview(self):
            self.frame_index += 1
            return frames[self.frame_index], (self.frame_index + 1) * 33_333_333

        def close(self):
            self.closed = True

    monkeypatch.setattr("macaque_tracker.calibration_video.VideoFileCamera", Camera)
    monkeypatch.setattr("macaque_tracker.calibration_video.time.monotonic", lambda: 10)
    video = CalibrationVideo(SimpleNamespace(fps=60), None)
    video.load("synthetic")
    camera = video.camera
    sampling = SyncCalibration(BOXES, 200, camera.video_fps)
    video.observer = sampling.observe
    sampling.observe(video.image, camera.frame_index, video.timestamp)
    video.playing = True
    assert np.array_equal(video.latest(), frames[-1])
    assert sampling.snapshot()[4:] == (2, 1, 4)
    assert video.observer is not None
    video.close()
    assert video.observer is None and camera.closed


def test_camera_feed_observes_each_capture_without_menu_rate_wait():
    class Camera:
        def __init__(self):
            self.sequence = 0

        def capture_preview(self):
            self.sequence += 1
            return pattern(30, 170), self.sequence * 16_666_667

    feed = CameraFeed(Camera())
    calls, waits = [], []
    feed._stop.wait = waits.append

    def observe(_image, sequence, timestamp):
        calls.append((sequence, timestamp))
        if sequence == 2:
            feed._stop.set()

    feed.set_observer(observe)
    feed._capture()
    assert calls == [(0, 16_666_667), (1, 33_333_334), (2, 50_000_001)]
    assert not waits
    feed.set_observer(None)
    assert feed._observer is None
