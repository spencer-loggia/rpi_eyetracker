from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from macaque_tracker.config import ConfigError, RoiLayout
from macaque_tracker.graphical_sync import SyncDetector, SyncRoi, sync_signal
from macaque_tracker.models import EyeImageSettings, PixelRoi

PERIOD = 16_666_667


def edges(signals, threshold=40):
    detector = SyncDetector(threshold, 60)
    return [event for i, value in enumerate(signals)
            if (event := detector.process(value, i, (i + 1) * PERIOD)) is not None]


@pytest.mark.parametrize("width", [2, 5, 20])
def test_left_right_signal_rejects_common_ambient_change_without_copying(width):
    roi = PixelRoi(3, 2, width, 4)
    gray = np.full((8, width + 6), 10, np.uint8)
    gray[2:6, 3:3 + width // 2] = 40
    gray[2:6, 3 + width // 2:3 + width] = 120
    before = gray.copy()
    assert sync_signal(gray, roi) == -80
    assert sync_signal(gray + 50, roi) == -80
    assert np.array_equal(gray, before)
    gray[2:6, 3:3 + width // 2] = 120
    gray[2:6, 3 + width // 2:3 + width] = 40
    assert sync_signal(gray, roi) == 80


@pytest.mark.parametrize("pulse_frames", [3, 12])
def test_50_to_200ms_pulses_emit_exact_sensor_timed_swaps(pulse_frames):
    signals = [-100] * 3 + [100] * pulse_frames + [-100] * 3
    events = edges(signals)
    assert [e.direction for e in events] == ["rising", "falling"]
    assert [e.frame_sequence for e in events] == [3, 3 + pulse_frames]
    for event in events:
        assert event.camera_time_ns == (event.frame_sequence + 1) * PERIOD
        assert event.previous_camera_time_ns == event.frame_sequence * PERIOD
        assert event.delta == event.signal - event.previous_signal


def test_startup_constant_drift_and_same_polarity_steps_do_not_emit():
    assert edges([100] * 10) == []
    assert edges(range(-100, 101, 2)) == []
    assert edges([-200, -100, -10, -100, -200]) == []


def test_opposite_edges_on_adjacent_frames_have_no_refractory_delay():
    assert [e.direction for e in edges([-100, 100, -100, 100])] == [
        "rising", "falling", "rising",
    ]


def test_zero_contrast_exposure_is_bridged_without_duplicate_edges():
    events = edges([-100, 0, 100, 200, 200, 0, -100, -200])
    assert [(e.frame_sequence, e.direction) for e in events] == [(2, "rising"), (6, "falling")]
    assert edges([0, 0, 100]) == []


@pytest.mark.parametrize("sequence,timestamp", [
    (2, 2 * PERIOD),  # discarded source sample
    (1, 3 * PERIOD),  # missing sensor frame despite consecutive capture calls
    (1, PERIOD),     # duplicate timestamp
    (1, 0),          # changed time availability
    (0, 2 * PERIOD), # seek/restart
])
def test_discontinuity_resets_baseline_without_fabricating_an_edge(sequence, timestamp):
    detector = SyncDetector(40, 60)
    assert detector.process(-100, 0, PERIOD) is None
    assert detector.process(100, sequence, timestamp) is None
    next_time = timestamp + PERIOD if timestamp else 0
    event = detector.process(-100, sequence + 1, next_time)
    assert event is not None and event.direction == "falling"


def test_sequence_wrap_and_unknown_sensor_time_are_not_host_timestamps():
    detector = SyncDetector(40, 60)
    assert detector.process(-100, 0xFFFFFFFF, 0) is None
    event = detector.process(100, 0, 0)
    assert event is not None
    assert event.camera_time_ns == event.previous_camera_time_ns == 0
    detector.reset()
    assert detector.process(-100, 1, 0) is None


def test_threshold_change_controls_detection_and_extreme_signed_delta():
    assert edges([-30, 30], threshold=61) == []
    assert edges([-30, 30], threshold=60)[0].delta == 60
    assert edges([-255, 255], threshold=510)[0].delta == 510


@pytest.mark.parametrize("threshold", [0, -1, 511, True, "20", float("nan"), float("inf")])
def test_invalid_thresholds_are_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        SyncRoi(0, 0, .2, .2, threshold)
    with pytest.raises(ValueError, match="threshold"):
        SyncDetector(threshold, 60)


def test_optional_sync_configuration_round_trips_without_changing_eye_settings(tmp_path):
    box = PixelRoi(100, 50, 80, 60)
    eye_settings = EyeImageSettings(gain=1.2, brightness=.1)
    old = RoiLayout.from_pixel_boxes((box,), (eye_settings,), 1000, 500)
    path = old.save(tmp_path / "rois.json")
    assert RoiLayout.load(path).sync_roi is None
    sync = SyncRoi.from_pixels(PixelRoi(400, 20, 101, 50), 1000, 500, 70)
    layout = replace(old, sync_roi=sync)
    layout.save(path)
    assert RoiLayout.load(path) == layout
    assert sync.to_pixels(1000, 500) == PixelRoi(400, 20, 101, 50)
    assert layout.rois == old.rois
    layout.validate_for_frame(1000, 500)
    replace(layout, sync_roi=SyncRoi(.4, .1, .001, .1)).save(path)
    with pytest.raises(ConfigError, match="two analysis pixels"):
        RoiLayout.load(path).validate_for_frame(1000, 500)
