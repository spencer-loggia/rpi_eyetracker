from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from macaque_tracker.config import ConfigError, RoiLayout
from macaque_tracker.graphical_sync import SyncBox, SyncDetector, SyncRoi, sync_signal
from macaque_tracker.models import EyeImageSettings, PixelRoi

PERIOD = 16_666_667


def edges(signals, threshold=40):
    detector = SyncDetector(threshold, 60)
    return [event for i, value in enumerate(signals)
            if (event := detector.process(value, i, (i + 1) * PERIOD)) is not None]


@pytest.mark.parametrize("boxes", [
    (PixelRoi(3, 2, 5, 4), PixelRoi(20, 12, 1, 1)),
    (PixelRoi(24, 12, 7, 6), PixelRoi(3, 2, 4, 3)),
    (PixelRoi(2, 15, 8, 2), PixelRoi(2, 2, 8, 2)),
])
def test_arbitrary_unequal_boxes_reject_common_ambient_change_without_copying(boxes):
    first, second = boxes
    gray = np.full((20, 40), 10, np.uint8)
    gray[first.y:first.y2, first.x:first.x2] = 40
    gray[second.y:second.y2, second.x:second.x2] = 120
    before = gray.copy()
    assert sync_signal(gray, boxes) == -80
    assert sync_signal(gray + 50, boxes) == -80
    assert np.array_equal(gray, before)
    gray[first.y:first.y2, first.x:first.x2] = 120
    gray[second.y:second.y2, second.x:second.x2] = 40
    assert sync_signal(gray, boxes) == 80


def test_initial_light_box_zero_is_baseline_then_detects_both_swaps():
    boxes = (PixelRoi(20, 10, 8, 3), PixelRoi(2, 2, 4, 5))
    detector = SyncDetector(40, 60)
    events = []
    for sequence, brightness in enumerate((180, 20, 180)):
        image = np.zeros((20, 40), np.uint8)
        for box, value in zip(boxes, (brightness, 200 - brightness), strict=True):
            image[box.y:box.y2, box.x:box.x2] = value
        events.append(detector.process(sync_signal(image, boxes), sequence, (sequence+1)*PERIOD))
    assert events[0] is None
    assert [event.direction for event in events[1:]] == ["falling", "rising"]


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
        SyncRoi((SyncBox(0, 0, .2, .2), SyncBox(.4, .2, .1, .1)), threshold)
    with pytest.raises(ValueError, match="threshold"):
        SyncDetector(threshold, 60)


def test_optional_sync_configuration_round_trips_without_changing_eye_settings(tmp_path):
    box = PixelRoi(100, 50, 80, 60)
    eye_settings = EyeImageSettings(gain=1.2, brightness=.1)
    old = RoiLayout.from_pixel_boxes((box,), (eye_settings,), 1000, 500)
    path = old.save(tmp_path / "rois.json")
    assert RoiLayout.load(path).sync_roi is None
    boxes = (PixelRoi(400, 20, 101, 50), PixelRoi(50, 350, 20, 30))
    sync = SyncRoi.from_pixels(boxes, 1000, 500, 70)
    layout = replace(old, sync_roi=sync)
    layout.save(path)
    assert RoiLayout.load(path) == layout
    assert sync.to_pixels(1000, 500) == boxes
    assert layout.rois == old.rois
    layout.validate_for_frame(1000, 500)


def test_legacy_split_box_migrates_at_source_grid_and_saves_as_two_boxes(tmp_path):
    eye = PixelRoi(100, 50, 80, 60)
    layout = RoiLayout.from_pixel_boxes((eye,), (EyeImageSettings(),), 1000, 500)
    path = layout.save(tmp_path / "rois.json")
    raw = json.loads(path.read_text())
    raw["sync_roi"] = {"x": .4, "y": .04, "width": .101, "height": .1, "threshold": 70.04}
    path.write_text(json.dumps(raw))
    migrated = RoiLayout.load(path)
    assert migrated.sync_roi.threshold == 70.04
    assert migrated.sync_roi.to_pixels(1000, 500) == (
        PixelRoi(400, 20, 50, 50), PixelRoi(450, 20, 51, 50),
    )
    assert migrated.rois == layout.rois
    migrated.save(path)
    saved = json.loads(path.read_text())["sync_roi"]
    assert set(saved) == {"boxes", "threshold"}
    assert len(saved["boxes"]) == 2
    assert RoiLayout.load(path) == migrated


@pytest.mark.parametrize("count", [0, 1, 3])
def test_sync_settings_require_exactly_two_boxes(count):
    with pytest.raises(ValueError, match="exactly two"):
        SyncRoi((SyncBox(0, 0, .1, .1),) * count)


@pytest.mark.parametrize("sync", [
    [], True,
    {"boxes": [{"x": 0, "y": 0, "width": .1, "height": .1}]},
    {"boxes": [{"x": -1, "y": 0, "width": .1, "height": .1}] * 2},
    {"boxes": [{"x": 0, "y": 0, "width": .1, "height": .1}] * 2, "threshold": 0},
])
def test_invalid_sync_geometry_or_threshold_cannot_be_loaded(tmp_path, sync):
    layout = RoiLayout.from_pixel_boxes((PixelRoi(1, 1, 30, 30),), (EyeImageSettings(),), 100, 50)
    path = layout.save(tmp_path / "rois.json")
    raw = json.loads(path.read_text())
    raw["sync_roi"] = sync
    path.write_text(json.dumps(raw))
    with pytest.raises(ConfigError):
        RoiLayout.load(path)
