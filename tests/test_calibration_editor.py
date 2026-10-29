from __future__ import annotations

from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from macaque_tracker import calibration_editor
from macaque_tracker.calibration_editor import ControlImageEditor
from macaque_tracker.config import TrackerConfig
from macaque_tracker.models import EyeFitSettings, EyeImageSettings
from macaque_tracker.roi_tool import EyeTuningEditor

cv2 = pytest.importorskip("cv2")


def labelled_editor():
    editor = ControlImageEditor(2)
    editor.capture(np.full((50, 80), 120, np.uint8))
    editor.paint((30, 25), (40, 25))
    editor.capture(np.full((50, 80), 160, np.uint8))
    editor.no_pupil()
    return editor


def test_capture_paint_erase_and_explicit_no_pupil_preserve_source_pixels():
    source = np.full((50, 80), 120, np.uint8)
    editor = ControlImageEditor(1)
    editor.capture(source)
    original = editor.sample
    source[:] = 250
    assert np.all(editor.sample.image == 120)
    with pytest.raises(ValueError, match="mark No pupil"):
        editor.export()
    editor.brush = 3
    editor.paint((10, 20), (60, 20))
    assert editor.sample.image is original.image
    assert not np.any(original.mask)
    assert np.all(editor.sample.mask[20, 10:61] == 255)
    assert np.all(editor.sample.image == 120)
    editor.paint((25, 20), (45, 20), erase=True)
    assert np.all(editor.sample.mask[20, 25:46] == 0)
    editor.no_pupil()
    assert not np.any(editor.export()[0][0].mask)
    editor.live()
    assert editor.sample is None
    editor.browse(1)
    assert editor.index == 0
    editor.delete()
    assert editor.index is None and editor.export() == ((),)


def test_more_than_six_controls_are_accepted_independently_per_eye():
    editor = ControlImageEditor(2)
    image = np.zeros((30, 40), np.uint8)
    editor.capture(image)
    editor.no_pupil()
    with pytest.raises(ValueError, match="at least 2"):
        editor.optimize(EyeImageSettings(), TrackerConfig())
    for _ in range(7):
        editor.capture(image)
        editor.no_pupil()
    with pytest.raises(ValueError, match="painted pupil"):
        editor.optimize(EyeImageSettings(), TrackerConfig())
    editor.select_eye(1)
    editor.capture(image)
    assert len(editor.samples[0]) == 8 and len(editor.samples[1]) == 1


def test_capture_validates_memory_budget_before_retaining_another_image(monkeypatch):
    editor = ControlImageEditor(1)
    image = np.zeros((30, 40), np.uint8)
    editor.capture(image)

    def full(samples):
        assert len(samples) == 2
        raise ValueError("Control images exceed memory budget")

    monkeypatch.setattr(calibration_editor, "validate_control_samples", full)
    with pytest.raises(ValueError, match="memory budget"):
        editor.capture(image)
    assert len(editor.samples[0]) == 1 and editor.index == 0


def test_optimization_runs_off_ui_thread_and_applies_only_when_polled(monkeypatch):
    editor = labelled_editor()
    started, finish = Event(), Event()
    initial = EyeImageSettings(gain=4.0, brightness=0.3)
    result = SimpleNamespace(settings=replace(initial, gain=1.2), before_score=0.4,
                             after_score=0.95, sample_scores=(0.9, 1.0), evaluations=20)

    def optimize(samples, settings, tracker_config, *, cancelled):
        assert settings == initial and isinstance(tracker_config, TrackerConfig)
        assert np.all(samples[0].image == 120)  # Original camera pixels, not gain-adjusted.
        assert np.any(samples[0].mask) and not np.any(samples[1].mask)
        started.set()
        assert finish.wait(2)
        assert not cancelled()
        return result

    monkeypatch.setattr(calibration_editor, "optimize_calibration", optimize)
    try:
        editor.optimize(initial, TrackerConfig())
        assert started.wait(2)
        assert editor.busy and editor.poll() is None
        with pytest.raises(ValueError, match="Cancel optimization"):
            editor.no_pupil()
        assert editor.results == [None, None]
        finish.set()
        editor._pending[1].result(timeout=2)
        assert editor.poll() == (0, result)
        assert not editor.busy and editor.index == 0
        assert editor.results[0] is result and "40% -> 95%" in editor.status
    finally:
        finish.set()
        editor.close()


def test_cancellation_discards_even_a_completed_worker_result(monkeypatch):
    editor = labelled_editor()
    result = SimpleNamespace(settings=EyeImageSettings(gain=2.0))
    monkeypatch.setattr(calibration_editor, "optimize_calibration", lambda *_a, **_k: result)
    editor.optimize(EyeImageSettings(), TrackerConfig())
    editor._pending[1].result(timeout=2)
    editor.cancel()
    assert editor.poll() is None
    assert editor.results == [None, None]
    assert "settings kept" in editor.status
    editor.close()
    editor.close()


def test_worker_failure_is_reported_without_changing_settings(monkeypatch):
    editor = labelled_editor()

    def fail(*_args, **_kwargs):
        raise ValueError("poor labels")

    monkeypatch.setattr(calibration_editor, "optimize_calibration", fail)
    try:
        editor.optimize(EyeImageSettings(), TrackerConfig())
        with pytest.raises(ValueError, match="poor labels"):
            editor._pending[1].result(timeout=2)
        assert editor.poll() is None
        assert editor.results == [None, None] and "poor labels" in editor.status
    finally:
        editor.close()


def test_raw_live_and_frozen_paint_canvas_ignores_aggressive_image_settings():
    source = np.full((80, 120), 100, np.uint8)
    editor = EyeTuningEditor(
        (source,), (EyeImageSettings(gain=8, contrast=8, brightness=1),),
        TrackerConfig(), frame_source=lambda: (source,),
        maximum_display_width=400, maximum_display_height=320,
    )
    editor.controls.show_prediction = False
    frame = editor._frame()
    x0, y0, _width, _height, scale = editor._panel_geometry[0]
    x = round((x0 + 30 * scale) * editor._frame_scale)
    y = round((y0 + 25 * scale) * editor._frame_scale)
    assert tuple(frame[y, x]) == (100, 100, 100)
    editor.control_action("capture")
    source[:] = 240
    editor._refresh_live_crops()
    frame = editor._frame()
    assert tuple(frame[y, x]) == (100, 100, 100)
    editor._mouse(cv2.EVENT_LBUTTONDOWN, x, y)
    editor._mouse(cv2.EVENT_LBUTTONUP, x, y)
    assert editor.controls.sample.mask[25, 30] == 255
    assert np.all(editor.controls.sample.image == 100)
    editor._mouse(cv2.EVENT_RBUTTONDOWN, x, y)
    editor._mouse(cv2.EVENT_RBUTTONUP, x, y)
    assert editor.controls.sample.mask[25, 30] == 0
    assert not editor.controls.labelled[0][0]
    editor.close()


def test_capture_freezes_displayed_video_frame_without_advancing_source():
    calls = []
    shown = np.full((40, 60), 80, np.uint8)

    def next_frame():
        calls.append(True)
        return (np.full_like(shown, 200),)

    editor = EyeTuningEditor((shown,), (EyeImageSettings(),), TrackerConfig(), frame_source=next_frame)
    editor._frame()
    editor.control_action("capture")
    assert not calls
    assert np.all(editor.controls.sample.image == 80)
    editor.control_action("capture")
    assert len(editor.controls.samples[0]) == 1
    assert "Choose Live" in editor.controls.status
    editor.close()


def test_painting_reuses_predictions_and_does_not_read_camera(monkeypatch):
    image = np.full((80, 120), 150, np.uint8)
    calls, captures = [0, 0], []

    def capture():
        captures.append(True)
        return image, image

    editor = EyeTuningEditor(
        (image, image), (EyeImageSettings(), EyeImageSettings()), TrackerConfig(), frame_source=capture,
    )
    for eye, detector in enumerate(editor.detectors):
        detect = detector.detect_with_mask

        def counted(*args, selected=eye, original=detect, **kwargs):
            calls[selected] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(detector, "detect_with_mask", counted)
    try:
        editor.control_action("capture")
        editor._frame()
        assert calls == [1, 1]
        for offset in range(5):
            editor.controls.paint((30, 25), (40 + offset, 25))
            editor._refresh_live_crops()
            editor._frame()
        assert not captures and calls == [1, 1]
        editor.controls.no_pupil()
        editor._frame()
        assert calls == [1, 1]
        editor.settings[0] = replace(editor.settings[0], gain=2)
        editor._frame()
        assert calls == [2, 1]
        editor.control_action("live")
        editor._refresh_live_crops()
        editor._frame()
        assert captures == [True] and calls == [3, 2]
    finally:
        editor.close()


@pytest.mark.parametrize("action", ["live", "eye1", "delete"])
def test_live_preview_immediately_reacquires_after_annotation_pause(action):
    before = np.full((160, 240), 185, np.uint8)
    after = before.copy()
    cv2.ellipse(before, (40, 80), (22, 16), 0, 0, 360, 25, -1)
    cv2.ellipse(after, (200, 80), (22, 16), 0, 0, 360, 25, -1)
    editor = EyeTuningEditor(
        (before, before), (EyeImageSettings(), EyeImageSettings()), TrackerConfig(),
        frame_source=lambda: (after, after),
    )
    try:
        editor._frame()
        assert all(editor._prediction(eye, crop)[0].x == pytest.approx(40, abs=1)
                   for eye, crop in enumerate(editor.crops))
        editor.control_action("capture")
        editor._frame()
        editor.control_action(action)
        editor._refresh_live_crops()
        editor._frame()
        assert all(editor._prediction(eye, crop)[0].x == pytest.approx(200, abs=1)
                   for eye, crop in enumerate(editor.crops))
    finally:
        editor.close()


def test_other_eye_is_marked_paused_while_annotating(monkeypatch):
    text, put_text = [], cv2.putText

    def record(image, label, *args, **kwargs):
        text.append(label)
        return put_text(image, label, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", record)
    image = np.full((80, 120), 150, np.uint8)
    editor = EyeTuningEditor((image, image), (EyeImageSettings(), EyeImageSettings()), TrackerConfig())
    try:
        editor.control_action("capture")
        editor._frame()
        assert any(label.startswith("EYE 0  CONTROL") for label in text)
        assert any(label.startswith("EYE 1  PAUSED") for label in text)
    finally:
        editor.close()


@pytest.mark.parametrize("outside", [False, True])
def test_mouse_commits_press_and_final_segment_with_no_intermediate_motion(outside):
    image = np.full((80, 120), 100, np.uint8)
    editor = EyeTuningEditor((image,), (EyeImageSettings(),), TrackerConfig())
    try:
        editor.control_action("capture")
        editor.controls.brush = 1
        editor._frame()
        x0, y0, _width, _height, scale = editor._panel_geometry[0]

        def pointer(x, y):
            return (round((x0 + x * scale) * editor._frame_scale),
                    round((y0 + y * scale) * editor._frame_scale))

        editor._mouse(cv2.EVENT_LBUTTONDOWN, *pointer(10, 25))
        assert editor.controls.sample.mask[25, 10] == 255
        editor._mouse(cv2.EVENT_LBUTTONUP, *pointer(140 if outside else 70, 25))
        last = 120 if outside else 71
        assert np.all(editor.controls.sample.mask[25, 10:last] == 255)
        editor._mouse(cv2.EVENT_MOUSEMOVE, *pointer(50, 50))
        assert editor.controls.sample.mask[50, 50] == 0
        assert editor._paint_point is None
        editor._mouse(cv2.EVENT_RBUTTONDOWN, *pointer(10, 25))
        editor._mouse(cv2.EVENT_RBUTTONUP, *pointer(70, 25))
        assert not np.any(editor.controls.sample.mask[25, 10:71])
        editor._mouse(cv2.EVENT_LBUTTONDOWN, *pointer(20, 50))
        # A native-window release lost outside its bounds must not turn hover
        # into a new stroke when the cursor re-enters without a button held.
        editor._mouse(cv2.EVENT_MOUSEMOVE, *pointer(80, 50), flags=0)
        assert editor.controls.sample.mask[50, 80] == 0
        assert editor._paint_point is None
    finally:
        editor.close()


def test_completed_optimization_updates_only_selected_eye_and_rebuilds_its_detector(monkeypatch):
    settings = EyeImageSettings(gain=1.5, fit=EyeFitSettings(10, 0.6, 0.5, 10, 0.7))
    result = SimpleNamespace(settings=settings, before_score=0.4, after_score=0.9,
                             sample_scores=(0.8, 1.0), evaluations=20)
    monkeypatch.setattr(calibration_editor, "optimize_calibration", lambda *_a, **_k: result)
    crop = np.full((50, 80), 120, np.uint8)
    other = EyeImageSettings(gain=2.5)
    editor = EyeTuningEditor((crop, crop), (EyeImageSettings(), other), TrackerConfig())
    editor.controls = labelled_editor()
    try:
        editor.control_action("optimize")
        editor.controls._pending[1].result(timeout=2)
        assert editor.settings == [EyeImageSettings(), other]
        editor._poll_optimization()
        assert editor.settings == [settings, other]
        assert editor.detectors[0].config.min_confidence == 0.7
        assert editor.detectors[1].config.min_confidence == TrackerConfig().min_confidence
        assert editor.controls.index == 0
    finally:
        editor.close()


def test_reset_refuses_to_interrupt_optimization_or_request_confirmation():
    image = np.full((50, 80), 120, np.uint8)
    editor = EyeTuningEditor((image,), (EyeImageSettings(),), TrackerConfig())
    editor.controls.capture(image)
    original = editor.controls.sample
    editor.controls._pending = object()
    try:
        editor.control_action("reset", confirm_reset=lambda: pytest.fail("Busy reset must not open a dialog"))
        assert editor.controls.sample is original
        assert "Cancel optimization" in editor.controls.status
    finally:
        editor.controls._pending = None
        editor.close()


def test_reopening_one_eye_keeps_inactive_bank_ready_for_second_eye():
    original = labelled_editor()
    original.select_eye(1)
    original.capture(np.full((60, 100), 155, np.uint8))
    original.no_pupil()
    reopened = ControlImageEditor(1, original.export())
    assert [len(eye) for eye in reopened.samples] == [2, 1]
    reopened.set_eye_count(2)
    reopened.select_eye(1)
    reopened.browse(1)
    reopened.paint((20, 20), (35, 20))
    assert len(reopened.results) == 2
    assert np.all(reopened.sample.image == 155)
    assert reopened.labelled[1] == [True]
    assert np.any(reopened.sample.mask)
