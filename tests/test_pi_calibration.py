from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from macaque_tracker.config import AppConfig, CameraConfig, RoiLayout, TrackerConfig
from macaque_tracker.experiments import ExperimentStore
from macaque_tracker.models import EyeImageSettings, NormalizedRoi, PixelRoi
from macaque_tracker.pi_calibration import CalibrationSession
from macaque_tracker.roi_tool import EyeTuningEditor, RoiEditor

cv2 = pytest.importorskip("cv2")


class FakeCamera:
    def __init__(self):
        self.controls = []
        self.fail_controls = False

    def image_control_limits(self):
        return {"exposure_us": (100.0, 40_000.0)}

    def set_image_controls(self, **values):
        if self.fail_controls:
            raise RuntimeError("Exposure unavailable")
        self.controls.append(values)


class FakeFeed:
    def __init__(self):
        self.camera = FakeCamera()
        self.image = np.full((90, 640), 100, np.uint8)

    def latest(self):
        return self.image


@pytest.fixture(autouse=True)
def forbid_opencv_windows(monkeypatch):
    calls = []

    def unexpected_gui(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("Embedded calibration must not access OpenCV windows")

    for name in (
        "namedWindow", "imshow", "waitKey", "destroyWindow", "destroyAllWindows",
        "createTrackbar", "setTrackbarPos", "setMouseCallback", "getWindowProperty",
    ):
        monkeypatch.setattr(cv2, name, unexpected_gui)
    yield
    assert not calls


def _session(tmp_path, *, calibrated=True):
    base_path = tmp_path / "settings" / "config.json"
    config = AppConfig(
        roi_config="base-calibration.json",
        camera=CameraConfig(
            analysis_width=640, analysis_height=90,
            fps=60, exposure_us=10_000,
        ),
        tracker=TrackerConfig(pupil_size_bias=-0.15),
    )
    config.save(base_path)
    if calibrated:
        RoiLayout(
            rois=(
                NormalizedRoi(0, "eye0", 0.55, 0.2, 0.15, 0.4,
                              EyeImageSettings(gain=1.3, brightness=0.1, pupil_size_bias=-0.25)),
                NormalizedRoi(1, "eye1", 0.80, 0.2, 0.15, 0.4,
                              EyeImageSettings(gain=1.7, contrast=1.8, pupil_size_bias=0.25)),
            ),
            source_width=640, source_height=90,
        ).save(base_path.parent / config.roi_config)
    store = ExperimentStore(tmp_path / "data", base_path)
    store.create("subject1", 1_792_000_000_000_000_000)
    experiment = store.create("subject1", 1_792_000_001_000_000_000)
    feed = FakeFeed()
    session = CalibrationSession(AppConfig.load(experiment.config_path), experiment, feed)
    return session, feed


def test_calibration_inherits_current_boxes_and_settings_and_updates_live_images(tmp_path):
    session, feed = _session(tmp_path)
    inherited = RoiLayout.load(session.experiment.calibration_path)
    assert session.phase == "boxes"
    assert isinstance(session.editor, RoiEditor)
    assert session.roi.boxes == [roi.to_pixels(640, 90) for roi in inherited.rois]
    assert session.initial_settings == {roi.eye_id: roi.settings for roi in inherited.rois}
    original = feed.image.copy()
    first = session.render()
    assert first.shape == (90, 640, 3)
    assert np.array_equal(feed.image, original)
    feed.image = np.full((90, 640), 130, np.uint8)
    assert not np.array_equal(session.render(), first)
    session.next()
    assert session.phase == "tuning"
    assert isinstance(session.editor, EyeTuningEditor)
    assert session.tuning.settings == [roi.settings for roi in inherited.rois]
    assert all(np.all(crop == 130) for crop in session.tuning.crops)
    feed.image = np.full((90, 640), 150, np.uint8)
    dashboard = session.render()
    assert dashboard.ndim == 3 and dashboard.shape[2] == 3
    assert all(np.all(crop == 150) for crop in session.tuning.crops)
    assert np.all(feed.image == 150)


def test_calibration_can_draw_first_roi_and_uses_tracker_defaults(tmp_path):
    session, _feed = _session(tmp_path, calibrated=False)
    assert session.roi.boxes == []
    with pytest.raises(ValueError, match="one or two"):
        session.next()
    session.mouse(cv2.EVENT_LBUTTONDOWN, 360, 20)
    session.mouse(cv2.EVENT_MOUSEMOVE, 460, 65)
    session.mouse(cv2.EVENT_LBUTTONUP, 460, 65)
    assert session.roi.boxes == [PixelRoi(360, 20, 101, 46)]
    session.next()
    assert session.tuning.settings == [EyeImageSettings(pupil_size_bias=-0.15)]
    selected = session.selected
    session.mouse(cv2.EVENT_LBUTTONDOWN, 500, 20)
    session.mouse(cv2.EVENT_LBUTTONUP, 600, 65)
    assert session.selected == selected  # Pupil tuning does not redraw ROIs.


def test_calibration_back_next_preserves_independent_eye_adjustments(tmp_path):
    session, _feed = _session(tmp_path)
    session.next()
    controls = list(session.sliders())
    assert len(controls) == 8
    for label, slider, _value, callback in controls:
        if label == "Eye 0: gain":
            callback(slider.position_for(2.2))
        elif label == "Eye 1: contrast":
            callback(slider.position_for(2.6))
        elif label == "Eye 1: pupil size bias":
            callback(slider.position_for(-0.5))
    tuned = list(session.tuning.settings)
    assert tuned[0].gain == 2.2
    assert tuned[0].contrast == 1
    assert tuned[1].gain == 1.7
    assert tuned[1].contrast == 2.6
    assert tuned[1].pupil_size_bias == -0.5
    session.previous()
    assert session.phase == "boxes" and session.editor is session.roi
    session.roi.boxes[0] = PixelRoi(350, 20, 100, 40)
    session.next()
    assert session.tuning.settings == tuned
    assert session.selected[0] == PixelRoi(350, 20, 100, 40)


def test_calibration_exposure_uses_live_camera_and_blocks_unapplied_changes(tmp_path):
    session, feed = _session(tmp_path)
    controls = list(session.sliders())
    assert len(controls) == 1
    _label, slider, value, callback = controls[0]
    assert value == 10_000
    assert slider.minimum == 100 and slider.maximum == 16_666
    callback(slider.position_for(12_500))
    assert feed.camera.controls == [{"exposure_us": 12_500}]
    assert session.roi.applied_camera_config == replace(session.config.camera, exposure_us=12_500)
    feed.camera.fail_controls = True
    callback(slider.position_for(13_000))
    assert session.roi.applied_camera_config.exposure_us == 12_500
    with pytest.raises(ValueError, match="Exposure was not applied"):
        session.next()
    assert session.phase == "boxes"
    feed.camera.fail_controls = False
    callback(slider.position_for(13_000))
    session.next()
    assert session.roi.applied_camera_config.exposure_us == 13_000


def test_calibration_save_updates_only_current_experiment(tmp_path):
    session, _feed = _session(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    with pytest.raises(ValueError, match="Complete pupil tuning"):
        session.save()
    _label, slider, _value, callback = next(session.sliders())
    callback(slider.position_for(12_500))
    session.roi.boxes[0] = PixelRoi(350, 20, 100, 40)
    session.next()
    session.tuning.settings[0] = replace(session.tuning.settings[0], gain=2.0, pupil_size_bias=0.4)
    session.save()
    updated = AppConfig.load(session.experiment.config_path)
    layout = RoiLayout.load(session.experiment.calibration_path)
    assert updated.camera.exposure_us == 12_500
    assert updated.recording.directory == str(session.experiment.directory)
    assert layout.source_width == 640 and layout.source_height == 90
    assert layout.rois[0].to_pixels(640, 90) == session.selected[0]
    assert [roi.settings for roi in layout.rois] == session.tuning.settings
    assert session.experiment.config_path.stat().st_mode & 0o777 == 0o666
    assert session.experiment.calibration_path.stat().st_mode & 0o777 == 0o666
    changed = {session.experiment.config_path, session.experiment.calibration_path}
    assert all(path.read_bytes() != content for path, content in before.items() if path in changed)
    assert all(path.read_bytes() == content for path, content in before.items() if path not in changed)


def test_calibration_navigation_without_save_leaves_files_unchanged(tmp_path):
    session, _feed = _session(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    session.next()
    session.tuning.settings[0] = replace(session.tuning.settings[0], gain=2.5)
    session.previous()
    session.roi.boxes.clear()
    session.render()
    assert {path: path.read_bytes() for path in before} == before
