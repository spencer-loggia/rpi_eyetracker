from __future__ import annotations

from dataclasses import replace
from threading import Event, Thread

import numpy as np
import pytest

from macaque_tracker.config import AppConfig, CameraConfig, RoiLayout, TrackerConfig
from macaque_tracker.experiments import ExperimentStore
from macaque_tracker.models import EyeFitSettings, EyeImageSettings, NormalizedRoi, PixelRoi
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


@pytest.mark.parametrize("operation", ["render", "next"])
def test_uart_close_waits_for_calibration_read_and_busy_redraw_is_skipped(
    tmp_path, monkeypatch, operation,
):
    session, feed = _session(tmp_path)
    reading, release, closing, closed = Event(), Event(), Event(), Event()
    failures = []
    close_video = session.video.close

    def read_frame():
        reading.set()
        assert release.wait(2), "Test did not release the decoder"
        assert not closed.is_set(), "Decoder closed during a calibration read"
        return feed.image

    def close_decoder():
        closed.set()
        close_video()

    def read():
        try:
            getattr(session, operation)()
        except Exception as exc:  # noqa: BLE001 - report worker failures in the test thread
            failures.append(exc)

    def close():
        closing.set()
        session.close()

    session.video.fallback = read_frame
    monkeypatch.setattr(session.video, "close", close_decoder)
    reader, closer = Thread(target=read), Thread(target=close)
    reader.start()
    try:
        assert reading.wait(1)
        closer.start()
        assert closing.wait(1)
        assert not closed.wait(0.05)
        assert session.render() is None  # UI must not wait for a decoder/lifecycle worker.
    finally:
        release.set()
        reader.join(2)
        if closer.ident is not None:
            closer.join(2)
        session.close()
    assert not reader.is_alive() and not closer.is_alive()
    assert not failures
    assert closed.is_set()
    assert session.render() is None  # A stale GUI view cannot reopen a closed session.
    with pytest.raises(RuntimeError, match="closed"):
        session.next()


def test_calibration_inherits_current_boxes_and_settings_and_updates_live_images(tmp_path):
    session, feed = _session(tmp_path)
    inherited = RoiLayout.load(session.experiment.calibration_path)
    assert session.phase == "boxes"
    assert isinstance(session.editor, RoiEditor)
    assert session.roi.boxes == [roi.to_pixels(640, 90) for roi in inherited.rois]
    assert session.initial_settings == {roi.eye_id: roi.settings for roi in inherited.rois}
    original = feed.image.copy()
    first = session.render()
    assert first.shape[0] > 90
    assert first.shape[1:] == (640, 3)
    assert tuple(first[0, 0]) == (100, 100, 100)  # No instruction overlay hides pixels.
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


@pytest.mark.parametrize("tuning", [False, True])
def test_pointer_redraw_does_not_advance_camera_frame(tmp_path, tuning):
    session, feed = _session(tmp_path)
    if tuning:
        session.next()
    captures = []

    def capture():
        captures.append(True)
        return feed.image

    feed.latest = capture
    # The editor stores a bound supplier, so guard those installed at creation.
    if not tuning:
        session.roi._frame_source = capture
    session.render(refresh=False)
    assert captures == []


def test_calibration_back_next_preserves_independent_eye_adjustments(tmp_path):
    session, _feed = _session(tmp_path)
    session.next()
    controls = list(session.sliders())
    assert len(controls) == 6
    for label, slider, _value, callback in controls:
        if label == "Eye 0: Pupil edge threshold":
            callback(slider.position_for(0.6))
        elif label == "Eye 1: Minimum confidence":
            callback(slider.position_for(0.7))
        elif label == "Eye 1: Pupil size (- small, + large)":
            callback(slider.position_for(-0.5))
    tuned = list(session.tuning.settings)
    assert tuned[0].gain == 1.3
    assert tuned[0].fit.threshold_range_fraction == 0.6
    assert tuned[0].contrast == 1
    assert tuned[1].gain == 1.7
    assert tuned[1].contrast == 1.8
    assert tuned[1].fit.min_confidence == 0.7
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


def test_calibration_rotation_persists_raw_boxes_and_rotates_tuning_crops(tmp_path):
    session, feed = _session(tmp_path)
    feed.image = np.arange(640 * 90, dtype=np.uint8).reshape(90, 640)
    original_boxes = list(session.roi.boxes)
    session.roi.rotate_camera(2)
    session.roi.rotate_camera(3)
    session.roi.rotate_camera(3)
    assert session.roi.camera_rotations == (0, 0, 90, 180)
    assert session.roi.boxes == original_boxes
    session.next()
    for index, (box, angle) in enumerate(zip(original_boxes, (90, 180), strict=True)):
        expected = np.rot90(box.extract(feed.image), -angle // 90)
        assert np.array_equal(session.tuning.crops[index], expected)
    session.save()
    saved = RoiLayout.load(session.experiment.calibration_path)
    assert saved.camera_rotations == (0, 0, 90, 180)
    assert [roi.to_pixels(640, 90) for roi in saved.rois] == original_boxes
    reopened = CalibrationSession(session.config, session.experiment, feed)
    assert reopened.roi.camera_rotations == saved.camera_rotations
    reopened.next()
    assert np.array_equal(reopened.tuning.crops[0], session.tuning.crops[0])


def test_pointer_selects_raw_box_through_rotated_camera_view(tmp_path):
    session, _feed = _session(tmp_path, calibrated=False)
    session.roi.rotate_camera(2)
    # Camera 2 raw size 160x90 becomes 90x160, positioned at display x=320.
    session.mouse(cv2.EVENT_LBUTTONDOWN, 330, 30)
    session.mouse(cv2.EVENT_MOUSEMOVE, 370, 100)
    session.mouse(cv2.EVENT_LBUTTONUP, 370, 100)
    assert session.roi.boxes == [PixelRoi(350, 39, 71, 41)]
    session.next()
    assert session.tuning.crops[0].shape == (71, 41)


def test_rotating_then_cancelling_does_not_overwrite_calibration(tmp_path):
    session, _feed = _session(tmp_path)
    original = session.experiment.calibration_path.read_bytes()
    session.roi.rotate_camera(2)
    session.next()
    session.previous()
    assert session.roi.camera_rotations == (0, 0, 90, 0)
    assert session.experiment.calibration_path.read_bytes() == original


def test_raw_controls_persist_reopen_and_keep_original_geometry(tmp_path):
    session, feed = _session(tmp_path)
    feed.image[:] = 130
    session.next()
    tuning = session.tuning
    tuning.control_action("capture")
    tuning.controls.paint((30, 18), (40, 18))
    assert np.all(tuning.controls.sample.image == 130)
    feed.image[:] = 180
    tuning.control_action("live")
    session.render()
    tuning.control_action("capture")
    tuning.control_action("no_pupil")
    session.previous()
    session.next()
    assert len(session.tuning.controls.samples[0]) == 2
    assert session.tuning.controls.labelled[0] == [True, True]
    session.save()
    sidecar = session.experiment.calibration_path.with_suffix(".controls.npz")
    assert sidecar.exists() and sidecar.stat().st_mode & 0o777 == 0o666
    reopened = CalibrationSession(session.config, session.experiment, feed)
    reopened.next()
    samples = reopened.tuning.controls.samples
    assert [len(eye) for eye in samples] == [2, 0]
    assert np.all(samples[0][0].image == 130)
    assert np.any(samples[0][0].mask)
    assert np.all(samples[0][1].image == 180)
    assert not np.any(samples[0][1].mask)
    original = sidecar.read_bytes()
    reopened.previous()
    reopened.roi.rotate_camera(2)
    reopened.next()
    assert reopened.tuning.controls.samples is samples
    assert samples[0][0].image.shape == (36, 96)  # Rotation applies only to new captures.
    assert reopened.tuning.crops[0].shape == (96, 36)
    reopened.close()
    assert sidecar.read_bytes() == original


def test_unfinished_control_requires_explicit_label_before_save(tmp_path):
    session, _feed = _session(tmp_path)
    before = session.experiment.calibration_path.read_bytes()
    session.next()
    session.tuning.control_action("capture")
    with pytest.raises(ValueError, match="mark No pupil"):
        session.save()
    assert session.experiment.calibration_path.read_bytes() == before
    assert not session.experiment.calibration_path.with_suffix(".controls.npz").exists()
    session.close()


def test_corrupt_control_archive_allows_fresh_calibration(tmp_path):
    session, _feed = _session(tmp_path)
    sidecar = session.experiment.calibration_path.with_suffix(".controls.npz")
    sidecar.write_bytes(b"damaged archive")
    session.next()
    assert session.tuning.controls.samples == [[], []]
    assert "recapture" in session.tuning.controls.status
    session.close()
    assert sidecar.read_bytes() == b"damaged archive"


def test_changed_geometry_preserves_all_learned_and_manual_settings(tmp_path):
    session, _feed = _session(tmp_path)
    learned = EyeFitSettings(8, 0.5, 0.45, 8, 0.48)
    session.initial_settings = {
        eye: replace(settings, fit=learned) for eye, settings in session.initial_settings.items()
    }
    session.next()
    assert all(settings.fit == learned for settings in session.tuning.settings)
    original = session.tuning.settings.copy()
    session.previous()
    session.roi.boxes[0] = PixelRoi(350, 20, 100, 40)
    session.next()
    assert session.tuning.settings == original
    session.close()


def test_saved_profile_survives_new_session_and_boxes_until_explicit_adjustment(tmp_path):
    session, feed = _session(tmp_path)
    session.next()
    fit = EyeFitSettings(9.3, 0.85, 0.35, 12, 0.645, 0.445, 0.7, 1.5, 0.12, 0.5)
    session.tuning.settings[0] = EyeImageSettings(0.7, -0.1, 1.2, 0.3, fit)
    session.tuning.settings[1] = EyeImageSettings(1.2, 0.1, 0.9, -0.4, replace(fit, min_contrast=16))
    expected = session.tuning.settings.copy()
    session.tuning.control_action("capture")
    session.tuning.control_action("no_pupil")
    session.save()
    session.close()

    store = ExperimentStore(tmp_path / "data", tmp_path / "settings" / "config.json")
    next_experiment = store.create("subject1", 1_792_000_002_000_000_000)
    reopened = CalibrationSession(AppConfig.load(next_experiment.config_path), next_experiment, feed)
    reopened.roi.boxes = [PixelRoi(350, 20, 100, 40), PixelRoi(520, 15, 110, 60)]
    reopened.next()
    assert reopened.tuning.settings == expected
    assert [len(eye) for eye in reopened.tuning.controls.samples] == [1, 0]
    assert reopened.tuning.controls.samples[0][0].image.shape == (36, 96)
    for label, slider, value, callback in reopened.sliders():
        if label == "Eye 0: Minimum confidence":
            assert value == 0.645  # Reading a rounded slider must not round the saved value.
            callback(slider.position_for(0.7))
    expected[0] = replace(expected[0], fit=replace(fit, min_confidence=0.7))
    assert reopened.tuning.settings == expected
    reopened.save()
    reopened.close()

    third = store.create("subject1", 1_792_000_003_000_000_000)
    third_session = CalibrationSession(AppConfig.load(third.config_path), third, feed)
    third_session.next()
    assert third_session.tuning.settings == expected
    for detector, settings in zip(third_session.tuning.detectors, expected, strict=True):
        assert all(getattr(detector.config, name) == value for name, value in settings.fit.as_dict().items())
    third_session.close()


def test_embedded_annotation_uses_native_buttons_and_preserves_raw_paint_coordinates(tmp_path):
    session, _feed = _session(tmp_path)
    session.next()
    tuning = session.tuning
    tuning.control_action("capture")
    tuning.controls.show_prediction = False
    tuning._frame()
    assert tuning._toolbar  # The standalone editor still draws its own buttons.
    frame = session.render()
    assert tuning._toolbar == []
    x0, y0, _width, _height, scale = tuning._panel_geometry[0]
    x = round((x0 + 30 * scale) * tuning._frame_scale)
    y = round((y0 + 18 * scale) * tuning._frame_scale)
    assert tuple(frame[y, x]) == (100, 100, 100)
    session.mouse(cv2.EVENT_LBUTTONDOWN, x, y)
    session.mouse(cv2.EVENT_LBUTTONUP, x, y)
    assert tuning.controls.sample.mask[18, 30] == 255
    assert np.all(tuning.controls.sample.image == 100)
    session.close()


def _recorded_video(tmp_path):
    path = tmp_path / "blink.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 30, (640, 90), False)
    assert writer.isOpened()
    frames = [np.full((90, 640), 30 + 20 * i, np.uint8) for i in range(8)]
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path, frames


def test_recorded_calibration_steps_exact_raw_frames_and_retains_labels(tmp_path):
    path, frames = _recorded_video(tmp_path)
    session, feed = _session(tmp_path)
    try:
        session.video_action("load_video", path=path)
        assert np.array_equal(session.roi.image, frames[0])
        assert not session.video.playing
        assert list(session.sliders()) == []  # No camera exposure controls on a recording.
        session.next()
        tuning = session.tuning
        controls = tuning.controls
        settings = tuple(tuning.settings)
        tuning.control_action("capture")
        controls.paint((20, 18), (40, 18))
        first = controls.sample
        assert np.all(first.image == 30)  # Inherited aggressive transforms are not painted.
        session.video_action("frame_back")
        assert controls.sample is first and len(controls.samples[0]) == 1
        session.video_action("frame_forward")
        assert len(controls.samples[0]) == 2
        assert np.all(controls.sample.image == 50)
        assert not np.any(controls.sample.mask)
        assert controls.samples[0][0] is first
        session.video_action("frame_forward")
        assert len(controls.samples[0]) == 2  # Move the unlabelled still instead of appending.
        assert np.all(controls.sample.image == 70)
        tuning.control_action("no_pupil")
        blink = controls.sample
        session.video_action("frame_forward")
        assert len(controls.samples[0]) == 3
        assert controls.labelled[0] == [True, True, False]
        assert controls.samples[0][1] is blink
        session.video_action("frame_back")
        assert np.all(controls.sample.image == 70)
        tuning.control_action("delete")
        controls.index = 0
        session.previous()
        session.next()
        tuning = session.tuning
        assert tuning.controls is controls
        session.video_action("frame_forward")
        assert np.all(controls.sample.image == 50)  # Step from the selected original still.
        tuning.control_action("no_pupil")
        session.save()
        assert tuple(tuning.settings) == settings
        assert feed.camera.controls == []
        assert np.all(feed.image == 100)
        camera = session.video.camera
    finally:
        session.close()
    assert not camera.started
    reopened = CalibrationSession(session.config, session.experiment, feed)
    try:
        reopened.next()
        assert [int(s.image[0, 0]) for s in reopened.tuning.controls.samples[0]] == [30, 70, 50]
    finally:
        reopened.close()


def test_video_load_cancel_failure_and_busy_optimization_preserve_current_still(tmp_path, monkeypatch):
    path, _frames = _recorded_video(tmp_path)
    session, _feed = _session(tmp_path)
    try:
        session.next()
        session.video_action("load_video", path=path)
        tuning = session.tuning
        tuning.control_action("capture")
        sample, camera = tuning.controls.sample, session.video.camera
        monkeypatch.setattr("macaque_tracker.calibration_video.choose_video", lambda: None)
        session.video_action("load_video")
        assert tuning.controls.sample is sample
        session.video_action("load_video", path=tmp_path / "missing.avi")
        assert tuning.controls.sample is sample
        assert session.video.camera is camera and camera.started
        tuning.controls._pending = object()
        session.video_action("frame_forward")
        assert "Cancel optimization" in tuning.controls.status
        assert tuning.controls.sample is sample
        tuning.controls._pending = None
    finally:
        session.close()


def test_video_calibration_without_camera_pauses_steps_and_stops_at_end(tmp_path, monkeypatch):
    from macaque_tracker.calibration_video import CalibrationVideo

    path, frames = _recorded_video(tmp_path)
    session, _feed = _session(tmp_path)
    session.close()
    session = CalibrationSession(session.config, session.experiment, None, video_path=path)
    try:
        assert np.array_equal(session.roi.image, frames[0])
        session.video_action("frame_back")
        assert session.video.camera.frame_index == 0
        session.video_action("play_video")
        clock = [1_000.0]
        monkeypatch.setattr("macaque_tracker.calibration_video.time.monotonic", lambda: clock[0])
        session.video._next_frame = clock[0]
        for _ in frames:
            session.render()
            clock[0] += 1 / 30
        assert session.video.camera.frame_index == len(frames) - 1
        assert not session.video.playing
        assert np.array_equal(session.roi.image, frames[-1])
        session.video_action("frame_forward")
        assert session.video.camera.frame_index == len(frames) - 1
        assert isinstance(session.video, CalibrationVideo)
    finally:
        session.close()


def test_unknown_length_video_stops_at_eof_and_can_restart(tmp_path, monkeypatch):
    from macaque_tracker.calibration_video import CalibrationVideo
    from macaque_tracker.camera import VideoFileCamera

    path, frames = _recorded_video(tmp_path)
    video = CalibrationVideo(CameraConfig(analysis_width=640, analysis_height=90), None)
    monkeypatch.setattr(VideoFileCamera, "frame_count", property(lambda _self: 0))
    try:
        video.load(path)
        for _ in range(len(frames) - 1):
            assert video.step(1)
        assert not video.step(1)
        assert video.camera.frame_index == len(frames) - 1
        assert np.array_equal(video.latest(), frames[-1])
        video.toggle()
        assert video.camera.frame_index == 0 and video.playing
        assert np.array_equal(video.image, frames[0])
    finally:
        video.close()


def test_changed_boxes_keep_old_and_new_labels_in_optimization_and_next_subject_session(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from macaque_tracker import calibration_editor

    session, feed = _session(tmp_path)
    session.next()
    tuning = session.tuning
    for eye in (0, 1):
        tuning.control_action(f"eye{eye}")
        tuning.control_action("capture")
        tuning.controls.paint((25, 18), (40, 18))
    originals = [eye[0] for eye in tuning.controls.samples]
    session.save()
    session.previous()
    session.roi.boxes = [PixelRoi(350, 15, 110, 60)]  # Temporarily configure only eye 0.
    session.next()
    tuning = session.tuning
    assert [len(eye) for eye in tuning.controls.samples] == [1, 1]
    assert tuning.controls.sample is None
    tuning.control_action("capture")
    tuning.control_action("no_pupil")
    assert [sample.image.shape for sample in tuning.controls.samples[0]] == [(36, 96), (60, 110)]
    seen = []

    def optimize(samples, settings, _config, **_kwargs):
        seen.extend(samples)
        return SimpleNamespace(settings=settings, before_score=0.5, after_score=0.6,
                               sample_scores=(0.6, 1.0))

    monkeypatch.setattr(calibration_editor, "optimize_calibration", optimize)
    tuning.control_action("optimize")
    tuning.controls._pending[1].result(timeout=2)
    tuning._poll_optimization()
    assert seen[0] is originals[0] and len(seen) == 2
    session.save()
    session.close()

    store = ExperimentStore(tmp_path / "data", tmp_path / "settings" / "config.json")
    next_experiment = store.create("subject1", 1_792_000_002_000_000_000)
    reopened = CalibrationSession(AppConfig.load(next_experiment.config_path), next_experiment, feed)
    try:
        reopened.roi.boxes.append(PixelRoi(520, 20, 100, 40))
        reopened.next()
        assert [len(eye) for eye in reopened.tuning.controls.samples] == [2, 1]
        for original, eye in zip(originals, reopened.tuning.controls.samples, strict=True):
            assert np.array_equal(eye[0].image, original.image)
            assert np.array_equal(eye[0].mask, original.mask)
        reopened.render()  # Original crop dimensions remain paintable in the new session.
    finally:
        reopened.close()


def test_reset_confirmation_cancel_save_and_subject_inheritance(tmp_path):
    session, feed = _session(tmp_path)
    session.next()
    tuning = session.tuning
    for eye in (0, 1):
        tuning.control_action(f"eye{eye}")
        tuning.control_action("capture")
        tuning.control_action("no_pupil")
    session.save()
    sidecar = session.experiment.calibration_path.with_suffix(".controls.npz")
    saved = sidecar.read_bytes()
    settings = tuple(tuning.settings)
    tuning.control_action("reset", confirm_reset=lambda: False)
    assert [len(eye) for eye in tuning.controls.samples] == [1, 1]
    tuning.control_action("reset", confirm_reset=lambda: True)
    assert tuning.controls.samples == [[], []] and tuning.controls.sample is None
    assert tuple(tuning.settings) == settings
    session.close()  # Cancel calibration: even a confirmed reset has not overwritten the saved bank.
    assert sidecar.read_bytes() == saved

    reopened = CalibrationSession(session.config, session.experiment, feed)
    reopened.next()
    assert [len(eye) for eye in reopened.tuning.controls.samples] == [1, 1]
    reopened.previous()
    reopened.roi.boxes.pop()
    reopened.next()
    reopened.tuning.control_action("reset", confirm_reset=lambda: True)
    reopened.save()
    reopened.close()
    store = ExperimentStore(tmp_path / "data", tmp_path / "settings" / "config.json")
    next_experiment = store.create("subject1", 1_792_000_002_000_000_000)
    following = CalibrationSession(AppConfig.load(next_experiment.config_path), next_experiment, feed)
    try:
        following.roi.boxes.append(PixelRoi(520, 20, 100, 40))
        following.next()
        assert following.tuning.controls.samples == [[], []]
    finally:
        following.close()
