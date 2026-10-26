from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

try:
    import cv2
except ImportError:
    cv2 = None

from macaque_tracker import roi_tool
from macaque_tracker.calibration_editor import ControlImageEditor
from macaque_tracker.config import AppConfig, CameraConfig, RoiLayout, TrackerConfig
from macaque_tracker.models import EyeFitSettings, EyeImageSettings, PixelRoi
from macaque_tracker.roi_tool import (
    EyeTuningEditor,
    RoiEditor,
    _exposure_slider,
    _tuning_sliders,
)
from macaque_tracker.tracker import PupilPrior, TemporalEyeTracker


class _FakeCv2:
    @staticmethod
    def setTrackbarPos(*args) -> None:
        del args


def test_exposure_slider_uses_hardware_and_frame_period_limits() -> None:
    slider = _exposure_slider(
        CameraConfig(fps=60.0, exposure_us=10_000),
        {"exposure_us": (100.0, 100_000.0)},
    )

    assert slider is not None
    assert slider.minimum == 100.0
    assert slider.maximum == 16_666.0
    assert slider.value_for(slider.position_for(12_345)) == 12_345
    assert _exposure_slider(CameraConfig(), {}) is None


def test_tuning_exposes_only_direct_fit_controls() -> None:
    assert [slider.config_name for slider in _tuning_sliders()] == [
        "pupil_size_bias",
        "threshold_range_fraction",
        "min_confidence",
    ]


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_roi_editor_uses_color_overlay_without_changing_gray_source() -> None:
    image = np.full((100, 160), 90, dtype=np.uint8)
    original = image.copy()
    editor = RoiEditor(
        image,
        (PixelRoi(20, 25, 50, 40),),
        maximum_display_width=160,
        maximum_display_height=100,
    )

    rendered = editor._frame()

    assert rendered.shape[0] > 100
    assert rendered.shape[1:] == (160, 3)
    assert editor._source_point(50, 101) is None
    assert np.any(rendered[:, :, 0] != rendered[:, :, 1])
    assert editor.image.ndim == 2
    assert np.array_equal(image, original)
    assert np.array_equal(editor.image, original)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_tuning_panel_colorizes_mask_without_changing_gray_crop() -> None:
    crop = np.full((80, 120), 100, dtype=np.uint8)
    mask = np.zeros_like(crop)
    mask[20:60, 35:85] = 255

    class FakeDetector:
        @staticmethod
        def detect_with_mask(image, prior):
            assert image.ndim == 2
            assert isinstance(prior, PupilPrior)
            return None, mask

    editor = EyeTuningEditor((crop,), (EyeImageSettings(),), TrackerConfig())
    editor.detectors = [FakeDetector()]

    rendered = editor._eye_panel(0)

    assert rendered.ndim == 3
    assert rendered.shape[2] == 3
    assert np.any(rendered[:, :, 0] != rendered[:, :, 1])
    assert np.array_equal(editor.crops[0], crop)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_tuning_and_tracking_share_immediate_reacquisition() -> None:
    open_eye = np.full((160, 240), 185, dtype=np.uint8)
    cv2.ellipse(open_eye, (124, 82), (28, 20), 11, 0, 360, 24, -1)
    closed_eye = np.full_like(open_eye, 185)
    config = TrackerConfig()
    editor = EyeTuningEditor((open_eye,), (EyeImageSettings(),), config)
    tracker = TemporalEyeTracker(0, config)
    reopened = np.full_like(open_eye, 185)
    cv2.ellipse(reopened, (65, 92), (17, 12), 11, 0, 360, 24, -1)

    for image, valid in ((open_eye, True), (closed_eye, False), (closed_eye, False),
                         (closed_eye, False), (reopened, True)):
        editor.crops = (image,)
        editor._eye_panel(0)
        assert tracker.process(image).valid == valid
        assert editor.priors[0] == tracker.state


def _roi_editor_for_test() -> RoiEditor:
    editor = object.__new__(RoiEditor)
    editor.cv2 = _FakeCv2()
    editor.camera_config = CameraConfig()
    editor.applied_camera_config = editor.camera_config
    editor._exposure = _exposure_slider(
        editor.camera_config,
        {"exposure_us": (100.0, 30_000.0)},
    )
    editor._initializing_slider = False
    editor._status = ""
    editor.image = np.zeros((30, 40), dtype=np.uint8)
    editor._frame_source = None
    editor._apply_exposure = None
    editor._recapture = None
    return editor


def test_exposure_remains_pending_until_explicit_recapture() -> None:
    editor = _roi_editor_for_test()

    editor._exposure_changed(12_000)

    assert editor.camera_config.exposure_us == 12_000
    assert editor.applied_camera_config.exposure_us != 12_000
    assert "press R" in editor._status

    calls: list[int] = []

    def recapture(config: CameraConfig) -> np.ndarray:
        calls.append(config.exposure_us)
        return np.full((30, 40), 7, dtype=np.uint8)

    editor._recapture = recapture
    editor._recapture_image()

    assert calls == [12_000]
    assert editor.applied_camera_config.exposure_us == 12_000
    assert np.all(editor.image == 7)


def test_live_preview_refreshes_and_applies_exposure_without_recapture() -> None:
    editor = _roi_editor_for_test()
    applied: list[int] = []
    editor._frame_source = lambda: np.full((30, 40), 9, dtype=np.uint8)
    editor._apply_exposure = lambda config: applied.append(config.exposure_us)

    editor._exposure_changed(12_500)
    editor._refresh_live_image()

    assert applied == [12_500]
    assert editor.applied_camera_config.exposure_us == 12_500
    assert np.all(editor.image == 9)
    assert editor._status == "Exposure applied to live camera"


def _tuning_editor_for_test() -> EyeTuningEditor:
    editor = object.__new__(EyeTuningEditor)
    editor.cv2 = _FakeCv2()
    editor.settings = [EyeImageSettings(), EyeImageSettings()]
    editor.detectors = [object(), object()]
    editor.priors = [PupilPrior(), PupilPrior()]
    editor._initializing_sliders = False
    editor._window_open = False
    editor.tracker_config = TrackerConfig()
    editor._status = ""
    editor._frame_source = None
    editor.controls = ControlImageEditor(2)

    def detector(settings: EyeImageSettings):
        return ("detector", settings)

    editor._detector = detector
    return editor


def test_fit_sliders_change_only_the_selected_parameter_and_eye() -> None:
    editor = _tuning_editor_for_test()
    fitted = EyeFitSettings(11, 0.8, 0.35, 16, 0.65, 0.445, 0.8, 1.5, 0.24, 0.5)
    original = EyeImageSettings(gain=1.75, brightness=-0.15, contrast=1.4, fit=fitted)
    editor.settings[1] = original
    edge = next(s for s in _tuning_sliders() if s.config_name == "threshold_range_fraction")
    assert editor._slider_value(1, edge) == 0.445
    editor._tuning_changed(1, edge, edge.position_for(0.6))
    assert editor.settings[0] == EyeImageSettings()
    assert editor.settings[1] == replace(original, fit=replace(fitted, threshold_range_fraction=0.6))
    assert editor.detectors[1] == ("detector", editor.settings[1])
    assert "Pupil edge" in editor._status


@pytest.mark.parametrize("initializing", [True, False])
def test_slider_initialization_does_not_round_saved_optimized_settings(initializing):
    editor = _tuning_editor_for_test()
    fitted = EyeFitSettings(8, 0.75, 0.45, 8, 0.48, 0.445)
    editor.settings[0] = EyeImageSettings(fit=fitted)
    editor._initializing_sliders = initializing
    edge = next(s for s in _tuning_sliders() if s.config_name == "threshold_range_fraction")
    editor._tuning_changed(0, edge, edge.position_for(0.445))
    assert editor.settings[0].fit == fitted


def test_pupil_size_bias_is_independent_per_eye() -> None:
    editor = _tuning_editor_for_test()
    slider = _tuning_sliders()[0]
    editor._tuning_changed(0, slider, slider.position_for(-0.4))
    assert editor.settings[0].pupil_size_bias == -0.4
    assert editor.settings[1].pupil_size_bias == 0.0
    assert editor.detectors[0] == ("detector", editor.settings[0])
    editor._tuning_changed(0, slider, slider.position_for(0))
    assert editor.settings[0].pupil_size_bias == 0.0


def test_live_tuning_refreshes_both_eye_crops_from_one_supplier() -> None:
    editor = _tuning_editor_for_test()
    editor.crops = (
        np.zeros((20, 30), dtype=np.uint8),
        np.zeros((20, 30), dtype=np.uint8),
    )
    calls = 0

    def next_crops() -> tuple[np.ndarray, ...]:
        nonlocal calls
        calls += 1
        return (
            np.full((20, 30), 4, dtype=np.uint8),
            np.full((20, 30), 8, dtype=np.uint8),
        )

    editor._frame_source = next_crops
    editor._refresh_live_crops()

    assert calls == 1
    assert np.all(editor.crops[0] == 4)
    assert np.all(editor.crops[1] == 8)


def test_video_configuration_supplies_live_frames_to_both_stages(
    monkeypatch,
    tmp_path,
) -> None:
    config = AppConfig()
    observed: dict[str, object] = {}

    class FakeVideoCamera:
        def __init__(self, video_path, camera_config, roi_layout=None, *, realtime=True):
            del camera_config, roi_layout
            observed["video_path"] = video_path
            observed["realtime"] = realtime
            self.calls = 0

        def start(self) -> None:
            observed["started"] = True

        def capture_preview(self):
            self.calls += 1
            image = np.full(
                (config.camera.analysis_height, config.camera.analysis_width),
                self.calls,
                dtype=np.uint8,
            )
            return image, self.calls

        def close(self) -> None:
            observed["closed"] = True

    class FakeRoiEditor:
        def __init__(self, image, **kwargs) -> None:
            self.image = image
            self.camera_rotations = kwargs["camera_rotations"]
            self.applied_camera_config = None
            frame_source = kwargs.get("frame_source")
            observed["roi_frame_source"] = frame_source
            observed["roi_values"] = [
                int(frame_source()[0, 0]),
                int(frame_source()[0, 0]),
            ]

        def run(self):
            return (PixelRoi(100, 50, 200, 100),)

    class FakeTuningEditor:
        def saved_control_samples(self):
            return ((),)

        def __init__(self, crops, initial, tracker_config, **kwargs) -> None:
            del crops, tracker_config
            self.initial = initial
            frame_source = kwargs.get("frame_source")
            observed["tuning_frame_source"] = frame_source
            observed["tuning_values"] = [
                int(frame_source()[0][0, 0]),
                int(frame_source()[0][0, 0]),
            ]

        def run(self):
            return self.initial

    monkeypatch.setattr(roi_tool, "_require_cv2", lambda: object())
    monkeypatch.setattr(roi_tool, "VideoFileCamera", FakeVideoCamera)
    monkeypatch.setattr(roi_tool, "RoiEditor", FakeRoiEditor)
    monkeypatch.setattr(roi_tool, "EyeTuningEditor", FakeTuningEditor)

    saved = roi_tool.configure_rois(
        config,
        tmp_path / "rois.json",
        video_path="fixture.mkv",
    )

    assert saved == tmp_path / "rois.json"
    assert observed["video_path"] == "fixture.mkv"
    assert observed["realtime"] is True
    assert observed["roi_values"] == [2, 3]
    assert observed["tuning_values"] == [4, 5]
    assert callable(observed["roi_frame_source"])
    assert callable(observed["tuning_frame_source"])
    assert observed["started"] is True
    assert observed["closed"] is True


def test_configuration_saves_exposure_and_per_eye_settings_separately(
    monkeypatch,
    tmp_path,
) -> None:
    config = AppConfig()
    config_path = config.save(tmp_path / "eye_tracker.json")
    roi_path = tmp_path / "rois.json"
    preview = np.full(
        (config.camera.analysis_height, config.camera.analysis_width),
        100,
        dtype=np.uint8,
    )
    tuned = EyeImageSettings(
        gain=1.5,
        brightness=-0.1,
        contrast=1.25,
        pupil_size_bias=-0.3,
    )

    class FakeCamera:
        def __init__(self, camera_config, recording_config, roi_layout=None) -> None:
            del camera_config, recording_config, roi_layout

        def start(self) -> None:
            pass

        def capture_preview(self):
            return preview.copy(), 1

        def image_control_limits(self):
            return {"exposure_us": (100.0, 30_000.0)}

        def close(self) -> None:
            pass

    class FakeRoiEditor:
        def __init__(self, image, **kwargs) -> None:
            self.image = image
            self.camera_rotations = kwargs["camera_rotations"]
            self.applied_camera_config = replace(
                kwargs["camera_config"],
                exposure_us=12_000,
            )

        def run(self):
            return (PixelRoi(100, 50, 200, 100),)

    live_sources: list[object] = []

    class FakeTuningEditor:
        def saved_control_samples(self):
            return ((),)

        def __init__(self, crops, initial, tracker_config, **kwargs) -> None:
            del crops, initial, tracker_config
            live_sources.append(kwargs.get("frame_source"))

        def run(self):
            return (tuned,)

    monkeypatch.setattr(roi_tool, "_require_cv2", lambda: object())
    monkeypatch.setattr(roi_tool, "Picamera2Camera", FakeCamera)
    monkeypatch.setattr(roi_tool, "RoiEditor", FakeRoiEditor)
    monkeypatch.setattr(roi_tool, "EyeTuningEditor", FakeTuningEditor)

    saved = roi_tool.configure_rois(
        config,
        roi_path,
        config_path=config_path,
        average_frames=1,
    )

    assert saved == roi_path
    updated_config = AppConfig.load(config_path)
    assert updated_config.camera.exposure_us == 12_000
    assert updated_config.camera.analogue_gain == config.camera.analogue_gain
    saved_roi = RoiLayout.load(roi_path).rois[0]
    assert saved_roi.settings == tuned
    assert updated_config.tracker == config.tracker
    assert live_sources[0] is not None


@pytest.mark.parametrize("previous_profile", [False, True])
def test_static_configuration_averages_frames_and_preserves_profile_in_new_box(
    monkeypatch,
    tmp_path,
    previous_profile,
) -> None:
    config = AppConfig()
    preview = np.full(
        (config.camera.analysis_height, config.camera.analysis_width),
        80,
        dtype=np.uint8,
    )
    captures = 0
    observed: dict[str, object] = {}

    class FakeCamera:
        def __init__(self, camera_config, recording_config, roi_layout=None) -> None:
            del camera_config, recording_config, roi_layout

        def start(self) -> None:
            pass

        def capture_preview(self):
            nonlocal captures
            captures += 1
            return preview.copy(), captures

        def image_control_limits(self):
            return {"exposure_us": (100.0, 30_000.0)}

        def close(self) -> None:
            pass

    class FakeRoiEditor:
        def __init__(self, image, **kwargs) -> None:
            self.image = image
            self.camera_rotations = kwargs["camera_rotations"]
            self.applied_camera_config = kwargs["camera_config"]
            observed["roi_frame_source"] = kwargs.get("frame_source")
            observed["recapture"] = kwargs.get("recapture")

        def run(self):
            return (PixelRoi(100, 50, 200, 100),)

    class FakeTuningEditor:
        def saved_control_samples(self):
            return ((),)

        def __init__(self, crops, initial, tracker_config, **kwargs) -> None:
            del crops, tracker_config
            self.initial = initial
            observed["tuning_frame_source"] = kwargs.get("frame_source")

        def run(self):
            return self.initial

    monkeypatch.setattr(roi_tool, "_require_cv2", lambda: object())
    monkeypatch.setattr(roi_tool, "Picamera2Camera", FakeCamera)
    monkeypatch.setattr(roi_tool, "RoiEditor", FakeRoiEditor)
    monkeypatch.setattr(roi_tool, "EyeTuningEditor", FakeTuningEditor)

    profile = EyeImageSettings(0.75, 0.1, 1.2, -0.3,
                              EyeFitSettings(9, 0.85, 0.35, 12, 0.64, 0.44, 0.7, 1.5, 0.12, 0.5))
    if previous_profile:
        RoiLayout.from_pixel_boxes(
            (PixelRoi(50, 30, 200, 100),), (profile,), preview.shape[1], preview.shape[0],
        ).save(tmp_path / "rois.json")

    saved = roi_tool.configure_rois(
        config,
        tmp_path / "rois.json",
        average_frames=3,
        static=True,
    )

    assert saved == tmp_path / "rois.json"
    assert captures == 3
    assert observed["roi_frame_source"] is None
    assert observed["tuning_frame_source"] is None
    assert callable(observed["recapture"])
    saved_roi = RoiLayout.load(saved).rois[0]
    assert saved_roi.settings == (profile if previous_profile else EyeImageSettings())
    assert saved_roi.to_pixels(preview.shape[1], preview.shape[0]) == PixelRoi(100, 50, 200, 100)


@pytest.mark.parametrize("scale", [1, 2])
def test_loading_layout_keeps_fit_when_analysis_resolution_changes(tmp_path, scale):
    settings = EyeImageSettings(fit=EyeFitSettings(10, 0.8, 0.4, 12, 0.65, 0.4, 0.7, 2, 0.12, 0.5))
    path = tmp_path / "rois.json"
    layout = RoiLayout.from_pixel_boxes((PixelRoi(350, 20, 100, 40),), (settings,), 640, 90)
    layout.save(path)
    boxes, loaded, _ = roi_tool._load_existing_layout(
        path, np.zeros((90 * scale, 640 * scale), np.uint8), TrackerConfig(),
    )
    assert loaded == {0: settings}
    assert boxes == (PixelRoi(350 * scale, 20 * scale, 100 * scale, 40 * scale),)
