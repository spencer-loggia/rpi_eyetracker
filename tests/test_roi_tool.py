from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

try:
    import cv2
except ImportError:
    cv2 = None

from macaque_tracker import roi_tool
from macaque_tracker.config import AppConfig, CameraConfig, RoiLayout
from macaque_tracker.models import EyeImageSettings, PixelRoi
from macaque_tracker.roi_tool import (
    EyeTuningEditor,
    RoiEditor,
    _exposure_slider,
    _software_sliders,
)
from macaque_tracker.tracker import PupilPrior


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


def test_phase_one_exposes_no_software_sliders() -> None:
    assert [slider.config_name for slider in _software_sliders()] == [
        "gain",
        "brightness",
        "contrast",
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

    assert rendered.shape == (100, 160, 3)
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

    editor = object.__new__(EyeTuningEditor)
    editor.cv2 = cv2
    editor.crops = (crop.copy(),)
    editor.settings = [EyeImageSettings()]
    editor.detectors = [FakeDetector()]
    editor.priors = [PupilPrior()]

    rendered = editor._eye_panel(0)

    assert rendered.ndim == 3
    assert rendered.shape[2] == 3
    assert np.any(rendered[:, :, 0] != rendered[:, :, 1])
    assert np.array_equal(editor.crops[0], crop)


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
    editor._status = ""
    editor._frame_source = None

    def detector(settings: EyeImageSettings):
        return ("detector", settings)

    editor._detector = detector
    return editor


def test_software_controls_are_independent_per_eye() -> None:
    editor = _tuning_editor_for_test()
    contrast = next(
        slider for slider in _software_sliders() if slider.config_name == "contrast"
    )

    editor._software_changed(1, contrast, contrast.position_for(1.75))

    assert editor.settings[0].contrast == 1.0
    assert editor.settings[1].contrast == 1.75
    assert "Eye 1 contrast" in editor._status


def test_pupil_size_bias_is_independent_per_eye() -> None:
    editor = _tuning_editor_for_test()

    editor._size_bias_changed(0, 60)

    assert editor.settings[0].pupil_size_bias == -0.4
    assert editor.settings[1].pupil_size_bias == 0.0
    assert editor.detectors[0] == ("detector", editor.settings[0])
    assert "smaller" in editor._status

    editor._size_bias_changed(0, 100)
    assert editor.settings[0].pupil_size_bias == 0.0
    assert "neutral" in editor._status


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
            self.applied_camera_config = replace(
                kwargs["camera_config"],
                exposure_us=12_000,
            )

        def run(self):
            return (PixelRoi(100, 50, 200, 100),)

    live_sources: list[object] = []

    class FakeTuningEditor:
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


def test_static_configuration_averages_frames_and_disables_live_sources(
    monkeypatch,
    tmp_path,
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
            self.applied_camera_config = kwargs["camera_config"]
            observed["roi_frame_source"] = kwargs.get("frame_source")
            observed["recapture"] = kwargs.get("recapture")

        def run(self):
            return (PixelRoi(100, 50, 200, 100),)

    class FakeTuningEditor:
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
