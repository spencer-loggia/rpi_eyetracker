"""Embedded adapter for the same ROI and pupil-tuning editors used by configure."""

from __future__ import annotations

from dataclasses import replace
from threading import RLock

from .calibration import save_control_samples
from .calibration_editor import load_controls
from .calibration_video import CalibrationVideo
from .config import RoiLayout
from .models import EyeImageSettings
from .roi_tool import (
    EyeTuningEditor,
    RoiEditor,
    _load_existing_layout,
    oriented_eye_crops,
)


class CalibrationSession:
    """Share raw ROI selection, orientation, and pupil tuning with standalone configure."""

    def __init__(self, config, experiment, feed, *, video_path=None) -> None:
        self._lock = RLock()
        self._closed = False
        self.config = config
        self.experiment = experiment
        self.feed = feed
        self.video = CalibrationVideo(config.camera, feed.latest if feed else None)
        if video_path is not None:
            self.video.load(video_path)
        try:
            self._initialize()
        except BaseException:
            self.video.close()
            raise

    def _initialize(self):
        config, experiment, feed = self.config, self.experiment, self.feed
        if self.video.camera is not None:
            feed = None
        image = self.video.latest()
        boxes, self.initial_settings, rotations = _load_existing_layout(
            experiment.calibration_path,
            image,
            config.tracker,
        )
        self.roi = RoiEditor(
            image,
            boxes,
            camera_rotations=rotations,
            camera_config=config.camera,
            control_limits=feed.camera.image_control_limits() if feed else {},
            frame_source=self.video.latest,
            apply_exposure=(None if feed is None else lambda camera: feed.camera.set_image_controls(
                exposure_us=camera.exposure_us,
            )),
            maximum_display_width=1280,
            maximum_display_height=600,
            video=self.video,
        )
        self.tuning = None
        self.phase = "boxes"
        self.selected = ()

    @property
    def editor(self):
        return self.roi if self.phase == "boxes" else self.tuning

    def render(self, *, refresh: bool = True):
        """Skip busy/closed sessions; never read a decoder while a UART transition closes it."""
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if self._closed:
                return None
            editor = self.editor
            if self.phase == "boxes":
                if refresh:
                    editor._refresh_live_image()
                return editor._frame(show_buttons=False)
            if refresh:
                editor._refresh_live_crops()
            return editor._frame(show_buttons=False)
        finally:
            self._lock.release()

    def mouse(self, event: int, x: int, y: int) -> None:
        """Forward pointer events to the shared box or raw control-image editor."""
        if self.phase == "boxes":
            self.roi._mouse(event, x, y, 0, None)
        elif self.tuning is not None:
            self.tuning._mouse(event, x, y)

    def next(self) -> None:
        """Build rotated tuning crops without changing the raw eye rectangles."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Calibration is busy; wait for the video operation to finish")
        try:
            if self._closed:
                raise RuntimeError("Calibration is closed")
            self._next()
        finally:
            self._lock.release()

    def _next(self) -> None:
        if not 1 <= len(self.roi.boxes) <= 2:
            raise ValueError("Draw one or two eye boxes before continuing")
        if self.roi.camera_config != self.roi.applied_camera_config:
            raise ValueError("Exposure was not applied; move its slider to retry")
        self.selected = tuple(self.roi.boxes)

        def crops():
            image = self.video.latest()
            return oriented_eye_crops(image, self.selected, self.roi.camera_rotations)

        initial = tuple(
            self.initial_settings.get(
                index,
                EyeImageSettings(pupil_size_bias=self.config.tracker.pupil_size_bias),
            )
            for index in range(len(self.selected))
        )
        warning = ""
        if self.tuning is not None:
            controls = self.tuning.controls
            controls.set_eye_count(len(self.selected))
            video_frames = self.tuning._video_frames
            samples = None
        else:
            controls = None
            video_frames = {}
            samples, warning = load_controls(self.experiment.calibration_path)
        self.tuning = EyeTuningEditor(
            crops(),
            initial,
            self.config.tracker,
            frame_source=crops,
            maximum_display_width=1280,
            maximum_display_height=540,
            control_samples=samples or (),
            video=self.video,
        )
        if controls is not None:
            self.tuning.controls = controls
            self.tuning._video_frames = video_frames
        if warning:
            self.tuning.controls.status = warning
        self.phase = "tuning"

    @property
    def control_signature(self):
        video = (id(self.video.camera), self.video.playing)
        if self.phase != "tuning":
            return video
        controls = self.tuning.controls
        return (controls.eye, controls.index, controls.busy, controls.erase, controls.brush,
                controls.show_prediction, tuple(self.tuning.settings), video)

    def video_action(self, action, *, path=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("Calibration is closed")
            if self.phase == "boxes":
                self.roi.video_action(action, path=path)
            else:
                self.tuning.control_action(action, path=path)
                if self.video.camera is not None:
                    self.roi._exposure = None
                    self.roi.camera_config = self.roi.applied_camera_config

    def previous(self) -> None:
        """Return to box/orientation editing, retaining the unsaved tuning values."""
        if self.tuning is not None:
            self.tuning.close()
            self.initial_settings.update(enumerate(self.tuning.settings))
        self.phase = "boxes"

    def sliders(self):
        """Return label, shared slider mapping, value, callback for toolkit controls."""
        if self.phase == "boxes":
            slider = self.roi._exposure
            if slider is not None:
                yield (
                    slider.label,
                    slider,
                    self.roi.camera_config.exposure_us,
                    self.roi._exposure_changed,
                )
        else:
            for index in range(len(self.tuning.settings)):
                for slider in self.tuning._sliders:
                    yield (
                        f"Eye {index}: {slider.label}",
                        slider,
                        self.tuning._slider_value(index, slider),
                        lambda position, i=index, s=slider: self.tuning._tuning_changed(i, s, position),
                    )

    def save(self) -> None:
        """Persist eye boxes, rotations, tuning, and exposure for this experiment."""
        if self.phase != "tuning" or self.tuning is None:
            raise ValueError("Complete pupil tuning before saving calibration")
        samples = self.tuning.saved_control_samples()
        height, width = self.roi.image.shape
        layout = RoiLayout.from_pixel_boxes(
            self.selected, tuple(self.tuning.settings), width, height,
            camera_rotations=self.roi.camera_rotations,
        )
        layout.validate_for_frame(width, height, tracker_config=self.config.tracker)
        updated = replace(self.config, camera=self.roi.applied_camera_config)
        # Config/ROI writers use atomic replacement; only the current experiment
        # is touched. Returning with Back instead discards unsaved adjustments.
        save_control_samples(
            self.experiment.calibration_path, samples, shared=True,
        )
        updated.save(self.experiment.config_path, shared=True)
        layout.save(self.experiment.calibration_path, shared=True)

    def close(self) -> None:
        """Cancel any offline work without delaying camera or application shutdown."""
        with self._lock:
            self._closed = True
            if self.tuning is not None:
                self.tuning.close()
            self.video.close()
