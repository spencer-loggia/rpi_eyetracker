from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np

from .calibration import save_control_samples
from .calibration_dialogs import confirm_reset_controls
from .calibration_editor import ControlImageEditor, load_controls
from .calibration_video import CalibrationVideo, choose_video
from .camera import CameraError, Picamera2Camera
from .config import AppConfig, CameraConfig, ConfigError, RoiLayout, TrackerConfig
from .models import EyeImageSettings, PixelRoi
from .orientation import DEFAULT_ROTATIONS, CameraCanvas, rotate_image
from .overlays import draw_roi_box
from .tracker import (
    AdaptivePupilDetector,
    apply_eye_image_settings,
    detector_config_for_eye,
    fit_settings_from_config,
)

Color = tuple[int, int, int]

# OpenCV display colours are BGR. Source images and detector inputs stay 2-D.
_BACKGROUND: Color = (24, 24, 24)
_TEXT: Color = (235, 235, 235)
_MUTED: Color = (160, 160, 160)
_TRACKED: Color = (80, 220, 80)
_FIT: Color = (0, 215, 255)
_NO_FIT: Color = (80, 80, 235)
VIDEO_KEYS = {ord("o"): "load_video", ord("O"): "load_video", ord(" "): "play_video",
              ord("["): "frame_back", ord("]"): "frame_forward"}


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "The ROI tool requires OpenCV with GUI support (Raspberry Pi package "
            "python3-opencv)."
        ) from exc
    return cv2


def _display_canvas(image: np.ndarray, cv2_module) -> np.ndarray:
    """Make a disposable BGR canvas from a grayscale source image."""

    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Display source must be a two-dimensional uint8 image")
    return cv2_module.cvtColor(image, cv2_module.COLOR_GRAY2BGR)


class _Slider:
    def __init__(
        self,
        config_name: str,
        label: str,
        minimum: float,
        maximum: float,
        scale: int,
        *,
        integer: bool = False,
    ) -> None:
        self.config_name = config_name
        self.label = label
        self.minimum = minimum
        self.maximum = maximum
        self.scale = scale
        self.integer = integer

    @property
    def position_offset(self) -> int:
        return max(0, -round(self.minimum * self.scale))

    @property
    def minimum_position(self) -> int:
        return round(self.minimum * self.scale) + self.position_offset

    @property
    def maximum_position(self) -> int:
        return max(1, round(self.maximum * self.scale) + self.position_offset)

    def position_for(self, value: float) -> int:
        clipped = min(max(float(value), self.minimum), self.maximum)
        return int(
            np.clip(
                round(clipped * self.scale) + self.position_offset,
                self.minimum_position,
                self.maximum_position,
            )
        )

    def value_for(self, position: int) -> int | float:
        position = int(np.clip(position, self.minimum_position, self.maximum_position))
        value = (position - self.position_offset) / self.scale
        return round(value) if self.integer else float(value)


def _exposure_slider(
    camera_config: CameraConfig,
    limits: dict[str, tuple[float, float]],
) -> _Slider | None:
    reported = limits.get("exposure_us")
    if reported is None:
        return None
    frame_maximum = max(1, math.ceil(1_000_000.0 / camera_config.fps) - 1)
    minimum = max(1, math.ceil(reported[0]))
    maximum = min(frame_maximum, math.floor(reported[1]))
    if minimum >= maximum:
        return None
    return _Slider(
        "exposure_us",
        "Exposure us",
        float(minimum),
        float(maximum),
        1,
        integer=True,
    )


def _tuning_sliders() -> tuple[_Slider, ...]:
    return (
        _Slider("pupil_size_bias", "Pupil size (- small, + large)", -1.0, 1.0, 100),
        _Slider("threshold_range_fraction", "Pupil edge threshold", 0.01, 1.0, 100),
        _Slider("min_confidence", "Minimum confidence", 0.01, 1.0, 100),
    )


def _set_trackbar_minimum(cv2_module, window: str, slider: _Slider) -> None:
    setter = getattr(cv2_module, "setTrackbarMin", None)
    if not callable(setter):
        return
    try:
        setter(slider.label, window, slider.minimum_position)
    except cv2_module.error:
        pass


class RoiEditor:
    """Phase one: choose eye boxes on the full four-camera preview."""

    WINDOW_NAME = "Macaque eye ROI selection"

    def __init__(
        self,
        image: np.ndarray,
        initial: tuple[PixelRoi, ...] = (),
        *,
        camera_rotations: tuple[int, ...] = DEFAULT_ROTATIONS,
        minimum_width: int = 24,
        minimum_height: int = 24,
        maximum_display_width: int = 1600,
        maximum_display_height: int = 900,
        camera_config: CameraConfig | None = None,
        control_limits: dict[str, tuple[float, float]] | None = None,
        frame_source: Callable[[], np.ndarray] | None = None,
        apply_exposure: Callable[[CameraConfig], None] | None = None,
        recapture: Callable[[CameraConfig], np.ndarray] | None = None,
        video: CalibrationVideo | None = None,
    ) -> None:
        if image.dtype != np.uint8 or image.ndim != 2:
            raise ValueError("ROI editor image must be two-dimensional uint8")
        if len(initial) > 2:
            raise ValueError("At most two initial boxes are allowed")
        self.cv2 = _require_cv2()
        self.image = image.copy()
        self.height, self.width = image.shape
        self.camera_rotations = tuple(camera_rotations)
        self._display_limits = (maximum_display_width, maximum_display_height)
        self.minimum_width = minimum_width
        self.minimum_height = minimum_height
        self.boxes = list(initial)
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None
        self.camera_config = camera_config
        self.applied_camera_config = camera_config
        self._frame_source = frame_source
        self._apply_exposure = apply_exposure
        self._recapture = recapture
        self.video = video
        self._video_toolbar = []
        self._exposure = (
            None
            if (
                camera_config is None
                or control_limits is None
                or (frame_source is None and recapture is None)
            )
            else _exposure_slider(camera_config, control_limits)
        )
        self._initializing_slider = False
        self._status = ""
        self._update_geometry()

    def _update_geometry(self) -> None:
        """Recompute the display map only when calibration orientation changes."""
        self.geometry = CameraCanvas(self.width, self.height, self.camera_rotations)
        maximum_width, maximum_height = self._display_limits
        self.scale = min(1.0, maximum_width / self.geometry.width,
                         maximum_height / self.geometry.height)
        self.display_width = max(1, round(self.geometry.width * self.scale))
        self.display_height = max(1, round(self.geometry.height * self.scale))

    def rotate_camera(self, index: int) -> None:
        """Rotate a camera clockwise, retaining its eye boxes in raw sensor coordinates."""
        rotations = list(self.camera_rotations)
        rotations[index] = (rotations[index] + 90) % 360
        geometry = CameraCanvas(self.width, self.height, tuple(rotations))
        for box in self.boxes:
            geometry.display_roi(box)  # Reject cross-camera boxes before changing state.
        self.camera_rotations = tuple(rotations)
        self.drag_start = self.drag_current = None
        self._update_geometry()
        self._status = f"Camera {index}: {rotations[index]} degrees clockwise; +x right, +y down"

    def _source_point(self, x: int, y: int) -> tuple[int, int] | None:
        """Map image pixels back to the sensor; ignore the footer and padded areas."""
        if not 0 <= x < self.display_width or not 0 <= y < self.display_height:
            return None
        return self.geometry.source_point(
            int(np.clip(round(x / self.scale), 0, self.geometry.width - 1)),
            int(np.clip(round(y / self.scale), 0, self.geometry.height - 1)),
        )

    def _mouse(self, event: int, x: int, y: int, flags: int, userdata) -> None:
        del flags, userdata
        if event == self.cv2.EVENT_LBUTTONDOWN:
            for left, top, width, action in self._video_toolbar:
                if left <= x < left + width and top <= y < top + 25:
                    self.video_action(action)
                    return
        source = self._source_point(x, y)
        if source is None:
            if event == self.cv2.EVENT_LBUTTONUP:
                self.drag_start = self.drag_current = None
            return
        if event == self.cv2.EVENT_LBUTTONDOWN and len(self.boxes) < 2:
            self.drag_start = source
            self.drag_current = source
        elif event == self.cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            self.drag_current = source
        elif event == self.cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x0, y0 = self.drag_start
            x1, y1 = source
            left, right = sorted((x0, x1))
            top, bottom = sorted((y0, y1))
            right = min(right + 1, self.width)
            bottom = min(bottom + 1, self.height)
            if right - left >= self.minimum_width and bottom - top >= self.minimum_height:
                box = PixelRoi(left, top, right - left, bottom - top)
                try:
                    self.geometry.display_roi(box)
                except ValueError as exc:
                    self._status = str(exc)
                else:
                    self.boxes.append(box)
            self.drag_start = None
            self.drag_current = None

    def _draw_box(self, canvas: np.ndarray, roi: PixelRoi, index: int, active=False) -> None:
        try:
            displayed = self.geometry.display_roi(roi)
        except ValueError:
            return
        draw_roi_box(
            canvas,
            displayed,
            index,
            cv2_module=self.cv2,
            scale=self.scale,
            active=active,
        )

    def _exposure_changed(self, position: int) -> None:
        if self.camera_config is None or self._exposure is None:
            return
        try:
            clipped = int(
                np.clip(
                    position,
                    self._exposure.minimum_position,
                    self._exposure.maximum_position,
                )
            )
            if clipped != position:
                self.cv2.setTrackbarPos(
                    self._exposure.label,
                    self.WINDOW_NAME,
                    clipped,
                )
            self.camera_config = replace(
                self.camera_config,
                exposure_us=self._exposure.value_for(clipped),
            )
            if not self._initializing_slider:
                if self._apply_exposure is not None:
                    self._apply_exposure(self.camera_config)
                    self.applied_camera_config = self.camera_config
                    self._status = "Exposure applied to live camera"
                else:
                    self._status = "Exposure changed; press R to apply and recapture"
        except Exception as exc:  # noqa: BLE001 - keep editor open for correction
            self._status = f"Exposure control failed: {exc}"

    def _refresh_live_image(self) -> None:
        if self._frame_source is None:
            return
        try:
            image = self._frame_source()
            if image.dtype != np.uint8 or image.ndim != 2 or image.shape != self.image.shape:
                raise ValueError("Live preview dimensions or type changed")
            self.image = image.copy()
        except Exception as exc:  # noqa: BLE001 - retain last frame and keep UI responsive
            self._status = f"Live preview failed: {exc}"

    def _create_trackbar(self) -> None:
        if self.camera_config is None or self._exposure is None:
            return
        self._initializing_slider = True
        try:
            initial = self._exposure.position_for(self.camera_config.exposure_us)
            self.cv2.createTrackbar(
                self._exposure.label,
                self.WINDOW_NAME,
                initial,
                self._exposure.maximum_position,
                self._exposure_changed,
            )
            _set_trackbar_minimum(self.cv2, self.WINDOW_NAME, self._exposure)
        finally:
            self._initializing_slider = False

    def _recapture_image(self) -> None:
        if self._recapture is None or self.camera_config is None:
            return
        try:
            image = self._recapture(self.camera_config)
            if image.dtype != np.uint8 or image.ndim != 2 or image.shape != self.image.shape:
                raise ValueError("Recaptured image dimensions or type changed")
            self.image = image.copy()
            self.applied_camera_config = self.camera_config
            self._status = "Exposure applied and preview recaptured"
        except Exception as exc:  # noqa: BLE001 - keep editor open for retry
            self._status = f"Recapture failed: {exc}"

    def video_action(self, action, *, path=None) -> None:
        if self.video is None:
            return
        try:
            if self.video.action(action, path=path):
                self._exposure = None
                self._recapture = None
                self._apply_exposure = None
                self.camera_config = self.applied_camera_config
                self._frame_source = self.video.latest
                self._refresh_live_image()
                self._status = ""
        except (OSError, RuntimeError, ValueError) as exc:
            self._status = str(exc)

    def _frame(self, *, show_buttons=True) -> np.ndarray:
        image = self.geometry.compose(self.image)
        if self.scale != 1.0:
            image = self.cv2.resize(
                image,
                (self.display_width, self.display_height),
                interpolation=self.cv2.INTER_AREA,
            )
        canvas = _display_canvas(image, self.cv2)
        for index, roi in enumerate(self.boxes):
            self._draw_box(canvas, roi, index)
        if self.drag_start is not None and self.drag_current is not None:
            x0, y0 = self.drag_start
            x1, y1 = self.drag_current
            left, right = sorted((x0, x1))
            top, bottom = sorted((y0, y1))
            if right > left and bottom > top:
                self._draw_box(
                    canvas,
                    PixelRoi(left, top, right - left, bottom - top),
                    len(self.boxes),
                    active=True,
                )
        instruction = "Drag 1-2 eye boxes | Enter/S next | U undo | C clear | 1-4 rotate cameras"
        if self._exposure is not None:
            instruction += (
                " | Exposure: live"
                if self._frame_source is not None
                else " | Exposure: R apply + recapture"
            )
        instruction += " | Q/Esc cancel"
        lines = [instruction, "Clockwise rotations (cams 0-3): "
                 + ", ".join(str(angle) for angle in self.camera_rotations)
                 + " | axes: +x right, +y down"]
        if self.camera_config is not None and self._exposure is not None:
            lines.append(f"exposure {self.camera_config.exposure_us} us")
        if self._status:
            lines.append(self._status)
        if self.video is not None:
            lines.append(self.video.description)
        # Instructions belong outside the image: this very wide camera strip
        # can be only 180 display pixels tall, so an overlay would hide eye boxes.
        canvas = self.cv2.copyMakeBorder(
            canvas, 0, 6 + 25 * len(lines), 0, 0, self.cv2.BORDER_CONSTANT, value=_BACKGROUND,
        )
        for index, line in enumerate(lines):
            self.cv2.putText(
                canvas,
                line,
                (8, self.display_height + 22 + 25 * index),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                _TEXT if index < 2 else _MUTED,
                1,
                self.cv2.LINE_AA,
            )
        self._video_toolbar = []
        if show_buttons and self.video is not None:
            y = canvas.shape[0]
            canvas = self.cv2.copyMakeBorder(
                canvas, 0, 31, 0, 0, self.cv2.BORDER_CONSTANT, value=_BACKGROUND,
            )
            x = 8
            for label, action in self.video.buttons():
                width = self.cv2.getTextSize(label, self.cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0][0] + 18
                self._video_toolbar.append((x, y, width, action))
                self.cv2.rectangle(canvas, (x, y), (x + width, y + 25), (65, 65, 65), -1)
                self.cv2.putText(canvas, label, (x + 8, y + 17), self.cv2.FONT_HERSHEY_SIMPLEX,
                                 0.43, _TEXT, 1, self.cv2.LINE_AA)
                x += width + 5
        return canvas

    def run(self) -> tuple[PixelRoi, ...] | None:
        self.cv2.namedWindow(self.WINDOW_NAME, self.cv2.WINDOW_AUTOSIZE)
        self.cv2.setMouseCallback(self.WINDOW_NAME, self._mouse)
        self._create_trackbar()
        try:
            while True:
                self._refresh_live_image()
                self.cv2.imshow(self.WINDOW_NAME, self._frame())
                wait_ms = 1 if self._frame_source is not None else 20
                key = self.cv2.waitKey(wait_ms) & 0xFF
                if key in VIDEO_KEYS:
                    self.video_action(VIDEO_KEYS[key])
                if key in (13, 10, ord("s"), ord("S")) and 1 <= len(self.boxes) <= 2:
                    if self.camera_config != self.applied_camera_config:
                        self._status = (
                            "Exposure was not applied; move the slider to retry"
                            if self._frame_source is not None
                            else "Press R to apply pending exposure before continuing"
                        )
                    else:
                        return tuple(self.boxes)
                elif key in (8, 127, ord("u"), ord("U")) and self.boxes:
                    self.boxes.pop()
                elif key in (ord("c"), ord("C")):
                    self.boxes.clear()
                elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                    try:
                        self.rotate_camera(key - ord("1"))
                    except ValueError as exc:
                        self._status = str(exc)
                elif key in (ord("r"), ord("R")):
                    self._recapture_image()
                elif key in (27, ord("q"), ord("Q")):
                    return None
                try:
                    visible = self.cv2.getWindowProperty(
                        self.WINDOW_NAME,
                        self.cv2.WND_PROP_VISIBLE,
                    )
                except self.cv2.error:
                    visible = 0.0
                if visible < 1.0:
                    return None
        finally:
            self.cv2.destroyWindow(self.WINDOW_NAME)


class EyeTuningEditor:
    """Phase two: label raw crops and review the single optimized per-eye fit."""

    WINDOW_NAME = "Macaque pupil tuning"

    def __init__(
        self,
        crops: tuple[np.ndarray, ...],
        initial: tuple[EyeImageSettings, ...],
        tracker_config: TrackerConfig,
        *,
        frame_source: Callable[[], tuple[np.ndarray, ...]] | None = None,
        maximum_display_width: int = 1600,
        maximum_display_height: int = 900,
        control_samples=(),
        video: CalibrationVideo | None = None,
    ) -> None:
        if not 1 <= len(crops) <= 2 or len(initial) != len(crops):
            raise ValueError("Pupil tuning requires matching settings for one or two crops")
        if any(crop.dtype != np.uint8 or crop.ndim != 2 for crop in crops):
            raise ValueError("Pupil tuning crops must be two-dimensional uint8")
        self.cv2 = _require_cv2()
        self.crops = tuple(crop.copy() for crop in crops)
        self._frame_source = frame_source
        self.settings = list(initial)
        self.tracker_config = tracker_config
        self.detectors = [self._detector(settings) for settings in self.settings]
        self._predictions = [None for _settings in self.settings]
        self._sliders = _tuning_sliders()
        self._initializing_sliders = False
        self._status = ""
        self.maximum_display_width = maximum_display_width
        self.maximum_display_height = maximum_display_height
        self.controls = ControlImageEditor(len(crops), control_samples)
        self._panel_geometry = {}
        self._paint_point = None
        self._paint_image = None
        self._paint_erase = False
        self._frame_scale = 1.0
        self._toolbar = []
        self._window_open = False
        self.video = video
        self._video_frames = {}

    def _refresh_live_crops(self) -> None:
        # Annotation is a still-image operation. Do not block strokes on a new
        # camera/video frame, or keep fitting the other eye while painting.
        if self._frame_source is None or self.controls.sample is not None:
            return
        try:
            crops = self._frame_source()
            if len(crops) != len(self.crops):
                raise ValueError("Live crop count changed")
            if any(
                crop.dtype != np.uint8
                or crop.ndim != 2
                or crop.shape != previous.shape
                for crop, previous in zip(crops, self.crops, strict=True)
            ):
                raise ValueError("Live crop dimensions or type changed")
            self.crops = tuple(crop.copy() for crop in crops)
        except Exception as exc:  # noqa: BLE001 - retain last crops and keep UI open
            self._status = f"Live eye preview failed: {exc}"

    def _detector(self, settings: EyeImageSettings) -> AdaptivePupilDetector:
        return AdaptivePupilDetector(detector_config_for_eye(self.tracker_config, settings))

    @staticmethod
    def _trackbar_label(eye_id: int, label: str) -> str:
        return f"Eye {eye_id} {label}"

    def _slider_value(self, eye_id: int, slider: _Slider) -> float:
        settings = self.settings[eye_id]
        source = settings if slider.config_name == "pupil_size_bias" else (
            settings.fit or self.tracker_config
        )
        return getattr(source, slider.config_name)

    def _tuning_changed(self, eye_id: int, slider: _Slider, position: int) -> None:
        try:
            if self.controls.busy or self._initializing_sliders:
                return
            clipped = int(np.clip(position, slider.minimum_position, slider.maximum_position))
            if clipped != position and self._window_open:
                self.cv2.setTrackbarPos(
                    self._trackbar_label(eye_id, slider.label), self.WINDOW_NAME, clipped,
                )
            # Toolkit initialization can queue a callback after setup ends.
            # Showing a rounded position must not round an optimized value.
            if clipped == slider.position_for(self._slider_value(eye_id, slider)):
                return
            value = slider.value_for(clipped)
            settings = self.settings[eye_id]
            if slider.config_name == "pupil_size_bias":
                settings = replace(settings, pupil_size_bias=value)
            else:
                fit = settings.fit or fit_settings_from_config(self.tracker_config)
                settings = replace(settings, fit=replace(fit, **{slider.config_name: value}))
            self.settings[eye_id] = settings
            self.detectors[eye_id] = self._detector(settings)
            self.controls.results[eye_id] = None
            self._status = f"Eye {eye_id} {slider.label}: {float(value):.2f}"
        except (TypeError, ValueError) as exc:
            self._status = f"Invalid eye {eye_id} adjustment: {exc}"

    def _create_trackbars(self) -> None:
        self._initializing_sliders = True
        try:
            for eye_id in range(len(self.settings)):
                for slider in self._sliders:
                    label = self._trackbar_label(eye_id, slider.label)
                    self.cv2.createTrackbar(
                        label,
                        self.WINDOW_NAME,
                        slider.position_for(self._slider_value(eye_id, slider)),
                        slider.maximum_position,
                        lambda position, selected=eye_id, control=slider: (
                            self._tuning_changed(selected, control, position)
                        ),
                    )
                    named_slider = _Slider(
                        slider.config_name, label, slider.minimum, slider.maximum, slider.scale,
                    )
                    _set_trackbar_minimum(self.cv2, self.WINDOW_NAME, named_slider)
        finally:
            self._initializing_sliders = False

    def _prediction(self, eye_id: int, raw: np.ndarray):
        """Evaluate the same current-image objective live, frozen, and in optimization."""
        settings = self.settings[eye_id]
        detector = self.detectors[eye_id]
        previous = self._predictions[eye_id]
        if (self.controls.sample is not None and previous is not None
                and previous[0] is raw and previous[1] == settings
                and previous[2] is detector):
            return previous[3], previous[4]
        candidate, mask = detector.detect_with_mask(apply_eye_image_settings(raw, settings))
        contours, _ = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)
        self._predictions[eye_id] = (raw, settings, detector, candidate, contours)
        return candidate, contours

    def _eye_panel(self, eye_id: int) -> np.ndarray:
        settings = self.settings[eye_id]
        controls = self.controls
        sample = controls.sample if eye_id == controls.eye else None
        raw = self.crops[eye_id] if sample is None else sample.image
        candidate, contours = self._prediction(eye_id, raw)
        # Labels always reference original pixels; transformed images are detector-only.
        canvas = _display_canvas(raw, self.cv2)
        if sample is not None:
            selected = sample.mask != 0
            canvas[selected] = (0.65 * canvas[selected] + 0.35 * np.asarray(_TRACKED)).astype(np.uint8)
        if controls.show_prediction:
            self.cv2.drawContours(canvas, contours, -1, _FIT, 1)
        if candidate is not None and controls.show_prediction:
            ellipse = (
                (candidate.ellipse_x, candidate.ellipse_y),
                (candidate.ellipse_width, candidate.ellipse_height),
                candidate.angle_degrees,
            )
            self.cv2.ellipse(canvas, ellipse, _FIT, 1, self.cv2.LINE_AA)

        target_height = int(np.clip(canvas.shape[0], 360, 540))
        scale = target_height / canvas.shape[0]
        canvas = self.cv2.resize(
            canvas,
            (max(1, round(canvas.shape[1] * scale)), target_height),
            interpolation=self.cv2.INTER_NEAREST if scale > 1.0 else self.cv2.INTER_AREA,
        )
        header_height = 35
        footer_height = 58
        panel_width = max(520, canvas.shape[1])
        panel = np.full(
            (header_height + canvas.shape[0] + footer_height, panel_width, 3),
            _BACKGROUND,
            dtype=np.uint8,
        )
        content_x = (panel_width - canvas.shape[1]) // 2
        panel[
            header_height : header_height + canvas.shape[0],
            content_x : content_x + canvas.shape[1],
        ] = canvas
        self._panel_geometry[eye_id] = (content_x, header_height, canvas.shape[1], canvas.shape[0], scale)
        fit = (
            "NO PUPIL FIT"
            if candidate is None
            else f"pupil {candidate.diameter:.1f}px conf {candidate.confidence:.2f}"
        )
        chosen_threshold = "--" if candidate is None else str(candidate.threshold)
        mode = "CONTROL" if sample is not None else "PAUSED" if controls.sample is not None else "LIVE"
        self.cv2.putText(
            panel,
            f"EYE {eye_id}  {mode}  {fit}",
            (8, 24),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            _NO_FIT if candidate is None else _FIT,
            1,
            self.cv2.LINE_AA,
        )
        footer_y = header_height + canvas.shape[0]
        details = (
            f"auto threshold {chosen_threshold}  size bias {settings.pupil_size_bias:+.2f}  "
            f"gain {settings.gain:.2f}"
        )
        details2 = (
            f"brightness {settings.brightness:.2f}  contrast {settings.contrast:.2f}"
        )
        if sample is not None:
            label = ("PUPIL" if np.any(sample.mask) else "NO PUPIL") if controls.labelled[eye_id][controls.index] else "PAINT OR LABEL"
            details = f"Control {controls.index + 1}/{len(controls.samples[eye_id])}  {label}  raw image"
            result = controls.results[eye_id]
            score = "" if result is None else f"  example agreement {result.sample_scores[controls.index]:.0%}"
            details2 = f"Green: your pupil | yellow: prediction{score}"
        for row, text in enumerate((details, details2)):
            self.cv2.putText(
                panel,
                text,
                (8, footer_y + 22 + 24 * row),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                _MUTED,
                1,
                self.cv2.LINE_AA,
            )
        return panel

    def control_action(self, action: str, *, path=None, confirm_reset=None) -> None:
        """Shared button commands for native and embedded calibration."""
        controls = self.controls
        was_frozen = controls.sample is not None
        previous_index = controls.index
        self._status = ""
        self._paint_point = None
        self._paint_image = None
        try:
            if action in VIDEO_KEYS.values() and self.video is not None:
                controls._editable()
                origin = self._video_frames.get(id(controls.sample.image)) if was_frozen else None
                origin = origin[1] if origin and origin[0] is self.video.camera else None
                if not self.video.action(action, path=path, origin=origin):
                    return
                if action == "load_video":
                    self._video_frames.clear()
                controls.live()
                self._refresh_live_crops()
                if action in ("frame_back", "frame_forward"):
                    # Move an unfinished still; retain every painted/blink label.
                    if was_frozen:
                        controls.index = previous_index
                    controls.capture(self.crops[controls.eye], replace_unlabelled=True)
            elif action.startswith("eye"):
                controls.select_eye(int(action[-1]))
            elif action == "capture":
                if controls.sample is not None:
                    raise ValueError("Choose Live, then Capture when the next useful image is visible")
                # Freeze the displayed crop, not an unseen later video frame.
                controls.capture(self.crops[controls.eye])
                if self.video is not None:
                    self.video.playing = False
            elif action in ("previous", "next"):
                controls.browse(-1 if action == "previous" else 1)
                if self.video is not None:
                    self.video.playing = False
            elif action == "live":
                controls.live()
            elif action == "no_pupil":
                controls.no_pupil()
            elif action == "delete":
                controls.delete()
            elif action == "reset":
                controls._editable()
                if (confirm_reset or confirm_reset_controls)():
                    controls.reset()
                    self._video_frames.clear()
                    self._predictions = [None for _ in self.settings]
            elif action == "erase":
                controls.erase = not controls.erase
                controls.status = f"Left drag {'erases' if controls.erase else 'paints'}; right drag always erases."
            elif action in ("smaller", "larger"):
                controls.brush = int(np.clip(controls.brush + (-1 if action == "smaller" else 1), 1, 30))
            elif action == "prediction":
                controls.show_prediction = not controls.show_prediction
            elif action == "optimize":
                controls.optimize(self.settings[controls.eye], self.tracker_config)
            elif action == "cancel":
                controls.cancel()
            if (action in ("capture", "frame_back", "frame_forward")
                    and self.video is not None and self.video.camera is not None):
                self._video_frames[id(controls.sample.image)] = (
                    self.video.camera, self.video.camera.frame_index,
                )
            retained = {id(sample.image) for eye in controls.samples for sample in eye}
            self._video_frames = {key: value for key, value in self._video_frames.items() if key in retained}
        except (ValueError, RuntimeError, OSError) as exc:
            controls.status = str(exc)

    def control_buttons(self):
        controls = self.controls
        buttons = [(f"Eye {i}{'*' if i == controls.eye else ''}", f"eye{i}") for i in range(len(self.crops))]
        buttons += [
            ("Capture", "capture"), ("Previous", "previous"), ("Next", "next"), ("Live", "live"),
            ("Use brush" if controls.erase else "Use eraser", "erase"),
            (f"Brush {controls.brush} -", "smaller"), ("Brush +", "larger"),
            ("No pupil", "no_pupil"), ("Delete", "delete"), ("Reset snapshots", "reset"),
            ("Hide fit" if controls.show_prediction else "Show fit", "prediction"),
            ("Cancel optimize", "cancel") if controls.busy else ("Optimize eye", "optimize"),
        ]
        if self.video is not None:
            buttons += self.video.buttons()
        return buttons

    def _poll_optimization(self) -> None:
        completed = self.controls.poll()
        if completed is None:
            return
        eye, result = completed
        self.settings[eye] = result.settings
        self.detectors[eye] = self._detector(result.settings)
        if self._window_open:
            self._initializing_sliders = True
            try:
                for slider in self._sliders:
                    self.cv2.setTrackbarPos(
                        self._trackbar_label(eye, slider.label), self.WINDOW_NAME,
                        slider.position_for(self._slider_value(eye, slider)),
                    )
            finally:
                self._initializing_sliders = False

    def _mouse(self, event: int, x: int, y: int, flags=None, userdata=None) -> None:
        del userdata
        x, y = round(x / self._frame_scale), round(y / self._frame_scale)
        if event == self.cv2.EVENT_LBUTTONDOWN:
            for x0, y0, width, height, action in self._toolbar:
                if x0 <= x < x0 + width and y0 <= y < y0 + height:
                    self.control_action(action)
                    return
        controls = self.controls
        geometry = self._panel_geometry.get(controls.eye)
        if geometry is None or controls.sample is None or controls.busy:
            self._paint_point = None
            self._paint_image = None
            return
        button = self.cv2.EVENT_FLAG_RBUTTON if self._paint_erase else self.cv2.EVENT_FLAG_LBUTTON
        if event == self.cv2.EVENT_MOUSEMOVE and flags is not None and not flags & button:
            # Native windows can lose a release outside their bounds. Tk sends
            # only button-drag motion and leaves flags unspecified.
            self._paint_point = None
            self._paint_image = None
            return
        x0, y0, width, height, scale = geometry
        point = (round((x - x0) / scale), round((y - y0) / scale))
        pressed = event in (self.cv2.EVENT_LBUTTONDOWN, self.cv2.EVENT_RBUTTONDOWN)
        released = event in (self.cv2.EVENT_LBUTTONUP, self.cv2.EVENT_RBUTTONUP)
        inside = x0 <= x < x0 + width and y0 <= y < y0 + height
        if pressed and not inside:
            self._paint_point = None
            self._paint_image = None
            return
        shape = controls.sample.image.shape
        # Commit the final segment even when release lands outside the canvas.
        # OpenCV clips the segment at the image edge without moving its endpoint.
        if pressed:
            point = (min(shape[1] - 1, point[0]), min(shape[0] - 1, point[1]))
            self._paint_point = point
            self._paint_image = controls.sample.image
            self._paint_erase = event == self.cv2.EVENT_RBUTTONDOWN
            controls.paint(point, point, erase=self._paint_erase)
        elif (self._paint_point is not None and self._paint_image is controls.sample.image
              and (event == self.cv2.EVENT_MOUSEMOVE or released)):
            if point != self._paint_point:
                controls.paint(self._paint_point, point, erase=self._paint_erase)
            self._paint_point = point
        if released:
            self._paint_point = None
            self._paint_image = None

    def saved_control_samples(self):
        if self.controls.busy:
            raise ValueError("Wait for optimization or cancel it before saving calibration")
        return self.controls.export()

    def close(self) -> None:
        self.controls.close()

    def _frame(self, *, show_buttons: bool = True) -> np.ndarray:
        self._poll_optimization()
        panels = [self._eye_panel(eye_id) for eye_id in range(len(self.crops))]
        height = max(panel.shape[0] for panel in panels)
        padded: list[np.ndarray] = []
        for panel in panels:
            if panel.shape[0] < height:
                panel = self.cv2.copyMakeBorder(
                    panel,
                    0,
                    height - panel.shape[0],
                    0,
                    0,
                    self.cv2.BORDER_CONSTANT,
                    value=_BACKGROUND,
                )
            padded.append(panel)
        body = self.cv2.hconcat(padded)
        status_lines = [""]
        status = self._status or self.controls.status
        if self.video is not None and self.video.camera is not None:
            status += " | " + self.video.description
            sample = self.controls.sample
            origin = self._video_frames.get(id(sample.image)) if sample else None
            if origin and origin[0] is self.video.camera:
                status += f" | Selected still: frame {origin[1] + 1}"
        for word in status.split():
            proposed = f"{status_lines[-1]} {word}".strip()
            width = self.cv2.getTextSize(proposed, self.cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
            if width > body.shape[1] - 16 and status_lines[-1]:
                status_lines.append(word)
            else:
                status_lines[-1] = proposed
        self._toolbar = []
        buttons = self.control_buttons() if show_buttons else ()
        x, y = 8, 59 + 18 * (len(status_lines) - 1)
        for label, action in buttons:
            width = self.cv2.getTextSize(label, self.cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1)[0][0] + 18
            if x + width > body.shape[1] - 8:
                x, y = 8, y + 29
            self._toolbar.append((x, y, width, 25, action))
            x += width + 5
        banner_height = y + (31 if buttons else 0)
        frame = np.full(
            (banner_height + body.shape[0], body.shape[1], 3),
            _BACKGROUND,
            dtype=np.uint8,
        )
        frame[banner_height:] = body
        self.cv2.putText(
            frame,
            "RAW | 2-6/eye recommended; extras welcome | S save | Esc cancel",
            (8, 22),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            _TEXT,
            1,
            self.cv2.LINE_AA,
        )
        for row, line in enumerate(status_lines):
            self.cv2.putText(
                frame, line, (8, 47 + 18 * row),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                _FIT,
                1,
                self.cv2.LINE_AA,
            )
        for (label, _), (x, y, width, height, _) in zip(buttons, self._toolbar, strict=True):
            self.cv2.rectangle(frame, (x, y), (x + width, y + height), (65, 65, 65), -1)
            self.cv2.putText(frame, label, (x + 8, y + 17), self.cv2.FONT_HERSHEY_SIMPLEX,
                             0.43, _TEXT, 1, self.cv2.LINE_AA)
        offset = 0
        for eye, panel in enumerate(panels):
            x, y, width, height, image_scale = self._panel_geometry[eye]
            self._panel_geometry[eye] = (x + offset, y + banner_height, width, height, image_scale)
            offset += panel.shape[1]
        scale = min(
            1.0,
            self.maximum_display_width / frame.shape[1],
            self.maximum_display_height / frame.shape[0],
        )
        self._frame_scale = scale
        if scale < 1.0:
            frame = self.cv2.resize(
                frame,
                (round(frame.shape[1] * scale), round(frame.shape[0] * scale)),
                interpolation=self.cv2.INTER_AREA,
            )
        return frame

    def run(self) -> tuple[EyeImageSettings, ...] | None:
        self.cv2.namedWindow(self.WINDOW_NAME, self.cv2.WINDOW_NORMAL)
        self._window_open = True
        try:
            self._create_trackbars()
            self.cv2.setMouseCallback(self.WINDOW_NAME, self._mouse)
            while True:
                self._refresh_live_crops()
                self.cv2.imshow(self.WINDOW_NAME, self._frame())
                wait_ms = (8 if self.controls.sample is not None
                           else 1 if self._frame_source is not None else 20)
                key = self.cv2.waitKey(wait_ms) & 0xFF
                if key in VIDEO_KEYS:
                    self.control_action(VIDEO_KEYS[key])
                if key in (13, 10, ord("s"), ord("S")):
                    try:
                        self.saved_control_samples()
                        return tuple(self.settings)
                    except ValueError as exc:
                        self.controls.status = str(exc)
                if key in (27, ord("q"), ord("Q")):
                    return None
                try:
                    visible = self.cv2.getWindowProperty(
                        self.WINDOW_NAME,
                        self.cv2.WND_PROP_VISIBLE,
                    )
                except self.cv2.error:
                    visible = 0.0
                if visible < 1.0:
                    return None
        finally:
            self.close()
            self._window_open = False
            self.cv2.destroyWindow(self.WINDOW_NAME)


def _average_camera_preview(
    camera: Picamera2Camera,
    frame_count: int,
) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    average: np.ndarray | None = None
    for index in range(frame_count):
        frame, _timestamp = camera.capture_preview()
        if average is None:
            average = frame.astype(np.float32)
        else:
            average += (frame.astype(np.float32) - average) / (index + 1)
    if average is None:
        raise RuntimeError("Camera returned no preview frames")
    return np.clip(average, 0, 255).astype(np.uint8)


def oriented_eye_crops(image, boxes, rotations):
    """Extract tuning crops in the same orientation used by live detection."""
    geometry = CameraCanvas(image.shape[1], image.shape[0], rotations)
    crops = []
    for box in boxes:
        geometry.display_roi(box)  # Validate a box does not span rotated cameras.
        index = min(3, int(box.x * 4 / image.shape[1]))
        crops.append(rotate_image(image[box.y:box.y2, box.x:box.x2], rotations[index]).copy(order="C"))
    return tuple(crops)


def _load_existing_layout(
    path: Path,
    image: np.ndarray,
    tracker_config: TrackerConfig,
) -> tuple[tuple[PixelRoi, ...], dict[int, EyeImageSettings], tuple[int, ...]]:
    """Load raw boxes, tuning, and rotations once; old calibrations default to zero."""
    if not path.is_file():
        return (), {}, DEFAULT_ROTATIONS
    layout = RoiLayout.load(path)
    layout.validate_for_frame(
        image.shape[1],
        image.shape[0],
        tracker_config=tracker_config,
    )
    boxes = tuple(
        roi.to_pixels(image.shape[1], image.shape[0]) for roi in layout.rois
    )
    settings = {roi.eye_id: roi.settings for roi in layout.rois}
    return boxes, settings, layout.camera_rotations


def configure_rois(
    config: AppConfig,
    output_path: str | Path,
    *,
    config_path: str | Path | None = None,
    image_path: str | Path | None = None,
    video_path: str | Path | None = None,
    average_frames: int = 8,
    static: bool = False,
) -> Path | None:
    """Edit live/still eye boxes and orientation, then save per-eye detector tuning."""
    if image_path is not None and video_path is not None:
        raise ValueError("image_path and video_path are mutually exclusive")
    cv2 = _require_cv2()
    path = Path(output_path).expanduser()
    source = None
    live_frame_source: Callable[[], np.ndarray] | None = None
    video = CalibrationVideo(config.camera, lambda: live_frame_source() if live_frame_source else image)
    editor: RoiEditor
    try:
        if image_path is not None:
            image = cv2.imread(str(Path(image_path).expanduser()), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ConfigError(f"Could not load preview image: {image_path}")
            configured_ratio = config.camera.analysis_width / config.camera.analysis_height
            image_ratio = image.shape[1] / image.shape[0]
            if abs(image_ratio / configured_ratio - 1.0) > 0.02:
                raise ConfigError(
                    "Preview image aspect ratio does not match the configured stitched stream"
                )
            if image.shape[::-1] != (
                config.camera.analysis_width,
                config.camera.analysis_height,
            ):
                image = cv2.resize(
                    image,
                    (config.camera.analysis_width, config.camera.analysis_height),
                    interpolation=cv2.INTER_AREA,
                )
        elif video_path is not None:
            video.load(video_path)
            image = video.latest()
        else:
            try:
                source = Picamera2Camera(config.camera, config.recording, roi_layout=None)
                source.start()
                image = (_average_camera_preview(source, average_frames) if static
                         else source.capture_preview()[0])
            except CameraError:
                # A desktop without Pi camera hardware can still enter the
                # same calibration interface by selecting a recording.
                video_path = choose_video()
                if not video_path:
                    return None
                if source is not None:
                    source.close()
                    source = None
                video.load(video_path)
                image = video.latest()
            if source is not None and not static:

                def next_live_frame() -> np.ndarray:
                    frame, _frame_timestamp = source.capture_preview()
                    return frame

                live_frame_source = next_live_frame

        existing, existing_settings, rotations = _load_existing_layout(
            path,
            image,
            config.tracker,
        )
        editor_kwargs = {}
        if image_path is None and video_path is None:
            camera = source

            def apply_exposure(camera_config: CameraConfig) -> None:
                camera.set_image_controls(exposure_us=camera_config.exposure_us)

            def recapture(camera_config: CameraConfig) -> np.ndarray:
                camera.set_image_controls(exposure_us=camera_config.exposure_us)
                camera.capture_preview()
                camera.capture_preview()
                return _average_camera_preview(camera, average_frames)

            editor_kwargs = {
                "camera_config": config.camera,
                "control_limits": camera.image_control_limits(),
                "frame_source": live_frame_source,
                "apply_exposure": None if static else apply_exposure,
                "recapture": recapture if static else None,
            }
        if live_frame_source is not None or video.camera is not None:
            editor_kwargs["frame_source"] = video.latest

        editor = RoiEditor(image, initial=existing, camera_rotations=rotations, video=video, **editor_kwargs)
        if live_frame_source is None:
            video.fallback = lambda: editor.image
        selected = editor.run()
        image = editor.image.copy()
        if selected is None:
            return None

        rotations = editor.camera_rotations
        crops = oriented_eye_crops(image, selected, rotations)
        initial_settings = tuple(
            existing_settings.get(
                eye_id,
                EyeImageSettings(pupil_size_bias=config.tracker.pupil_size_bias),
            )
            for eye_id in range(len(crops))
        )

        def next_live_crops() -> tuple[np.ndarray, ...]:
            return oriented_eye_crops(video.latest(), selected, rotations)

        samples, control_warning = load_controls(path)
        tuning = EyeTuningEditor(
            crops,
            initial_settings,
            config.tracker,
            frame_source=next_live_crops,
            control_samples=samples,
            video=video,
        )
        if control_warning:
            tuning.controls.status = control_warning
        tuned_settings = tuning.run()
        if tuned_settings is None:
            return None
    finally:
        video.close()
        if source is not None:
            source.close()

    layout = RoiLayout.from_pixel_boxes(
        selected, tuple(tuned_settings), image.shape[1], image.shape[0],
        camera_rotations=rotations,
    )
    layout.validate_for_frame(
        image.shape[1],
        image.shape[0],
        tracker_config=config.tracker,
    )
    save_control_samples(
        path, tuning.saved_control_samples(),
    )
    saved = layout.save(path)
    if (
        config_path is not None
        and editor.applied_camera_config is not None
        and editor.applied_camera_config.exposure_us != config.camera.exposure_us
    ):
        replace(
            config,
            camera=replace(
                config.camera,
                exposure_us=editor.applied_camera_config.exposure_us,
            ),
        ).save(config_path)
    return saved
