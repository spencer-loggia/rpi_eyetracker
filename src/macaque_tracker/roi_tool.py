from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from .camera import Picamera2Camera, VideoFileCamera
from .config import AppConfig, CameraConfig, ConfigError, RoiLayout, TrackerConfig
from .models import EyeImageSettings, NormalizedRoi, PixelRoi
from .tracker import AdaptivePupilDetector, apply_eye_image_settings


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "The ROI tool requires OpenCV with GUI support (Raspberry Pi package "
            "python3-opencv)."
        ) from exc
    return cv2


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


def _software_sliders() -> tuple[_Slider, ...]:
    return (
        _Slider("gain", "Gain x100", 0.01, 8.0, 100),
        _Slider("brightness", "Brightness +100", -1.0, 1.0, 100),
        _Slider("contrast", "Contrast x100", 0.01, 8.0, 100),
        _Slider("sharpness", "Sharpness x100", 0.0, 8.0, 100),
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
        minimum_width: int = 24,
        minimum_height: int = 24,
        maximum_display_width: int = 1600,
        maximum_display_height: int = 900,
        camera_config: CameraConfig | None = None,
        control_limits: dict[str, tuple[float, float]] | None = None,
        recapture: Callable[[CameraConfig], np.ndarray] | None = None,
    ) -> None:
        if image.dtype != np.uint8 or image.ndim != 2:
            raise ValueError("ROI editor image must be two-dimensional uint8")
        if len(initial) > 2:
            raise ValueError("At most two initial boxes are allowed")
        self.cv2 = _require_cv2()
        self.image = image.copy()
        self.height, self.width = image.shape
        self.minimum_width = minimum_width
        self.minimum_height = minimum_height
        self.boxes = list(initial)
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None
        self.camera_config = camera_config
        self.applied_camera_config = camera_config
        self._recapture = recapture
        self._exposure = (
            None
            if camera_config is None or control_limits is None or recapture is None
            else _exposure_slider(camera_config, control_limits)
        )
        self._initializing_slider = False
        self._status = ""
        self.scale = min(
            1.0,
            maximum_display_width / self.width,
            maximum_display_height / self.height,
        )
        self.display_width = max(1, round(self.width * self.scale))
        self.display_height = max(1, round(self.height * self.scale))

    def _source_point(self, x: int, y: int) -> tuple[int, int]:
        return (
            int(np.clip(round(x / self.scale), 0, self.width - 1)),
            int(np.clip(round(y / self.scale), 0, self.height - 1)),
        )

    def _mouse(self, event: int, x: int, y: int, flags: int, userdata) -> None:
        del flags, userdata
        source = self._source_point(x, y)
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
                self.boxes.append(PixelRoi(left, top, right - left, bottom - top))
            self.drag_start = None
            self.drag_current = None

    def _draw_box(self, canvas: np.ndarray, roi: PixelRoi, index: int, active=False) -> None:
        colour = (
            (0, 210, 255)
            if active
            else ((20, 230, 20) if index == 0 else (255, 170, 20))
        )
        x1 = round(roi.x * self.scale)
        y1 = round(roi.y * self.scale)
        x2 = round(roi.x2 * self.scale)
        y2 = round(roi.y2 * self.scale)
        self.cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        self.cv2.putText(
            canvas,
            f"eye {index}",
            (x1 + 4, max(18, y1 - 5)),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            self.cv2.LINE_AA,
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
                self._status = "Exposure changed; press R to apply and recapture"
        except (ConfigError, TypeError, ValueError) as exc:
            self._status = f"Invalid exposure: {exc}"

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

    def _frame(self) -> np.ndarray:
        canvas = self.cv2.cvtColor(self.image, self.cv2.COLOR_GRAY2BGR)
        if self.scale != 1.0:
            canvas = self.cv2.resize(
                canvas,
                (self.display_width, self.display_height),
                interpolation=self.cv2.INTER_AREA,
            )
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
        instruction = "Drag 1-2 eye boxes | Enter/S next | Backspace/U undo | C clear"
        if self._exposure is not None:
            instruction += " | Exposure: R apply + recapture"
        instruction += " | Q/Esc cancel"
        lines = [instruction]
        if self.camera_config is not None and self._exposure is not None:
            lines.append(f"exposure {self.camera_config.exposure_us} us")
        if self._status:
            lines.append(self._status)
        banner_bottom = 6 + 25 * len(lines)
        self.cv2.rectangle(
            canvas,
            (0, 0),
            (canvas.shape[1] - 1, banner_bottom),
            (0, 0, 0),
            -1,
        )
        for index, line in enumerate(lines):
            self.cv2.putText(
                canvas,
                line,
                (8, 22 + 25 * index),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255) if index < 2 else (0, 210, 255),
                1,
                self.cv2.LINE_AA,
            )
        return canvas

    def run(self) -> tuple[PixelRoi, ...] | None:
        self.cv2.namedWindow(self.WINDOW_NAME, self.cv2.WINDOW_AUTOSIZE)
        self.cv2.setMouseCallback(self.WINDOW_NAME, self._mouse)
        self._create_trackbar()
        try:
            while True:
                self.cv2.imshow(self.WINDOW_NAME, self._frame())
                key = self.cv2.waitKey(20) & 0xFF
                if key in (13, 10, ord("s"), ord("S")) and 1 <= len(self.boxes) <= 2:
                    if self.camera_config != self.applied_camera_config:
                        self._status = "Press R to apply pending exposure before continuing"
                    else:
                        return tuple(self.boxes)
                elif key in (8, 127, ord("u"), ord("U")) and self.boxes:
                    self.boxes.pop()
                elif key in (ord("c"), ord("C")):
                    self.boxes.clear()
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
    """Phase two: tune detection and software controls per eye crop."""

    WINDOW_NAME = "Macaque pupil tuning"

    def __init__(
        self,
        crops: tuple[np.ndarray, ...],
        initial: tuple[EyeImageSettings, ...],
        tracker_config: TrackerConfig,
        *,
        maximum_display_width: int = 1600,
        maximum_display_height: int = 900,
    ) -> None:
        if not 1 <= len(crops) <= 2 or len(initial) != len(crops):
            raise ValueError("Pupil tuning requires matching settings for one or two crops")
        if any(crop.dtype != np.uint8 or crop.ndim != 2 for crop in crops):
            raise ValueError("Pupil tuning crops must be two-dimensional uint8")
        self.cv2 = _require_cv2()
        self.crops = tuple(crop.copy() for crop in crops)
        self.settings = list(initial)
        self.tracker_config = tracker_config
        self.detectors = [self._detector(settings) for settings in self.settings]
        self._sliders = _software_sliders()
        self._initializing_sliders = False
        self._status = ""
        self.maximum_display_width = maximum_display_width
        self.maximum_display_height = maximum_display_height

    def _detector(self, settings: EyeImageSettings) -> AdaptivePupilDetector:
        return AdaptivePupilDetector(
            replace(
                self.tracker_config,
                pupil_threshold=settings.pupil_threshold,
            )
        )

    @staticmethod
    def _trackbar_label(eye_id: int, label: str) -> str:
        return f"Eye {eye_id} {label}"

    def _threshold_changed(self, eye_id: int, position: int) -> None:
        try:
            threshold = None if position == 0 else position
            self.settings[eye_id] = replace(
                self.settings[eye_id],
                pupil_threshold=threshold,
            )
            self.detectors[eye_id] = self._detector(self.settings[eye_id])
            if not self._initializing_sliders:
                value = "AUTO" if threshold is None else str(threshold)
                self._status = f"Eye {eye_id} pupil threshold {value}"
        except (TypeError, ValueError) as exc:
            self._status = f"Invalid eye {eye_id} threshold: {exc}"

    def _software_changed(self, eye_id: int, slider: _Slider, position: int) -> None:
        try:
            clipped = int(
                np.clip(position, slider.minimum_position, slider.maximum_position)
            )
            label = self._trackbar_label(eye_id, slider.label)
            if clipped != position:
                self.cv2.setTrackbarPos(label, self.WINDOW_NAME, clipped)
            value = slider.value_for(clipped)
            self.settings[eye_id] = replace(
                self.settings[eye_id],
                **{slider.config_name: value},
            )
            if not self._initializing_sliders:
                self._status = f"Eye {eye_id} {slider.config_name} {float(value):.2f}"
        except (TypeError, ValueError) as exc:
            self._status = f"Invalid eye {eye_id} image adjustment: {exc}"

    def _create_trackbars(self) -> None:
        self._initializing_sliders = True
        try:
            for eye_id, settings in enumerate(self.settings):
                self.cv2.createTrackbar(
                    self._trackbar_label(eye_id, "Pupil threshold (0 auto)"),
                    self.WINDOW_NAME,
                    settings.pupil_threshold or 0,
                    254,
                    lambda position, selected=eye_id: self._threshold_changed(
                        selected,
                        position,
                    ),
                )
                for slider in self._sliders:
                    label = self._trackbar_label(eye_id, slider.label)
                    self.cv2.createTrackbar(
                        label,
                        self.WINDOW_NAME,
                        slider.position_for(getattr(settings, slider.config_name)),
                        slider.maximum_position,
                        lambda position, selected=eye_id, control=slider: (
                            self._software_changed(selected, control, position)
                        ),
                    )
                    named_slider = _Slider(
                        slider.config_name,
                        label,
                        slider.minimum,
                        slider.maximum,
                        slider.scale,
                    )
                    _set_trackbar_minimum(self.cv2, self.WINDOW_NAME, named_slider)
        finally:
            self._initializing_sliders = False

    def _eye_panel(self, eye_id: int) -> np.ndarray:
        settings = self.settings[eye_id]
        adjusted = apply_eye_image_settings(self.crops[eye_id], settings)
        candidate, mask = self.detectors[eye_id].detect_with_mask(adjusted)
        canvas = self.cv2.cvtColor(adjusted, self.cv2.COLOR_GRAY2BGR)
        selected = mask != 0
        colour = np.asarray((255, 30, 220), dtype=np.float32)
        canvas[selected] = np.clip(
            0.25 * canvas[selected].astype(np.float32) + 0.75 * colour,
            0,
            255,
        ).astype(np.uint8)
        if candidate is not None:
            ellipse = (
                (candidate.ellipse_x, candidate.ellipse_y),
                (candidate.ellipse_width, candidate.ellipse_height),
                candidate.angle_degrees,
            )
            self.cv2.ellipse(canvas, ellipse, (255, 255, 255), 1, self.cv2.LINE_AA)

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
        panel = np.zeros(
            (header_height + canvas.shape[0] + footer_height, panel_width, 3),
            dtype=np.uint8,
        )
        content_x = (panel_width - canvas.shape[1]) // 2
        panel[
            header_height : header_height + canvas.shape[0],
            content_x : content_x + canvas.shape[1],
        ] = canvas
        fit = (
            "NO PUPIL FIT"
            if candidate is None
            else f"pupil {candidate.diameter:.1f}px conf {candidate.confidence:.2f}"
        )
        threshold = "auto" if settings.pupil_threshold is None else settings.pupil_threshold
        self.cv2.putText(
            panel,
            f"EYE {eye_id}  {fit}",
            (8, 24),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            1,
            self.cv2.LINE_AA,
        )
        footer_y = header_height + canvas.shape[0]
        details = (
            f"threshold {threshold}  gain {settings.gain:.2f}  "
            f"brightness {settings.brightness:.2f}"
        )
        details2 = f"contrast {settings.contrast:.2f}  sharpness {settings.sharpness:.2f}"
        for row, text in enumerate((details, details2)):
            self.cv2.putText(
                panel,
                text,
                (8, footer_y + 22 + 24 * row),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.47,
                (220, 220, 220),
                1,
                self.cv2.LINE_AA,
            )
        return panel

    def _frame(self) -> np.ndarray:
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
                    value=(0, 0, 0),
                )
            padded.append(panel)
        body = self.cv2.hconcat(padded)
        banner_height = 58 if self._status else 33
        frame = np.zeros((banner_height + body.shape[0], body.shape[1], 3), dtype=np.uint8)
        frame[banner_height:] = body
        self.cv2.putText(
            frame,
            "Tune each eye independently | Enter/S save | Q/Esc cancel",
            (8, 22),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            self.cv2.LINE_AA,
        )
        if self._status:
            self.cv2.putText(
                frame,
                self._status,
                (8, 47),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 210, 255),
                1,
                self.cv2.LINE_AA,
            )
        scale = min(
            1.0,
            self.maximum_display_width / frame.shape[1],
            self.maximum_display_height / frame.shape[0],
        )
        if scale < 1.0:
            frame = self.cv2.resize(
                frame,
                (round(frame.shape[1] * scale), round(frame.shape[0] * scale)),
                interpolation=self.cv2.INTER_AREA,
            )
        return frame

    def run(self) -> tuple[EyeImageSettings, ...] | None:
        self.cv2.namedWindow(self.WINDOW_NAME, self.cv2.WINDOW_NORMAL)
        self._create_trackbars()
        try:
            while True:
                self.cv2.imshow(self.WINDOW_NAME, self._frame())
                key = self.cv2.waitKey(20) & 0xFF
                if key in (13, 10, ord("s"), ord("S")):
                    return tuple(self.settings)
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
            self.cv2.destroyWindow(self.WINDOW_NAME)


def _average_camera_preview(camera: Picamera2Camera, frame_count: int) -> np.ndarray:
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


def _load_existing_layout(
    path: Path,
    image: np.ndarray,
) -> tuple[tuple[PixelRoi, ...], dict[int, EyeImageSettings]]:
    if not path.is_file():
        return (), {}
    layout = RoiLayout.load(path)
    layout.validate_for_frame(image.shape[1], image.shape[0])
    boxes = tuple(
        roi.to_pixels(image.shape[1], image.shape[0]) for roi in layout.rois
    )
    settings = {roi.eye_id: roi.settings for roi in layout.rois}
    return boxes, settings


def configure_rois(
    config: AppConfig,
    output_path: str | Path,
    *,
    config_path: str | Path | None = None,
    image_path: str | Path | None = None,
    video_path: str | Path | None = None,
    average_frames: int = 8,
) -> Path | None:
    if image_path is not None and video_path is not None:
        raise ValueError("image_path and video_path are mutually exclusive")
    cv2 = _require_cv2()
    path = Path(output_path).expanduser()
    editor: RoiEditor

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
        existing, existing_settings = _load_existing_layout(path, image)
        editor = RoiEditor(image, initial=existing)
        selected = editor.run()
    elif video_path is not None:
        video = VideoFileCamera(
            video_path,
            config.camera,
            roi_layout=None,
            realtime=False,
        )
        try:
            video.start()
            image = _average_camera_preview(video, average_frames)
        finally:
            video.close()
        existing, existing_settings = _load_existing_layout(path, image)
        editor = RoiEditor(image, initial=existing)
        selected = editor.run()
    else:
        camera = Picamera2Camera(config.camera, config.recording, roi_layout=None)
        try:
            camera.start()
            image = _average_camera_preview(camera, average_frames)
            existing, existing_settings = _load_existing_layout(path, image)

            def recapture(camera_config: CameraConfig) -> np.ndarray:
                camera.set_image_controls(exposure_us=camera_config.exposure_us)
                camera.capture_preview()
                camera.capture_preview()
                return _average_camera_preview(camera, average_frames)

            editor = RoiEditor(
                image,
                initial=existing,
                camera_config=config.camera,
                control_limits=camera.image_control_limits(),
                recapture=recapture,
            )
            selected = editor.run()
            image = editor.image.copy()
        finally:
            camera.close()

    if selected is None:
        return None

    crops = tuple(box.extract(image) for box in selected)
    initial_settings = tuple(
        existing_settings.get(
            eye_id,
            EyeImageSettings(pupil_threshold=config.tracker.pupil_threshold),
        )
        for eye_id in range(len(crops))
    )
    tuning = EyeTuningEditor(crops, initial_settings, config.tracker)
    tuned_settings = tuning.run()
    if tuned_settings is None:
        return None

    rois = tuple(
        NormalizedRoi.from_pixels(
            eye_id=index,
            label=f"eye_{index}",
            roi=box,
            frame_width=image.shape[1],
            frame_height=image.shape[0],
            settings=tuned_settings[index],
        )
        for index, box in enumerate(selected)
    )
    saved = RoiLayout(
        rois=rois,
        source_width=image.shape[1],
        source_height=image.shape[0],
    ).save(path)
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
