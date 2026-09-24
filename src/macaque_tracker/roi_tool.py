from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .camera import Picamera2Camera
from .config import AppConfig, CameraConfig, ConfigError, RoiLayout, TrackerConfig
from .models import NormalizedRoi, PixelRoi
from .tracker import AdaptivePupilDetector, PupilCandidate


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "The ROI tool requires OpenCV with GUI support (Raspberry Pi package "
            "python3-opencv)."
        ) from exc
    return cv2


@dataclass(frozen=True)
class _ControlSlider:
    config_name: str
    label: str
    minimum: float
    maximum: float
    scale: int

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
        return round(value) if self.config_name == "exposure_us" else float(value)


def _control_sliders(
    camera_config: CameraConfig,
    limits: dict[str, tuple[float, float]],
) -> tuple[_ControlSlider, ...]:
    frame_exposure_limit = max(1, math.ceil(1_000_000.0 / camera_config.fps) - 1)
    definitions = (
        ("exposure_us", "Exposure us", 1.0, float(frame_exposure_limit), 1),
        ("analogue_gain", "Gain x100", 0.01, 32.0, 100),
        ("brightness", "Brightness +100", -1.0, 1.0, 100),
        ("contrast", "Contrast x100", 0.01, 32.0, 100),
        ("sharpness", "Sharpness x100", 0.0, 16.0, 100),
    )
    sliders: list[_ControlSlider] = []
    for config_name, label, safe_minimum, safe_maximum, scale in definitions:
        if config_name not in limits:
            continue
        reported_minimum, reported_maximum = limits[config_name]
        minimum = max(float(reported_minimum), safe_minimum)
        maximum = min(float(reported_maximum), safe_maximum)
        if config_name == "exposure_us":
            minimum = float(math.ceil(minimum))
            maximum = float(math.floor(maximum))
        if minimum <= maximum:
            sliders.append(_ControlSlider(config_name, label, minimum, maximum, scale))
    return tuple(sliders)


class RoiEditor:
    WINDOW_NAME = "Macaque eye ROI configuration"

    def __init__(
        self,
        image: np.ndarray,
        initial: tuple[PixelRoi, ...] = (),
        *,
        minimum_width: int = 24,
        minimum_height: int = 24,
        maximum_display_width: int = 1600,
        maximum_display_height: int = 900,
        tracker_config: TrackerConfig | None = None,
        camera_config: CameraConfig | None = None,
        control_limits: dict[str, tuple[float, float]] | None = None,
        recapture: Callable[[CameraConfig], np.ndarray] | None = None,
    ) -> None:
        if image.dtype != np.uint8 or image.ndim != 2:
            raise ValueError("ROI editor image must be two-dimensional uint8")
        if len(initial) > 2:
            raise ValueError("At most two initial boxes are allowed")
        self.cv2 = _require_cv2()
        self.image = image
        self.height, self.width = image.shape
        self.minimum_width = minimum_width
        self.minimum_height = minimum_height
        self.boxes = list(initial)
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None
        self.cancelled = False
        self.detector = (
            None if tracker_config is None else AdaptivePupilDetector(tracker_config)
        )
        self.camera_config = camera_config
        self.applied_camera_config = camera_config
        self._recapture = recapture
        self._sliders = (
            ()
            if camera_config is None or control_limits is None or recapture is None
            else _control_sliders(camera_config, control_limits)
        )
        self._initializing_sliders = False
        self._analysis_image: np.ndarray | None = None
        self._detections: dict[int, PupilCandidate | None] = {}
        self._status = ""
        self.scale = min(
            1.0,
            maximum_display_width / self.width,
            maximum_display_height / self.height,
        )
        self.display_width = max(1, round(self.width * self.scale))
        self.display_height = max(1, round(self.height * self.scale))

    def _invalidate_analysis(self) -> None:
        self._analysis_image = None
        self._detections.clear()

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
                self._invalidate_analysis()
            self.drag_start = None
            self.drag_current = None

    def _draw_box(self, canvas: np.ndarray, roi: PixelRoi, index: int, active=False) -> None:
        colour = (0, 210, 255) if active else ((20, 230, 20) if index == 0 else (255, 170, 20))
        x1 = round(roi.x * self.scale)
        y1 = round(roi.y * self.scale)
        x2 = round(roi.x2 * self.scale)
        y2 = round(roi.y2 * self.scale)
        self.cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        candidate = self._detections.get(index)
        detail = (
            ""
            if active
            else (
                "  NO PUPIL FIT"
                if candidate is None
                else f"  pupil {candidate.diameter:.1f}px conf {candidate.confidence:.2f}"
            )
        )
        self.cv2.putText(
            canvas,
            f"eye {index}{detail}",
            (x1 + 4, max(18, y1 - 5)),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            self.cv2.LINE_AA,
        )

    def _analysed_source_image(self) -> np.ndarray:
        if self._analysis_image is not None:
            return self._analysis_image
        canvas = self.cv2.cvtColor(self.image, self.cv2.COLOR_GRAY2BGR)
        self._detections = {}
        if self.detector is not None:
            colours = ((255, 30, 220), (30, 170, 255))
            for index, roi in enumerate(self.boxes):
                candidate, mask = self.detector.detect_with_mask(roi.extract(self.image))
                self._detections[index] = candidate
                if candidate is None:
                    continue
                region = canvas[roi.y : roi.y2, roi.x : roi.x2]
                selected = mask != 0
                colour = np.asarray(colours[index], dtype=np.float32)
                region[selected] = np.clip(
                    0.25 * region[selected].astype(np.float32) + 0.75 * colour,
                    0,
                    255,
                ).astype(np.uint8)
                ellipse = (
                    (candidate.ellipse_x + roi.x, candidate.ellipse_y + roi.y),
                    (candidate.ellipse_width, candidate.ellipse_height),
                    candidate.angle_degrees,
                )
                self.cv2.ellipse(canvas, ellipse, (255, 255, 255), 1, self.cv2.LINE_AA)
        self._analysis_image = canvas
        return canvas

    def _settings_text(self) -> str:
        config = self.camera_config
        if config is None or not self._sliders:
            return ""
        return (
            f"exposure {config.exposure_us} us | gain {config.analogue_gain:.2f} | "
            f"brightness {config.brightness:.2f} | contrast {config.contrast:.2f} | "
            f"sharpness {config.sharpness:.2f}"
        )

    def _slider_changed(self, slider: _ControlSlider, position: int) -> None:
        if self.camera_config is None:
            return
        try:
            clipped_position = int(
                np.clip(position, slider.minimum_position, slider.maximum_position)
            )
            if clipped_position != position:
                self.cv2.setTrackbarPos(
                    slider.label,
                    self.WINDOW_NAME,
                    clipped_position,
                )
            self.camera_config = replace(
                self.camera_config,
                **{slider.config_name: slider.value_for(clipped_position)},
            )
            if not self._initializing_sliders:
                self._status = "Controls changed; press R to apply and recapture"
        except (ConfigError, TypeError, ValueError) as exc:
            self._status = f"Invalid control value: {exc}"

    def _create_trackbars(self) -> None:
        if self.camera_config is None:
            return
        self._initializing_sliders = True
        try:
            for slider in self._sliders:
                initial = slider.position_for(getattr(self.camera_config, slider.config_name))
                self.cv2.createTrackbar(
                    slider.label,
                    self.WINDOW_NAME,
                    initial,
                    slider.maximum_position,
                    lambda position, selected=slider: self._slider_changed(selected, position),
                )
                set_trackbar_minimum = getattr(self.cv2, "setTrackbarMin", None)
                if callable(set_trackbar_minimum):
                    try:
                        set_trackbar_minimum(
                            slider.label,
                            self.WINDOW_NAME,
                            slider.minimum_position,
                        )
                    except self.cv2.error:
                        pass
        finally:
            self._initializing_sliders = False

    def _recapture_image(self) -> None:
        if self._recapture is None or self.camera_config is None:
            return
        try:
            image = self._recapture(self.camera_config)
            if image.dtype != np.uint8 or image.ndim != 2 or image.shape != self.image.shape:
                raise ValueError("Recaptured image dimensions or type changed")
            self.image = image
            self.applied_camera_config = self.camera_config
            self._invalidate_analysis()
            self._status = "Recaptured with displayed controls"
        except Exception as exc:  # noqa: BLE001 - keep editor open for correction/retry
            self._status = f"Recapture failed: {exc}"

    def _frame(self) -> np.ndarray:
        canvas = self._analysed_source_image().copy()
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
        recapture_help = (
            " | R apply controls + recapture" if self._recapture is not None else ""
        )
        instruction = (
            "Drag 1-2 eye boxes | Enter/S save | Backspace/U undo | C clear"
            f"{recapture_help} | Q/Esc cancel"
        )
        settings = self._settings_text()
        banner_height = 31 + (25 if settings else 0) + (25 if self._status else 0)
        self.cv2.rectangle(
            canvas,
            (0, 0),
            (canvas.shape[1] - 1, banner_height),
            (0, 0, 0),
            -1,
        )
        self.cv2.putText(
            canvas,
            instruction,
            (8, 22),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            self.cv2.LINE_AA,
        )
        if settings:
            self.cv2.putText(
                canvas,
                settings,
                (8, 47),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                self.cv2.LINE_AA,
            )
        if self._status:
            self.cv2.putText(
                canvas,
                self._status,
                (8, 47 + (25 if settings else 0)),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 210, 255),
                1,
                self.cv2.LINE_AA,
            )
        return canvas

    def run(self) -> tuple[PixelRoi, ...] | None:
        self.cv2.namedWindow(self.WINDOW_NAME, self.cv2.WINDOW_AUTOSIZE)
        self.cv2.setMouseCallback(self.WINDOW_NAME, self._mouse)
        self._create_trackbars()
        try:
            while True:
                self.cv2.imshow(self.WINDOW_NAME, self._frame())
                key = self.cv2.waitKey(20) & 0xFF
                if key in (13, 10, ord("s"), ord("S")) and 1 <= len(self.boxes) <= 2:
                    if self.camera_config != self.applied_camera_config:
                        self._status = "Press R to apply pending controls before saving"
                    else:
                        return tuple(self.boxes)
                if key in (8, 127, ord("u"), ord("U")) and self.boxes:
                    self.boxes.pop()
                    self._invalidate_analysis()
                elif key in (ord("c"), ord("C")):
                    self.boxes.clear()
                    self._invalidate_analysis()
                elif key in (ord("r"), ord("R")):
                    self._recapture_image()
                elif key in (27, ord("q"), ord("Q")):
                    self.cancelled = True
                    return None
                try:
                    visible = self.cv2.getWindowProperty(
                        self.WINDOW_NAME, self.cv2.WND_PROP_VISIBLE
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


def configure_rois(
    config: AppConfig,
    output_path: str | Path,
    *,
    config_path: str | Path | None = None,
    image_path: str | Path | None = None,
    average_frames: int = 8,
) -> Path | None:
    cv2 = _require_cv2()
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
    existing: tuple[PixelRoi, ...] = ()
    path = Path(output_path).expanduser()
    editor: RoiEditor
    if image_path is not None:
        if path.is_file():
            layout = RoiLayout.load(path)
            layout.validate_for_frame(image.shape[1], image.shape[0])
            existing = tuple(
                roi.to_pixels(image.shape[1], image.shape[0]) for roi in layout.rois
            )
        editor = RoiEditor(image, initial=existing, tracker_config=config.tracker)
        selected = editor.run()
    else:
        camera = Picamera2Camera(config.camera, config.recording, roi_layout=None)
        try:
            camera.start()
            image = _average_camera_preview(camera, average_frames)
            if path.is_file():
                layout = RoiLayout.load(path)
                layout.validate_for_frame(image.shape[1], image.shape[0])
                existing = tuple(
                    roi.to_pixels(image.shape[1], image.shape[0]) for roi in layout.rois
                )

            control_limits = camera.image_control_limits()
            adjustable_names = {
                slider.config_name for slider in _control_sliders(config.camera, control_limits)
            }

            def recapture(camera_config: CameraConfig) -> np.ndarray:
                camera.set_image_controls(
                    **{
                        name: getattr(camera_config, name)
                        for name in adjustable_names
                    }
                )
                # Do not include requests that were already queued when the
                # controls changed in the newly displayed average.
                camera.capture_preview()
                camera.capture_preview()
                return _average_camera_preview(camera, average_frames)

            editor = RoiEditor(
                image,
                initial=existing,
                tracker_config=config.tracker,
                camera_config=config.camera,
                control_limits=control_limits,
                recapture=recapture,
            )
            selected = editor.run()
        finally:
            camera.close()
    if selected is None:
        return None
    rois = tuple(
        NormalizedRoi.from_pixels(
            eye_id=index,
            label=f"eye_{index}",
            roi=box,
            frame_width=image.shape[1],
            frame_height=image.shape[0],
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
        and editor.applied_camera_config != config.camera
    ):
        replace(config, camera=editor.applied_camera_config).save(config_path)
    return saved
