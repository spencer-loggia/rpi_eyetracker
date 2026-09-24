from __future__ import annotations

from pathlib import Path

import numpy as np

from .camera import Picamera2Camera
from .config import AppConfig, ConfigError, RoiLayout
from .models import NormalizedRoi, PixelRoi


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "The ROI tool requires OpenCV with GUI support (Raspberry Pi package "
            "python3-opencv)."
        ) from exc
    return cv2


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
        colour = (0, 210, 255) if active else ((20, 230, 20) if index == 0 else (255, 170, 20))
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
        instruction = (
            "Drag 1-2 eye boxes | Enter/S save | Backspace/U undo | C clear | Q/Esc cancel"
        )
        self.cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 1050), 31), (0, 0, 0), -1)
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
        return canvas

    def run(self) -> tuple[PixelRoi, ...] | None:
        self.cv2.namedWindow(self.WINDOW_NAME, self.cv2.WINDOW_AUTOSIZE)
        self.cv2.setMouseCallback(self.WINDOW_NAME, self._mouse)
        try:
            while True:
                self.cv2.imshow(self.WINDOW_NAME, self._frame())
                key = self.cv2.waitKey(20) & 0xFF
                if key in (13, 10, ord("s")) and 1 <= len(self.boxes) <= 2:
                    return tuple(self.boxes)
                if key in (8, 127, ord("u")) and self.boxes:
                    self.boxes.pop()
                elif key == ord("c"):
                    self.boxes.clear()
                elif key in (27, ord("q")):
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
    else:
        camera = Picamera2Camera(config.camera, config.recording, roi_layout=None)
        try:
            camera.start()
            image = _average_camera_preview(camera, average_frames)
        finally:
            camera.close()

    existing: tuple[PixelRoi, ...] = ()
    path = Path(output_path).expanduser()
    if path.is_file():
        layout = RoiLayout.load(path)
        layout.validate_for_frame(image.shape[1], image.shape[0])
        existing = tuple(roi.to_pixels(image.shape[1], image.shape[0]) for roi in layout.rois)
    selected = RoiEditor(image, initial=existing).run()
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
    return RoiLayout(
        rois=rois,
        source_width=image.shape[1],
        source_height=image.shape[0],
    ).save(path)
