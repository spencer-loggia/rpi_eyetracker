from __future__ import annotations

import multiprocessing as mp
import queue
import sys
import threading
import time
from typing import Any

import numpy as np

from .config import PreviewConfig
from .models import AnalysisFrame, EyeImageSettings, EyeMeasurement, FrameResult, GrayImage
from .tracker import apply_eye_image_settings, pupil_mask_for_threshold

try:  # Configuration and protocol tools remain usable without the vision extra.
    import cv2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised on hosts without the vision extra
    cv2 = None


Color = tuple[int, int, int]

# OpenCV display colours are BGR. Camera and tracker arrays remain single-channel;
# these values are used only on disposable preview canvases.
_TRACKED: Color = (80, 220, 80)
_FIT: Color = (0, 215, 255)
_BLINK: Color = (0, 165, 255)
_NO_FIT: Color = (80, 80, 235)
_MASK: Color = (255, 190, 50)
_WHITE: Color = (235, 235, 235)
_GRAY: Color = (160, 160, 160)
_BACKGROUND: Color = (24, 24, 24)


def _require_opencv() -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for the live preview. Install the vision extra "
            "or Raspberry Pi OS package python3-opencv."
        )


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: Color = _WHITE,
    scale: float = 0.48,
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _measurement_status(measurement: EyeMeasurement) -> tuple[str, Color]:
    if measurement.valid:
        return "TRACKING", _TRACKED
    if measurement.blink:
        return "BLINK / OCCLUDED", _BLINK
    return "NO FIT", _NO_FIT


def _draw_crosshair(
    image: np.ndarray,
    x: float,
    y: float,
    color: Color,
    radius: int = 7,
) -> None:
    center = (round(x), round(y))
    cv2.line(
        image,
        (center[0] - radius, center[1]),
        (center[0] + radius, center[1]),
        color,
        1,
    )
    cv2.line(
        image,
        (center[0], center[1] - radius),
        (center[0], center[1] + radius),
        color,
        1,
    )


def _eye_panel(
    eye_id: int,
    crop: GrayImage,
    measurement: EyeMeasurement,
    diagnostics: dict[str, Any],
    preview_config: PreviewConfig,
    image_settings: EyeImageSettings,
) -> np.ndarray:
    processed = apply_eye_image_settings(crop, image_settings)
    crop_canvas = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    status, status_color = _measurement_status(measurement)

    if diagnostics.get("detected"):
        ellipse = (
            (
                float(diagnostics["ellipse_x"]),
                float(diagnostics["ellipse_y"]),
            ),
            (
                float(diagnostics["ellipse_width"]),
                float(diagnostics["ellipse_height"]),
            ),
            float(diagnostics["ellipse_angle_degrees"]),
        )
        cv2.ellipse(crop_canvas, ellipse, _FIT, 2, cv2.LINE_AA)
        _draw_crosshair(
            crop_canvas,
            float(diagnostics["candidate_x"]),
            float(diagnostics["candidate_y"]),
            _FIT,
            radius=5,
        )
    if measurement.x or measurement.y or measurement.valid:
        _draw_crosshair(
            crop_canvas,
            measurement.x,
            measurement.y,
            _TRACKED,
            radius=8,
        )

    views = [crop_canvas]
    if preview_config.show_threshold_mask:
        threshold = diagnostics.get("threshold")
        if isinstance(threshold, (int, float)):
            # Recomputed in this separate process using the detector's shared
            # segmentation helper, so it cannot delay acquisition or tracking.
            mask = pupil_mask_for_threshold(processed, int(threshold))
            mask_canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
            mask_canvas[mask != 0] = _MASK
        else:
            mask_canvas = np.zeros((*processed.shape, 3), dtype=np.uint8)
            _put_text(
                mask_canvas,
                "no selected mask",
                (8, max(18, crop.shape[0] // 2)),
                color=_GRAY,
            )
        views.append(mask_canvas)
    content = cv2.hconcat(views)
    content_scale = min(3.0, max(1.0, 280.0 / content.shape[0]))
    if content_scale > 1.0:
        content = cv2.resize(
            content,
            None,
            fx=content_scale,
            fy=content_scale,
            interpolation=cv2.INTER_NEAREST,
        )

    header_height = 34
    footer_height = 66
    panel_width = max(720, content.shape[1])
    panel = np.full(
        (header_height + content.shape[0] + footer_height, panel_width, 3),
        _BACKGROUND,
        dtype=np.uint8,
    )
    content_x = (panel_width - content.shape[1]) // 2
    panel[
        header_height : header_height + content.shape[0],
        content_x : content_x + content.shape[1],
    ] = content
    cv2.rectangle(
        panel,
        (0, 0),
        (panel.shape[1] - 1, panel.shape[0] - 1),
        status_color,
        2,
    )
    _put_text(
        panel,
        f"EYE {eye_id}  {status}",
        (9, 23),
        color=status_color,
        scale=0.58,
        thickness=2,
    )

    footer_y = header_height + content.shape[0]
    _put_text(
        panel,
        f"x {measurement.x:.1f}  y {measurement.y:.1f}  "
        f"diameter {measurement.pupil_diameter:.1f}px  conf {measurement.confidence:.2f}",
        (9, footer_y + 23),
    )
    if diagnostics.get("detected"):
        detail = (
            f"fit {float(diagnostics['ellipse_width']):.1f}x"
            f"{float(diagnostics['ellipse_height']):.1f}px  "
            f"threshold {int(diagnostics['threshold'])}  "
            f"contrast {float(diagnostics['contrast']):.1f}  "
            f"size bias {image_settings.pupil_size_bias:+.2f}"
        )
    else:
        detail = f"missing frames {int(diagnostics.get('missing_frames', 0))}"
    _put_text(panel, detail, (9, footer_y + 48), color=_GRAY)
    return panel


def render_preview(
    frame: AnalysisFrame,
    result: FrameResult,
    preview_config: PreviewConfig,
    *,
    display_fps: float = 0.0,
    now_ns: int | None = None,
    image_settings: dict[int, EyeImageSettings] | None = None,
) -> np.ndarray:
    """Render one diagnostic dashboard frame without opening a window."""

    _require_opencv()
    crops = dict(frame.crops)
    measurements = {eye.eye_id: eye for eye in result.eyes}
    if set(crops) != set(measurements):
        raise ValueError("Preview frame eye IDs do not match tracker results")
    raw_eye_diagnostics = result.diagnostics.get("eyes", {})
    eye_diagnostics = raw_eye_diagnostics if isinstance(raw_eye_diagnostics, dict) else {}
    configured_settings = {} if image_settings is None else image_settings

    panels = [
        _eye_panel(
            eye_id,
            crops[eye_id],
            measurements[eye_id],
            eye_diagnostics.get(str(eye_id), {}),
            preview_config,
            configured_settings.get(eye_id, EyeImageSettings()),
        )
        for eye_id in sorted(crops)
    ]
    maximum_height = max(panel.shape[0] for panel in panels)
    padded: list[np.ndarray] = []
    for panel in panels:
        if panel.shape[0] < maximum_height:
            panel = cv2.copyMakeBorder(
                panel,
                0,
                maximum_height - panel.shape[0],
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=_BACKGROUND,
            )
        padded.append(panel)
    body = cv2.hconcat(padded)

    top_height = 42
    bottom_height = 32
    dashboard = np.full(
        (top_height + body.shape[0] + bottom_height, body.shape[1], 3),
        _BACKGROUND,
        dtype=np.uint8,
    )
    dashboard[top_height : top_height + body.shape[0]] = body
    current_ns = time.monotonic_ns() if now_ns is None else now_ns
    age_ms = max(0.0, (current_ns - result.produced_timestamp_ns) / 1_000_000.0)
    _put_text(
        dashboard,
        f"frame {result.frame_sequence}  tracker {result.processing_time_us / 1000.0:.2f} ms  "
        f"display {display_fps:.1f} Hz  drops {result.dropped_analysis_frames}  "
        f"display age {age_ms:.1f} ms",
        (10, 27),
        scale=0.55,
    )
    _put_text(
        dashboard,
        "yellow: fitted ellipse/center   green: reported center   Q or Esc: close preview",
        (10, top_height + body.shape[0] + 22),
        color=_GRAY,
        scale=0.43,
    )

    height, width = dashboard.shape[:2]
    scale = min(
        1.0,
        preview_config.max_display_width / width,
        preview_config.max_display_height / height,
    )
    if scale < 1.0:
        dashboard = cv2.resize(
            dashboard,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return dashboard


def _preview_process(
    packets: Any,
    errors: Any,
    stop_event: Any,
    closed_event: Any,
    preview_config: PreviewConfig,
    image_settings: dict[int, EyeImageSettings],
) -> None:
    window_created = False
    try:
        _require_opencv()
        cv2.namedWindow(preview_config.window_name, cv2.WINDOW_NORMAL)
        window_created = True
        previous_display_ns: int | None = None
        display_fps = 0.0
        while not stop_event.is_set():
            try:
                packet = packets.get(timeout=0.1)
            except queue.Empty:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                continue

            # Drain the size-one/latest queue defensively in case the platform's
            # multiprocessing feeder briefly allowed more than one item through.
            while True:
                try:
                    packet = packets.get_nowait()
                except queue.Empty:
                    break
            frame, result = packet
            display_ns = time.monotonic_ns()
            if previous_display_ns is not None and display_ns > previous_display_ns:
                instantaneous = 1e9 / (display_ns - previous_display_ns)
                display_fps = (
                    instantaneous
                    if display_fps == 0.0
                    else 0.15 * instantaneous + 0.85 * display_fps
                )
            previous_display_ns = display_ns
            dashboard = render_preview(
                frame,
                result,
                preview_config,
                display_fps=display_fps,
                now_ns=display_ns,
                image_settings=image_settings,
            )
            cv2.imshow(preview_config.window_name, dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            try:
                visible = cv2.getWindowProperty(
                    preview_config.window_name,
                    cv2.WND_PROP_VISIBLE,
                )
            except cv2.error:
                break
            if visible < 1.0:
                break
    except BaseException as exc:  # noqa: BLE001 - process boundary must report GUI failure
        message = f"{type(exc).__name__}: {exc}"
        print(f"eye-tracker preview: {message}", file=sys.stderr, flush=True)
        try:
            errors.put_nowait(message)
        except queue.Full:
            pass
    finally:
        if window_created:
            try:
                cv2.destroyWindow(preview_config.window_name)
                cv2.waitKey(1)
            except Exception:  # noqa: BLE001, S110 - display may already be gone
                pass
        closed_event.set()


class LivePreview:
    """Non-blocking latest-frame publisher backed by a display process."""

    _STOP_TIMEOUT_SECONDS = 2.0

    def __init__(
        self,
        preview_config: PreviewConfig,
        *,
        image_settings: dict[int, EyeImageSettings] | None = None,
    ) -> None:
        self.preview_config = preview_config
        self.image_settings = {} if image_settings is None else dict(image_settings)
        self._context = mp.get_context("spawn")
        self._packets: Any | None = None
        self._errors: Any | None = None
        self._stop_event: Any | None = None
        self._closed_event: Any | None = None
        self._process: mp.Process | None = None
        self._error_message: str | None = None
        self._lock = threading.RLock()

    @property
    def closed(self) -> bool:
        with self._lock:
            if self._process is None:
                return False
            return bool(self._closed_event.is_set()) or not self._process.is_alive()

    @property
    def error_message(self) -> str | None:
        with self._lock:
            self._read_error_locked()
            return self._error_message

    def _read_error_locked(self) -> None:
        if self._errors is not None:
            try:
                while True:
                    self._error_message = self._errors.get_nowait()
            except queue.Empty:
                pass
        process = self._process
        requested_stop = self._stop_event is not None and self._stop_event.is_set()
        if (
            self._error_message is None
            and process is not None
            and process.exitcode not in (None, 0)
            and not requested_stop
        ):
            self._error_message = (
                f"preview process exited with status {process.exitcode}; "
                "check the graphical display session"
            )

    def start(self) -> None:
        with self._lock:
            if self._process is not None and self._process.is_alive():
                return
            self._close_queues_locked()
            self._error_message = None
            self._packets = self._context.Queue(maxsize=1)
            self._errors = self._context.Queue(maxsize=1)
            self._stop_event = self._context.Event()
            self._closed_event = self._context.Event()
            self._process = self._context.Process(
                target=_preview_process,
                args=(
                    self._packets,
                    self._errors,
                    self._stop_event,
                    self._closed_event,
                    self.preview_config,
                    self.image_settings,
                ),
                name="eye-preview",
                daemon=True,
            )
            try:
                self._process.start()
            except BaseException:
                self._process = None
                self._close_queues_locked()
                raise

    def publish(self, frame: AnalysisFrame, result: FrameResult) -> None:
        with self._lock:
            if self._process is None or not self._process.is_alive():
                self._read_error_locked()
                return
            try:
                self._packets.put_nowait((frame, result))
            except queue.Full:
                try:
                    self._packets.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._packets.put_nowait((frame, result))
                except queue.Full:
                    pass

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                self._close_queues_locked()
                return
            self._stop_event.set()
        process.join(timeout=self._STOP_TIMEOUT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(timeout=self._STOP_TIMEOUT_SECONDS)
        with self._lock:
            self._read_error_locked()
            self._process = None
            self._close_queues_locked()

    def _close_queues_locked(self) -> None:
        for item in (self._packets, self._errors):
            if item is not None:
                try:
                    item.cancel_join_thread()
                    item.close()
                except (OSError, ValueError):
                    pass
        self._packets = None
        self._errors = None
        self._stop_event = None
        self._closed_event = None
