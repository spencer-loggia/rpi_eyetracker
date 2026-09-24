from __future__ import annotations

import math
import queue
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import replace
from typing import Protocol, Self

from .camera import Picamera2Camera
from .config import AppConfig, RoiLayout
from .models import AnalysisFrame, FrameResult
from .protocol import CommandCode, CommandPacket, StatusCode, StatusSnapshot
from .session import RecordingSession
from .tracker import MultiEyeTracker


class CameraSource(Protocol):
    @property
    def recording(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def capture_analysis(self) -> AnalysisFrame: ...

    def start_recording(self, destination) -> object: ...

    def stop_recording(self) -> None: ...

    def set_image_controls(self, **controls) -> object: ...

    def image_control_limits(self) -> dict[str, tuple[float, float]]: ...


class PreviewSink(Protocol):
    @property
    def closed(self) -> bool: ...

    @property
    def error_message(self) -> str | None: ...

    def start(self) -> None: ...

    def publish(self, frame: AnalysisFrame, result: FrameResult) -> None: ...

    def read_camera_controls(self) -> dict[str, int | float]: ...

    def stop(self) -> None: ...


class InvalidStateError(RuntimeError):
    pass


class EyeTrackingService:
    """Threaded capture/tracking state machine controlled by protocol commands."""

    _COMMAND_CACHE_SIZE = 64
    _WORKER_JOIN_TIMEOUT_S = 3.0

    def __init__(
        self,
        config: AppConfig,
        layout: RoiLayout,
        *,
        camera: CameraSource | None = None,
        tracker: MultiEyeTracker | None = None,
        preview: PreviewSink | None = None,
    ) -> None:
        self.config = config
        self.layout = layout
        layout.validate_for_frame(
            config.camera.analysis_width,
            config.camera.analysis_height,
        )
        self.camera = (
            Picamera2Camera(config.camera, config.recording, layout)
            if camera is None
            else camera
        )
        eye_ids = tuple(roi.eye_id for roi in layout.rois)
        self._tracker = (
            tracker if tracker is not None else MultiEyeTracker(eye_ids, config.tracker)
        )
        if preview is None and config.preview.enabled:
            from .preview import LivePreview

            frame_exposure_limit = max(
                1,
                math.ceil(1_000_000.0 / config.camera.fps) - 1,
            )
            limits_method = getattr(self.camera, "image_control_limits", None)
            reported_limits = limits_method() if callable(limits_method) else {}
            raw_exposure_limits = reported_limits.get("exposure_us")
            exposure_limits: tuple[int, int] | None = None
            if raw_exposure_limits is not None:
                minimum = max(1, math.ceil(raw_exposure_limits[0]))
                maximum = min(frame_exposure_limit, int(raw_exposure_limits[1]))
                if minimum < maximum:
                    exposure_limits = (minimum, maximum)
            preview = LivePreview(
                config.preview,
                exposure_us=config.camera.exposure_us,
                exposure_limits=exposure_limits,
            )
        self._preview = preview
        self._preview_publish_enabled = preview is not None
        self._preview_error_message: str | None = None
        self.session_id = secrets.randbits(32)
        self._lock = threading.RLock()
        self._frames: queue.Queue[AnalysisFrame] = queue.Queue(maxsize=1)
        self._run = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._tracker_thread: threading.Thread | None = None
        self._tracking = False
        self._session: RecordingSession | None = None
        self._latest: FrameResult | None = None
        self._dropped = 0
        self._command_cache: OrderedDict[int, tuple[CommandPacket, StatusCode]] = OrderedDict()
        self._error_message: str | None = None
        self._status = StatusCode.OK

    @property
    def tracking(self) -> bool:
        with self._lock:
            return self._tracking

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._session is not None

    @property
    def error_message(self) -> str | None:
        with self._lock:
            return self._error_message

    @property
    def preview_closed(self) -> bool:
        with self._lock:
            preview = self._preview
        return False if preview is None else preview.closed

    @property
    def preview_error_message(self) -> str | None:
        with self._lock:
            preview = self._preview
            saved = self._preview_error_message
        if preview is None:
            return saved
        try:
            current = preview.error_message
        except Exception as exc:  # noqa: BLE001 - optional display must not stop tracking
            current = f"could not read preview status: {exc}"
        if current is not None:
            with self._lock:
                self._preview_error_message = current
            return current
        return saved

    def _record_preview_error(self, message: str) -> None:
        with self._lock:
            self._preview_error_message = message
            self._preview_publish_enabled = False

    def _record_error(
        self,
        message: str,
        owner_event: threading.Event | None = None,
    ) -> None:
        with self._lock:
            if owner_event is not None and owner_event is not self._run:
                return
            self._error_message = message
            self._status = StatusCode.INTERNAL_ERROR
            self._run.clear()
            should_stop = self._tracking
        if should_stop:
            try:
                # This is safe from either worker: stop_tracking never joins
                # the calling thread, and it prevents a failed camera/encoder
                # session from continuing unattended.
                self.stop_tracking()
            except Exception as cleanup_error:  # noqa: BLE001 - preserve primary fault
                with self._lock:
                    self._error_message = (
                        f"{self._error_message}; cleanup failed: {cleanup_error}"
                    )

    def _capture_loop(self, run_event: threading.Event) -> None:
        try:
            while run_event.is_set():
                with self._lock:
                    preview = self._preview if self._preview_publish_enabled else None
                if preview is not None:
                    try:
                        read_controls = getattr(preview, "read_camera_controls", None)
                        controls = read_controls() if callable(read_controls) else {}
                        if controls:
                            set_controls = getattr(self.camera, "set_image_controls", None)
                            if not callable(set_controls):
                                raise RuntimeError(
                                    "camera does not support live image controls"
                                )
                            set_controls(**controls)
                    except Exception as exc:  # noqa: BLE001 - control is best effort
                        self._record_preview_error(f"preview camera control failed: {exc}")
                frame = self.camera.capture_analysis()
                try:
                    self._frames.put_nowait(frame)
                except queue.Full:
                    try:
                        self._frames.get_nowait()
                    except queue.Empty:
                        pass
                    with self._lock:
                        self._dropped += 1
                    self._frames.put_nowait(frame)
        except Exception as exc:  # noqa: BLE001 - worker boundary
            if run_event.is_set():
                self._record_error(f"capture failed: {exc}", run_event)

    def _tracking_loop(self, run_event: threading.Event) -> None:
        try:
            while run_event.is_set() or not self._frames.empty():
                try:
                    frame = self._frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                with self._lock:
                    dropped = self._dropped
                result = self._tracker.process(frame, dropped_frames=dropped)
                with self._lock:
                    self._latest = result
                    session = self._session
                    preview = self._preview if self._preview_publish_enabled else None
                if session is not None:
                    session.write_result(result)
                if preview is not None:
                    try:
                        preview.publish(frame, result)
                    except Exception as exc:  # noqa: BLE001 - preview is best effort
                        self._record_preview_error(f"preview publish failed: {exc}")
        except Exception as exc:  # noqa: BLE001 - worker boundary
            if run_event.is_set():
                self._record_error(f"tracking failed: {exc}", run_event)

    def start_tracking(self) -> None:
        with self._lock:
            if self._tracking:
                return
            lingering = tuple(
                thread
                for thread in (self._capture_thread, self._tracker_thread)
                if thread is not None and thread.is_alive()
            )
            if lingering:
                raise InvalidStateError("Previous tracking workers have not terminated")
            self._error_message = None
            self._status = StatusCode.OK
            self._latest = None
            self._dropped = 0
            self._preview_error_message = None
            self._preview_publish_enabled = self._preview is not None
            reset = getattr(self._tracker, "reset", None)
            if callable(reset):
                reset()
            while not self._frames.empty():
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    break
            self.camera.start()
            run_event = threading.Event()
            self._run = run_event
            self._tracking = True
            run_event.set()
            try:
                if self.config.recording.record_on_tracking:
                    self._start_recording_locked()
                if self._preview is not None:
                    try:
                        self._preview.start()
                    except Exception as exc:  # noqa: BLE001 - acquisition remains usable
                        self._record_preview_error(f"preview start failed: {exc}")
                self._capture_thread = threading.Thread(
                    target=self._capture_loop,
                    args=(run_event,),
                    name="eye-capture",
                    daemon=True,
                )
                self._tracker_thread = threading.Thread(
                    target=self._tracking_loop,
                    args=(run_event,),
                    name="eye-analysis",
                    daemon=True,
                )
                self._tracker_thread.start()
                self._capture_thread.start()
            except BaseException as start_error:
                run_event.clear()
                self._tracking = False
                cleanup_errors: list[Exception] = []
                if self._session is not None:
                    try:
                        self._stop_recording_locked()
                    except Exception as exc:  # noqa: BLE001 - continue camera cleanup
                        cleanup_errors.append(exc)
                if self._preview is not None:
                    try:
                        self._preview.stop()
                    except Exception as exc:  # noqa: BLE001 - continue camera cleanup
                        self._record_preview_error(f"preview stop failed: {exc}")
                try:
                    self.camera.stop()
                except Exception as exc:  # noqa: BLE001 - preserve original start failure
                    cleanup_errors.append(exc)
                for thread in (self._capture_thread, self._tracker_thread):
                    if thread is not None and thread.is_alive():
                        thread.join(timeout=self._WORKER_JOIN_TIMEOUT_S)
                if self._capture_thread is not None and not self._capture_thread.is_alive():
                    self._capture_thread = None
                if self._tracker_thread is not None and not self._tracker_thread.is_alive():
                    self._tracker_thread = None
                for cleanup_error in cleanup_errors:
                    start_error.add_note(f"cleanup also failed: {cleanup_error}")
                raise

    def _start_recording_locked(self) -> None:
        if not self._tracking:
            raise InvalidStateError("Tracking must be active before recording")
        if self._session is not None:
            return
        session = RecordingSession(
            self.camera, self.config, self.layout, session_id=self.session_id
        )
        session.start()
        self._session = session

    def start_recording(self) -> None:
        with self._lock:
            self._start_recording_locked()

    def _stop_recording_locked(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.stop()

    def stop_recording(self) -> None:
        with self._lock:
            self._stop_recording_locked()

    def stop_tracking(self) -> None:
        with self._lock:
            if not self._tracking:
                return
            run_event = self._run
            run_event.clear()
            capture_thread = self._capture_thread
            tracker_thread = self._tracker_thread
        current = threading.current_thread()
        workers = (capture_thread, tracker_thread)
        deadline = time.monotonic() + self._WORKER_JOIN_TIMEOUT_S
        for thread in workers:
            if thread is not None and thread is not current:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))

        errors: list[Exception] = []
        if self._preview is not None:
            try:
                self._preview.stop()
            except Exception as exc:  # noqa: BLE001 - optional preview cleanup
                self._record_preview_error(f"preview stop failed: {exc}")
        with self._lock:
            try:
                self._stop_recording_locked()
            except Exception as exc:  # noqa: BLE001 - still stop the camera
                errors.append(exc)
        try:
            # Stopping the camera also releases a capture request that did not
            # return during the first bounded join.
            self.camera.stop()
        except Exception as exc:  # noqa: BLE001 - still verify worker shutdown
            errors.append(exc)

        deadline = time.monotonic() + self._WORKER_JOIN_TIMEOUT_S
        for thread in workers:
            if thread is not None and thread is not current and thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
        survivors = tuple(
            thread
            for thread in workers
            if thread is not None and thread is not current and thread.is_alive()
        )
        if survivors:
            names = ", ".join(thread.name for thread in survivors)
            errors.append(RuntimeError(f"Tracking workers did not terminate: {names}"))

        with self._lock:
            self._tracking = False
            self._capture_thread = capture_thread if capture_thread in survivors else None
            self._tracker_thread = tracker_thread if tracker_thread in survivors else None
        if errors:
            primary = errors[0]
            for secondary in errors[1:]:
                primary.add_note(f"additional shutdown failure: {secondary}")
            raise primary

    def clear_error(self) -> None:
        with self._lock:
            if self._tracking and not self._run.is_set():
                raise InvalidStateError("Stop the failed tracking session before clearing")
            self._error_message = None
            self._status = StatusCode.OK

    def snapshot(self) -> StatusSnapshot:
        with self._lock:
            latest = self._latest
            stale_after_ns = int(max(0.100, 3.0 / self.config.camera.fps) * 1e9)
            stale = bool(
                self._tracking
                and (
                    latest is None
                    or time.monotonic_ns() - latest.produced_timestamp_ns > stale_after_ns
                )
            )
            return StatusSnapshot(
                tracking=self._tracking,
                recording=self._session is not None,
                configured=True,
                result=latest,
                session_id=self.session_id,
                configured_eye_count=len(self.layout.rois),
                status=self._status,
                error=self._error_message is not None,
                stale=stale,
            )

    def execute(self, command: CommandPacket) -> StatusSnapshot:
        if command.code is CommandCode.POLL:
            return self.snapshot()
        with self._lock:
            cached = self._command_cache.get(command.sequence)
            if cached is not None:
                previous, previous_status = cached
                if previous == command:
                    self._command_cache.move_to_end(command.sequence)
                    snapshot = self.snapshot()
                    return replace(
                        snapshot,
                        status=previous_status,
                        error=snapshot.error or previous_status is not StatusCode.OK,
                    )
                return replace(self.snapshot(), status=StatusCode.SEQUENCE_CONFLICT, error=True)
        try:
            if command.code is CommandCode.START_TRACKING:
                self.start_tracking()
            elif command.code is CommandCode.STOP_TRACKING:
                self.stop_tracking()
            elif command.code is CommandCode.START_RECORDING:
                self.start_recording()
            elif command.code is CommandCode.STOP_RECORDING:
                self.stop_recording()
            elif command.code is CommandCode.CLEAR_ERROR:
                self.clear_error()
        except InvalidStateError:
            outcome = replace(self.snapshot(), status=StatusCode.INVALID_STATE, error=True)
        except Exception as exc:  # noqa: BLE001 - command boundary returns protocol error
            self._record_error(str(exc))
            outcome = self.snapshot()
        else:
            # The status code is the outcome of this command. Service health is
            # carried independently by ERROR/RESULT_STALE flags; for example,
            # STOP can succeed while preserving an earlier tracker fault until
            # CLEAR_ERROR is issued.
            outcome = replace(self.snapshot(), status=StatusCode.OK)
        with self._lock:
            self._command_cache[command.sequence] = (command, outcome.status)
            self._command_cache.move_to_end(command.sequence)
            while len(self._command_cache) > self._COMMAND_CACHE_SIZE:
                self._command_cache.popitem(last=False)
        return outcome

    def close(self) -> None:
        try:
            self.stop_tracking()
        finally:
            self.camera.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
