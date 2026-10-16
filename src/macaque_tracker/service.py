from __future__ import annotations

import logging
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
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...

    def capture_analysis(
        self, *, overview_size: tuple[int, int] | None = None,
    ) -> AnalysisFrame: ...

    def start_recording(self, destination) -> object: ...

    def stop_recording(self) -> None: ...

class PreviewSink(Protocol):
    @property
    def closed(self) -> bool: ...

    @property
    def error_message(self) -> str | None: ...

    def start(self) -> None: ...

    def publish(self, frame: AnalysisFrame, result: FrameResult | None) -> None: ...

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
            tracker_config=config.tracker,
        )
        self.camera = (
            Picamera2Camera(config.camera, config.recording, layout)
            if camera is None
            else camera
        )
        eye_ids = tuple(roi.eye_id for roi in layout.rois)
        self._tracker = (
            tracker
            if tracker is not None
            else MultiEyeTracker(
                eye_ids,
                config.tracker,
                {roi.eye_id: roi.settings for roi in layout.rois},
            )
        )
        if preview is None and config.preview.enabled:
            from .preview import LivePreview

            preview = LivePreview(
                config.preview,
                image_settings={roi.eye_id: roi.settings for roi in layout.rois},
                rois=layout.rois,
                camera_rotations=layout.camera_rotations,
            )
        self._preview = preview
        self._preview_publish_enabled = preview is not None
        self._preview_started = False
        self._preview_error_message: str | None = None
        self.session_id = secrets.randbits(32)
        self._lock = threading.RLock()
        # Serialize detector resets, result commits, and recording transitions.
        # UART snapshots do not acquire this lock or wait for image processing.
        self._processing_lock = threading.RLock()
        self._generation = 0
        self._frames: queue.Queue[tuple[int, AnalysisFrame]] = queue.Queue(maxsize=1)
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
        self._warning_message: str | None = None
        self._recording_warning: str | None = None

    def bind_telemetry_rates(self, rates) -> None:
        """Share transport counters with the optional display, never with detection."""
        binder = getattr(self._preview, "bind_telemetry_rates", None)
        if binder is not None:
            binder(rates)

    @property
    def warning_message(self) -> str | None:
        """Return nonfatal recording/power warnings without marking tracking failed."""
        with self._lock:
            return self._warning_message

    def report_warning(self, message: str) -> None:
        """Publish a persistent warning without interrupting the tracking worker."""
        with self._lock:
            self._warning_message = message
        logging.getLogger(__name__).warning(message)
        notify = getattr(self._preview, "notify_warning", None)
        if notify is not None:
            notify(message)

    def _report_recording_warning(self, message: str) -> None:
        """Remember which warning a successful video restart can clear."""
        with self._lock:
            self._recording_warning = message
            self.report_warning(message)

    @property
    def tracking(self) -> bool:
        with self._lock:
            return self._tracking

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._session is not None and self._session.video_recording

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
        try:
            # A fault also stops idle preview capture; a normal STOP command
            # leaves its acquisition loop running. Neither closes the window.
            self.stop_tracking()
        except Exception as cleanup_error:  # noqa: BLE001 - preserve primary fault
            with self._lock:
                self._error_message = (
                    f"{self._error_message}; cleanup failed: {cleanup_error}"
                )

    def _capture_loop(self, run_event: threading.Event) -> None:
        overview_size = (
            min(1280, self.config.preview.max_display_width),
            min(720, self.config.preview.max_display_height),
        )
        try:
            while run_event.is_set():
                with self._lock:
                    generation = self._generation
                    show_overview = self._preview_started and self._preview_publish_enabled
                frame = (
                    self.camera.capture_analysis(overview_size=overview_size)
                    if show_overview else self.camera.capture_analysis()
                )
                try:
                    self._frames.put_nowait((generation, frame))
                except queue.Full:
                    try:
                        discarded_generation, _discarded = self._frames.get_nowait()
                    except queue.Empty:
                        discarded_generation = None
                    with self._lock:
                        if (
                            self._tracking
                            and generation == self._generation == discarded_generation
                        ):
                            self._dropped += 1
                    self._frames.put_nowait((generation, frame))
        except Exception as exc:  # noqa: BLE001 - worker boundary
            if run_event.is_set():
                self._record_error(f"capture failed: {exc}", run_event)

    def _tracking_loop(self, run_event: threading.Event) -> None:
        try:
            while run_event.is_set() or not self._frames.empty():
                try:
                    generation, frame = self._frames.get(timeout=0.1)
                except queue.Empty:
                    continue
                with self._processing_lock:
                    with self._lock:
                        if not run_event.is_set() or generation != self._generation:
                            continue
                        tracking = self._tracking
                        dropped = self._dropped
                        session = self._session
                        preview = self._preview if self._preview_publish_enabled else None
                    result = None
                    if tracking:
                        result = self._tracker.process(frame, dropped_frames=dropped)
                        with self._lock:
                            if not run_event.is_set():
                                continue
                            self._latest = result
                        if session is not None:
                            session.write_result(result)
                    if preview is not None:
                        try:
                            if preview.closed:
                                with self._lock:
                                    self._preview_publish_enabled = False
                            else:
                                preview.publish(frame, result)
                        except Exception as exc:  # noqa: BLE001 - preview is best effort
                            self._record_preview_error(f"preview publish failed: {exc}")
        except Exception as exc:  # noqa: BLE001 - worker boundary
            if run_event.is_set():
                self._record_error(f"tracking failed: {exc}", run_event)

    def _start_preview_locked(self) -> None:
        if self._preview is None or self._preview_started:
            return
        self._preview_error_message = None
        try:
            self._preview.start()
        except Exception as exc:  # noqa: BLE001 - acquisition remains usable
            self._record_preview_error(f"preview start failed: {exc}")
        else:
            self._preview_started = True
            self._preview_publish_enabled = True

    def _start_acquisition_locked(self) -> None:
        if self._run.is_set():
            return
        if any(
            thread is not None and thread.is_alive()
            for thread in (self._capture_thread, self._tracker_thread)
        ):
            raise InvalidStateError("Previous tracking workers have not terminated")
        self.camera.start()
        self._run = run_event = threading.Event()
        run_event.set()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, args=(run_event,), name="eye-capture", daemon=True,
        )
        self._tracker_thread = threading.Thread(
            target=self._tracking_loop, args=(run_event,), name="eye-analysis", daemon=True,
        )
        self._tracker_thread.start()
        self._capture_thread.start()

    def start_preview(self) -> None:
        """Start live display acquisition without enabling tracking or recording."""
        try:
            with self._processing_lock, self._lock:
                self._start_preview_locked()
                if self._preview_started:
                    self._start_acquisition_locked()
        except Exception as exc:
            self._record_error(str(exc))
            raise

    def start_tracking(self) -> None:
        try:
            with self._processing_lock, self._lock:
                if self._tracking:
                    return
                self._start_acquisition_locked()
                self._error_message = None
                self._status = StatusCode.OK
                self._latest = None
                self._dropped = 0
                self._generation += 1
                reset = getattr(self._tracker, "reset", None)
                if callable(reset):
                    reset()
                self._tracking = True
                if self.config.recording.record_on_tracking:
                    self._start_recording_locked()
                self._start_preview_locked()
        except InvalidStateError:
            raise
        except Exception as exc:
            self._record_error(str(exc))
            raise

    def _start_recording_locked(self) -> None:
        if not self._tracking:
            raise InvalidStateError("Tracking must be active before recording")
        if self._session is not None:
            if self._session.video_recording:
                return
            # Explicit restart after a cutoff rechecks disk space and opens a
            # new synchronized video/log pair; never auto-retry every frame.
            self._session.stop()
            self._session = None
        session = RecordingSession(
            self.camera, self.config, self.layout, session_id=self.session_id,
            warning_callback=self._report_recording_warning,
        )
        session.start()
        self._session = session
        if session.video_recording:
            if self._warning_message == self._recording_warning:
                self._warning_message = None
                notify = getattr(self._preview, "notify_warning", None)
                if notify is not None:
                    notify(None)
            self._recording_warning = None

    def start_recording(self) -> None:
        with self._processing_lock, self._lock:
            self._start_recording_locked()

    def _stop_recording_locked(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            session.stop()

    def stop_recording(self) -> None:
        with self._processing_lock, self._lock:
            self._stop_recording_locked()

    def stop_tracking(self) -> None:
        error = None
        if not self._processing_lock.acquire(timeout=self._WORKER_JOIN_TIMEOUT_S):
            # Preserve bounded shutdown even if analysis is stuck. Clearing
            # the run event prevents that result being committed if it returns.
            with self._lock:
                self._run.clear()
            self._stop_acquisition()
            raise RuntimeError("Tracking did not finish its current frame before stop")
        try:
            with self._lock:
                self._generation += 1
                self._latest = None
                keep_preview = self._preview_started and self._run.is_set()
                try:
                    self._stop_recording_locked()
                except Exception as exc:  # noqa: BLE001 - still stop camera on failure
                    error = exc
                    keep_preview = False
                if keep_preview:
                    self._tracking = False
                else:
                    self._run.clear()
        finally:
            self._processing_lock.release()
        if not keep_preview:
            self._stop_acquisition()
        if error is not None:
            raise error

    def _stop_acquisition(self) -> None:
        with self._lock:
            self._run.clear()
            capture_thread = self._capture_thread
            tracker_thread = self._tracker_thread
        current = threading.current_thread()
        workers = (capture_thread, tracker_thread)
        deadline = time.monotonic() + self._WORKER_JOIN_TIMEOUT_S
        for thread in workers:
            if thread is not None and thread is not current and thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.monotonic()))

        errors: list[Exception] = []
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
                recording=self._session is not None and self._session.video_recording,
                configured=True,
                result=latest,
                session_id=self.session_id,
                configured_eye_count=len(self.layout.rois),
                status=self._status,
                error=self._error_message is not None,
                stale=stale,
            )

    def execute(self, command: CommandPacket) -> StatusSnapshot:
        if command.code in (
            CommandCode.START_EXPERIMENT, CommandCode.STOP_EXPERIMENT, CommandCode.SHUTDOWN
        ):
            return replace(self.snapshot(), status=StatusCode.UNKNOWN_COMMAND, error=True)
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
        with self._lock:
            self._preview_started = False
            self._run.clear()
        try:
            self.stop_tracking()
        finally:
            try:
                if self._preview is not None:
                    try:
                        self._preview.stop()
                    except Exception as exc:  # noqa: BLE001 - still release acquisition
                        self._record_preview_error(f"preview stop failed: {exc}")
                self.camera.close()
            finally:
                close_tracker = getattr(self._tracker, "close", None)
                if callable(close_tracker):
                    close_tracker()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
