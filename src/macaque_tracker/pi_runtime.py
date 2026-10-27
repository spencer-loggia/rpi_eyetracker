"""Experiment lifecycle and camera ownership, independent of the GUI toolkit."""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .camera import Picamera2Camera, VideoFileCamera
from .config import AppConfig, RoiLayout
from .data_sync import ExperimentDataSync, SyncReport, SyncStatus
from .experiments import Block, Experiment, ExperimentStore
from .preview import LivePreview
from .protocol import (
    CommandCode,
    CommandPacket,
    StatusCode,
    StatusSnapshot,
    validate_run_number,
    validate_subject_name,
)
from .service import EyeTrackingService, InvalidStateError
from .system_control import power_off_system

_LOGGER = logging.getLogger(__name__)


class CameraFeed:
    """A bounded, latest-frame live feed, used only outside tracking."""

    def __init__(self, camera) -> None:
        self.camera = camera
        self.image = None
        self.error_message: str | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.camera.start()
        self._thread = threading.Thread(target=self._capture, name="menu-camera", daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            raise RuntimeError("Camera did not supply a preview frame within five seconds")
        if self.error_message:
            raise RuntimeError(self.error_message)

    def _capture(self) -> None:
        try:
            while not self._stop.is_set():
                self.image, _timestamp = self.camera.capture_preview()
                self._ready.set()
                self._stop.wait(1 / 15)
        except Exception as exc:  # noqa: BLE001 - report camera worker faults
            if not self._stop.is_set():
                self.error_message = str(exc)
            self._ready.set()

    def latest(self):
        if self.image is None:
            raise RuntimeError(self.error_message or "Waiting for camera")
        return self.image

    def close(self) -> None:
        self._stop.set()
        errors = []
        try:
            self.camera.stop()
        except Exception as exc:  # noqa: BLE001 - report boundary failures and keep cleanup available
            errors.append(exc)
        if self._thread is not None:
            self._thread.join(timeout=3)
        try:
            self.camera.close()
        except Exception as exc:  # noqa: BLE001 - report boundary failures and keep cleanup available
            errors.append(exc)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                raise RuntimeError(
                    "Preview camera did not stop; refusing to open a second owner"
                )
        if errors:
            raise errors[0]


@dataclass(frozen=True)
class RuntimeView:
    mode: str
    experiment: Experiment | None
    feed: CameraFeed | None
    preview: LivePreview | None
    calibration: object | None
    error_message: str | None
    busy: bool = False
    shutdown_requested: bool = False
    shutdown_submitted: bool = False
    camera_rotations: tuple[int, ...] = (0, 0, 0, 0)
    warning_message: str | None = None
    block: Block | None = None
    sync_active: bool = False
    sync_message: str = ""
    sync_report: SyncReport | None = None
    shutdown_decision: bool = False


def _clock_permission_details() -> str:
    """Inspect the calling thread after failure; never change privileges."""
    details = [f"pid={os.getpid()}", f"uid={os.geteuid()}", f"tid={threading.get_native_id()}"]
    try:
        status = dict(
            line.split(":", 1)
            for line in Path("/proc/thread-self/status").read_text(encoding="utf-8").splitlines()
            if ":" in line
        )
        effective = int(status["CapEff"].strip(), 16)
        # CAP_SYS_TIME is capability 25 in the Linux UAPI.
        present = bool(effective & (1 << 25))
        details.append(f"CAP_SYS_TIME effective={'yes' if present else 'no'}")
        for key, label in (("CapPrm", "permitted"), ("CapBnd", "bounding"), ("CapAmb", "ambient")):
            if key in status:
                present = bool(int(status[key].strip(), 16) & (1 << 25))
                details.append(f"{label}={'yes' if present else 'no'}")
        if "Seccomp" in status:
            details.append(f"Seccomp={status['Seccomp'].strip()}")
    except (OSError, KeyError, ValueError):
        details.append("effective capabilities unavailable")
    return ", ".join(details)


def set_system_time(timestamp_ns: int) -> None:
    """Set UTC wall time; capture and tracking timing remain monotonic."""
    try:
        time.clock_settime_ns(time.CLOCK_REALTIME, timestamp_ns)
    except AttributeError as exc:
        raise RuntimeError(
            "Cannot set system time: this Python/platform does not provide "
            f"time.clock_settime_ns and CLOCK_REALTIME ({exc}). "
            "Run the launcher with Python 3.11+ on Raspberry Pi OS/Linux. "
            "No experiment was created."
        ) from exc
    except PermissionError as exc:
        raise RuntimeError(
            f"Cannot set system time: {type(exc).__name__}: {exc} "
            f"({_clock_permission_details()}). The running launcher needs "
            "CAP_SYS_TIME. If using systemd, its ProtectClock setting must be disabled "
            "(not checked here). See docs/protocol.md. No experiment was created."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Cannot set system time: {type(exc).__name__}: {exc}; "
            f"requested UTC timestamp_ns={timestamp_ns}. No experiment was created."
        ) from exc


class ExperimentRuntime:
    """One transport and one camera owner across every launcher screen.

    Transitions are serialized separately from snapshots. Snapshots use the
    last status during a transition; the serial link acknowledges each command
    after it completes. Rendering never runs in the tracker worker.
    """

    def __init__(
        self,
        store: ExperimentStore,
        *,
        video_path: Path | None = None,
        clock_setter: Callable[[int], None] = set_system_time,
        shutdown_executor: Callable[[], None] = power_off_system,
        camera_factory=None,
        service_factory=EyeTrackingService,
        preview_factory=LivePreview,
        feed_factory=CameraFeed,
        sync_factory=ExperimentDataSync,
    ) -> None:
        self.store = store
        self.base_config = AppConfig.load(store.base_config_path)
        self.config = self.base_config
        self.video_path = video_path
        self._clock_setter = clock_setter
        self._shutdown_executor = shutdown_executor
        self._shutdown_requested = False
        self._shutdown_ready = False
        self._shutdown_submitted = False
        self._shutdown_acknowledged = False
        self._shutdown_decision = False
        self._sync_factory = sync_factory
        self._sync_active = False
        self._sync_message = ""
        self._sync_report: SyncReport | None = None
        self._sync_warning: str | None = None
        self._sync_cancel = threading.Event()
        self._sync_thread: threading.Thread | None = None
        self._sync_again = False
        self._camera_factory = camera_factory or self._new_camera
        self._service_factory = service_factory
        self._preview_factory = preview_factory
        self._feed_factory = feed_factory
        self._transition = threading.RLock()
        self._state = threading.RLock()
        self._mode = "home"
        self._experiment: Experiment | None = None
        self._block: Block | None = None
        self._requested_block_number: int | None = None
        self._feed = None
        self._service = None
        self._preview = None
        self._calibration = None
        self._camera_rotations = (0, 0, 0, 0)
        self._error: str | None = None
        self._session_id = secrets.randbits(32)
        self._commands: OrderedDict[int, tuple[CommandPacket, StatusCode]] = OrderedDict()
        self._closed = False
        self._close_complete = False
        self._telemetry_rates = None
        self._warning = None
        self._last_snapshot = StatusSnapshot(
            False, False, False, None, session_id=self._session_id
        )

    def _new_camera(self, config, layout=None):
        if self.video_path is not None:
            return VideoFileCamera(self.video_path, config.camera, roi_layout=layout)
        return Picamera2Camera(config.camera, config.recording, roi_layout=layout)

    def view(self) -> RuntimeView:
        available = self._transition.acquire(blocking=False)
        try:
            with self._state:
                error = self._error
                warning = self._warning
                if available and self._service is not None:
                    error = error or self._service.error_message
                    warning = getattr(self._service, "warning_message", None) or warning
                return RuntimeView(
                    self._mode,
                    self._experiment,
                    self._feed,
                    self._preview,
                    self._calibration,
                    error,
                    not available or self._sync_active,
                    self._shutdown_requested,
                    self._shutdown_submitted,
                    self._camera_rotations,
                    warning,
                    self._block,
                    self._sync_active,
                    self._sync_message,
                    self._sync_report,
                    self._shutdown_decision,
                )
        finally:
            if available:
                self._transition.release()

    def _change(self, **fields) -> None:
        with self._state:
            for name, value in fields.items():
                setattr(self, f"_{name}", value)

    def bind_telemetry_rates(self, rates) -> None:
        """Keep one transmitter counter across launcher screens and tracking services."""
        self._telemetry_rates = rates

    def report_warning(self, message: str) -> None:
        """Expose a nonfatal launcher warning to the GUI."""
        _LOGGER.warning(message)
        self._change(warning=message)

    def report_error(self, message: str) -> None:
        self._change(error=message or None)

    def clear_error(self) -> None:
        with self._transition:
            if self._service is not None:
                self._service.clear_error()
            self._change(error=None)

    def snapshot(self) -> StatusSnapshot:
        if not self._transition.acquire(blocking=False):
            with self._state:
                return replace(
                    self._last_snapshot,
                    stale=True,
                    experiment_active=self._experiment is not None,
                    calibrating=self._mode == "calibration",
                    shutting_down=self._shutdown_requested,
                    error=self._last_snapshot.error or self._error is not None,
                )
        try:
            snapshot = self._snapshot()
            self._last_snapshot = snapshot
            return snapshot
        finally:
            self._transition.release()

    def _snapshot(self) -> StatusSnapshot:
        with self._state:
            service, experiment = self._service, self._experiment
            mode, error = self._mode, self._error
        if service is not None:
            snapshot = service.snapshot()
        else:
            configured = experiment is not None and experiment.calibration_path.is_file()
            snapshot = StatusSnapshot(
                tracking=False,
                recording=False,
                configured=configured,
                result=None,
                session_id=self._session_id,
            )
        return replace(
            snapshot,
            experiment_active=experiment is not None,
            calibrating=mode == "calibration",
            shutting_down=self._shutdown_requested,
            error=snapshot.error or error is not None,
            status=StatusCode.INTERNAL_ERROR if error else snapshot.status,
        )

    def _stop_activity(self) -> None:
        # Keep references until shutdown succeeds: a failed close must never
        # allow another component to open the still-owned camera.
        if self._calibration is not None:
            self._calibration.close()
        if self._service is not None:
            self._service.close()
            self._change(service=None, preview=None, block=None)
        if self._feed is not None:
            self._feed.close()
            self._change(feed=None)
        self._change(calibration=None)

    def ensure_feed(self) -> None:
        with self._transition:
            if (
                self._closed
                or self._shutdown_requested
                or self._sync_active
                or self._feed is not None
                or self._service is not None
                or self._calibration is not None
            ):
                return
            # Load once per menu feed, never in the per-frame renderer.
            config_path = (self.store.base_config_path if self._experiment is None
                           else self._experiment.config_path)
            calibration_path = Path(self.config.roi_config).expanduser()
            if not calibration_path.is_absolute():
                calibration_path = config_path.parent / calibration_path
            rotations = (RoiLayout.load(calibration_path).camera_rotations
                         if calibration_path.is_file() else (0, 0, 0, 0))
            self._change(camera_rotations=rotations)
            camera = self._camera_factory(self.config)
            try:
                feed = self._feed_factory(camera)
            except Exception:
                camera.close()
                raise
            self._change(feed=feed)
            try:
                feed.start()
            except Exception:
                feed.close()
                self._change(feed=None)
                raise

    def show_subjects(self) -> None:
        with self._transition:
            self._require_open()
            if self._experiment is not None:
                self.end_experiment()
            self._change(mode="subjects", error=None)

    def start_experiment(
        self, subject: str, timestamp_ns: int | None = None, *,
        number: int | None = None, remote=False,
    ):
        """Open a daily numbered experiment; only UART starts update the system clock."""
        with self._transition:
            self._require_open()
            validate_subject_name(subject)
            timestamp_ns = time.time_ns() if timestamp_ns is None and not remote else timestamp_ns
            number = self.store.check_available(subject, timestamp_ns, number)
            if remote:
                if not isinstance(timestamp_ns, int) or timestamp_ns <= 0:
                    raise ValueError("START_EXPERIMENT requires a positive UTC timestamp")
                # Validate range/collision before changing the system clock or
                # disturbing an active experiment. The store rechecks on create.
                self._clock_setter(timestamp_ns)
            self._stop_activity()
            replaced_experiment = self._experiment is not None
            if self._experiment is not None:
                self.store.finish(self._experiment)
                self._change(experiment=None, mode="home")
            experiment = self.store.create(subject, timestamp_ns, number)
            self.config = AppConfig.load(experiment.config_path)
            self._change(experiment=experiment, mode="experiment", error=None,
                         sync_message="", sync_report=None)
            if replaced_experiment:
                self._start_sync_locked()
            return experiment

    def end_experiment(self) -> None:
        with self._transition:
            self._stop_activity()
            if self._experiment is not None:
                self.store.finish(self._experiment)
            self.config = self.base_config
            self._change(experiment=None, block=None, mode="home", error=None)
            self._start_sync_locked()

    def _start_sync_locked(self) -> None:
        """Schedule data-only cleanup after writers close, without blocking UART replies."""
        if not self.base_config.sync.remote_data_url.strip():
            return
        if self._sync_active:
            self._sync_again = True
            return
        self._sync_cancel = threading.Event()
        self._sync_again = False
        if self._warning == self._sync_warning:
            self._change(warning=None)
        self._sync_warning = None
        self._change(sync_active=True, sync_report=None, shutdown_decision=False,
                     sync_message="Preparing experiment synchronization")
        self._sync_thread = threading.Thread(target=self._sync_worker,
                                             name="experiment-sync", daemon=True)
        self._sync_thread.start()

    def _sync_worker(self) -> None:
        """Transfer immutable experiments while new acquisition is held at menu boundaries."""
        while True:
            with self._transition:
                exclude = () if self._experiment is None else (self._experiment.directory,)
            try:
                report = self._sync_factory(self.store, self.base_config.sync).run(
                    self._sync_cancel, lambda message: self._change(sync_message=message),
                    exclude=exclude,
                )
            except Exception as exc:
                _LOGGER.exception("Experiment synchronization failed")
                report = SyncReport(SyncStatus.FAILED, str(exc))
            with self._transition:
                if self._sync_again and report.status is SyncStatus.SUCCESS:
                    self._sync_again = False
                    continue
                self._change(sync_active=False, sync_report=report, sync_message=report.summary)
                if report.status is SyncStatus.CANCELLED:
                    if self._shutdown_requested:
                        self._cancel_shutdown(StatusCode.INVALID_STATE)
                    self._report_sync_warning(report)
                elif report.status is SyncStatus.FAILED:
                    if self._shutdown_requested:
                        self._change(shutdown_decision=True, error=report.detail)
                    else:
                        self._report_sync_warning(report)
                elif report.cleanup_errors:
                    self._report_sync_warning(report)
                self._finish_shutdown_if_ready()
                return

    def _report_sync_warning(self, report: SyncReport) -> None:
        """Track cleanup warnings separately so a later successful retry clears only these."""
        self._sync_warning = f"{report.summary}\n{report.detail}"
        self.report_warning(self._sync_warning)

    def cancel_sync(self) -> None:
        """Signal cancellation without acquiring a transition lock or touching Tk."""
        self._sync_cancel.set()

    def resolve_sync_failure(self, action: str) -> None:
        """Apply the reference launcher's Retry / Shut down anyway / Cancel choices."""
        with self._transition:
            if not self._shutdown_decision or self._sync_active:
                return
            if action == "retry":
                self._change(error=None)
                self._start_sync_locked()
            elif action == "shutdown":
                self._change(shutdown_decision=False, error=None)
                self._finish_shutdown_if_ready()
            elif action == "cancel":
                self._cancel_shutdown(StatusCode.INVALID_STATE)
            else:
                raise ValueError(f"Unknown shutdown cleanup choice: {action}")

    def _cancel_shutdown(self, status: StatusCode) -> None:
        """Cancel power-off and prevent a cached UART retry from resubmitting it."""
        self._shutdown_ready = self._shutdown_acknowledged = False
        self._change(shutdown_requested=False, shutdown_decision=False, error=None)
        for sequence, (command, outcome) in list(self._commands.items()):
            if command.code is CommandCode.SHUTDOWN and outcome is StatusCode.OK:
                self._commands[sequence] = (command, status)

    def _block_config(self, config: AppConfig, layout: RoiLayout) -> AppConfig:
        """Allocate a block once, at a saved-run boundary, using the loaded calibration."""
        block, saved = self.store.create_block(
            self._experiment, config, layout, self._requested_block_number,
        )
        self._change(block=block)
        return saved

    def open_tracking(self) -> None:
        with self._transition:
            self._require_open()
            if self._experiment is None or self._mode == "calibration":
                raise InvalidStateError("Select an experiment and finish calibration first")
            if self._service is not None:
                return
            config = AppConfig.load(self._experiment.config_path)
            layout = RoiLayout.load(self._experiment.calibration_path)
            # Always constrain every recording and metadata artifact to this
            # experiment, even if its copied config was edited externally.
            config = replace(
                config,
                recording=replace(
                    config.recording,
                    directory=str(self._experiment.directory),
                    layout="hands_eyes",
                    shared_permissions=True,
                    record_on_tracking=(
                        config.recording.record_on_tracking and self.video_path is None
                    ),
                ),
                preview=replace(
                    config.preview, enabled=True, max_display_width=1280, max_display_height=680
                ),
            )
            layout.validate_for_frame(
                config.camera.analysis_width,
                config.camera.analysis_height,
                tracker_config=config.tracker,
            )
            self._stop_activity()
            preview = self._preview_factory(
                config.preview,
                embedded=True,
                image_settings={roi.eye_id: roi.settings for roi in layout.rois},
                rois=layout.rois,
                camera_rotations=layout.camera_rotations,
            )
            camera = self._camera_factory(config, layout)
            try:
                service = self._service_factory(
                    config, layout, camera=camera, preview=preview,
                    recording_config_factory=self._block_config,
                )
            except Exception:
                camera.close()
                preview.stop()
                raise
            binder = getattr(service, "bind_telemetry_rates", None)
            if binder is not None and self._telemetry_rates is not None:
                binder(self._telemetry_rates)
            self._change(service=service, preview=preview)
            try:
                service.start_preview()  # Deliberately leaves tracking/saving off.
            except Exception:
                service.close()
                self._change(service=None, preview=None)
                raise
            self.config = config
            self._change(mode="tracking", error=None)

    def begin_calibration(self, *, video_path=None) -> None:
        from .pi_calibration import CalibrationSession

        with self._transition:
            self._require_open()
            if self._experiment is None:
                raise InvalidStateError("Select an experiment first")
            self._stop_activity()
            self.config = AppConfig.load(self._experiment.config_path)
            if video_path is None:
                self.ensure_feed()
            calibration = CalibrationSession(
                self.config, self._experiment, self._feed,
                video_path=video_path,
            )
            self._change(calibration=calibration, mode="calibration", error=None)

    def calibration_video_action(self, action, *, path=None) -> None:
        with self._transition:
            self._require_open()
            if self._calibration is None:
                raise InvalidStateError("Calibration is not open")
            self._calibration.video_action(action, path=path)

    def save_calibration(self) -> None:
        with self._transition:
            self._require_open()
            if self._calibration is None:
                raise InvalidStateError("Calibration is not open")
            self._calibration.save()
            self._stop_activity()
            self.config = AppConfig.load(self._experiment.config_path)
            self._change(mode="experiment", error=None)

    def back(self) -> None:
        with self._transition:
            self._require_open()
            if self._mode in {"tracking", "calibration"}:
                self._stop_activity()
                self.config = AppConfig.load(self._experiment.config_path)
                self._change(mode="experiment", error=None)
            elif self._mode == "experiment":
                self.end_experiment()
                self._change(mode="subjects")
            else:
                self._change(mode="home", error=None)

    def _require_open(self) -> None:
        if self._closed or self._shutdown_requested:
            raise InvalidStateError("Launcher is shutting down")
        if self._sync_active:
            raise InvalidStateError("Experiment synchronization is in progress; wait or cancel it")

    @property
    def response_action_pending(self) -> bool:
        return self._shutdown_ready

    def after_response(self) -> None:
        """Called by the transport after its best-effort shutdown acknowledgement."""
        with self._transition:
            if not self._shutdown_ready:
                return
            self._shutdown_ready = False
            self._shutdown_acknowledged = True
            self._finish_shutdown_if_ready()

    def _finish_shutdown_if_ready(self) -> None:
        """Power off only after acknowledgement and successful/explicitly bypassed cleanup."""
        if (not self._shutdown_requested or not self._shutdown_acknowledged
                or self._sync_active or self._shutdown_decision or self._shutdown_submitted):
            return
        try:
            self._shutdown_executor()
        except Exception as exc:
            _LOGGER.exception("System power-off failed")
            self._cancel_shutdown(StatusCode.INTERNAL_ERROR)
            self._change(error=f"System power-off failed: {exc}")
        else:
            self._change(shutdown_submitted=True)
            self._closed = True

    def _perform(self, command: CommandPacket) -> StatusCode:
        code = command.code
        if code is CommandCode.SHUTDOWN:
            if not self._shutdown_requested:
                self._change(shutdown_requested=True)
                try:
                    self.end_experiment()
                except Exception:
                    self._change(shutdown_requested=False)
                    raise
                self._shutdown_ready = True
            return StatusCode.OK
        self._require_open()
        if code is CommandCode.START_EXPERIMENT:
            self.start_experiment(
                command.subject, command.controller_timestamp_ns,
                number=command.experiment_number, remote=True,
            )
        elif code is CommandCode.STOP_EXPERIMENT:
            self.end_experiment()
        elif code is CommandCode.CLEAR_ERROR:
            self.clear_error()
        else:
            if self._experiment is None or self._mode == "calibration":
                raise InvalidStateError(
                    "Tracking commands require an experiment outside calibration"
                )
            if code is CommandCode.START_TRACKING:
                self._start_saved_run(command.block_number, tracking=True)
            elif self._service is None:
                if code not in (CommandCode.STOP_TRACKING, CommandCode.STOP_RECORDING):
                    raise InvalidStateError("Open tracking before starting recording")
            elif code is CommandCode.STOP_TRACKING:
                self._service.stop_tracking()
            elif code is CommandCode.START_RECORDING:
                self._start_saved_run(command.block_number, tracking=False)
            elif code is CommandCode.STOP_RECORDING:
                self._service.stop_recording()
            else:
                return StatusCode.UNKNOWN_COMMAND
        return StatusCode.OK

    def _start_saved_run(self, number: int | None = None, *, tracking: bool) -> None:
        """Apply a requested block ID only to this start; retries are handled by execute."""
        validate_run_number(number)
        if tracking:
            self.open_tracking()
        elif self._service is None or not self._service.tracking:
            raise InvalidStateError("Tracking must be active before recording")
        if number is not None:
            # Close every writer before replacing an existing block directory.
            if tracking:
                self._service.stop_tracking()
            else:
                self._service.stop_recording()
        self._requested_block_number = number
        try:
            if tracking:
                self._service.start_tracking()
            else:
                self._service.start_recording()
        finally:
            self._requested_block_number = None

    def execute(self, command: CommandPacket) -> StatusSnapshot:
        if command.code is CommandCode.POLL:
            return self.snapshot()
        with self._transition:
            previous = self._commands.get(command.sequence)
            if previous is not None:
                if previous[0] != command:
                    return replace(
                        self.snapshot(), status=StatusCode.SEQUENCE_CONFLICT, error=True
                    )
                self._commands.move_to_end(command.sequence)
                status = previous[1]
            else:
                try:
                    if self._closed:
                        raise InvalidStateError("Launcher is closing")
                    status = self._perform(command)
                except (InvalidStateError, ValueError) as exc:
                    self.report_error(str(exc))
                    status = StatusCode.INVALID_STATE
                except Exception as exc:
                    _LOGGER.exception("Pi tracker command %s failed", command.code.name)
                    self.report_error(str(exc))
                    status = StatusCode.INTERNAL_ERROR
                self._commands[command.sequence] = (command, status)
                while len(self._commands) > 64:
                    self._commands.popitem(last=False)
            snapshot = self.snapshot()
            return replace(
                snapshot, status=status, error=snapshot.error or status != StatusCode.OK
            )

    def toggle_tracking(self) -> None:
        with self._transition:
            self._require_open()
            if self._service is None:
                raise InvalidStateError("Open the tracking view first")
            if self._service.tracking:
                self._service.stop_tracking()
            else:
                self._start_saved_run(tracking=True)

    def close(self) -> None:
        """Close acquisition and await cleanup; closing the app never requests power-off."""
        with self._transition:
            if self._close_complete or self._shutdown_submitted:
                return
            self._closed = True
            if self._shutdown_requested:
                self._cancel_shutdown(StatusCode.INVALID_STATE)
            self.end_experiment()
            worker = self._sync_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        self._close_complete = True
