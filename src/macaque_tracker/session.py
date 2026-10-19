from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .camera import (
    H264_ENCODER_PRESET,
    Picamera2Camera,
    _recording_frame_skip_count,
)
from .config import AppConfig, RoiLayout
from .file_access import SHARED_FILE_MODE, share_output
from .models import FrameResult
from .orientation import rotated_size
from .recording import recording_layout
from .storage import DISK_CHECK_INTERVAL_S, LowDiskSpace, recording_space_warning


class RecordingSession:
    """H.264 recording plus an append-only scientific metadata sidecar."""

    def __init__(
        self,
        camera: Picamera2Camera,
        config: AppConfig,
        layout: RoiLayout,
        session_id: int,
        *,
        warning_callback=None,
    ) -> None:
        self.camera = camera
        self.config = config
        self.layout = layout
        self.session_id = session_id
        self.video_path: Path | None = None
        self.metadata_path: Path | None = None
        self._stream: Any | None = None
        self._rows = 0
        self._lock = threading.RLock()
        self._video_lock = threading.Lock()
        self._video_recording = False
        self._video_stop_error = None
        self._disk_stop = threading.Event()
        self._warning_callback = warning_callback
        self.warning_message = None

    @property
    def video_recording(self) -> bool:
        """Read video state without waiting for encoder finalization."""
        return self._video_recording

    def _warn(self, message: str) -> None:
        """Report once per incident; callbacks may not interfere with cleanup."""
        self.warning_message = message
        logging.getLogger(__name__).warning(message)
        if self._warning_callback is not None:
            try:
                self._warning_callback(message)
            except Exception:
                logging.getLogger(__name__).exception("Could not display recording warning")

    def _check_disk_space(self) -> bool:
        """Perform one guard check; return false after a cutoff or video stop."""
        if not self._video_recording:
            return False
        warning = recording_space_warning(
            self.metadata_path.parent, self.config.recording.minimum_free_gib,
        )
        if warning is None:
            return True
        self._warn(warning)
        try:
            self.stop_video(reason="low_disk_space")
        except Exception as exc:  # noqa: BLE001 - report flush failure without stopping tracking
            self._warn(f"{warning} Video finalization failed: {exc}")
        return False

    def _monitor_disk(self) -> None:
        """Check only while video is active, on a low-frequency background worker."""
        while not self._disk_stop.wait(DISK_CHECK_INTERVAL_S):
            if not self._check_disk_space():
                return

    def stop_video(self, *, reason: str = "requested") -> None:
        """Finalize video without holding the metadata lock or stopping position logs."""
        with self._video_lock:
            if not self._video_recording:
                return
            self._video_recording = False
            self._disk_stop.set()
            try:
                self.camera.stop_recording()
            except Exception as exc:
                self._video_stop_error = exc
                raise
            with self._lock:
                if self._stream is not None and reason != "requested":
                    self._write({"type": "video_stop", "reason": reason,
                                 "stopped_utc": datetime.now(UTC).isoformat()})
                    self._stream.flush()

    @staticmethod
    def _configuration_hash(config: AppConfig, layout: RoiLayout) -> str:
        """Identify the exact image, detector, orientation, and recording settings."""
        value = {"config": asdict(config), "roi_layout": asdict(layout)}
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _eye_coordinate_systems(self) -> list[dict[str, Any]]:
        """Describe each rotated eye crop at analysis and recording resolutions."""
        systems = []
        for roi in self.layout.rois:
            rotation = self.layout.rotation_for_roi(roi)
            analysis = roi.to_pixels(
                self.config.camera.analysis_width, self.config.camera.analysis_height,
            )
            recording = roi.to_pixels(self.config.recording.width, self.config.recording.height)
            systems.append({
                "eye_id": roi.eye_id,
                "rotation_degrees": rotation,
                "analysis_crop_size": list(rotated_size(analysis.width, analysis.height, rotation)),
                "recording_crop_size": list(rotated_size(
                    recording.width, recording.height, rotation,
                )),
            })
        return systems

    def _video_scope(self) -> str:
        """Distinguish selected views from full-frame recordings, including rotation."""
        if self.config.recording.layout == "full":
            if any(self.layout.camera_rotations):
                return "all four camera views with calibration rotations; not ROI crops"
            return "full stitched recording stream; all cameras; not ROI crops"
        cameras = self.config.recording.pose_cameras
        return (
            "pose cameras " + ", ".join(str(index) for index in cameras)
            + " (indices from left to right) plus unscaled eye ROI crops"
            if cameras else "unscaled eye ROI crops only"
        )

    def start(self, *, record_video: bool = True) -> Path | None:
        """Start metadata and, when space permits, video; remove partial failed starts."""
        if self._stream is not None:
            raise RuntimeError("Recording session is already active")
        directory = Path(self.config.recording.directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if self.config.recording.shared_permissions:
            share_output(directory, directory=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        stem = f"eye_{stamp}_{self.session_id:08x}"
        video = directory / f"{stem}.{self.config.recording.container}"
        metadata = directory / f"{stem}.jsonl"
        if video.exists() or metadata.exists():
            raise RuntimeError(f"Recording collision for session stem {stem}")

        stream = metadata.open("x", encoding="utf-8", buffering=64 * 1024)
        recording_started = False
        try:
            if self.config.recording.shared_permissions:
                os.fchmod(stream.fileno(), SHARED_FILE_MODE)
            packed_layout = recording_layout(self.config.recording, self.layout)
            frame_skip = _recording_frame_skip_count(
                self.config.camera.fps, self.config.recording.fps,
            )
            warning = (recording_space_warning(directory, self.config.recording.minimum_free_gib)
                       if record_video else None)
            if record_video and warning is None:
                try:
                    self.camera.start_recording(video)
                    recording_started = True
                except LowDiskSpace as exc:
                    warning = str(exc)
            if warning is not None:
                self._warn(warning)
            self._video_recording = recording_started
            if self.config.recording.shared_permissions and video.exists():
                share_output(video)
            self.video_path = video if recording_started else None
            self.metadata_path = metadata
            self._stream = stream
            self._write(
                {
                    "type": "session_start",
                    "schema_version": (
                        3 if any(self.layout.camera_rotations)
                        else 2 if packed_layout is not None else 1
                    ),
                    "session_id": self.session_id,
                    "started_utc": datetime.now(UTC).isoformat(),
                    "video": video.name if recording_started else None,
                    "video_disabled_reason": warning,
                    "configuration_sha256": self._configuration_hash(self.config, self.layout),
                    "sensor_size": [
                        self.config.camera.sensor_width,
                        self.config.camera.sensor_height,
                    ],
                    "video_size": [packed_layout.width, packed_layout.height]
                    if packed_layout is not None else [
                        self.config.recording.width,
                        self.config.recording.height,
                    ],
                    "camera_fps": self.config.camera.fps,
                    "video_fps": self.config.recording.fps,
                    "encoder_frame_skip_count": frame_skip,
                    "encoder_frame_selection": (
                        "phase_accumulator" if frame_skip is None else "fixed_stride"
                    ),
                    "pixel_model": "grayscale luma",
                    "encoded_pixel_format": "YUV420 with neutral chroma",
                    "video_codec": "H.264/libx264",
                    "encoder_preset": H264_ENCODER_PRESET,
                    "encoder_crf": self.config.recording.crf,
                    "encoder_maximum_bitrate": self.config.recording.bitrate,
                    "video_scope": self._video_scope(),
                    "pose_cameras": list(self.config.recording.pose_cameras),
                    "video_layout": None if packed_layout is None else packed_layout.as_dict(),
                    "analysis_size": [
                        self.config.camera.analysis_width,
                        self.config.camera.analysis_height,
                    ],
                    "coordinate_space": (
                        "crop-local analysis-stream pixels after calibration rotation; "
                        "origin top-left; +x right; +y down; not gaze degrees"
                    ),
                    "camera_rotations": list(self.layout.camera_rotations),
                    "rotation_convention": (
                        "clockwise degrees; cameras indexed left to right in the raw sensor frame"
                    ),
                    "eye_coordinate_systems": self._eye_coordinate_systems(),
                    "pupil_size": "external-contour area-equivalent diameter in pixels",
                    "rois": [roi.as_dict() for roi in self.layout.rois],
                }
            )
            stream.flush()
        except BaseException as start_error:
            if recording_started:
                try:
                    self.camera.stop_recording()
                except Exception as cleanup_error:  # noqa: BLE001 - preserve start failure
                    start_error.add_note(f"encoder cleanup also failed: {cleanup_error}")
            try:
                stream.close()
            except Exception as cleanup_error:  # noqa: BLE001 - preserve start failure
                start_error.add_note(f"sidecar cleanup also failed: {cleanup_error}")
            metadata.unlink(missing_ok=True)
            video.unlink(missing_ok=True)
            self._video_recording = False
            self._disk_stop.set()
            self.video_path = None
            self.metadata_path = None
            self._stream = None
            raise
        if recording_started:
            threading.Thread(target=self._monitor_disk, name="recording-disk-guard",
                             daemon=True).start()
        return self.video_path

    def _write(self, value: dict[str, Any]) -> None:
        """Append one finite-valued JSON record to the buffered sidecar."""
        if self._stream is None:
            raise RuntimeError("Recording session is not active")
        self._stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")

    def write_result(self, result: FrameResult) -> None:
        """Persist the same oriented pupil coordinates used by preview and UART."""
        with self._lock:
            if self._stream is None:
                return
            self._write(
                {
                    "type": "frame",
                    "frame_sequence": result.frame_sequence,
                    "sensor_timestamp_ns": result.sensor_timestamp_ns,
                    "produced_timestamp_ns": result.produced_timestamp_ns,
                    "processing_time_us": result.processing_time_us,
                    "dropped_analysis_frames": result.dropped_analysis_frames,
                    "eyes": [
                        {
                            "eye_id": eye.eye_id,
                            "x": eye.x,
                            "y": eye.y,
                            "pupil_diameter": eye.pupil_diameter,
                            "confidence": eye.confidence,
                            "valid": eye.valid,
                            "blink": eye.blink,
                        }
                        for eye in result.eyes
                    ],
                }
            )
            self._rows += 1
            if self._rows % 30 == 0:
                self._stream.flush()

    def stop(self) -> tuple[Path | None, Path | None]:
        """Finalize the encoder and sidecar, retaining cleanup errors for the caller."""
        self._disk_stop.set()
        error: Exception | None = self._video_stop_error
        try:
            self.stop_video()
        except Exception as exc:  # noqa: BLE001 - still close metadata after video errors
            error = exc
        # The disk guard may have been finalizing video while stop_video waited.
        error = error or self._video_stop_error
        with self._lock:
            if self._stream is None:
                return self.video_path, self.metadata_path
            stream = self._stream
            try:
                self._write(
                    {
                        "type": "session_end",
                        "ended_utc": datetime.now(UTC).isoformat(),
                        "tracked_rows": self._rows,
                        "encoder_stop_error": None if error is None else str(error),
                    }
                )
                stream.flush()
            finally:
                try:
                    stream.close()
                finally:
                    self._stream = None
            if error is not None:
                raise error
            return self.video_path, self.metadata_path
