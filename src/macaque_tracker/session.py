from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .camera import Picamera2Camera
from .config import AppConfig, RoiLayout
from .models import FrameResult


class RecordingSession:
    """H.264 recording plus an append-only scientific metadata sidecar."""

    def __init__(
        self,
        camera: Picamera2Camera,
        config: AppConfig,
        layout: RoiLayout,
        session_id: int,
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

    @staticmethod
    def _configuration_hash(config: AppConfig, layout: RoiLayout) -> str:
        value = {"config": asdict(config), "roi_layout": asdict(layout)}
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def start(self) -> Path:
        if self._stream is not None:
            raise RuntimeError("Recording session is already active")
        directory = Path(self.config.recording.directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        stem = f"eye_{stamp}_{self.session_id:08x}"
        video = directory / f"{stem}.{self.config.recording.container}"
        metadata = directory / f"{stem}.jsonl"
        if video.exists() or metadata.exists():
            raise RuntimeError(f"Recording collision for session stem {stem}")

        stream = metadata.open("x", encoding="utf-8", buffering=64 * 1024)
        recording_started = False
        try:
            self.camera.start_recording(video)
            recording_started = True
            self.video_path = video
            self.metadata_path = metadata
            self._stream = stream
            self._write(
                {
                    "type": "session_start",
                    "schema_version": 1,
                    "session_id": self.session_id,
                    "started_utc": datetime.now(UTC).isoformat(),
                    "video": video.name,
                    "configuration_sha256": self._configuration_hash(self.config, self.layout),
                    "sensor_size": [
                        self.config.camera.sensor_width,
                        self.config.camera.sensor_height,
                    ],
                    "video_size": [
                        self.config.camera.video_width,
                        self.config.camera.video_height,
                    ],
                    "video_scope": "full stitched main stream; all cameras; not ROI crops",
                    "analysis_size": [
                        self.config.camera.analysis_width,
                        self.config.camera.analysis_height,
                    ],
                    "coordinate_space": "crop-local analysis-stream pixels; origin top-left",
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
            self.video_path = None
            self.metadata_path = None
            self._stream = None
            raise
        return video

    def _write(self, value: dict[str, Any]) -> None:
        if self._stream is None:
            raise RuntimeError("Recording session is not active")
        self._stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")

    def write_result(self, result: FrameResult) -> None:
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
        with self._lock:
            if self._stream is None:
                return self.video_path, self.metadata_path
            error: Exception | None = None
            try:
                self.camera.stop_recording()
            except Exception as exc:  # noqa: BLE001 - still close the sidecar
                error = exc
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
