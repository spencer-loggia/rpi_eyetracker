from __future__ import annotations

import math
import shutil
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar, Self

from .config import CameraConfig, RecordingConfig, RoiLayout
from .models import AnalysisFrame, GrayImage, decoded_monochrome_frame

H264_ENCODER_PRESET = "ultrafast"


def _crf_h264_encoder_type(base_encoder: type) -> type:
    """Add explicit libx264 CRF/VBV controls to Picamera2's software encoder."""

    class CrfH264Encoder(base_encoder):
        def __init__(
            self,
            *args: Any,
            crf: int,
            maximum_bitrate: int,
            preset: str,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._configured_crf = crf
            self._configured_maximum_bitrate = maximum_bitrate
            self._configured_preset = preset
            # Picamera2 already defaults to ultrafast, but setting the public
            # property makes the requested policy explicit before _start.
            self.preset = preset

        def _setup(self, quality: Any) -> None:
            # Preserve any setup work added by Picamera2, then disable its
            # quality-to-bitrate fallback so libx264 actually uses CRF.
            super()._setup(quality)
            self.bitrate = None

        def _start(self) -> None:
            super()._start()
            options = self._stream.codec_context.options
            options["preset"] = self._configured_preset
            options["crf"] = str(self._configured_crf)
            options["maxrate"] = str(self._configured_maximum_bitrate)
            options["bufsize"] = str(self._configured_maximum_bitrate * 2)

    CrfH264Encoder.__name__ = "CrfH264Encoder"
    return CrfH264Encoder


class CameraError(RuntimeError):
    pass


class VideoFileCamera:
    """Camera-compatible, real-time source backed by a prerecorded video.

    The file is resized to the configured analysis stream and loops at its
    reported frame rate. This keeps the normal threaded tracker and preview
    paths usable on development machines without Raspberry Pi camera bindings.
    """

    def __init__(
        self,
        video_path: str | Path,
        camera_config: CameraConfig,
        roi_layout: RoiLayout | None = None,
        *,
        realtime: bool = True,
    ) -> None:
        self.video_path = Path(video_path).expanduser()
        self.config = camera_config
        self.roi_layout = roi_layout
        self.realtime = realtime
        self._lock = threading.RLock()
        self._capture: Any | None = None
        self._cv2: Any | None = None
        self._pending_frame: GrayImage | None = None
        self._started = False
        self._closed = False
        self._sequence = 0
        self._frame_period_s = 1.0 / camera_config.fps
        self._next_frame_time = 0.0

    @property
    def started(self) -> bool:
        return self._started

    @property
    def recording(self) -> bool:
        return False

    def _open(self) -> tuple[Any, Any]:
        try:
            import cv2
        except ImportError as exc:
            raise CameraError(
                "Prerecorded-video playback requires OpenCV. Install the vision extra "
                "or the Raspberry Pi python3-opencv package."
            ) from exc
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"Could not open video: {self.video_path}")
        return cv2, capture

    def _gray_analysis_frame(self, image: Any) -> GrayImage:
        cv2 = self._cv2
        if cv2 is None:
            raise CameraError("Video source is not started")
        if image is None or not hasattr(image, "ndim"):
            raise CameraError(f"Video returned an invalid frame: {self.video_path}")
        try:
            gray = decoded_monochrome_frame(image)
        except ValueError as exc:
            raise CameraError(f"Invalid monochrome video frame: {exc}") from exc
        height, width = gray.shape
        configured_ratio = self.config.analysis_width / self.config.analysis_height
        frame_ratio = width / height
        if abs(frame_ratio / configured_ratio - 1.0) > 0.02:
            raise CameraError(
                f"Video aspect ratio {width}x{height} does not match configured "
                f"analysis stream {self.config.analysis_width}x"
                f"{self.config.analysis_height}"
            )
        target = (self.config.analysis_width, self.config.analysis_height)
        if (width, height) != target:
            gray = cv2.resize(gray, target, interpolation=cv2.INTER_AREA)
        if gray.dtype.name != "uint8":
            raise CameraError("Video frames must decode to 8-bit images")
        return gray.copy(order="C")

    def _read_frame_locked(self) -> GrayImage:
        if self._capture is None or self._cv2 is None:
            raise CameraError("Video source is not started")
        ok, image = self._capture.read()
        if not ok:
            # Some OpenCV/FFmpeg backends claim to seek after EOF but never
            # decode another frame. Try the cheap rewind first, then reopen the
            # file so configuration and tracking previews loop reliably for
            # containers/codecs whose decoder cannot seek backwards.
            rewound = self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
            if rewound:
                ok, image = self._capture.read()
            if not ok:
                previous = self._capture
                cv2, replacement = self._open()
                ok, image = replacement.read()
                if not ok:
                    replacement.release()
                else:
                    self._cv2 = cv2
                    self._capture = replacement
                    previous.release()
        if not ok:
            raise CameraError(f"Video contained no decodable frames: {self.video_path}")
        return self._gray_analysis_frame(image)

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise CameraError("Video source is closed")
            if self._started:
                return
            cv2, capture = self._open()
            self._cv2 = cv2
            self._capture = capture
            try:
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(fps) or fps <= 0.0:
                    fps = self.config.fps
                self._frame_period_s = 1.0 / fps
                # Decode and validate one frame now so startup errors are
                # reported synchronously rather than from the capture thread.
                self._pending_frame = self._read_frame_locked()
            except BaseException:
                capture.release()
                self._capture = None
                self._cv2 = None
                self._pending_frame = None
                raise
            self._sequence = 0
            self._next_frame_time = time.monotonic()
            self._started = True

    def stop(self) -> None:
        with self._lock:
            capture, self._capture = self._capture, None
            self._pending_frame = None
            self._started = False
            self._cv2 = None
        if capture is not None:
            capture.release()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.stop()

    def _next_gray_frame(self) -> tuple[GrayImage, int]:
        if not self._started:
            raise CameraError("Video source is not started")
        if self.realtime:
            delay = self._next_frame_time - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
        with self._lock:
            if not self._started:
                raise CameraError("Video source is not started")
            if self._pending_frame is not None:
                gray, self._pending_frame = self._pending_frame, None
            else:
                gray = self._read_frame_locked()
            timestamp = time.monotonic_ns()
            if self.realtime:
                now = time.monotonic()
                self._next_frame_time = max(
                    self._next_frame_time + self._frame_period_s,
                    now,
                )
        return gray, timestamp

    def capture_preview(self) -> tuple[GrayImage, int]:
        return self._next_gray_frame()

    def capture_analysis(self) -> AnalysisFrame:
        if self.roi_layout is None:
            raise CameraError("No ROI layout was supplied")
        gray, timestamp = self._next_gray_frame()
        pixel_rois = tuple(
            (
                roi.eye_id,
                roi.to_pixels(
                    self.config.analysis_width,
                    self.config.analysis_height,
                ),
            )
            for roi in self.roi_layout.rois
        )
        crops = tuple((eye_id, roi.extract(gray)) for eye_id, roi in pixel_rois)
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        return AnalysisFrame(
            frame_sequence=self._sequence,
            sensor_timestamp_ns=timestamp,
            crops=crops,
        )

    def image_control_limits(self) -> dict[str, tuple[float, float]]:
        return {}

    def set_image_controls(self, **controls: float) -> CameraConfig:
        if controls:
            raise CameraError("Prerecorded video does not support camera image controls")
        return self.config

    def start_recording(self, destination: str | Path) -> Path:
        del destination
        raise CameraError("Prerecorded video input cannot be recorded")

    def stop_recording(self) -> None:
        return None


class Picamera2Camera:
    """Single owner of the CamArray capture, analysis, and recording streams."""

    def __init__(
        self,
        camera_config: CameraConfig,
        recording_config: RecordingConfig,
        roi_layout: RoiLayout | None = None,
    ) -> None:
        self.config = camera_config
        self.recording_config = recording_config
        self.roi_layout = roi_layout
        try:
            from libcamera import controls as libcamera_controls
            from picamera2 import MappedArray, Picamera2
            from picamera2.encoders import LibavH264Encoder
            from picamera2.outputs import FileOutput, PyavOutput
        except ImportError as exc:
            raise CameraError(
                "Raspberry Pi camera packages are missing. Install python3-picamera2 "
                "and python3-libcamera from Raspberry Pi OS."
            ) from exc

        self._MappedArray = MappedArray
        self._H264Encoder = _crf_h264_encoder_type(LibavH264Encoder)
        self._FileOutput = FileOutput
        self._PyavOutput = PyavOutput
        self._controls = libcamera_controls
        self._camera = Picamera2(camera_config.camera_index)
        self._lock = threading.RLock()
        self._started = False
        self._recording = False
        self._encoder: Any | None = None
        self._output_error: Exception | None = None
        self._sequence = 0
        self._led: Any | None = None
        self._closed = False
        try:
            self._configure()
        except BaseException as configure_error:
            try:
                self._camera.close()
            except Exception as cleanup_error:  # noqa: BLE001 - preserve configure failure
                configure_error.add_note(f"camera close also failed: {cleanup_error}")
            self._closed = True
            raise

    def _fast_noise_reduction(self) -> Any | None:
        draft = getattr(self._controls, "draft", None)
        enum_type = getattr(draft, "NoiseReductionModeEnum", None)
        if enum_type is None:
            enum_type = getattr(self._controls, "NoiseReductionModeEnum", None)
        return None if enum_type is None else getattr(enum_type, "Fast", None)

    def _available_control(
        self,
        name: str,
        value: Any,
        *,
        required: bool = False,
    ) -> tuple[str, Any] | None:
        info = self._camera.camera_controls.get(name)
        if info is None:
            if required:
                raise CameraError(f"Camera does not expose required control: {name}")
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                minimum, maximum = float(info[0]), float(info[1])
            except (IndexError, TypeError, ValueError):
                pass
            else:
                if not minimum <= float(value) <= maximum:
                    raise CameraError(
                        f"Camera control {name}={value} is outside "
                        f"[{minimum:g}, {maximum:g}]"
                    )
        return name, value

    def _camera_controls(self) -> dict[str, Any]:
        cfg = self.config
        frame_duration_us = max(1, round(1_000_000.0 / cfg.fps))
        requested: list[tuple[str, Any] | None] = [
            self._available_control(
                "FrameDurationLimits", (frame_duration_us, frame_duration_us)
            ),
            self._available_control("AeEnable", False),
            self._available_control("AwbEnable", False),
            self._available_control("ExposureTime", cfg.exposure_us, required=True),
            self._available_control("AnalogueGain", cfg.analogue_gain, required=True),
            self._available_control("Brightness", cfg.brightness),
            self._available_control("Contrast", cfg.contrast),
            self._available_control("Saturation", 0.0),
            self._available_control("Sharpness", cfg.sharpness),
        ]
        fast_denoise = self._fast_noise_reduction()
        if fast_denoise is not None:
            requested.append(
                self._available_control("NoiseReductionMode", fast_denoise)
            )
        return dict(item for item in requested if item is not None)

    _IMAGE_CONTROL_NAMES: ClassVar[dict[str, str]] = {
        "exposure_us": "ExposureTime",
        "analogue_gain": "AnalogueGain",
        "brightness": "Brightness",
        "contrast": "Contrast",
        "sharpness": "Sharpness",
    }

    def image_control_limits(self) -> dict[str, tuple[float, float]]:
        """Return numeric min/max values for adjustable monochrome image controls."""

        limits: dict[str, tuple[float, float]] = {}
        available = self._camera.camera_controls
        for config_name, control_name in self._IMAGE_CONTROL_NAMES.items():
            info = available.get(control_name)
            if info is None:
                continue
            try:
                minimum, maximum = float(info[0]), float(info[1])
            except (IndexError, TypeError, ValueError):
                continue
            if minimum <= maximum:
                limits[config_name] = (minimum, maximum)
        return limits

    def set_image_controls(
        self,
        *,
        exposure_us: int | None = None,
        analogue_gain: float | None = None,
        brightness: float | None = None,
        contrast: float | None = None,
        sharpness: float | None = None,
    ) -> CameraConfig:
        """Apply validated manual image controls without reconfiguring the streams."""

        requested = {
            name: value
            for name, value in {
                "exposure_us": exposure_us,
                "analogue_gain": analogue_gain,
                "brightness": brightness,
                "contrast": contrast,
                "sharpness": sharpness,
            }.items()
            if value is not None
        }
        if not requested:
            return self.config
        updated = replace(self.config, **requested)
        controls: dict[str, Any] = {}
        for config_name, value in requested.items():
            control_name = self._IMAGE_CONTROL_NAMES[config_name]
            self._available_control(control_name, value, required=True)
            controls[control_name] = value
        with self._lock:
            if self._closed:
                raise CameraError("Camera is closed")
            self._camera.set_controls(controls)
            self.config = updated
        return updated

    def _configure(self) -> None:
        cfg = self.config
        # Picamera2's H.264 path accepts YUV420 but not Y8. Saturation is fixed
        # to zero in _camera_controls, so the required chroma planes are
        # neutral while both analysis and preview consume only the Y plane.
        configuration = self._camera.create_video_configuration(
            main={"format": "YUV420", "size": (cfg.video_width, cfg.video_height)},
            lores={
                "format": "YUV420",
                "size": (cfg.analysis_width, cfg.analysis_height),
            },
            sensor={
                "output_size": (cfg.sensor_width, cfg.sensor_height),
                "bit_depth": cfg.sensor_bit_depth,
            },
            controls=self._camera_controls(),
            buffer_count=cfg.buffer_count,
            queue=False,
            encode="main",
        )
        self._camera.configure(configuration)
        applied = self._camera.camera_configuration()
        sensor = applied.get("sensor", {})
        selected_size = tuple(sensor.get("output_size", ()))
        selected_depth = sensor.get("bit_depth")
        expected_size = (cfg.sensor_width, cfg.sensor_height)
        if selected_size != expected_size or int(selected_depth or -1) != cfg.sensor_bit_depth:
            raise CameraError(
                "Camera did not negotiate the requested aggregate sensor mode: "
                f"requested {expected_size}/{cfg.sensor_bit_depth}-bit, got "
                f"{selected_size}/{selected_depth}-bit. Check `rpicam-hello --list-cameras` "
                "and update the configuration to an advertised CamArray mode."
            )
        for stream_name, expected in (
            ("main", (cfg.video_width, cfg.video_height)),
            ("lores", (cfg.analysis_width, cfg.analysis_height)),
        ):
            actual = tuple(applied[stream_name]["size"])
            if actual != expected:
                raise CameraError(
                    f"Applied {stream_name} stream is {actual}, expected {expected}"
                )

    @property
    def started(self) -> bool:
        return self._started

    @property
    def recording(self) -> bool:
        return self._recording

    def _start_led(self) -> None:
        if self.config.ir_led_pin is None:
            return
        if self._led is None:
            try:
                from gpiozero import LED
            except ImportError as exc:
                raise CameraError(
                    "gpiozero is required when camera.ir_led_pin is configured"
                ) from exc
            self._led = LED(self.config.ir_led_pin)
        self._led.on()

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise CameraError("Camera is closed")
            if self._started:
                return
            self._start_led()
            camera_started = False
            try:
                if self.config.ir_led_warmup_seconds:
                    time.sleep(self.config.ir_led_warmup_seconds)
                self._camera.start()
                camera_started = True
                self._camera.set_controls(self._camera_controls())
                self._started = True
            except BaseException as start_error:
                if camera_started:
                    try:
                        self._camera.stop()
                    except Exception as cleanup_error:  # noqa: BLE001 - preserve start failure
                        start_error.add_note(f"camera stop also failed: {cleanup_error}")
                if self._led is not None:
                    try:
                        self._led.off()
                    except Exception as cleanup_error:  # noqa: BLE001 - preserve start failure
                        start_error.add_note(f"IR light shutdown also failed: {cleanup_error}")
                raise

    def stop(self) -> None:
        with self._lock:
            error: Exception | None = None
            if self._recording:
                try:
                    self.stop_recording()
                except Exception as exc:  # noqa: BLE001 - continue hardware cleanup
                    error = exc
            if self._started:
                try:
                    self._camera.stop()
                except Exception as exc:  # noqa: BLE001 - still switch off illumination
                    if error is None:
                        error = exc
                finally:
                    self._started = False
            if self._led is not None:
                try:
                    self._led.off()
                except Exception as exc:  # noqa: BLE001 - report after all cleanup
                    if error is None:
                        error = exc
            if error is not None:
                raise error

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            error: Exception | None = None
            try:
                self.stop()
            except Exception as exc:  # noqa: BLE001 - still release camera and GPIO
                error = exc
            try:
                self._camera.close()
            except Exception as exc:  # noqa: BLE001 - still release GPIO
                if error is None:
                    error = exc
            finally:
                if self._led is not None:
                    try:
                        self._led.close()
                    except Exception as exc:  # noqa: BLE001 - preserve first error
                        if error is None:
                            error = exc
                    self._led = None
                self._closed = True
            if error is not None:
                raise error

    def capture_analysis(self) -> AnalysisFrame:
        self._raise_recording_output_error()
        if not self._started:
            raise CameraError("Camera is not started")
        if self.roi_layout is None:
            raise CameraError("No ROI layout was supplied")
        cfg = self.config
        pixel_rois = tuple(
            (roi.eye_id, roi.to_pixels(cfg.analysis_width, cfg.analysis_height))
            for roi in self.roi_layout.rois
        )
        with self._camera.captured_request() as request:
            metadata = request.get_metadata()
            sensor_timestamp = int(metadata.get("SensorTimestamp", time.monotonic_ns()))
            with self._MappedArray(request, "lores") as mapped:
                y_plane = mapped.array[: cfg.analysis_height, : cfg.analysis_width]
                crops = tuple((eye_id, roi.extract(y_plane)) for eye_id, roi in pixel_rois)
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        return AnalysisFrame(
            frame_sequence=self._sequence,
            sensor_timestamp_ns=sensor_timestamp,
            crops=crops,
        )

    def capture_preview(self) -> tuple[GrayImage, int]:
        if not self._started:
            raise CameraError("Camera is not started")
        cfg = self.config
        with self._camera.captured_request() as request:
            metadata = request.get_metadata()
            timestamp = int(metadata.get("SensorTimestamp", time.monotonic_ns()))
            with self._MappedArray(request, "lores") as mapped:
                # Always copy before MappedArray releases the camera request;
                # ascontiguousarray may otherwise return the mapped view itself.
                gray = mapped.array[: cfg.analysis_height, : cfg.analysis_width].copy(order="C")
        return gray, timestamp

    def _record_output_error(self, error: Exception) -> None:
        """Remember the first asynchronous container/muxing failure."""

        with self._lock:
            if self._output_error is None:
                self._output_error = error

    def _raise_recording_output_error(self) -> None:
        with self._lock:
            error = self._output_error
        if error is not None:
            raise CameraError(f"Recording output failed: {error}") from error

    def start_recording(self, destination: str | Path) -> Path:
        with self._lock:
            if not self._started:
                raise CameraError("Tracking camera must be running before recording starts")
            if self._recording:
                raise CameraError("Recording is already active")
            path = Path(destination).expanduser().resolve()
            if path.suffix.lower() not in {".h264", ".264", ".mkv", ".mp4"}:
                raise CameraError("Recording destination must end in .h264, .mkv, or .mp4")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise CameraError(f"Refusing to overwrite existing recording: {path}")
            free = shutil.disk_usage(path.parent).free
            required = int(self.recording_config.minimum_free_gib * 1024**3)
            if free < required:
                raise CameraError(
                    f"Only {free / 1024**3:.1f} GiB free; recording requires at least "
                    f"{self.recording_config.minimum_free_gib:.1f} GiB"
                )
            encoder = self._H264Encoder(
                bitrate=None,
                repeat=True,
                iperiod=self.recording_config.intra_period,
                framerate=max(1, round(self.config.fps)),
                preset=H264_ENCODER_PRESET,
                crf=self.recording_config.crf,
                maximum_bitrate=self.recording_config.bitrate,
            )
            output = (
                self._FileOutput(str(path))
                if path.suffix.lower() in {".h264", ".264"}
                else self._PyavOutput(str(path))
            )
            self._output_error = None
            if hasattr(output, "error_callback"):
                output.error_callback = self._record_output_error
            try:
                self._camera.start_encoder(encoder, output, name="main")
            except Exception as exc:
                # start_encoder can fail after opening the destination. Best-effort
                # cleanup keeps a retry from finding a stale partial file.
                try:
                    self._camera.stop_encoder(encoder)
                except Exception as cleanup_error:  # noqa: BLE001 - preserve encoder failure
                    exc.add_note(f"encoder cleanup also failed: {cleanup_error}")
                stop_output = getattr(output, "stop", None)
                if callable(stop_output):
                    try:
                        stop_output()
                    except Exception as cleanup_error:  # noqa: BLE001 - preserve failure
                        exc.add_note(f"output cleanup also failed: {cleanup_error}")
                path.unlink(missing_ok=True)
                raise CameraError(f"Could not start H.264 encoder: {exc}") from exc
            self._encoder = encoder
            self._recording = True
            return path

    def stop_recording(self) -> None:
        with self._lock:
            if not self._recording:
                return
            stop_error: Exception | None = None
            try:
                self._camera.stop_encoder(self._encoder)
            except Exception as exc:  # noqa: BLE001 - report after clearing state
                stop_error = exc
            finally:
                self._recording = False
                self._encoder = None
            output_error, self._output_error = self._output_error, None
            if output_error is not None:
                error = CameraError(f"Recording output failed: {output_error}")
                if stop_error is not None:
                    error.add_note(f"encoder stop also failed: {stop_error}")
                raise error from output_error
            if stop_error is not None:
                raise stop_error

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
