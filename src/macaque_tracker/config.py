from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from .models import EyeImageSettings, NormalizedRoi

DEFAULT_CONFIG_PATH = Path("config/eye_tracker.json")
DEFAULT_ROI_PATH = Path("rois.json")


class ConfigError(ValueError):
    """Raised when a configuration is missing or unsafe."""


@dataclass(frozen=True)
class CameraConfig:
    camera_index: int = 0
    sensor_width: int = 5120
    sensor_height: int = 720
    sensor_bit_depth: int = 8
    video_width: int = 3840
    video_height: int = 540
    analysis_width: int = 2560
    analysis_height: int = 360
    fps: float = 30.0
    buffer_count: int = 6
    exposure_us: int = 19_000
    analogue_gain: float = 4.0
    brightness: float = 0.0
    contrast: float = 1.0
    sharpness: float = 0.0
    ir_led_pin: int | None = None
    ir_led_warmup_seconds: float = 0.2

    def __post_init__(self) -> None:
        integer_positive = (
            "sensor_width",
            "sensor_height",
            "sensor_bit_depth",
            "video_width",
            "video_height",
            "analysis_width",
            "analysis_height",
            "buffer_count",
            "exposure_us",
        )
        for name in integer_positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"camera.{name} must be a positive integer")
        if (
            isinstance(self.camera_index, bool)
            or not isinstance(self.camera_index, int)
            or self.camera_index < 0
        ):
            raise ConfigError("camera.camera_index must be non-negative")
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise ConfigError("camera.fps must be positive")
        if self.exposure_us >= 1_000_000.0 / self.fps:
            raise ConfigError("camera.exposure_us must be shorter than one frame period")
        for name, minimum, inclusive in (
            ("analogue_gain", 0.0, False),
            ("contrast", 0.0, False),
            ("sharpness", 0.0, True),
            ("ir_led_warmup_seconds", 0.0, True),
        ):
            value = getattr(self, name)
            valid = math.isfinite(value) and (
                value >= minimum if inclusive else value > minimum
            )
            if not valid:
                comparison = "non-negative" if inclusive else "positive"
                raise ConfigError(f"camera.{name} must be finite and {comparison}")
        if not math.isfinite(self.brightness):
            raise ConfigError("camera.brightness must be finite")
        if self.analysis_width % 2 or self.analysis_height % 2:
            raise ConfigError("analysis stream dimensions must be even for YUV420")
        if self.video_width % 2 or self.video_height % 2:
            raise ConfigError("video stream dimensions must be even for YUV420")
        sensor_ratio = self.sensor_width / self.sensor_height
        for label, width, height in (
            ("video", self.video_width, self.video_height),
            ("analysis", self.analysis_width, self.analysis_height),
        ):
            if abs((width / height) / sensor_ratio - 1.0) > 0.02:
                raise ConfigError(
                    f"camera.{label} aspect ratio must match the stitched sensor mode"
                )
        if self.ir_led_pin is not None and (
            isinstance(self.ir_led_pin, bool)
            or not isinstance(self.ir_led_pin, int)
            or not 0 <= self.ir_led_pin <= 27
        ):
            raise ConfigError("camera.ir_led_pin must be null or a BCM GPIO number in [0, 27]")


@dataclass(frozen=True)
class TrackerConfig:
    min_pupil_diameter_px: float = 8.0
    max_pupil_diameter_fraction: float = 0.75
    min_axis_ratio: float = 0.45
    min_contrast: float = 8.0
    pupil_size_bias: float = 0.0
    min_confidence: float = 0.48
    threshold_percentiles: tuple[float, ...] = (2.0, 5.0, 9.0, 14.0)
    max_position_jump_fraction: float = 0.35
    max_diameter_change_fraction: float = 0.55
    # Defaults report the per-frame measurement without temporal smoothing so
    # saccade kinetics are not delayed. Values below 1 enable optional EMA.
    position_smoothing: float = 1.0
    diameter_smoothing: float = 1.0
    blink_after_missing_frames: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_pupil_diameter_px) or self.min_pupil_diameter_px <= 0.0:
            raise ConfigError("tracker.min_pupil_diameter_px must be positive")
        for name in (
            "max_pupil_diameter_fraction",
            "min_axis_ratio",
            "min_confidence",
            "max_position_jump_fraction",
            "max_diameter_change_fraction",
            "position_smoothing",
            "diameter_smoothing",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ConfigError(f"tracker.{name} must be in (0, 1]")
        if not math.isfinite(self.min_contrast) or self.min_contrast < 0.0:
            raise ConfigError("tracker.min_contrast must be non-negative")
        if (
            isinstance(self.pupil_size_bias, bool)
            or not math.isfinite(self.pupil_size_bias)
            or not -1.0 <= self.pupil_size_bias <= 1.0
        ):
            raise ConfigError("tracker.pupil_size_bias must be finite and in [-1, 1]")
        if (
            isinstance(self.blink_after_missing_frames, bool)
            or not isinstance(self.blink_after_missing_frames, int)
            or self.blink_after_missing_frames < 1
        ):
            raise ConfigError("tracker.blink_after_missing_frames must be >= 1")
        if not self.threshold_percentiles:
            raise ConfigError("tracker.threshold_percentiles cannot be empty")
        if any(not 0.0 < value < 50.0 for value in self.threshold_percentiles):
            raise ConfigError("tracker threshold percentiles must lie in (0, 50)")


@dataclass(frozen=True)
class RecordingConfig:
    directory: str = "recordings"
    container: str = "mkv"
    crf: int = 24
    # CRF controls normal output size; this is a conservative VBV ceiling.
    bitrate: int = 64_000_000
    intra_period: int = 60
    record_on_tracking: bool = True
    minimum_free_gib: float = 4.0

    def __post_init__(self) -> None:
        if not isinstance(self.directory, str) or not self.directory.strip():
            raise ConfigError("recording.directory cannot be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.bitrate, self.intra_period)
        ):
            raise ConfigError("recording bitrate and intra_period must be positive")
        if (
            isinstance(self.crf, bool)
            or not isinstance(self.crf, int)
            or not 0 <= self.crf <= 51
        ):
            raise ConfigError("recording.crf must be an integer in [0, 51]")
        if self.container not in {"mkv", "mp4", "h264"}:
            raise ConfigError("recording.container must be mkv, mp4, or h264")
        if not isinstance(self.record_on_tracking, bool):
            raise ConfigError("recording.record_on_tracking must be boolean")
        if not math.isfinite(self.minimum_free_gib) or self.minimum_free_gib < 0.0:
            raise ConfigError("recording.minimum_free_gib must be non-negative")


@dataclass(frozen=True)
class TransportConfig:
    backend: str = "unix"
    uart_device: str = "/dev/ttyAMA0"
    uart_baud: int = 460_800
    unix_socket: str = "/tmp/macaque-eye-tracker.sock"
    transaction_size: int = 64

    def __post_init__(self) -> None:
        if self.backend not in {"uart", "unix", "none"}:
            raise ConfigError("transport.backend must be uart, unix, or none")
        if self.transaction_size != 64:
            raise ConfigError("Protocol version 1 requires 64-byte transactions")
        if self.backend == "uart" and (
            not isinstance(self.uart_device, str) or not self.uart_device.strip()
        ):
            raise ConfigError("transport.uart_device cannot be empty")
        if self.uart_baud not in {115_200, 230_400, 460_800, 921_600}:
            raise ConfigError("transport.uart_baud must be 115200, 230400, 460800, or 921600")
        if self.backend == "unix" and (
            not isinstance(self.unix_socket, str) or not self.unix_socket.strip()
        ):
            raise ConfigError("transport.unix_socket cannot be empty")


@dataclass(frozen=True)
class PreviewConfig:
    """Optional local diagnostic display for the analysis crops."""

    enabled: bool = False
    max_display_width: int = 1920
    max_display_height: int = 1080
    show_threshold_mask: bool = True
    window_name: str = "Macaque eye tracker"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigError("preview.enabled must be boolean")
        if not isinstance(self.show_threshold_mask, bool):
            raise ConfigError("preview.show_threshold_mask must be boolean")
        for name in ("max_display_width", "max_display_height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"preview.{name} must be a positive integer")
        if not isinstance(self.window_name, str) or not self.window_name.strip():
            raise ConfigError("preview.window_name cannot be empty")


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    roi_config: str = str(DEFAULT_ROI_PATH)

    def __post_init__(self) -> None:
        if not isinstance(self.roi_config, str) or not self.roi_config.strip():
            raise ConfigError("roi_config must be a non-empty path string")
        if self.transport.backend == "uart" and self.camera.ir_led_pin in {14, 15}:
            raise ConfigError(
                "camera.ir_led_pin cannot use GPIO14/15 while transport.backend is uart"
            )

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
        config_path = Path(path).expanduser()
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"Configuration does not exist: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError("Top-level configuration must be a JSON object")
        _reject_unknown(raw, cls, "config")
        try:
            return cls(
                camera=_dataclass_from_dict(CameraConfig, raw.get("camera", {}), "camera"),
                tracker=_tracker_from_dict(raw.get("tracker", {})),
                recording=_dataclass_from_dict(
                    RecordingConfig, raw.get("recording", {}), "recording"
                ),
                transport=_dataclass_from_dict(
                    TransportConfig, raw.get("transport", {}), "transport"
                ),
                preview=_dataclass_from_dict(PreviewConfig, raw.get("preview", {}), "preview"),
                roi_config=raw.get("roi_config", str(DEFAULT_ROI_PATH)),
            )
        except TypeError as exc:
            raise ConfigError(f"Invalid configuration values: {exc}") from exc

    def save(self, path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
        """Atomically save the complete application configuration as JSON."""

        config_path = Path(path).expanduser()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = config_path.with_suffix(config_path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        temporary.replace(config_path)
        return config_path


@dataclass(frozen=True)
class RoiLayout:
    rois: tuple[NormalizedRoi, ...]
    source_width: int
    source_height: int
    version: int = 1

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ConfigError(f"Unsupported ROI configuration version: {self.version}")
        if self.source_width <= 0 or self.source_height <= 0:
            raise ConfigError("ROI source dimensions must be positive")
        if not 1 <= len(self.rois) <= 2:
            raise ConfigError("ROI configuration must contain one or two eye boxes")
        ids = [roi.eye_id for roi in self.rois]
        if ids != list(range(len(ids))):
            raise ConfigError("ROI eye_id values must be contiguous and ordered from zero")

    @classmethod
    def load(cls, path: str | Path) -> RoiLayout:
        roi_path = Path(path).expanduser()
        try:
            raw = json.loads(roi_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(
                f"ROI configuration does not exist: {roi_path}. Run `eye-tracker configure`."
            ) from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid ROI JSON in {roi_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError("ROI configuration must be a JSON object")
        expected = {"version", "source_width", "source_height", "rois"}
        unknown = set(raw) - expected
        if unknown:
            raise ConfigError(f"Unknown ROI fields: {', '.join(sorted(unknown))}")
        raw_rois = raw.get("rois")
        if not isinstance(raw_rois, list):
            raise ConfigError("ROI field `rois` must be a list")
        try:
            rois: list[NormalizedRoi] = []
            for item in raw_rois:
                if not isinstance(item, dict):
                    raise TypeError("Each ROI must be a JSON object")
                values = dict(item)
                raw_settings = values.pop("settings", {})
                if not isinstance(raw_settings, dict):
                    raise TypeError("ROI settings must be a JSON object")
                # Fixed per-eye thresholds were supported briefly. They cannot
                # be translated into a size preference, so old files migrate
                # to the neutral bias and regain fully adaptive segmentation.
                raw_settings = dict(raw_settings)
                raw_settings.pop("pupil_threshold", None)
                # Software sharpening amplified sensor/codec noise before the
                # detector blurred it again; retain compatibility without
                # carrying that ineffective control forward.
                raw_settings.pop("sharpness", None)
                settings = EyeImageSettings(**raw_settings)
                rois.append(NormalizedRoi(**values, settings=settings))
            return cls(
                rois=tuple(rois),
                source_width=int(raw["source_width"]),
                source_height=int(raw["source_height"]),
                version=int(raw.get("version", 1)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid ROI configuration: {exc}") from exc

    def save(self, path: str | Path) -> Path:
        roi_path = Path(path).expanduser()
        roi_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.version,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "rois": [roi.as_dict() for roi in self.rois],
        }
        temporary = roi_path.with_suffix(roi_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(roi_path)
        return roi_path

    def validate_for_frame(
        self,
        frame_width: int,
        frame_height: int,
        *,
        minimum_crop_size: int = 24,
        tracker_config: TrackerConfig | None = None,
    ) -> None:
        """Validate that normalized boxes remain usable on an analysis stream."""
        if frame_width <= 0 or frame_height <= 0:
            raise ConfigError("Analysis frame dimensions must be positive")
        source_ratio = self.source_width / self.source_height
        frame_ratio = frame_width / frame_height
        if abs(frame_ratio / source_ratio - 1.0) > 0.02:
            raise ConfigError(
                "ROI source aspect ratio does not match the configured analysis stream; "
                "rerun `eye-tracker configure`"
            )
        for roi in self.rois:
            pixels = roi.to_pixels(frame_width, frame_height)
            if pixels.width < minimum_crop_size or pixels.height < minimum_crop_size:
                raise ConfigError(
                    f"ROI eye_id={roi.eye_id} becomes {pixels.width}x{pixels.height} pixels; "
                    f"both dimensions must be at least {minimum_crop_size}"
                )
            if (
                tracker_config is not None
                and tracker_config.max_pupil_diameter_fraction
                * min(pixels.width, pixels.height)
                < tracker_config.min_pupil_diameter_px
            ):
                raise ConfigError(
                    f"ROI eye_id={roi.eye_id} is too small for the configured pupil "
                    "diameter limits"
                )


T = TypeVar("T")


def _reject_unknown(raw: dict[str, Any], cls: type[Any], section: str) -> None:
    known = {item.name for item in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown {section} fields: {', '.join(sorted(unknown))}")


def _dataclass_from_dict(cls: type[T], raw: Any, section: str) -> T:
    if not isinstance(raw, dict):
        raise ConfigError(f"{section} must be a JSON object")
    _reject_unknown(raw, cls, section)
    return cls(**raw)


def _tracker_from_dict(raw: Any) -> TrackerConfig:
    if not isinstance(raw, dict):
        raise ConfigError("tracker must be a JSON object")
    values = dict(raw)
    # Accept old configuration files while deliberately returning their fixed
    # threshold override to neutral, frame-adaptive behavior.
    values.pop("pupil_threshold", None)
    _reject_unknown(values, TrackerConfig, "tracker")
    if "threshold_percentiles" in values:
        if not isinstance(values["threshold_percentiles"], list):
            raise ConfigError("tracker.threshold_percentiles must be a list")
        values["threshold_percentiles"] = tuple(values["threshold_percentiles"])
    return TrackerConfig(**values)
