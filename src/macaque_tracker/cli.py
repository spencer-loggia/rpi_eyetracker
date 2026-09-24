from __future__ import annotations

import argparse
import signal
import sys
import threading
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .benchmark import benchmark_video, format_benchmark
from .config import DEFAULT_CONFIG_PATH, AppConfig, ConfigError, RoiLayout


def _load(config_path: Path) -> tuple[AppConfig, RoiLayout]:
    config = AppConfig.load(config_path)
    roi_path = Path(config.roi_config).expanduser()
    if not roi_path.is_absolute():
        roi_path = (config_path.resolve().parent / roi_path).resolve()
    layout = RoiLayout.load(roi_path)
    layout.validate_for_frame(
        config.camera.analysis_width,
        config.camera.analysis_height,
    )
    return config, layout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eye-tracker",
        description="Head-fixed macaque pupil tracker for Arducam CamArray",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    configure = subparsers.add_parser(
        "configure",
        help="tune the camera, inspect pupil fits, and save 1-2 eye ROIs",
    )
    configure.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    configure.add_argument("--output", type=Path)
    configure_source = configure.add_mutually_exclusive_group()
    configure_source.add_argument(
        "--image",
        type=Path,
        help="use an image instead of the camera",
    )
    configure_source.add_argument(
        "--video",
        type=Path,
        help="average frames from a prerecorded video instead of the camera",
    )
    configure.add_argument("--average-frames", type=int, default=8)

    run = subparsers.add_parser("run", help="run the command service")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    run.add_argument(
        "--start-tracking",
        action="store_true",
        help="start locally without waiting for a command",
    )
    preview_override = run.add_mutually_exclusive_group()
    preview_override.add_argument(
        "--preview",
        dest="preview_override",
        action="store_true",
        help="show the diagnostic eye-crop display while tracking",
    )
    preview_override.add_argument(
        "--no-preview",
        dest="preview_override",
        action="store_false",
        help="disable the diagnostic display even when enabled in the configuration",
    )
    run.set_defaults(preview_override=None)

    preview = subparsers.add_parser(
        "preview",
        help="track live camera crops in a local display without recording",
    )
    preview.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    preview.add_argument(
        "--video",
        type=Path,
        help="run the tracker on a looping prerecorded video instead of the camera",
    )

    validate = subparsers.add_parser("validate", help="validate configuration files")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    validate.add_argument("--hardware", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="offline tracker timing check")
    benchmark.add_argument("video", type=Path)
    benchmark.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    benchmark.add_argument("--frames", type=int)
    return parser


def _run_service(
    config_path: Path,
    start_tracking: bool,
    preview_override: bool | None,
) -> int:
    from .service import EyeTrackingService
    from .transport import ProtocolEngine, UartServer, UnixSocketServer

    config, layout = _load(config_path)
    if preview_override is not None:
        config = replace(
            config,
            preview=replace(config.preview, enabled=preview_override),
        )
    stop = threading.Event()

    def request_stop(signum=None, frame=None) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with EyeTrackingService(config, layout) as service:
        if start_tracking:
            service.start_tracking()
        engine = ProtocolEngine(service)
        if config.transport.backend == "unix":
            server = UnixSocketServer(config.transport.unix_socket, engine)
            print(f"Ready on Unix socket {config.transport.unix_socket}", flush=True)
            server.serve(stop)
        elif config.transport.backend == "uart":
            server = UartServer(
                config.transport.uart_device,
                config.transport.uart_baud,
                engine,
            )
            print(
                f"Ready on UART {config.transport.uart_device} "
                f"at {config.transport.uart_baud} baud",
                flush=True,
            )
            server.serve(stop)
        else:
            if not start_tracking:
                raise ConfigError(
                    "transport.backend=none requires --start-tracking or no work can start"
                )
            print("Tracking; press Ctrl-C to stop", flush=True)
            while not stop.wait(0.5):
                if service.error_message:
                    raise RuntimeError(service.error_message)
    return 0


def _run_local_preview(config_path: Path, video_path: Path | None = None) -> int:
    from .camera import VideoFileCamera
    from .service import EyeTrackingService

    config, layout = _load(config_path)
    config = replace(
        config,
        recording=replace(config.recording, record_on_tracking=False),
        preview=replace(config.preview, enabled=True),
    )
    stop = threading.Event()

    def request_stop(signum=None, frame=None) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    camera = (
        None
        if video_path is None
        else VideoFileCamera(video_path, config.camera, layout, realtime=True)
    )
    with EyeTrackingService(config, layout, camera=camera) as service:
        service.start_tracking()
        source = "Live camera" if video_path is None else f"Looping video {video_path}"
        print(
            f"{source} diagnostic preview; press Q or Esc in the window to stop",
            flush=True,
        )
        while not stop.wait(0.1):
            if service.error_message:
                raise RuntimeError(service.error_message)
            if service.preview_error_message:
                raise RuntimeError(service.preview_error_message)
            if service.preview_closed:
                # Give a child-process error message time to traverse the queue
                # before treating a closed window as an intentional exit.
                stop.wait(0.05)
                if service.preview_error_message:
                    raise RuntimeError(service.preview_error_message)
                break
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "configure":
            from .roi_tool import configure_rois

            config = AppConfig.load(args.config)
            output = args.output
            if output is None:
                output = Path(config.roi_config).expanduser()
                if not output.is_absolute():
                    output = args.config.resolve().parent / output
            saved = configure_rois(
                config,
                output,
                config_path=args.config,
                image_path=args.image,
                video_path=args.video,
                average_frames=args.average_frames,
            )
            if saved is None:
                print("ROI configuration cancelled", file=sys.stderr)
                return 2
            if args.image is None and args.video is None:
                print(f"Saved {saved} and exposure in {args.config}")
            else:
                print(f"Saved {saved}")
            return 0
        if args.command == "validate":
            config, layout = _load(args.config)
            if args.hardware:
                from .camera import Picamera2Camera

                camera = Picamera2Camera(config.camera, config.recording, layout)
                camera.close()
            print(
                f"Configuration valid: {len(layout.rois)} eye ROI(s), {config.camera.fps:g} fps"
            )
            return 0
        if args.command == "benchmark":
            config, layout = _load(args.config)
            print(
                format_benchmark(
                    benchmark_video(args.video, config, layout, maximum_frames=args.frames)
                )
            )
            return 0
        if args.command == "preview":
            return _run_local_preview(args.config, args.video)
        if args.command == "run":
            return _run_service(
                args.config,
                args.start_tracking,
                args.preview_override,
            )
    except (ConfigError, RuntimeError, ValueError, OSError) as exc:
        print(f"eye-tracker: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
