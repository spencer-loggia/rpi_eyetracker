from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean

import numpy as np

from .config import AppConfig, RoiLayout
from .models import AnalysisFrame
from .tracker import MultiEyeTracker


def benchmark_video(
    video_path: str | Path,
    config: AppConfig,
    layout: RoiLayout,
    *,
    maximum_frames: int | None = None,
) -> dict[str, object]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Offline benchmark requires OpenCV") from exc
    capture = cv2.VideoCapture(str(Path(video_path).expanduser()))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = config.camera.fps
    width, height = config.camera.analysis_width, config.camera.analysis_height
    pixel_rois = tuple((roi.eye_id, roi.to_pixels(width, height)) for roi in layout.rois)
    tracker = MultiEyeTracker(
        tuple(roi.eye_id for roi in layout.rois),
        config.tracker,
        {roi.eye_id: roi.settings for roi in layout.rois},
    )
    durations_ms: list[float] = []
    valid_counts = {roi.eye_id: 0 for roi in layout.rois}
    blink_counts = {roi.eye_id: 0 for roi in layout.rois}
    frames = 0
    wall_start = time.perf_counter()
    try:
        while maximum_frames is None or frames < maximum_frames:
            ok, image = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if gray.shape != (height, width):
                gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
            crops = tuple((eye_id, roi.extract(gray)) for eye_id, roi in pixel_rois)
            frames += 1
            sensor_ns = int(frames * 1e9 / fps)
            start = time.perf_counter_ns()
            result = tracker.process(AnalysisFrame(frames, sensor_ns, crops), dropped_frames=0)
            durations_ms.append((time.perf_counter_ns() - start) / 1e6)
            for eye in result.eyes:
                valid_counts[eye.eye_id] += int(eye.valid)
                blink_counts[eye.eye_id] += int(eye.blink)
    finally:
        capture.release()
    elapsed = time.perf_counter() - wall_start
    if not durations_ms:
        raise RuntimeError("Video contained no decodable frames")
    values = np.asarray(durations_ms)
    return {
        "frames": frames,
        "video_fps": fps,
        "wall_fps": frames / elapsed if elapsed else 0.0,
        "processing_ms": {
            "mean": mean(durations_ms),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "max": float(np.max(values)),
        },
        "valid_fraction": {
            str(eye_id): count / frames for eye_id, count in valid_counts.items()
        },
        "blink_fraction": {
            str(eye_id): count / frames for eye_id, count in blink_counts.items()
        },
        "realtime_at_video_rate": float(np.percentile(values, 99)) < 1000.0 / fps,
    }


def format_benchmark(result: dict[str, object]) -> str:
    return json.dumps(result, indent=2, sort_keys=True)
