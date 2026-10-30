from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean

import numpy as np

from .camera import VideoFileCamera
from .config import AppConfig, RoiLayout
from .tracker import MultiEyeTracker


def benchmark_video(
    video_path: str | Path,
    config: AppConfig,
    layout: RoiLayout,
    *,
    maximum_frames: int | None = None,
) -> dict[str, object]:
    """Measure the same oriented eye crops and detector used by live acquisition."""
    if maximum_frames is not None and maximum_frames < 1:
        raise ValueError("maximum_frames must be positive")
    # Reuse playback decoding, layout, crop geometry, and authoritative timestamps.
    camera = VideoFileCamera(video_path, config.camera, layout, realtime=False, loop=False)
    # Compare compute time with the configured live acquisition rate.
    fps = config.camera.fps
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
        camera.start()
        while maximum_frames is None or frames < maximum_frames:
            try:
                frame = camera.capture_analysis()
            except EOFError:
                break
            frames += 1
            start = time.perf_counter_ns()
            result = tracker.process(frame, dropped_frames=0)
            durations_ms.append((time.perf_counter_ns() - start) / 1e6)
            for eye in result.eyes:
                valid_counts[eye.eye_id] += int(eye.valid)
                blink_counts[eye.eye_id] += int(eye.blink)
    finally:
        camera.close()
        tracker.close()
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
    """Format benchmark statistics for the command-line report."""
    return json.dumps(result, indent=2, sort_keys=True)
