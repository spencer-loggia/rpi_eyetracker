from __future__ import annotations

import queue
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import cv2
except ImportError:
    cv2 = None

import macaque_tracker.preview as preview_module
from macaque_tracker.config import PreviewConfig
from macaque_tracker.models import AnalysisFrame, EyeMeasurement, FrameResult, NormalizedRoi
from macaque_tracker.preview import (
    LivePreview,
    _preview_process,
    _tracking_rate_sample,
    render_preview,
)


def _result(sequence: int, produced_ns: int, dropped: int) -> FrameResult:
    return FrameResult(
        frame_sequence=sequence,
        sensor_timestamp_ns=produced_ns,
        produced_timestamp_ns=produced_ns,
        eyes=(EyeMeasurement(0, 1.0, 1.0, 1.0, 1.0, True, False),),
        dropped_analysis_frames=dropped,
    )


def test_tracking_rate_is_independent_of_preview_and_analysis_drops() -> None:
    previous = _result(100, 1_000_000_000, 3)
    current = _result(112, 1_200_000_000, 3)
    with_analysis_drops = _result(112, 1_200_000_000, 5)

    assert _tracking_rate_sample(previous, current) == pytest.approx(60.0)
    assert _tracking_rate_sample(previous, with_analysis_drops) == pytest.approx(50.0)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_render_preview_draws_fit_dashboard_and_respects_size_limit() -> None:
    crop = np.full((100, 160), 120, dtype=np.uint8)
    second_crop = np.full((80, 120), 135, dtype=np.uint8)
    original_crop = crop.copy()
    original_second_crop = second_crop.copy()
    frame = AnalysisFrame(5, 100, ((0, crop), (1, second_crop)))
    result = FrameResult(
        frame_sequence=5,
        sensor_timestamp_ns=100,
        produced_timestamp_ns=200,
        eyes=(
            EyeMeasurement(0, 81.0, 52.0, 40.0, 0.85, True, False),
            EyeMeasurement(1, 55.0, 38.0, 0.0, 0.0, False, True),
        ),
        dropped_analysis_frames=3,
        processing_time_us=1400,
        diagnostics={
            "eyes": {
                "0": {
                    "detected": True,
                    "missing_frames": 0,
                    "candidate_x": 80.0,
                    "candidate_y": 51.0,
                    "ellipse_x": 80.0,
                    "ellipse_y": 51.0,
                    "ellipse_width": 45.0,
                    "ellipse_height": 35.0,
                    "ellipse_angle_degrees": 12.0,
                    "equivalent_diameter": 40.0,
                    "confidence": 0.85,
                    "contrast": 42.0,
                    "threshold": 55,
                },
                "1": {"detected": False, "missing_frames": 2},
            }
        },
    )
    rendered = render_preview(
        frame,
        result,
        PreviewConfig(max_display_width=300, max_display_height=200),
        display_fps=29.8,
        now_ns=1_000,
    )

    assert rendered.dtype == np.uint8
    assert rendered.ndim == 3
    assert rendered.shape[2] == 3
    assert rendered.shape[1] <= 300
    assert rendered.shape[0] <= 200
    assert np.any(rendered[:, :, 0] != rendered[:, :, 1])
    assert np.array_equal(crop, original_crop)
    assert np.array_equal(second_crop, original_second_crop)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_render_preview_rejects_mismatched_eye_ids() -> None:
    frame = AnalysisFrame(1, 1, ((0, np.zeros((30, 30), dtype=np.uint8)),))
    result = FrameResult(
        frame_sequence=1,
        sensor_timestamp_ns=1,
        produced_timestamp_ns=1,
        eyes=(EyeMeasurement(1, 0.0, 0.0, 0.0, 0.0, False, False),),
    )
    with pytest.raises(ValueError, match="do not match"):
        render_preview(frame, result, PreviewConfig())


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_idle_preview_renders_live_crops_without_measurements_or_masks(monkeypatch) -> None:
    crop = np.full((100, 160), 120, dtype=np.uint8)
    original = crop.copy()
    frame = AnalysisFrame(10, 100, ((0, crop),))
    labels = []
    draw_text = preview_module._put_text

    def record_text(image, text, *args, **kwargs):
        labels.append(text)
        return draw_text(image, text, *args, **kwargs)

    def unexpected_mask(*_args):
        raise AssertionError("Idle preview must not reconstruct a pupil mask")

    monkeypatch.setattr(preview_module, "_put_text", record_text)
    monkeypatch.setattr(preview_module, "pupil_mask_for_threshold", unexpected_mask)
    first = render_preview(frame, None, PreviewConfig())
    assert any("TRACKING IDLE" in label for label in labels)
    assert "EYE 0  IDLE" in labels
    assert not any(label.startswith("x ") for label in labels)
    assert np.array_equal(crop, original)
    next_frame = AnalysisFrame(11, 200, ((0, crop + 30),))
    second = render_preview(next_frame, None, PreviewConfig())
    assert not np.array_equal(first, second)


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("tracking", [False, True])
@pytest.mark.parametrize("eye_count", [1, 2])
def test_overview_shows_four_views_and_configured_boxes(monkeypatch, tracking, eye_count):
    overview = np.repeat(np.array([[40, 80, 120, 160]], dtype=np.uint8), 320, axis=1)
    overview = np.repeat(overview, 180, axis=0)
    original = overview.copy()
    crops = tuple((i, np.full((100, 160), 11, dtype=np.uint8)) for i in range(eye_count))
    frame = AnalysisFrame(1, 1, crops, overview=overview)
    rois = tuple(
        NormalizedRoi(i, f"eye {i}", 0.13 + i * 0.5, 0.4, 0.07, 0.3)
        for i in range(eye_count)
    )
    result = (
        FrameResult(1, 1, 1, tuple(
            EyeMeasurement(i, 80, 50, 20, 0.9, True, False) for i in range(eye_count)
        )) if tracking else None
    )
    drawn = []
    draw_box = preview_module.draw_roi_box

    def record_box(canvas, roi, eye_id, **kwargs):
        draw_box(canvas, roi, eye_id, **kwargs)
        drawn.append((canvas, roi, eye_id))

    monkeypatch.setattr(preview_module, "draw_roi_box", record_box)
    config = PreviewConfig(max_display_width=2000, max_display_height=2000)
    rendered = render_preview(frame, result, config, rois=rois)
    details = render_preview(AnalysisFrame(1, 1, crops), result, config)
    assert rendered.shape[0] > details.shape[0]
    assert rendered.shape[1] == details.shape[1]
    assert len(drawn) == eye_count
    for canvas, pixel_roi, eye_id in drawn:
        height, width = canvas.shape[:2]
        assert pixel_roi == rois[eye_id].to_pixels(width, height)
        color = ((80, 220, 80), (220, 90, 220))[eye_id]
        assert tuple(canvas[pixel_roi.y + pixel_roi.height // 2, pixel_roi.x]) == color
        assert np.any(np.all(rendered == color, axis=2))
        # All four camera views remain visible, including those without an eye.
        for index, shade in enumerate((40, 80, 120, 160)):
            assert np.all(canvas[height - 5, round((index + 0.5) * width / 4)] == shade)
            assert np.any(np.all(rendered == shade, axis=2))
    assert np.array_equal(overview, original)
    assert all(np.all(crop == 11) for _eye_id, crop in crops)
    limited = render_preview(
        frame, result, PreviewConfig(max_display_width=640, max_display_height=360),
        rois=rois,
    )
    assert limited.shape[1] <= 640 and limited.shape[0] <= 360


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
@pytest.mark.parametrize("output_full", [False, True])
def test_embedded_renderer_never_opens_window_or_waits_for_consumer(monkeypatch, output_full):
    class RenderedQueue(queue.Queue):
        cancelled = False

        def cancel_join_thread(self):
            self.cancelled = True

    def unexpected_gui_call(*_args, **_kwargs):
        raise AssertionError("Embedded rendering must not access OpenCV windows")

    for name in ("namedWindow", "imshow", "waitKey", "getWindowProperty", "destroyWindow"):
        monkeypatch.setattr(cv2, name, unexpected_gui_call)
    monkeypatch.setattr(cv2, "setNumThreads", lambda _count: None)
    monkeypatch.setattr(preview_module.os, "nice", lambda _value: None)
    packets = queue.Queue()
    for sequence in (1, 2):
        packets.put((AnalysisFrame(sequence, 1, ((0, np.full((40, 60), 80, np.uint8)),)), None))
    errors, rendered = queue.Queue(1), RenderedQueue(1)
    sentinel = object()
    if output_full:
        rendered.put(sentinel)
    stop, closed = threading.Event(), threading.Event()
    render = render_preview
    rendered_sequences = []

    def render_and_stop(frame, result, config, **kwargs):
        rendered_sequences.append(frame.frame_sequence)
        dashboard = render(frame, result, config, **kwargs)
        stop.set()
        return dashboard

    monkeypatch.setattr(preview_module, "render_preview", render_and_stop)
    _preview_process(
        packets, errors, stop, closed,
        PreviewConfig(max_display_width=320, max_display_height=240), {}, (), rendered,
    )
    assert closed.is_set()
    assert rendered.cancelled
    assert errors.empty()
    assert rendered_sequences == [2]
    assert rendered.qsize() == 1
    output = rendered.get_nowait()
    if output_full:
        assert output is sentinel
    else:
        assert output.dtype == np.uint8
        assert output.shape[2] == 3
        assert output.shape[0] <= 240 and output.shape[1] <= 320


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_ordinary_renderer_keeps_window_display_and_close_behavior(monkeypatch):
    calls = []
    monkeypatch.setattr(cv2, "setNumThreads", lambda _count: None)
    monkeypatch.setattr(preview_module.os, "nice", lambda _value: None)
    monkeypatch.setattr(cv2, "namedWindow", lambda name, _flags: calls.append(("open", name)))
    monkeypatch.setattr(cv2, "imshow", lambda name, image: calls.append(("show", name, image)))
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: -1)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda _name, _prop: 0.0)
    monkeypatch.setattr(cv2, "destroyWindow", lambda name: calls.append(("close", name)))
    frame = AnalysisFrame(1, 1, ((0, np.full((40, 60), 80, np.uint8)),))
    packets, errors = queue.Queue(1), queue.Queue(1)
    packets.put((frame, None))
    stop, closed = threading.Event(), threading.Event()
    config = PreviewConfig(max_display_width=320, max_display_height=240)
    _preview_process(packets, errors, stop, closed, config, {}, ())
    assert errors.empty()
    assert closed.is_set()
    assert [call[0] for call in calls] == ["open", "show", "close"]
    assert all(call[1] == config.window_name for call in calls)
    assert np.array_equal(calls[1][2], render_preview(frame, None, config))


@pytest.mark.parametrize("embedded", [False, True])
def test_publication_throttle_only_applies_to_embedded_preview(monkeypatch, embedded):
    preview = LivePreview(PreviewConfig(), embedded=embedded)
    preview._process = SimpleNamespace(is_alive=lambda: True)
    preview._packets = queue.Queue(1)
    now = [1_000_000_000]
    monkeypatch.setattr(preview_module.time, "monotonic_ns", lambda: now[0])
    accepted = []
    for sequence, elapsed in enumerate((0, 10_000_000, 70_000_000), start=1):
        now[0] = 1_000_000_000 + elapsed
        frame = AnalysisFrame(sequence, 1, ((0, np.zeros((40, 60), np.uint8)),))
        preview.publish(frame, None)
        try:
            accepted.append(preview._packets.get_nowait()[0].frame_sequence)
        except queue.Empty:
            pass
    assert accepted == ([1, 3] if embedded else [1, 2, 3])


@pytest.mark.parametrize("embedded", [False, True])
def test_preview_publication_replaces_pending_frame_without_growing_queue(monkeypatch, embedded):
    preview = LivePreview(PreviewConfig(), embedded=embedded)
    preview._process = SimpleNamespace(is_alive=lambda: True)
    preview._packets = queue.Queue(1)
    now = [1_000_000_000]
    monkeypatch.setattr(preview_module.time, "monotonic_ns", lambda: now[0])
    for sequence in range(1, 5):
        frame = AnalysisFrame(sequence, 1, ((0, np.zeros((40, 60), np.uint8)),))
        now[0] += 100_000_000
        preview.publish(frame, None)
        assert preview._packets.qsize() == 1
    assert preview._packets.get_nowait()[0].frame_sequence == 4


def test_embedded_preview_stop_bounds_unresponsive_process_and_releases_queues():
    class ClosingQueue(queue.Queue):
        def __init__(self):
            super().__init__(maxsize=1)
            self.cancelled = False
            self.closed = False

        def cancel_join_thread(self):
            self.cancelled = True

        def close(self):
            self.closed = True

    class UnresponsiveProcess:
        exitcode = None

        def __init__(self):
            self.joins = []
            self.terminated = False

        def join(self, timeout):
            self.joins.append(timeout)

        def is_alive(self):
            return not self.terminated

        def terminate(self):
            self.terminated = True
            self.exitcode = -15

    preview = LivePreview(PreviewConfig(), embedded=True)
    process = UnresponsiveProcess()
    preview._process = process
    preview._stop_event = stop = threading.Event()
    preview._closed_event = threading.Event()
    preview._packets, preview._errors, preview._rendered = queues = [ClosingQueue() for _ in range(3)]
    preview.stop()
    assert stop.is_set()
    assert process.terminated
    assert process.joins == [preview._STOP_TIMEOUT_SECONDS] * 2
    assert all(item.cancelled and item.closed for item in queues)
    assert preview._process is None
    assert preview._packets is None and preview._errors is None and preview._rendered is None
    assert preview.read_rendered() is None
    assert preview.error_message is None
    preview.stop()  # Shutdown is idempotent.


@pytest.mark.skipif(cv2 is None, reason="OpenCV is not installed")
def test_embedded_preview_renders_in_spawned_process_and_stops():
    # An importable subprocess entry point also works when the parent test suite
    # was launched through stdin, which multiprocessing.spawn cannot re-import.
    script = textwrap.dedent("""
        import sys
        import time
        import numpy as np
        sys.path.insert(0, sys.argv[1])
        from macaque_tracker.config import PreviewConfig
        from macaque_tracker.models import AnalysisFrame
        from macaque_tracker.preview import LivePreview, render_preview

        if __name__ == '__main__':
            config = PreviewConfig(max_display_width=320, max_display_height=240)
            preview = LivePreview(config, embedded=True)
            frame = AnalysisFrame(1, 1, ((0, np.full((40, 60), 80, np.uint8)),))
            preview.start()
            process = preview._process
            try:
                preview.publish(frame, None)
                deadline = time.monotonic() + 5
                output = None
                while output is None and time.monotonic() < deadline:
                    assert preview.error_message is None, preview.error_message
                    output = preview.read_rendered()
                    time.sleep(0.01)
                assert output is not None, 'renderer produced no output'
                assert np.array_equal(output, render_preview(frame, None, config))
                # Leave the next dashboard unconsumed when stopping. A feeder
                # holding image data must not indefinitely delay process shutdown.
                preview.publish(frame, None)
                time.sleep(0.15)
            finally:
                started = time.monotonic()
                preview.stop()
                assert time.monotonic() - started < 5
            assert not process.is_alive()
            assert process.exitcode == 0, 'renderer needed forced termination'
            assert preview.read_rendered() is None
            assert preview.error_message is None
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve().parents[1] / "src")],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_overview_rotates_pixels_and_roi_overlays_together(monkeypatch, angle):
    from macaque_tracker.orientation import CameraCanvas

    raw = np.arange(400 * 60, dtype=np.uint8).reshape(60, 400)
    roi = NormalizedRoi(0, "eye", 0.55, 0.2, 0.1, 0.4)
    rotations = (0, 0, angle, 0)
    captured = []
    monkeypatch.setattr(preview_module, "draw_roi_box", lambda c, r, i, **kw: captured.append((c, r)))
    preview_module._overview_panel(raw, (roi,), 600, rotations)
    geometry = CameraCanvas(400, 60, rotations)
    canvas, displayed = captured[0]
    expected = cv2.resize(geometry.compose(raw), (canvas.shape[1], canvas.shape[0]),
                          interpolation=cv2.INTER_AREA)
    assert np.array_equal(canvas[:, :, 0], expected)
    box = geometry.display_roi(roi.to_pixels(400, 60))
    expected_roi = NormalizedRoi.from_pixels(
        eye_id=0, label="eye", roi=box, frame_width=geometry.width, frame_height=geometry.height,
    ).to_pixels(canvas.shape[1], canvas.shape[0])
    assert displayed == expected_roi
