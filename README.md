# Macaque eye tracker

Low-latency pupil tracking and synchronized video recording for an Arducam
CamArray on Raspberry Pi 5. The runtime tracks one or two independently tuned
eye regions, records the stitched camera stream, and exposes results over a
Unix socket or direct Pi-to-Pi UART.

This is a pupil tracker, not an iris tracker or a calibrated gaze system.
Reported positions are crop-local pixels until a separate gaze calibration is
applied.

## Core behavior

- Camera, recorded video, decoded frames, and detector inputs are treated as
  grayscale. Some H.264 decoders expose a monochrome recording as three BGR
  channels; the loader collapses those channels to one luminance plane.
- Configuration and tracking previews use colored overlays on disposable BGR
  canvases. Those colors never enter tracking or recorded video.
- Pupil thresholds are selected automatically for every frame. The per-eye
  pupil-size slider only biases candidate selection toward smaller or larger
  fits; it does not set a fixed threshold.
- Prerecorded videos loop in both configuration stages and in the tracking
  preview.
- Recording uses software H.264/libx264 with the `ultrafast` preset. CRF is
  configurable and defaults to 24 when omitted.

## Install

Use a 64-bit Raspberry Pi OS image supported by the installed Arducam
Pivariety driver. Install the camera and vision packages from the OS so they
match the system camera stack:

```bash
sudo apt install python3-picamera2 python3-libcamera python3-opencv python3-av
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e .
```

Install `python3-gpiozero` too if `camera.ir_led_pin` is configured. On a
non-Pi development machine:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[vision,dev]'
pytest
```

## Configure eye regions

Interactive live-camera configuration:

```bash
eye-tracker configure --config config/eye_tracker.json
```

Use a looping recording for repeatable tuning:

```bash
eye-tracker configure --config config/eye_tracker.json --video session.mkv
```

Use `--image frame.png` for a still image, or add `--static` to average a
camera/video source instead of showing it live. Configuration has two stages:

1. Draw one or two eye boxes.
2. Tune each eye independently while inspecting the automatic pupil fit.

The exposure control is camera-wide. Gain, brightness, contrast, and pupil-size
bias are software settings saved per eye in `rois.json`.
Keep each crop tight enough to exclude unrelated dark structures, but large
enough to contain the full pupil at every required gaze direction.

## Preview and test recordings

Track a live camera without recording:

```bash
eye-tracker preview --config config/eye_tracker.json
```

Run exactly the same tracker on a looping recording:

```bash
eye-tracker preview --config config/eye_tracker.json --video session.mkv
```

Prerecorded input is paced at `camera.fps` from the selected configuration;
container frame-rate metadata is ignored. Playback skips stale frames before
image conversion and bounds catch-up work, so a slow decoder or configuration
window cannot accumulate an ever-growing backlog. If decoding cannot sustain
the configured rate, playback rebases its clock and runs at the achievable rate.

Both live and video previews serve the configured transport. By default this
is UART on `/dev/ttyAMA10` at 460800 baud. A downstream device can poll video
tracking results with the same commands and packet format as live tracking:

```bash
eye-tracker-controller monitor --rate 100
```

`STOP_TRACKING` leaves the preview open with live camera views, eye crops, and an `IDLE`
label. Pupil analysis and measurement output resume on `START_TRACKING`.
Closing the window in the standalone `preview` command exits that command.

For playback without a display, use:

```bash
eye-tracker run --config config/eye_tracker.json --video session.mkv --start-tracking
```

Omit `--start-tracking` to let the downstream controller start video tracking.
Video input does not start a new recording. Set `transport.backend` to `none`
for display-only development without a UART device, or `unix` to test the
same protocol through the configured local socket.

Run an offline timing benchmark:

```bash
eye-tracker benchmark session.mkv --config config/eye_tracker.json
eye-tracker benchmark session.mkv --config config/eye_tracker.json --frames 1000
```

The benchmark measures detector throughput, not camera-to-controller latency or
tracking accuracy.

## Run

Validate files, then optionally validate camera negotiation on the Pi:

```bash
eye-tracker validate --config config/eye_tracker.json
eye-tracker validate --config config/eye_tracker.json --hardware
```

Start the command service:

```bash
eye-tracker run --config config/eye_tracker.json
```

For a local run that starts immediately:

```bash
eye-tracker run --config config/eye_tracker.json --start-tracking --preview
```

`--preview` and `--no-preview` override `preview.enabled` for one run.
An enabled preview opens immediately, including when the service is waiting
for `START_TRACKING`. `STOP_TRACKING` stops pupil analysis and recording but
keeps live camera views and eye crops visible; `STOP_RECORDING` leaves tracking and the display
running. Idle preview frames are not processed by the pupil detector, saved,
or returned as eye-position measurements over UART. The camera and any
configured IR LED stay on to supply the live preview. Without an enabled
preview, stopping tracking also stops the camera as before.

Both `preview` and `run --preview` show an overview pane with all four camera
views and the configured eye boxes above the eye-detail panes. The boxes use
the same colors and labels as the configuration menu. The overview is a small
grayscale copy sampled from the same frame as the eye crops, with no extra
camera read; headless runs skip this copy. Overlays are drawn only in the
preview process and do not change detector inputs or recordings.

Closing the preview in `run` does not stop tracking or recording. Display rendering is
isolated behind a latest-frame queue, so a slow display drops preview frames
instead of blocking acquisition. The dashboard reports completed `tracking`
Hz separately from `display` Hz: a low display rate is harmless, but
`analysis drops` must remain at zero for a complete result stream.

With two configured eyes, their independent fits run concurrently on two
long-lived worker threads. OpenCV's nested worker pool is disabled for this
path, and the separate preview process runs at lower priority. CPU affinity is
left to the OS so the threaded software encoder can use otherwise-idle cores.

## Configuration

The example profiles are:

- `config/eye_tracker.json`: 5120x720, 60 Hz tracking with UART; selected views at 60 fps.
- `config/eye_tracker.uart.json`: reduced-resolution, 30 Hz UART profile.

Both profiles default to the Pi 5 dedicated UART connector, `/dev/ttyAMA10`.
Use `transport.backend: "unix"` for local Unix-socket development.

Important fields:

| Section | Fields |
|---|---|
| `camera` | sensor/analysis size, acquisition FPS, exposure, analogue gain, optional IR LED GPIO |
| `tracker` | pupil diameter bounds, minimum axis ratio/contrast/confidence, adaptive threshold percentiles, temporal limits |
| `recording` | layout, source size/FPS, directory, container, CRF, rate ceiling, keyframe interval, free-space guard |
| `transport` | `unix`, `uart`, or `none`, plus socket/device/baud |
| `preview` | enabled state, size limit, and threshold-mask display |
| `roi_config` | ROI file path, relative to the main config file when not absolute |

`recording.crf` accepts libx264 values 0–51; lower values preserve more detail
and produce larger files. The default is 24. The bitrate setting is a
conservative VBV ceiling, not the expected average bitrate.

Recording resolution and rate are independent of analysis resolution and
`camera.fps`. The recording rate must divide the camera rate exactly: at 60 Hz,
selected-view recording accepts 60 or 30 Hz. Libcamera's ISP sizes the recording
stream before encoding, and Picamera2's frame skipper discards unrecorded
camera requests before libx264. No Python image resizing is performed. When
recording and analysis sizes match, both paths share one camera stream instead
of producing duplicate full-resolution ISP outputs. Reduced recording FPS
requires Picamera2 0.3.19 or newer and is rejected at startup on older versions.

The example files are starting points, not validated experimental settings.
Revalidate after changing resolution, crop, exposure, illumination, frame
rate, or tracker parameters.

## Pupil fitting

For each eye crop the detector:

1. applies the saved per-eye grayscale adjustments;
2. lightly blurs sensor noise and evaluates several frame-adaptive dark-pixel
   thresholds;
3. builds one adaptive bright-reflection mask, including a small halo, and
   excludes affected boundary points before fitting and refining ellipses;
4. rejects candidates that are too small, large, elongated, irregular, weakly
   contrasted, or inconsistent with the recent pupil;
5. ranks the remaining pupil candidates using local contrast, shape, mask
   support, darkness, and threshold/size continuity.

Intact pupil boundaries use contour moments for the center and area-equivalent
contour diameter. When a glint intersects the boundary, the detector uses the
fitted ellipse center and area-equivalent ellipse diameter only if the remaining
arcs pass stricter checks: at least 60% of contour points remain, at least 11 of
16 angular sectors support the ellipse, and residuals and reconstructed area
stay within bounded limits. It does not inpaint pixels, merge disconnected
fragments, or run an iterative random search.

Reflections are excluded from appearance and fill statistics. At least 60% of
the fitted interior and 65% of the surrounding ring must remain visible.
Glint-affected fits also require strong dark-mask support and a uniform visible
interior to reject iris-sized fits containing a darker pupil. Masked area lowers
confidence; all configured size, axis-ratio, contrast, confidence, and temporal
limits still apply. Diagnostic eye records include `glint_fraction` (including
the masked halo) and `boundary_recovered`. Appearance scoring operates only on
the ellipse and its local ring to keep the added computation small.

A fit that is too elliptical is rejected via `tracker.min_axis_ratio`.
The pupil-size bias changes only candidate ranking:
negative favors smaller plausible pupils, positive favors larger ones, and zero
is neutral. Threshold selection remains automatic at every setting.

No image-only method can recover a boundary that is not visible. Prefer stable
IR illumination, avoid clipped highlights and eyelid-heavy crops, and tune with
representative video from all required gaze directions.

## Recording and outputs

With `recording.record_on_tracking: true`, tracking starts the configured
MKV/MP4/H.264 recording and a neighboring JSONL sidecar. The sidecar stores
timestamps, sequence/drop information, per-eye results, and encoder settings.
Stopping tracking flushes recording before it acknowledges completion.

There is one `frame` row per completed tracking result. For a lossless 60 Hz
analysis run, `frame_sequence` is consecutive and
`dropped_analysis_frames` remains zero. This is independent of how many frames
the optional preview displays.

The `ultrafast` preset and the configured CRF minimize encoder work. The
profiles currently choose different CRFs for their intended tests, while 24 is
the code default when a value is omitted. Actual size is scene-dependent;
perform a full-duration storage and thermal test with the final camera settings.
The shipped `config/eye_tracker.json` uses `recording.layout: "hands_eyes"`.
One synchronized video contains the full views from camera 0 and camera 1
(the first two views from the left in configuration) plus the configured eye
crops. Each hand-camera view retains 1280x720 pixels; each eye crop retains
its source pixels without resizing or software image adjustments. The target
is 60 fps, with 5120x720/60 Hz acquisition and analysis unchanged. CRF remains 20.

For this layout, `recording.width` and `recording.height` specify the stitched
**source** size before cropping. The encoded dimensions depend on the eye
boxes. Eye crops are placed beside or below the hand views, choosing the smaller
canvas and padding only as needed for YUV420. For example, two 256x216 eye
crops produce a 2816x720 video instead of 5120x720. The JSONL header records
the encoded size and every source/destination rectangle under `video_layout`.
`eye-tracker validate` reports the actual encoded size for the saved ROIs.

The camera callback copies selected luma pixels into an owned buffer. A
separate worker performs compression and file output, without holding camera
requests. Its queue is bounded to eight frames. Sustained overload is reported
as a recording error rather than silently discarding video frames. Sixty fps
is a target that still needs testing on the Pi; if necessary, set only
`recording.fps` to 30. Selected-view recording rejects rates below 30 fps.

Keep the video and its `.jsonl` sidecar together for `--video` playback and
offline benchmarks. Playback restores the saved views to their original
coordinates, with unsaved regions black. Eye fitting therefore uses the saved
ROI positions. The live preview still shows all four camera views.
Older configurations without a `layout` field retain full stitched recording;
set `recording.layout: "full"` explicitly to select that behavior.

Pi 5 H.264 encoding uses the CPU. Picamera2 calls the encoder while processing
camera requests, before notifying waiting capture jobs, so the original full-frame
recording path can delay analysis frames. The selected-view worker removes
compression and output from that callback, though encoding still consumes CPU.
See the upstream
[camera request loop](https://github.com/raspberrypi/picamera2/blob/main/picamera2/picamera2.py)
and [libx264 encoder](https://github.com/raspberrypi/picamera2/blob/main/picamera2/encoders/libav_h264_encoder.py).
`preview` disables automatic recording, which can make it substantially faster
than an overloaded `run`. Compare these commands sequentially with the same
configuration and camera scene (Ctrl-C between runs):

```bash
eye-tracker run --start-tracking --preview --no-recording
eye-tracker run --start-tracking --preview
```

`--no-recording` overrides `recording.record_on_tracking` for that invocation;
explicit controller `START_RECORDING` commands can still start a recording.
Compare the dashboard's **tracking** Hz, not display Hz. Check elapsed sensor
timestamps in the JSONL sidecar as well: zero `dropped_analysis_frames` only
means no drops in the application's analysis queue, and does not detect camera
acquisition slowed by recording. Confirm recorded frame rate from the video's
timestamps. The selected-view default still needs validation on the recording Pi.

The legacy zero-copy-style recorder remains in `video/` for isolated camera
tests:

```bash
python -m video.recorder session.h264 --duration 10
python -m video.recorder session.h264 --duration 10 --dry-run
```

Its exposure and CRF are configured in `video/example_config.json`. See
[video/README.md](video/README.md) for camera setup and storage details.

## Controller link

Production uses fixed 64-byte request/response records over 3.3 V UART at
460800 baud. The tracker and controller default to `/dev/ttyAMA10` on the Pi 5
dedicated three-pin UART connector. Do not connect either Pi's 5 V rail.
Typical commands are:

```bash
eye-tracker-controller start-tracking
eye-tracker-controller monitor --rate 100
eye-tracker-controller stop-tracking
```

UART results are request/response rather than unsolicited. Polling at 100 Hz is
designed to observe each new 60 Hz result; at the default 460800 baud this rate
is within the checked wire-rate limit. UART v1 returns the latest result rather
than buffered history, so consumers must detect sequence gaps. The JSONL
sidecar is the authoritative complete local stream. Status records expose
cumulative analysis drops and per-frame processing time.

For local development with `transport.backend` set to `unix`:

```bash
eye-tracker-controller --unix /tmp/macaque-eye-tracker.sock status
```

The complete pinout, packet layout, retry rules, and validity flags are in
[docs/protocol.md](docs/protocol.md). Pi service and cooling guidance is in
[docs/deployment.md](docs/deployment.md).

## Experimental validation

Before collecting data:

1. Hand-label representative frames across gaze direction, lighting, pupil
   size, glints, eyelid occlusion, and blinks.
2. Report center/diameter error, valid fraction, lost-track rate, and blink
   precision/recall for each eye.
3. Measure capture-to-result latency, frame loss, controller timing, CPU
   temperature, throttling, storage use, and clean recording finalization.
4. Run an hour-scale burn-in, then a full-duration recording test.
5. Calibrate crop-local pupil coordinates to display coordinates on held-out
   fixation trials; never treat raw x/y as gaze.

Retain raw x/y/diameter and validity flags even after gaze calibration. Invalid
samples retain the last reliable x/y for diagnostics but report zero diameter;
downstream code must ignore their coordinates.

## Repository layout

```text
src/macaque_tracker/  runtime, detector, preview, transport, and controller
config/               example runtime profiles
docs/                 deployment and wire-protocol details
deploy/               example systemd unit
tests/                unit and synthetic-image regression tests
video/                legacy standalone recorder and preview
```

The codebase deliberately keeps capture, pupil detection, display, recording,
and transport separate. New features should preserve grayscale data paths and
avoid adding alternate detector pipelines unless they replace an existing one.
