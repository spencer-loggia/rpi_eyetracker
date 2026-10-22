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
sudo apt install python3-picamera2 python3-libcamera python3-opencv python3-av python3-tk
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

## Pi tracker experiment interface

From the repository, in a graphical desktop session:

```bash
python pi_tracker.py config/eye_tracker.json
```

After installation, `pi-tracker config/eye_tracker.json` is equivalent. Tk is
provided by the OS package `python3-tk`; no additional Python GUI dependency
is required. The opening screen shows all four live cameras. Choose **Start
experiment**, select a subject (or add one), then **Open experiment**.
Subject names use 1–32 ASCII letters, digits, underscores, periods or hyphens,
starting with a letter or digit.

Each experiment has its own directory:

```text
data/<subject>/exp_<subject>_<YYYYMMDD>_<experiment_number>_<HHMMSS>/
    experiment.json       # subject, number, UTC start/close times, calibration source
    state.json            # launcher Git commit, host, OS and Python version
    config.json           # this experiment's settings
    calibration.json      # eye boxes, camera rotations and per-eye tuning
    blocks/
        1/
            block.json
            config.json       # effective settings for this saved run
            calibration.json  # snapshot of the calibration actually loaded
            <recording>.mkv    # or configured MP4/H.264 container
            <recording>.jsonl  # synchronized eye traces and video layout
            <recording>.frames.csv  # saved video frame -> exact camera_time_ns
        2/
            ...
```

Dates and the additional `HHMMSS` time are **UTC**, matching the UART experiment-start
datetime. Frame telemetry uses the separate sensor clock described below.

`state.json` records the full source-checkout commit SHA as `git_commit`, plus
`hostname`, `os` (system, release and machine), and `python_version`. This state is
captured once when Pi Tracker launches and saved for every new experiment, whether
opened locally or over UART. Calibration edits and later checkout changes do not
rewrite it; restart the launcher after updating code. The commit identifies Git
HEAD and does not describe uncommitted edits. If Git or the source checkout is
unavailable, `git_commit` is explicitly `null`. State lookup adds no per-frame work,
and `state.json` is transferred with the rest of the experiment during sync.

Local experiment numbers start at 1 for each subject/day, then use one more than
the highest used number, including experiments numbered over UART or already synced. Existing
experiment numbers on the same subject/day are rejected, even if the start time
differs. Old timestamp-only directories remain available as calibration sources.

`experiment_history.tsv` in the local data root (default `data/`, configurable with
`--data`) is a permanent local table with one row per experiment. It records the directory name, subject,
experiment number, UTC date/start/end times, wall-clock runtime in seconds, block
count, status, calibration file/source, and launch Git commit. Paths identify the
original calibration artifacts; the complete experiment and its block calibration
snapshots are retained on the remote storage after sync. End time and runtime are
blank while an experiment is open. Runtime covers the whole experiment, including
calibration and idle time; block count and calibration details are refreshed
when it closes and again before sync cleanup. Existing local experiments are added
on launcher startup. The table also reserves past experiment numbers independently
of the hidden calibration cache.

This TSV is **never synchronized or pruned**. Updates preserve all earlier rows,
use atomic replacement, and flush to local storage before any uploaded experiment
is removed. A history-write failure prevents sync cleanup. The small writer lock
file `.experiment_history.lock` also stays local. No history work runs per frame.

Each continuous saved run gets a block. Opening the idle tracking view creates
no block; starting tracking creates the next numbered block and logs eye traces
even when automatic video is disabled. Stopping tracking ends the block.
`STOP_RECORDING` closes the saved run while tracking/UART continue;
`START_RECORDING` starts another block. A disk-space video cutoff leaves the
current block's eye log running; explicitly restarting recording starts a new
block. Block numbers use the next available integer above the existing maximum.

`block.json` records `started_timestamp_ns` and `ended_timestamp_ns` as system
Unix UTC time, plus readable `started_utc` and `ended_utc` strings in ISO 8601
to the second (for example, `2026-10-20T14:30:00+00:00`). End fields are `null`
while the block is active and are written after its video and eye-trace writers
close. A failed recording start/close also ends the block, with `status: "error"`
and its error message. These lifecycle times are not frame acquisition times.

UART starts can supply `--experiment-number` and `--block-number` through the
controller CLI. An explicit block number **replaces that block's entire directory**
after its previous writer has closed, including when that block is already active.
An unnumbered start while tracking/video is already active remains a no-op.
Retry the identical command sequence/bytes to avoid creating or replacing a block
twice. See [the UART protocol](docs/protocol.md) for byte offsets and examples.

Launcher output directories (`data`, subject, experiment and block folders) use mode
`0777`; calibration, settings, manifests, videos and measurement sidecars use
`0666`. All local users can read and modify these files, including files created
by a root launcher. Directory write access also allows deleting or replacing
outputs. Files retain their original owner, and data files are not executable.
Permissions are set at creation or replacement, with no per-frame work and no
change to the process-wide umask. Standalone `eye-tracker` keeps its usual
permissions unless `recording.shared_permissions` is enabled; launcher
experiment configs enable it automatically.

For recordings created before this change, run once from the project directory:

```bash
sudo chmod -R a+rwX data
```

If using `--data`, substitute that directory. Parent directories above the data
root must also allow other users to traverse them; a private home directory
can still block access. The launcher does not change permissions outside its
configured data root or recursively change earlier experiments.

Use `--data /path/to/data` to change the data root (default: `./data`). All
tracking recordings and generated artifacts stay inside the active experiment.
The newest earlier calibration for the same subject is copied into the new
experiment, together with its calibrated exposure/image controls and tracker
settings. Capture sizes/rates, pose-camera choices, encoder and transport settings
come from the supplied base config. Launcher experiments use the selected-view
layout: full pose-camera views plus cropped eyes, even with a legacy base
config that uses full stitched recording. For a subject without an earlier
calibration, the base config's ROI file is used if present; otherwise choose
**Calibrate** before tracking. Earlier experiments and the base config are
never overwritten by calibration.

The experiment screen offers **Calibrate** and **Start tracking**. Calibration
embeds the same box-selection and pupil-tuning code as `eye-tracker configure`.
During box selection, **Cam 0–3: …° ↻** rotates each view clockwise in 90°
steps. The standalone `eye-tracker configure` editor uses keys **1–4** for
cameras 0–3. Camera indices always mean the original physical views, counted
from left to right. Draw boxes on the oriented view; existing boxes stay attached
to the same sensor pixels when rotated. A box must stay within one camera when
rotation is enabled. Back from tuning lets you adjust orientation again.

Saved `camera_rotations` in `calibration.json` apply to menu feeds, configuration,
tracking previews, video recordings, playback, and offline benchmarks. Old
calibrations default to `[0, 0, 0, 0]`. Rotation is lossless, with no interpolation
or change to capture/recording frame rates. Rotated full pose views and eye crops
retain every source pixel; 90°/270° swaps their width and height. Only the small
eye crops are rotated in the tracking path, as part of their existing owned copy.

Coordinates remain **pixels with origin at the rotated eye crop's top-left**:
+x is right and +y is down in the oriented preview and eye video. Rotation
therefore changes the physical direction that increases/decreases each axis;
it does not center coordinates or convert them to gaze angles. UART and JSONL
use the same rotated analysis coordinates. Recorded eye video may use a different
pixel resolution; its sidecar gives both crop sizes and each camera's angle.
Any downstream gaze calibration must use the same orientation.

**Save calibration** writes the current experiment's files. **Back** from pupil
tuning returns to box selection; **Cancel calibration** discards unsaved changes.
Subject and experiment remain visible throughout.

**Start tracking** opens the tracking view with tracking and saving initially
**OFF**, using the same service as `eye-tracker run --preview`. Its **Start
tracking / saving** button or UART `START_TRACKING` starts measurement and the
configured automatic recording. `STOP_TRACKING` leaves the live preview open.
**Back** stops and flushes tracking before returning to the experiment menu;
**End experiment** returns to the opening screen. Back from the experiment
menu closes that experiment and returns to subject selection.

Only one component owns the camera and one server owns UART. Menus and
calibration release the camera before tracking starts. The embedded dashboard
renders in a separate, lower-priority process at up to 15 fps with bounded
queues; this display limit does not lower the camera, tracking, or recording
rates. When tracking is off, the live camera stays open but detection and
measurement saving stop. For desktop testing, add `--video recording.mkv` and
use a config with transport `none` or `unix`; video playback does not record.

UART `START_EXPERIMENT` carries a subject and UTC datetime, sets the Pi system
clock, and opens the requested experiment. `STOP_EXPERIMENT` flushes recording
and returns home from any screen. Clock setting needs Linux `CAP_SYS_TIME` in
the launcher's service context; without that permission a remote start returns
an error and leaves the current experiment unchanged. Manual starts do not
need clock-setting permission. See [the UART protocol](docs/protocol.md) for
wire layout, controller examples, status flags, retries and clock behavior.
To shut down the Pi from the downstream computer, send UART `SHUTDOWN` (code 8)
or run `eye-tracker-controller --device /dev/ttyAMA10 shutdown`. Pi Tracker
finalizes recordings and the experiment, acknowledges the command, synchronizes
experiment data when configured, then requests clean system power-off. Run the launcher as root for this operation;
clock-setting permission alone is insufficient. Cleanup failures prevent
power-off and appear in the interface. See the [shutdown protocol](docs/protocol.md#clean-system-shutdown)
for acknowledgment and retry behavior.

Do not run another `eye-tracker` process on the same camera/UART alongside the
launcher.

### Experiment synchronization

The launcher's base configuration accepts:

```json
"sync": {
  "remote_data_url": "/mnt/data/",
  "remote_git_url": "/mnt/data/pitracker_mirror"
}
```

`remote_data_url` is an **absolute local path to a mounted NFS share**, not an
HTTP/SSH URL. Missing/empty `remote_data_url` disables synchronization.
`remote_git_url` is retained in saved configurations but unused: this feature
transfers experiment data only and does not reset, pull, or modify source code.

Ending an experiment (GUI or UART), replacing an active experiment, closing
the launcher, or sending `SHUTDOWN` attempts to transfer **all pending local
experiments** to:

```text
<remote_data_url>/pitracker/<subject>/exp_<subject>_<YYYYMMDD>_<number>_<HHMMSS>/
```

The complete experiment directory is copied, including configuration,
calibration, every block, videos and eye traces. A newly opened experiment is
excluded when replacing an older one. Recording writers close before copying;
new tracking/recording is blocked during sync, keeping transfer work out of the
tracking loop. A progress dialog offers **Cancel Sync**.

The implementation follows `neuro_tasks`' NFS checks and rsync error handling:
the remote root must be on NFS/NFS4 and nonempty, filesystem probes have five-second
deadlines, rsync uses a 30-second I/O timeout, and 60 seconds without progress
aborts the attempt. Active transfers have no total time limit. Install `rsync`
and `findmnt` (`util-linux`) on the Pi. The remote share must be writable by the
launcher, including when NFS root squashing maps root to an unprivileged user.
See [deployment notes](docs/deployment.md#experiment-transfer-storage).

Only after rsync succeeds for the entire planned transfer are those local
experiment folders removed. Failed/cancelled transfers retain all local
experiments for retry; partial remote copies can be reused by rsync. Local
removal failures generate a warning. Empty subject folders remain available
in the menu. `data/.history/<subject>/` retains a small numbering ledger and the
latest calibration/configuration so new experiments can inherit calibration
and continue numbering without contacting the server. It contains no videos
or eye traces. The permanent `data/experiment_history.tsv` keeps readable experiment
summaries after removal and also preserves numbering. Diagnostics are appended to
`data/interface_sync.log` (under `--data` when overridden); these logs and the history
cache stay local. Experiments already removed before the TSV was introduced still
reserve their numbers through the old cache, but their full summaries cannot be
recovered from that cache alone.

A failed shutdown sync offers **Retry**, **Shut down anyway**, or **Cancel
shutdown**. Cancelling an in-progress shutdown sync also cancels power-off.
The explicit bypass leaves unsynced data on the Pi. A normal experiment-stop
failure returns to the interface with a warning; the next experiment stop or
shutdown retries pending data. `STOP_EXPERIMENT` with no active experiment can
also trigger a retry.

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
Each cropped eye image is automatically normalized for broad spatial lighting
differences and overall intensity before its software controls. Start
with gain **1**, brightness **0**, and contrast **1**, then tune as needed; existing
calibrations with large brightness/gain boosts may need a one-time retune.
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

- `config/eye_tracker.json`: 5120x720, 60 Hz tracking with UART; selected views from a 3840x540 source at 30 fps.
- `config/eye_tracker.uart.json`: reduced-resolution, 30 Hz UART profile.

Both profiles default to the Pi 5 dedicated UART connector, `/dev/ttyAMA10`.
Use `transport.backend: "unix"` for local Unix-socket development.

Important fields:

| Section | Fields |
|---|---|
| `camera` | sensor/analysis size, acquisition FPS, exposure, analogue gain, optional IR LED GPIO |
| `tracker` | pupil diameter bounds, minimum axis ratio/contrast/confidence, adaptive threshold percentiles, temporal limits |
| `recording` | layout, pose_cameras, source size/FPS, directory, shared_permissions, container, CRF, rate ceiling, keyframe interval, free-space guard |
| `transport` | `unix`, `uart`, or `none`, plus socket/device/baud |
| `preview` | enabled state, size limit, and threshold-mask display |
| `roi_config` | ROI file path, relative to the main config file when not absolute |

`recording.crf` accepts libx264 values 0–51; lower values preserve more detail
and produce larger files. The default is 24. The bitrate setting is a
conservative VBV ceiling, not the expected average bitrate.

Recording resolution and rate are independent of analysis resolution and
`camera.fps`. Recording cannot exceed acquisition FPS; selected-view recording
must be at least 30 fps. At 60 Hz, 60, 45 and 30 fps recording are supported.
Libcamera's ISP sizes the recording stream before encoding. Exact divisors
use Picamera2's frame skipper; fractional ratios use an integer phase
accumulator (60→45 keeps three of four frames), before packing or encoding.
Sensor timestamps are preserved. No Python image resizing is performed. When
recording and analysis sizes match, both paths share one camera stream instead
of producing duplicate full-resolution ISP outputs. Reduced recording FPS
requires Picamera2 0.3.19 or newer and is rejected at startup on older versions.

The example files are starting points, not validated experimental settings.
Revalidate after changing resolution, crop, exposure, illumination, frame
rate, or tracker parameters.

## Pupil fitting

For each eye crop the detector:

1. normalizes the cropped eye's intensity, then applies the saved per-eye
   grayscale adjustments;
2. lightly blurs sensor noise and evaluates several frame-adaptive dark-pixel
   thresholds;
3. builds one adaptive bright-reflection mask, including a small halo, and
   excludes affected boundary points before fitting and refining ellipses;
4. rejects candidates that are too small, large, elongated, irregular, weakly
   contrasted, or inconsistent with the recent pupil;
5. ranks the remaining pupil candidates using local contrast, shape, mask
   support, darkness, and threshold/size continuity.

Normalization runs independently on each eye crop. It estimates a smooth
quadratic lighting surface on a copy at most **64 pixels across**, excluding
dark pupil/eyelid structures and small bright reflections. Only six lighting
coefficients are estimated, using cached spatial coordinates. Poorly supported
surfaces are skipped. A bounded brightness correction is interpolated to the
original crop; the actual eye pixels are never warped, resized, or blurred by
normalization. This reduces broad bright/dark regions while retaining the
original sampling of the pupil boundary.

The corrected crop's 2nd and 70th intensity percentiles are then mapped toward
32 and 176, leaving headroom for bright reflections. These statistics sample
actual corrected pixels on a grid at most 128 pixels across. Combined illumination and
range amplification is capped at 2x before the manual controls. Nearly flat
input crops (percentiles fewer than 24 levels apart) bypass normalization to
avoid amplifying noise. Calculations stay in floating point until the final
8-bit clipping after the software controls. There is one pupil-detection pass
per eye, with no image-setting search or added temporal smoothing.
Calibration, tracking, playback, and eye previews share the same processing.
Camera frames, overview images, and saved video pixels are not normalized.

This compensates for broad illumination variations and changes in the crop's
overall intensity range. It cannot remove arbitrary sharp glare, separate a
reflection from a pupil contour at the same spatial scale, or restore detail
already clipped by the camera. Extreme software settings can still clip the
normalized image. Validate pupil center and diameter on representative footage
after changing the lighting or calibration.

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

Confidence evaluates the pupil's contrast against its immediate surrounding
ring, interior variation, shape, mask support, and temporal consistency. A clear
grey pupil on white can receive the same confidence as a dark pupil; unrelated
dark eyelashes elsewhere in the crop do not lower its appearance confidence.
Weak contrast and interior variation still lower the score. Whole-crop darkness
and threshold preferences remain in candidate **ranking**, separate from the
confidence used for acceptance and reporting. This retains the preference for
the pupil over a larger iris containing a darker core.

This scoring does not add a post-adjustment image stretch or change the pixels
used for fitting. It reuses the existing contrast and interior statistics, with
no extra image pass or pupil fit. Noise, flat frames, and occlusions still pass
through the existing geometry/visibility checks and minimum contrast/confidence
limits. Diagnostics include `local_separation` and `local_uniformity` scores.
Confidence is a heuristic quality score, not a calibrated probability of a
correct pupil measurement.

To reduce false locks on eyelashes or eyebrows during blinks, candidates that
move at least a quarter of the previous pupil diameter, or follow a missed
frame, must have a well-supported ellipse boundary and a major-axis length
within 30% of the last accepted pupil. Major-axis length allows gaze-related
foreshortening without treating it as a sudden change in pupil scale. This
guard retains scale evidence for up to 30 consecutive missed frames, even
after the short position prior is released; it does not restrict where the
eye may look after a blink. Longer losses release the scale memory so tracking
can recover after a large change in eye/crop scale. Concentric iris-to-pupil
corrections during continuous tracking remain allowed.

Rejected candidates produce an invalid measurement with zero diameter and
confidence, retain the last accepted position, and count toward the existing
blink delay. They never replace the pupil prior. The guard reuses contour
evidence and adds no image pass, ellipse fit, smoothing, or confirmation-frame
delay. Gain, contrast, thresholds, and confidence settings remain unchanged.
It reduces inconsistent switches; a distractor with the same scale and a
fully pupil-like boundary can still pass. Check representative blink footage
when validating performance.

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
`camera_time_ns`, sequence/drop information, per-eye results, and encoder settings.
Stopping tracking flushes recording before it acknowledges completion.

**`camera_time_ns` is the unmodified libcamera `SensorTimestamp` in nanoseconds.**
Libcamera defines it as exposure of the sensor's first row on the system's
boot-time clock ([control definition](https://github.com/raspberrypi/libcamera/blob/main/src/libcamera/control_ids_core.yaml)).
Tracking, UART status, and video-frame indexes use this same sensor clock. They
may run at different rates or include different frames; order events by
`camera_time_ns`, without matching row numbers or assuming equal sampling rates.
It is not Unix time, is not rebased at a block/recording start, and does not change
when a UART experiment command sets the system wall clock. Compare values within
the same Pi boot; a reboot starts a new sensor-clock timeline.

JSONL schema **4** replaces `sensor_timestamp_ns` with `camera_time_ns` and removes
`produced_timestamp_ns`. The software completion clock remains internal to
freshness checks and display-rate calculations; it is neither saved nor sent.
Processing duration and result age remain useful durations, not extra timestamps.
Block start/end dates live in `block.json`; JSONL lifecycle records no longer
duplicate wall-clock timestamps.

Each video has a compact `<recording>.frames.csv` with exactly these columns:

```csv
video_frame_index,camera_time_ns
0,4123456789123
1,4123473455790
```

The index is **zero-based decoded video-frame order**, not the tracker sequence.
Rows are written only after the encoded frame's successful mux/file write,
including frames drained when recording stops. Frame-rate skips create no rows.
The original nanoseconds are preserved even though the video container's internal
presentation timestamps use lower precision and a relative origin. No timestamps
are interpolated. The index uses buffered writes and bounded bookkeeping; it adds
no image copies, resampling, or coordination between tracking and recording.

Missing/invalid live `SensorTimestamp` is an explicit acquisition/recording error;
software monotonic time is never substituted. Codec/mux/index failures are
reported in the interface and JSONL `session_end.encoder_stop_error`; a failed or
interrupted recording may contain only a partial index and must not be treated as
a completed recording. Timestamp association uses Picamera2's
[encoder timestamp path](https://github.com/raspberrypi/picamera2/blob/main/picamera2/encoders/encoder.py)
and [post-write output hook](https://github.com/raspberrypi/picamera2/blob/main/picamera2/outputs/pyavoutput.py).

Playback reads the adjacent CSV and preserves its original sensor times, including
when looping or skipping frames to catch up. An older video without the CSV still
supports visual testing but has unavailable timing (`null` in JSONL, zero on UART);
its UART responses are marked stale with invalid eye slots. No host clock or
container timestamp is passed off as sensor time.

There is one `frame` row per completed tracking result. For a lossless 60 Hz
analysis run, `frame_sequence` is consecutive and
`dropped_analysis_frames` remains zero. This is independent of how many frames
the optional preview displays.

The `ultrafast` preset and the configured CRF minimize encoder work. The
profiles currently choose different CRFs for their intended tests, while 24 is
the code default when a value is omitted. Actual size is scene-dependent;
perform a full-duration storage and thermal test with the final camera settings.
The shipped `config/eye_tracker.json` uses `recording.layout: "hands_eyes"`.
One synchronized video contains the full views selected by
`recording.pose_cameras` (default `[0, 1]`, the first two views from the user's
left in configuration) plus the configured eye crops. Camera indices are 0–3;
use `[1, 3]` for different pose cameras, or `[]` for eyes only. Each selected
view in the shipped profile is 960x540 pixels, taken from the ISP's 3840x540
recording stream at 30 fps (45 or 60 fps can also be selected). Eye crops retain their recording-source pixels
without software image adjustments. Acquisition and pupil analysis remain
5120x720 at 60 Hz. CRF remains 20.

For this layout, `recording.width` and `recording.height` specify the stitched
**source** size before cropping. The encoded dimensions depend on the eye
boxes and camera rotations. Views are arranged in a compact montage at recording
start, padding only as needed for YUV420. Encoded dimensions therefore
change with the selected pose cameras and eye boxes. The JSONL header records
the encoded size, source/destination rectangles and per-region rotations under
`video_layout`. Rotated recordings use JSONL schema 3; earlier unrotated schemas
remain readable.
`eye-tracker validate` reports the actual encoded size for the saved ROIs.

The camera callback copies selected luma pixels into an owned buffer. A
separate worker performs compression and file output, without holding camera
requests. Its queue is bounded to eight frames. Sustained overload is reported
as a recording error rather than silently discarding video frames. Actual
tracking and recording throughput still need validation on the Pi with the
chosen views; if necessary, set only `recording.fps` to 30. Selected-view recording rejects rates below 30 fps.

Keep the video and its `.jsonl` sidecar together for `--video` playback and
offline benchmarks. Playback restores the saved views to their original
coordinates, with unsaved regions black. Eye fitting therefore uses the saved
ROI positions, then applies the loaded calibration rotation once. Keep the same
calibration with the video when reproducing its coordinates. The live preview
still shows all four camera views.
Older configurations without a `layout` field retain full stitched recording;
set `recording.layout: "full"` explicitly to select that behavior. With nonzero
rotations, all four full views use the same asynchronous montage encoder.

Video saving checks the **recording filesystem** before starting and every
**10 seconds** while recording. Below **10 GB free** (or a higher configured
`recording.minimum_free_gib` reserve), it stops and finalizes video on a background
worker, displays a nonmodal warning popup, and continues pupil tracking, UART,
and the smaller position/metadata log. `RECORDING` clears while `TRACKING` stays
set. No video restarts automatically: free space and explicitly start recording
again, or start a new tracking session. A run started below the reserve creates
only its JSONL log, with `video: null` and `video_disabled_reason` in the header.
A cutoff during recording adds a `video_stop` row with reason `low_disk_space`.
Headless runs log the warning; the popup appears in Pi Tracker or an open preview.

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
The dashboard separates these measurements:

- **compute … ms/frame**: detector processing time. The former `tracker 20 ms`
  label meant 20 milliseconds per frame, consistent with about 50 Hz.
- **results … Hz**: completed tracking results, including invalid detections.
  This replaces the former `tracking` label.
- **UART valid unique: eye … Hz**: distinct, fresh, valid pupil positions
  successfully written to the transport, measured per eye over two-second windows.
  Repeated replies for the same frame, stale results, blinks, and invalid positions
  do not inflate it. This is the relevant rate for the **approximately 50 Hz valid
  UART output** target. It measures the Pi's writes; no receiver validation is
  performed. With the development Unix transport it counts socket writes instead.
- **display … Hz**: preview refresh rate, independent of pupil output rate.

Compare **UART valid unique** with the usual task polling, and **results** to
diagnose acquisition/detection throughput. Check elapsed sensor
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

For a black camera pane or a monitor that loses its entire image while tracking
continues, see [Pi display troubleshooting](docs/display-troubleshooting.md).
The application retains the last frame across status-only changes; whole-monitor
sleep/HDMI faults require checking the Pi display session and hardware.

The horizontal-motion audit found no axis-specific gain or filter: both axes
share temporal smoothing, distance gates, and UART precision. Synthetic tests
with noise and glints preserved matched horizontal and vertical displacements.
This does not establish equal angular sensitivity in real animals: camera angle,
eye/eyelid geometry, and behavior can change measured pixel displacement. Keep
`VALID`/`LOST` flags in downstream analysis; invalid fits hold the last position
and must not be interpreted as stationary eyes.

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
