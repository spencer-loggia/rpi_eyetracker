# Macaque eye tracker

Experimental low-latency pupil tracking and video recording for a head-fixed
macaque imaged by an Arducam B0267 four-camera CamArray on a Raspberry Pi 5.
One or two user-selected regions are tracked independently. A controller polls
the latest measurements through a fixed-size binary command protocol.

Pose detection, gaze calibration, multi-view fusion, and behavioral-event
synchronization are intentionally out of scope for this version. Sustained
30 Hz operation and calibrated gaze performance over approximately +/-10
degrees horizontally and vertically are acceptance targets that still require
validation on the target Pi and representative macaque recordings. The current
minimum acceptable held-out accuracy and short-term precision are both below
1 degree throughout that working area.

## What is implemented

- One Picamera2 owner for the synchronized four-camera aggregate.
- A full-field stitched recording stream containing all four cameras and a
  separate grayscale ROI analysis stream from the same camera configuration.
- An initializer that draws, saves, reloads, and edits one or two eye boxes.
- Independent pupil center, equivalent diameter, confidence, blink, and lost
  state for every box.
- Adaptive dark-pupil segmentation that does not require a corneal glint and
  tolerates multiple bright reflections inside the pupil.
- A latest-frame queue: analysis overload drops stale work instead of adding
  latency; recording remains on the camera/encoder path.
- Start/stop tracking and recording commands with idempotent sequence handling.
- A versioned 64-byte, little-endian, CRC-32C protocol over direct Pi-to-Pi
  UART, plus a Unix-socket development transport.
- MKV/MP4 recording through Picamera2's encoder/output path, plus an append-only
  JSONL result sidecar.
- A controller-side command/poll client for the direct UART link.
- Offline replay benchmarking and unit tests.
- An optional on-tracker diagnostic display with live eye crops, selected
  pupil masks, fitted ellipses, centers, confidence, timing, and drop counts.

The Raspberry Pi camera, Arducam driver, encoder, GPIO, thermal behavior, and
electrical link cannot be exercised on this development Mac. Treat the default
values as a starting profile and complete the on-Pi validation gates below
before experimental use.

## Data definitions

Each configured box is a channel (`eye_id` 0 or 1). Channels are never fused,
even when they are two views of the same biological eye.

- `x`, `y`: crop-local analysis-stream pixels, origin at the crop's upper-left,
  x rightward and y downward.
- `pupil_diameter`: equal-area diameter of the selected external dark contour,
  `2 * sqrt(contour_area / pi)`, in the same pixels.
- `confidence`: a heuristic contour score, not a calibrated probability;
  `0..1` internally and `0..255` on the wire.
- invalid/lost: `valid=false`, `blink=false`, diameter exactly `0`.
- inferred blink: `valid=false`, `blink=true`, diameter exactly `0` after the
  configured missing-frame hysteresis. This can also mean occlusion, poor
  illumination, ROI loss, or detector failure; it is not yet physiologically
  validated. Until reacquisition, x/y retain the last reliable center (or zero
  before the first lock), but are not marked valid.

The runtime deliberately reports unsmoothed per-frame positions by default.
Optional exponential smoothing is configurable, but it adds temporal lag and
should not be enabled for saccade timing without validation.

## Architecture

```text
OV9281 x4 CamArray (one synchronized horizontal frame)
                    |
                 Picamera2
            +-------+--------+
            |                |
     main YUV420              lores Y plane
          |                        |
 full-field H.264 + MKV/MP4   copy 1-2 ROIs only
          |                        |
     video file            size-one latest queue
                             |
                    pupil tracker worker
                             |
                atomic latest result + JSONL
                             |
            Unix test link / direct UART
                             |
                     controller Pi
```

Picamera2 documents both multiple streams and `MappedArray`, which avoids a
full image copy before the small ROI copies are made. Its H.264 encoder on Pi 5
is software/libx264; the main stream defaults to 3840x540, approximately the
same pixel count as 1080p. See the
[Picamera2 manual](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf)
and [Raspberry Pi camera documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html).

## Set up a new Raspberry Pi 5

Use current 64-bit Raspberry Pi OS and active cooling. Choose the Desktop image
when the attached-display preview is required; a Lite image is appropriate for
a headless tracker. Arducam's Pivariety driver is kernel-version sensitive, so
record and pin the qualified OS, kernel, firmware, Picamera2, and Arducam
versions before experiments.

1. Update the fresh OS and install the system camera libraries:

   ```bash
   sudo apt update
   sudo apt full-upgrade
   sudo apt install git rpicam-apps python3-venv python3-picamera2 \
     python3-libcamera python3-opencv python3-av
   ```

   Install `python3-gpiozero` as well only when `camera.ir_led_pin` will control
   an external illuminator. Picamera2, libcamera, OpenCV, and PyAV should come
   from APT so their native libraries match Raspberry Pi OS. Do not replace
   Picamera2 or libcamera with unrelated PyPI builds.

2. With power disconnected, attach the B0267 bundle as directed by the current
   [Arducam CamArray quick start](https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamArray/quick-start/)
   and install the matching Pivariety driver. Arducam recommends the Pi 5 CAM1
   connector by default. If the vendor instructions for the installed bundle
   require CAM0, add its documented overlay to `/boot/firmware/config.txt`.
   Reboot, then confirm that the aggregate camera and modes are visible:

   ```bash
   rpicam-hello --list-cameras
   rpicam-vid --list-cameras
   ```

3. Clone or copy this repository, create a virtual environment that can see the
   APT-installed camera bindings, and install the project:

   ```bash
   git clone YOUR_REPOSITORY_URL macaque-eye-tracker
   cd macaque-eye-tracker
   python3 -m venv --system-site-packages .venv
   . .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m pip install --no-deps -e .
   ```

   `requirements.txt` contains the pip-managed runtime packages. Hardware-bound
   libraries remain APT packages and are exposed through
   `--system-site-packages`. Verify the resulting environment:

   ```bash
   python -c 'import av, cv2, libcamera, numpy; print("core imports OK")'
   python -c 'from picamera2 import Picamera2; print("Picamera2 import OK")'
   ```

4. Edit `config/eye_tracker.json` to match a sensor mode actually reported by
   `rpicam-hello`, choose an absolute recording directory on the NVMe volume,
   and set the transport profile. The B0267 documentation advertises aggregate
   RAW8 modes including 5120x800, 5120x720, and 2560x400, but the installed
   driver output is authoritative. Keep sensor, video, and analysis aspect
   ratios equal. Then create the one or two eye ROIs and test the live fit:

   ```bash
   eye-tracker configure --config config/eye_tracker.json
   eye-tracker validate --config config/eye_tracker.json --hardware
   eye-tracker preview --config config/eye_tracker.json
   ```

5. For the production direct-UART link, configure **both** Pi 5 boards. Run
   `sudo raspi-config`, then under **Interface Options > Serial Port** disable
   the serial login shell while enabling serial hardware. Add
   `dtoverlay=uart0-pi5` to `/boot/firmware/config.txt`, reboot, and verify that
   `/dev/ttyAMA0` exists and is not owned by a serial getty.
   Connect GPIO14/TX to the other Pi's GPIO15/RX, GPIO15/RX to GPIO14/TX, and
   ground to ground. Use only 3.3 V logic. The exact wiring and verification
   rules are in [docs/protocol.md](docs/protocol.md).

6. Validate the production profile before installing the service:

   ```bash
   eye-tracker validate --config config/eye_tracker.uart.json --hardware
   eye-tracker run --config config/eye_tracker.uart.json --start-tracking --preview
   ```

   Stop the smoke test with Ctrl-C, inspect the full-field video and JSONL
   sidecar, and then follow [docs/deployment.md](docs/deployment.md) to set the
   service user, NVMe path, graphical-session access, and systemd unit.

For a controller-only Pi, skip the Arducam and camera-library steps. Install
`git` and `python3-venv`, install this repository into a normal virtual
environment using `requirements.txt`, configure UART as in step 5, and use the
`eye-tracker-controller` commands. NumPy is the only non-standard runtime
package needed by that role.

The official Raspberry Pi documentation recommends installing Picamera2 with
APT and creating virtual environments with `--system-site-packages`; Raspberry
Pi OS Bookworm and later also require pip installs to occur inside a virtual
environment. See the
[camera software documentation](https://www.raspberrypi.com/documentation/computers/camera_software.html)
and [Python environment guidance](https://www.raspberrypi.com/documentation/usage/spi/).

`camera.ir_led_pin` defaults to `null`. Set it only when that GPIO drives the
logic input of a correctly current-limited IR illuminator; never power a
high-current emitter from a GPIO pin. Always evaluate illumination safety and
lock exposure/gain before collecting pupillometry data.

## Configuration notes

The three configured image sizes have different roles: `sensor_*` selects an
exact native aggregate mode, `video_*` is the encoded stitched recording, and
`analysis_*` is the grayscale stream used for ROI crops. All three must keep
the same aspect ratio. With the default four-across 5120x720 mode, each camera
contributes about 960x540 recording pixels and 640x360 analysis pixels. The
related two-camera project used 5120x800 on its particular setup, so use the
mode actually printed by `rpicam-hello --list-cameras`, not either value by
assumption. Exposure and gain are shared across the four CamArray channels.

Every saved MKV, MP4, or H.264 file is encoded from Picamera2's full stitched
`main` stream. It therefore contains the complete field from all four cameras,
not the configured eye boxes. The default profile scales the native 5120x720
aggregate to 3840x540 before encoding without spatially cropping it. Eye boxes
are copied only from the separate 2560x360 analysis stream. Changing or moving
an ROI cannot change the saved video's field of view. The JSONL session header
also records this as `video_scope` and records the encoded dimensions. Each
session produces one stitched aggregate video, not four separate camera files.

`roi_config` is resolved relative to the main configuration file. A relative
`recording.directory` is instead resolved from the process working directory;
use an absolute NVMe path in production. At 8 Mbit/s, video is approximately
3.6 GB/hour before small container/filesystem overhead. `minimum_free_gib` is a
startup threshold, not a duration-aware reservation or an out-of-space guard.

## Configure the eye boxes

The initializer captures and averages eight preview frames, then displays that
frozen image so the user can draw one or two boxes anywhere in the stitched
frame:

```bash
eye-tracker configure --config config/eye_tracker.json
```

Controls:

- drag with the left mouse button to add a box;
- inspect the coloured pixels inside each completed box: these are the exact
  contour pixels selected by the pupil detector, with the fitted ellipse shown
  in white;
- adjust `Gain`, `Brightness`, `Contrast`, and `Sharpness` to transform the
  cached frozen image in memory and refresh the detector overlay immediately;
  these controls never capture another camera frame;
- adjust `Pupil threshold` to rerun the detector overlay immediately; lower
  values admit only darker pixels and can exclude the lighter iris, while `0`
  restores adaptive threshold selection;
- adjust `Exposure`, then press `R` to apply it and recapture the averaged
  image;
- Enter or `S` saves after one or two boxes;
- Backspace or `U` removes the last box;
- `C` clears all boxes;
- Escape or `Q` cancels without replacing the saved file.

Exposure is limited to the range reported by the attached camera and to less
than one configured frame period. Saving writes the applied values back to the
main JSON configuration as well as writing the ROI file. Gain, brightness,
contrast, and sharpness are applied to the original cached preview and marked
saveable as they move; only exposure communicates with the camera and requires
`R` to recapture. The pupil threshold is also saved and used by the live
tracker. An exposure change remains pending until `R` is pressed, and the
editor asks for that recapture before it will save.

In JSON, `tracker.pupil_threshold: null` selects adaptive mode; an integer from
1 through 254 forces that threshold and supersedes `threshold_percentiles`.
The `--image` offline mode also provides all four in-memory image-control
sliders and the detector selection, but has no exposure slider or recapture
action.

Drawing order assigns `eye_id` 0 then 1; the IDs do not inherently mean left
and right eye. Boxes are stored as normalized stitched-frame coordinates in
`config/rois.json`, so they map cleanly between preview, analysis, and recording
sizes with the same aspect ratio. Changing analysis resolution changes the
pixel-valued output. To develop without camera hardware:

```bash
eye-tracker configure --config config/eye_tracker.json --image preview.png
```

## Live diagnostic preview

On the tracking Pi with an attached display, start a display-only camera and
tracker check with:

```bash
eye-tracker preview --config config/eye_tracker.json
```

This shows each configured live eye crop and, independently for each channel:

- the selected threshold mask;
- the raw fitted pupil ellipse and candidate center in cyan;
- the reported pupil center in green;
- tracking, no-fit, or blink/occlusion state;
- x/y, equivalent pupil diameter, confidence, fit axes, threshold, and
  pupil/background contrast;
- tracker time, display rate, result age, sequence, and analysis drop count.

The `Exposure us` slider changes the running Picamera2 exposure in real time;
the requested live value is shown in the dashboard header. This live adjustment
is not written back to the JSON configuration. Use the configuration window and
save there when the new exposure should become the next run's default.

Press `Q` or Escape in the window to close it. The standalone `preview` command
never starts the video encoder, regardless of `record_on_tracking`, so it is
safe for alignment and fit checks without creating a recording.

To show the same diagnostics during a normal controller-driven run, use:

```bash
eye-tracker run --config config/eye_tracker.uart.json --preview
```

Alternatively set `preview.enabled` to `true` in the JSON configuration. Use
`--no-preview` for a one-run override. This changes only the local display:
normal full-field video, JSONL results, and UART behavior are unchanged.

The GUI runs in a separate process behind its own latest-frame queue. Slow
rendering drops old display frames rather than blocking capture or tracking,
and closing or losing the display does not stop an active experiment. The
display process does copy the one or two eye crops, so re-run the on-Pi timing
and thermal validation with preview enabled before relying on it during data
collection. `show_threshold_mask: false` reduces display work.

## Run

Validate configuration only:

```bash
eye-tracker validate --config config/eye_tracker.json
```

Validate camera negotiation too (on the Pi):

```bash
eye-tracker validate --config config/eye_tracker.json --hardware
```

The hardware validation opens/configures the camera and checks negotiated
sizes. It does not start capture, encode video, switch illumination, exercise
UART, or prove 30 Hz end-to-end throughput.

Run the command service with the default Unix development transport:

```bash
eye-tracker run --config config/eye_tracker.json
```

Run the direct-UART production profile on the tracking Pi:

```bash
eye-tracker run --config config/eye_tracker.uart.json
```

For a local camera smoke test without a controller:

```bash
eye-tracker run --config config/eye_tracker.json --start-tracking
```

With `record_on_tracking: true`, `START_TRACKING` starts the MKV recording
before analysis threads begin. Every recording gets a neighboring JSONL file
with sensor timestamp, processing-completion timestamp, sequence, drop count,
and both independent eye results. `STOP_TRACKING` stops acquisition and flushes
the encoder and sidecar before acknowledging completion.

The sidecar records analysis results, but it does not currently store an exact
encoder-PTS-to-analysis-frame mapping. Do not claim frame-exact video/result
synchronization. It is flushed periodically and at graceful stop, but it is not
an `fsync`-per-row crash journal.

Explicit recording start is valid only while tracking is active. Stopping
tracking always stops and flushes recording first. Configuration is loaded at
process start; there is no hot reload or remote process-shutdown command.

## Direct Pi-to-Pi UART

The complete wire layout and controller behavior are in
[docs/protocol.md](docs/protocol.md).

The production profile uses UART0 at 460800 baud with no intermediary device:

```text
tracking Pi GPIO14 TX  --->  GPIO15 RX controller Pi
tracking Pi GPIO15 RX  <---  GPIO14 TX controller Pi
tracking Pi GND        -----  GND controller Pi
```

The link is 3.3 V logic. Do not connect either 5 V rail. On each Pi 5, enable
UART0 on header pins 8/10 with `dtoverlay=uart0-pi5`, disable any serial console
on `/dev/ttyAMA0`, and verify the resulting device before starting the service.
Use `config/eye_tracker.uart.json` for the tracking Pi.

On the controller Pi, the included client sends commands and prints JSONL
telemetry. It defaults to `/dev/ttyAMA0`, 460800 baud:

```bash
eye-tracker-controller start-tracking
eye-tracker-controller monitor --rate 100
eye-tracker-controller stop-tracking
```

Run exactly one owner of the UART on each Pi. The service and controller claim
exclusive tty access, so a second process fails instead of flushing or
interleaving an active experiment's traffic.

For a same-machine development test, select the Unix socket instead:

```bash
eye-tracker-controller --unix /tmp/macaque-eye-tracker.sock status
```

The controller must poll. Use 100 Hz rather than exactly 30 Hz; this avoids
phase-locking against a 30 Hz camera and easily observes every new result.
At 460800 baud, one 64-byte 8-N-1 record takes about 1.39 ms on one direction;
100 request/response polls per second use about 28% of the sequential wire time.
Duplicate frame sequences are normal and should be discarded by the controller.
Every request has a nonzero sequence that its response echoes, so the client can
drain a late response without shifting subsequent transactions. The theoretical
poll ceiling is `baud / 1280` before timing margin; consequently the CLI rejects
the default 100 Hz rate at 115200 baud. Use a lower rate or the production
460800-baud profile.
State-changing calls are synchronous and use a separate 10-second response
timeout because camera startup and recording finalization can take much longer
than a telemetry poll. A lost response is handled by retransmitting the exact
same command sequence, never by interpreting a later poll as its outcome.
The monitor keeps running through isolated timeout/CRC failures and reports
consecutive link errors on stderr; persistent errors still require operator
attention and should be included in the on-Pi burn-in test.

UART was selected because the PL011 path is simple and supported without an
intermediary. RP1 exposes target-capable SPI4/SPI7 hardware, but this repository
and its pinned production image do not provide a qualified userspace SPI-target
path. An I2C address conflict is likewise solvable with a normal unused address
such as `0x42` (I2C address `0x00` is the reserved general-call address), but the
currently evaluated stock Pi 5 kernel lacks `CONFIG_I2C_SLAVE`. I2C therefore
remains an option only after qualifying the exact kernel and a target backend.
Relevant primary references are the
[Raspberry Pi UART documentation](https://www.raspberrypi.com/documentation/computers/configuration.html#configure-uarts),
[RP1 peripheral specification](https://datasheets.raspberrypi.com/rp1/rp1-peripherals.pdf),
[current RP1 device tree](https://github.com/raspberrypi/linux/blob/rpi-6.18.y/arch/arm64/boot/dts/broadcom/rp1.dtsi),
and the
[evaluated Pi 5 kernel configuration](https://github.com/raspberrypi/linux/blob/rpi-6.18.y/arch/arm64/configs/bcm2712_defconfig).

## Using eye data downstream

The display/controller Pi receives one decoded status record containing one or
two eye slots for the same tracker frame. These are not separate UART streams,
and their `x`/`y` values are pupil centers in their respective crop-local pixel
coordinates—not screen coordinates or calibrated gaze.

For a real-time display application, the recommended pipeline is:

```text
poll + decode + CRC check
          |
reject duplicate, stale, errored, or invalid samples
          |
apply the calibration for each usable eye/view
          |
optionally fuse calibrated estimates
          |
apply a validated causal filter
          |
screen coordinate + validity + timestamp
```

### Accepting a sample

Poll at about 100 Hz and process only new camera frames. The included
`EyeTrackerController.poll_new()` and `iter_new_samples()` already deduplicate
using `(session_id, frame_sequence, sensor_timestamp_ns)`. Before using a
sample, require all of the following:

- `status == OK`;
- `TRACKING` is set, while `ERROR` and `RESULT_STALE` are clear;
- the service `session_id` has not changed unexpectedly;
- `result_age_ms` is within the latency budget for the display task;
- the selected eye slot has `valid == true` and acceptable confidence.

When `valid` is false, ignore its x/y values: they intentionally retain the
last reliable center. `blink == true` means inferred blink/occlusion; otherwise
an invalid slot is lost/no-fit. Both states report pupil diameter zero. Do not
silently turn a long invalid interval into a stationary gaze estimate. The
application should emit an invalid-gaze state and explicitly choose whether to
hide, freeze briefly, or otherwise change the display.

The command-line monitor prints one JSON object per new camera frame:

```bash
eye-tracker-controller monitor --rate 100
```

A record has this shape:

```json
{
  "status": "OK",
  "flags": ["TRACKING", "RECORDING", "CONFIGURED"],
  "frame_sequence": 1234,
  "sensor_timestamp_ns": 567890123456,
  "session_id": 305419896,
  "configured_eye_count": 2,
  "result_age_ms": 8,
  "eyes": [
    {"eye_id": 0, "x": 121.5, "y": 78.25, "pupil_diameter": 42.0,
     "confidence": 0.91, "valid": true, "blink": false},
    {"eye_id": 1, "x": 98.75, "y": 81.0, "pupil_diameter": 39.5,
     "confidence": 0.87, "valid": true, "blink": false}
  ]
}
```

For a latency-sensitive application, import `SerialPacketChannel` and
`EyeTrackerController` directly instead of parsing the monitor's JSON text.
The complete binary field definitions and failure rules are in
[docs/protocol.md](docs/protocol.md).

### Mapping pupil position to the display

Collect a gaze calibration for each configured channel by presenting known
targets across the display and retaining stable, valid fixation samples. Fit a
mapping from crop coordinates `(x, y)` to display coordinates `(screen_x,
screen_y)`. An affine x/y-only transform is a useful baseline, but it must not
be assumed sufficient:

```text
screen_x = a0 + a1*x + a2*y
screen_y = b0 + b1*x + b2*y
```

Pupil dilation can move the measured pupil centroid without an eye rotation.
This pupil-size artefact has been reported for both centroid and ellipse pupil
tracking, and its size and direction can depend on gaze direction and the
camera axis. That makes diameter a scientifically justified candidate
calibration feature for this pupil-only tracker. For each eye/view, compare at
least these nested models on held-out calibration samples:

```text
x/y affine:       [1, x, y]
size-aware:       [1, x, y, dd]
interaction:      [1, x, y, dd, x*dd, y*dd, dd*dd]
```

Here `dd = pupil_diameter - reference_diameter`, where the reference is fixed
from the calibration set. Fit separate coefficient vectors for `screen_x` and
`screen_y`. A modest ridge penalty or robust regression is preferable to an
unconstrained high-order fit when calibration data are limited. Spatial
curvature terms such as `x*x`, `x*y`, and `y*y` may also help, but should be
retained only when they improve held-out error.

For the present minimum-good-enough implementation, start with the size-aware
interaction model `[1, x, y, dd, x*dd, y*dd]` for each channel. Compare it with
the x/y-only affine baseline using target-wise cross-validation. Add spatial
quadratic terms only if the simpler model misses the acceptance threshold. If
that still fails, improve the ROI, camera alignment, illumination, or pupil fit
before adding a more flexible model that may merely overfit sparse targets.

The calibration protocol must vary pupil size at the *same gaze targets*, for
example by interleaving at least two safe, controlled background luminances.
Otherwise gaze location, stimulus content, and pupil size may be confounded and
the model cannot identify a pupil-size correction. Cover the pupil-diameter
range expected during the task; extrapolation beyond it is unsafe. Compare
x/y-only and diameter-aware models by target-wise cross-validation, including
the display edges and each luminance condition. Prefer the simplest model that
meets the preregistered accuracy and latency requirements.

The transmitted diameter is an image-space, equal-area contour diameter. It is
zero for invalid samples and can itself change with gaze angle, eyelid
occlusion, and pupil foreshortening; never feed zero/invalid values into the
calibration. Preserve the raw x/y/diameter alongside calibrated gaze so that a
correction can be audited or re-fit offline.

This recommendation is supported by evidence that pupil-size regression can
substantially reduce false gaze shifts during fixation, although the published
effect is primarily characterized in humans and must be measured in this
macaque/camera geometry. The macaque-focused Oculomatic work used an affine
offline calibration as a practical baseline but also observed systematic edge
error that could benefit from a higher-order calibration. See
[Choe et al. 2016](https://doi.org/10.1016/j.visres.2014.12.018),
[Hessels et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516786/), and
[Zimmermann et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4981506/).

Store the chosen calibration and its validation metrics together with the ROI
layout, analysis resolution, camera/exposure setup, illumination conditions,
animal, display geometry, and software configuration hash. Recalibrate after
changing any of those inputs.

### Current calibration acceptance target

For this project, treat the required working area as `-10..+10` degrees on
both horizontal and vertical display axes. Because “accuracy” and “precision”
describe different failures, both must pass on held-out fixation data:

- **Accuracy:** at every tested target and pupil-size condition, the median
  Euclidean distance between calibrated gaze and target is less than 1 degree.
- **Precision:** at every tested target and pupil-size condition, the RMS
  sample-to-sample displacement during accepted fixation samples is less than
  1 degree. Also report horizontal/vertical standard deviation so directional
  noise is not hidden by a single radial value.
- **Coverage:** validation includes the center, +/-10-degree horizontal and
  vertical limits, corners of the required area, and intermediate locations.
- **Validity:** report the valid-sample fraction and blink/lost fraction at
  every target; a sub-degree result based on a small surviving subset does not
  pass.
- **Latency:** report these metrics on unfiltered gaze and on the exact causal
  filter used online, together with that filter's added delay. Heavy smoothing
  cannot be used to claim that the underlying tracker meets the requirement.

A practical minimum dataset is repeated fixation at a 3x3 grid spanning
`{-10, 0, +10}` degrees on both axes, under at least two approved luminance
conditions, plus held-out intermediate targets around 5 degrees. Split by
entire fixation trial and target—not individual video frames—so temporally
adjacent samples from one fixation cannot leak into both fit and validation
sets. Calibration and validation targets should be presented in randomized
order.

One valid channel is sufficient to produce a calibrated 2-D estimate. With two
channels:

- For two biological eyes, calibrate each eye independently, then combine the
  two *screen-coordinate* estimates using a validated rule such as a weighted
  mean based on confidence and per-eye calibration error. Retain the two
  estimates separately if binocular disparity is scientifically relevant.
- For two camera views of the same eye, either calibrate each view separately
  and fuse in screen space, or train one joint calibration using
  `(x0, y0, d0, x1, y1, d1)` and selected interactions. Do not average their
  raw crop coordinates: the origins, scales, and view geometry differ.

Pupil diameter remains a pupillometry output in its own right. Using it as a
gaze-correction feature does not make the corrected value an anatomical pupil
diameter, and the uncorrected diameter should still be retained for analysis.

### Filtering and timing

Filter only accepted, calibrated samples. A causal low-pass, One Euro, or
state-space filter can reduce display jitter, but every filter adds lag; choose
parameters from measured noise and task latency rather than appearance alone.
Do not interpolate across blinks or loss unless the downstream record preserves
that the output was imputed. For gaze-contingent stimuli, compare the complete
capture-to-display latency and not just tracker processing time.

`sensor_timestamp_ns` is the camera time on the tracking Pi. UART receipt time
is later and variable, and monotonic clocks on two Pis are not automatically in
the same time domain. Use a qualified clock-synchronization method and/or a
shared hardware/visual timing event when aligning gaze to display events. The
validation section below describes the required end-to-end timing check.

## Offline benchmark

Run the same tracker over a saved stitched recording:

```bash
eye-tracker benchmark session.mkv --config config/eye_tracker.json
eye-tracker benchmark session.mkv --config config/eye_tracker.json --frames 1000
```

The report includes mean/p50/p95/p99 detector time, effective FPS, valid/blink
fractions, and whether p99 stays under the source frame period. It excludes
camera capture, encoding, sidecar I/O, UART, scheduling, and thermal throttling;
it is a detector timing check, not an end-to-end or accuracy measurement.

## Tracker design and tuning

The tracker works only within the small selected regions:

1. Gaussian noise filtering.
2. Several frame-adaptive dark thresholds.
3. Small opening/closing operations.
4. External contours, so bright glints wholly inside the pupil remain holes and
   are not treated as required landmarks.
5. Robust ellipse fit for shape validation and outlier rejection.
6. Contour-moment center and contour equal-area diameter for output.
7. Candidate ranking by pupil/annulus contrast, ellipse residual, fill,
   boundary support, and distance/size change from the previous frame.
8. Missing-frame hysteresis for blink versus transient loss.

This follows the practical direction of
[Oculomatic](https://pmc.ncbi.nlm.nih.gov/articles/PMC4981506/), which was built
for head-restrained macaques and found threshold/contour methods effective at
high rate. No code is copied from that project or from restrictively licensed
pupil detectors.

Tune per animal, camera, and illumination. In particular, quantify stimulus
luminance effects: pupil size changes can couple into apparent pupil position.

## Validation required before experiments

Do not describe the system as precise solely because it runs at the configured
frame rate. Validate it with representative macaque video:

1. Hand-label center, pupil boundary/diameter, blink, lid occlusion, glints,
   gaze range, and stimulus luminance strata.
2. Report center and diameter error, blink precision/recall, and lost-track
   rate independently for every crop.
3. Measure actual sensor and result rates plus p50/p95/p99 capture-to-result
   latency and sequence gaps.
4. Use a controller TTL captured by both systems or a switched LED in the
   camera field to measure end-to-end timing; UART receipt time is not exposure
   time.
5. Run an hour-scale recording/tracking/polling burn-in while logging Pi
   temperature, throttling, storage throughput, and dropped frames.
6. Recheck after any OS, kernel, camera driver, exposure, illumination, crop,
   or detector-parameter change.

At 30-50 fps this design suits fixation, gross eye position, and pupillometry.
It is not sufficient for high-quality saccade-onset, peak-velocity, or
microsaccade measurements. The CamArray's smaller aggregate mode may reach 150
fps, but that profile still needs target-hardware and encoder validation.

## Repository layout

```text
src/macaque_tracker/   runtime, tracker, protocol, controller, initializer
config/                example runtime profiles (generated rois.json is local)
docs/                  wire protocol and deployment notes
deploy/                example systemd service
requirements.txt       pip-managed runtime dependencies and Pi APT notes
tests/                 configuration, protocol, service, synthetic-eye tests
video/                 original standalone rpicam-vid recorder (legacy fallback)
```

The standalone code in `video/` remains useful for isolated camera/encoder
testing, but it cannot share the camera with real-time tracking and is not the
primary runtime.
