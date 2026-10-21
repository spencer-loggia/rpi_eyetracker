# Eye tracker wire protocol v1

All fields are little-endian. UART uses 8 data bits, no parity, one stop bit,
and no flow control (8-N-1), normally at 460800 baud. Each request and response
is exactly 64 bytes. CRC-32C covers bytes 0-59; its reflected polynomial is
`0x82f63b78`, initial value and final XOR are `0xffffffff`. The standard check
is:

```text
CRC32C("123456789") = 0xe3069283
```

## Common 24-byte header

| Offset | Size | Field |
|---:|---:|---|
| 0 | 2 | ASCII magic `ET` |
| 2 | 1 | protocol version, currently `1` |
| 3 | 1 | packet type: command `1`, status `2` |
| 4 | 2 | flags |
| 6 | 2 | command or status code |
| 8 | 4 | command sequence, or result frame sequence |
| 12 | 4 | start number for commands (see below); echoed request sequence for status (`0` if undecodable) |
| 16 | 8 | experiment UTC datetime for `START_EXPERIMENT` (otherwise zero); `camera_time_ns` for status |

Bytes 60-63 contain the little-endian CRC-32C.

## Commands

Command packets use type `1`. Bytes 24-59 are zero except for
`START_EXPERIMENT`. Command flags retain their existing meaning; send zero
unless an extension explicitly specifies otherwise. Header bytes **12-15** hold
a little-endian uint32 experiment number for `START_EXPERIMENT`, or block number
for `START_TRACKING`/`START_RECORDING`. **Zero requests automatic numbering**;
explicit numbers are 1–4294967295. These bytes must remain zero for all other
commands. Send zero at bytes 16-23 for all commands except `START_EXPERIMENT`.
Older command packets containing a controller monotonic timestamp remain accepted
but that value is ignored. Only
`START_EXPERIMENT` uses **nonzero Unix UTC nanoseconds since 1970-01-01T00:00:00Z**.

| Code | Name | Meaning |
|---:|---|---|
| 0 | `POLL` | fetch telemetry; response echoes this poll's request sequence |
| 1 | `START_TRACKING` | start camera, optional configured recording, and analysis; Pi Tracker accepts a block number at offset 12 |
| 2 | `STOP_TRACKING` | stop pupil analysis and flush recording; keep live preview if enabled |
| 3 | `START_RECORDING` | start recording while tracking is active; Pi Tracker accepts a block number at offset 12 |
| 4 | `STOP_RECORDING` | flush recording while tracking continues |
| 5 | `CLEAR_ERROR` | clear an idle error state |
| 6 | `START_EXPERIMENT` | Pi Tracker: set system UTC time and open an experiment for the supplied subject and number at offset 12 |
| 7 | `STOP_EXPERIMENT` | Pi Tracker: stop activity, flush recording, close the experiment, return home, and queue configured data sync |
| 8 | `SHUTDOWN` | Pi Tracker: finalize activity and request immediate clean system power-off |

### Pi Tracker experiment commands

These commands are supported by the `python pi_tracker.py CONFIG` interface.
The standalone `eye-tracker run` and `eye-tracker preview` services reject
experiment commands with `UNKNOWN_COMMAND` and numbered block starts with
`INVALID_STATE`; their unnumbered tracking commands are unchanged.
The interface owns the UART throughout all of its screens, so the controller
uses the same port and protocol during setup, calibration, and tracking.

`START_EXPERIMENT` has the following payload:

| Offset | Size | Field |
|---:|---:|---|
| 24 | 1 | subject length, 1–32 ASCII bytes |
| 25 | 32 | subject bytes followed by zero padding |
| 57 | 3 | reserved, zero |

The experiment number is in header bytes **12-15**, not in the subject payload.
The fixed 64-byte packet, protocol version 1, subject capacity, CRC and status
layout are unchanged. Existing controllers that send zero at offset 12 use
automatic numbering. Update the tracker before sending explicit numbers: older
tracker versions reject nonzero bytes at offset 12 as reserved data.

Subject names must match `[A-Za-z0-9][A-Za-z0-9_.-]{0,31}`. They are case
sensitive, are not rewritten, and cannot contain whitespace, slashes, or
non-ASCII characters. All bytes after the subject, up to and including byte
59, must be zero. A missing subject, zero timestamp, invalid name, or nonzero
padding is rejected as `BAD_PACKET_TYPE` before any experiment or clock action.
`STOP_EXPERIMENT` and `SHUTDOWN` require the usual all-zero payload.

On a valid start, Pi Tracker sets the system wall clock to the supplied UTC
time and creates `data/SUBJECT/exp_SUBJECT_YYYYMMDD_NUMBER_HHMMSS/`.
Both the date and the additional `HHMMSS` time use UTC. With number zero or a
local GUI start, NUMBER is one greater than the highest used experiment
number for that subject and UTC day, starting at 1. An explicit number already
used on that subject/day is rejected before setting the clock or stopping the
active experiment; experiments are never overwritten. Numbers already uploaded
and removed locally remain reserved in the local history ledger. Numbers can be skipped.
The new experiment inherits the most recent prior experiment's calibration
when available. Experiment files, calibration, tracking data, and videos
remain under the experiment directory, with saved runs under `blocks/NUMBER/`.
An experiment-level `config.json` stores its settings, and `calibration.json`
remains editable in calibration. Starting an experiment does not start
eye tracking. If an experiment is already active, a new start command stops
its activity, flushes recording, closes it, and opens the requested experiment.
A stop command when no experiment is active also retries pending data sync
when configured; it creates no experiment or block.

With `sync.remote_data_url` configured, closing an experiment queues background
transfer of all pending experiments to `<remote_data_url>/pitracker/`. Replacing
an experiment excludes the newly created experiment from this transfer. An
`OK` response confirms recording finalization and queuing, **not completed
upload**. No packet layout or command codes change. `POLL` remains available
during copying; new acquisition/state commands return `INVALID_STATE` until
sync finishes or is cancelled in the GUI. `SHUTDOWN` is accepted during sync
and waits for it. If a start was rejected during copying, retry it with a new
sequence after completion because the rejected response is cached.

Successful transfers remove their local experiment directories. Failures keep
local data and show a warning with diagnostics in `<data-root>/interface_sync.log`.
A normal stop's transfer failure is a GUI warning, not a retroactive change to
its `OK` response. Use a new `STOP_EXPERIMENT` sequence to retry later. Calibration
reuse and daily numbering survive removal through a small local history cache.
See [sync configuration and error handling](../README.md#experiment-synchronization).

The Pi Tracker process needs OS permission to set the system clock, such as
Linux `CAP_SYS_TIME` granted by its service configuration. It does not launch
an interactive `sudo` prompt over UART. A clock-setting failure returns
`INTERNAL_ERROR`; no new experiment directory is created and the active
experiment is unchanged. UTC changes affect wall-clock dates, not camera
sensor timestamps or the monotonic clocks used for tracking and timeouts.
The supplied timestamp is applied when handled; the protocol does not estimate
wire transit delay. Send it immediately before the command.

Clock-setting failures are shown in the GUI and logged with the underlying
exception. A permission error includes the launcher's PID, effective UID and
whether the calling UART thread's **effective**, permitted, bounding and ambient
capability sets contain `CAP_SYS_TIME`. Linux capabilities are per-thread, so
the diagnostic reads `/proc/thread-self/status`. An ambient
capability configured on a service is not by itself proof that the GUI process
received it. If the effective capability is present, check service restrictions
such as `ProtectClock` (it must be `no`) and any container/security policy.
The GUI's advice about `ProtectClock` is not a measurement of that service setting.
An unavailable Python clock API is reported separately from permission denial.
These diagnostics run only after failure and never alter the clock or permissions.

For an offline permission test, close the existing interface and launch from
the project directory with `sudo -E .venv/bin/python pi_tracker.py config/eye_tracker.json`
(adjust `.venv/bin/python` if your environment is elsewhere). A normal root
process on Raspberry Pi OS has clock-setting permission. Files produced during
that run are root-owned but the launcher makes experiment files readable and
writable by all local users (mode `0666`, directories `0777`). When using a multiline `systemd-run` command instead,
each continuation backslash must be immediately followed by a newline; remove
the backslashes when entering the command on a single line.

Within an active experiment, `START_TRACKING` enters tracking mode and starts
analysis/recording according to the configuration. It is rejected with
`INVALID_STATE` during calibration or without an experiment. The GUI's
**Start tracking** button instead opens the tracking view with tracking idle,
ready for a manual start or `START_TRACKING`. `STOP_TRACKING` leaves the live
view and experiment open. `STOP_EXPERIMENT` also exits calibration or tracking
and returns to the opening screen.

### Numbered blocks

`START_TRACKING` and `START_RECORDING` accept a block number at bytes **12-15**;
their bytes 24-59 remain zero. A local start or a zero number selects one greater
than the highest block number already present in this experiment, starting at 1.
The idle preview creates no block. Each new saved run writes its videos and eye
trace JSONL files to `EXPERIMENT/blocks/NUMBER/`, alongside `calibration.json`,
`config.json` and `block.json`. Snapshots describe the settings actually loaded
for that run, including camera rotations, even if experiment files are later
edited. Pi Tracker saves eye traces when automatic video recording is disabled.

**A positive block number explicitly authorizes replacement of the entire existing
block directory.** A new numbered `START_TRACKING` first stops the current run,
then replaces/creates the requested block and restarts tracking. A numbered
`START_RECORDING` closes the current saved run and starts the requested block
while tracking remains enabled. Other blocks and experiment settings are preserved.
Snapshot preparation errors leave an existing block's contents intact. Symlink
destinations are rejected. Number allocation and copying run only at starts,
with no added per-frame file operations.

An unnumbered start while the requested activity is already active is a no-op.
`STOP_TRACKING` ends the saved block; `STOP_RECORDING` ends saving but leaves
tracking/UART running. A following start allocates the next block unless numbered
explicitly. A low-disk video cutoff keeps the current block's position log open;
an explicit recording restart creates a new block. Back, experiment stop and
shutdown close the current block's files before leaving.

The same sequence cache described below covers experiment commands. Retry a
timed-out start with **identical bytes**, including the original timestamp,
subject, experiment/block number, and sequence. A cached duplicate returns its
original outcome without changing the clock, allocating another directory, or
erasing block files again. A fresh sequence is a new request; an explicit block
number can replace that block again. The 64-command cache is in memory and resets
when Pi Tracker restarts; this retry protection does not persist across restarts.

### Controller examples

Controller examples (ISO 8601 dates require `Z` or an explicit UTC offset):

```sh
eye-tracker-controller start-experiment subject_01 --experiment-number 2 --datetime 2026-10-14T14:30:00Z
eye-tracker-controller start-tracking --block-number 3
eye-tracker-controller stop-tracking
eye-tracker-controller start-tracking  # next automatic block
eye-tracker-controller stop-experiment
```

Omitting `--datetime` sends the controller's current UTC wall-clock time.
For Python clients, use
`controller.start_experiment("subject_01", experiment_number=2, timestamp_ns=...)`,
`controller.start_tracking(block_number=3)`, and `controller.stop_experiment()`.
Omit either number for automatic numbering. `controller.start_recording(block_number=...)`
also selects a block while tracking is running.
The wire timestamp supports nanoseconds; the ISO 8601 CLI accepts Python
datetime precision (microseconds). Subsequent telemetry polling is unchanged.

With preview enabled, the camera stays running before `START_TRACKING` and
after `STOP_TRACKING` to display live crops. Idle frames are not pupil
measurements: `TRACKING` and `RECORDING` are clear, frame sequence and sensor
timestamp are zero, and the eye slots are invalid. Starting tracking resets
the detector and excludes frames captured during the preceding idle period.
Without preview, `STOP_TRACKING` also stops the camera.

Use a unique, nonzero, monotonically increasing sequence for every request,
including `POLL`, within a persistent controller process. Every valid response
echoes that request sequence at header offset 12, which lets the controller
discard a late response from an earlier transaction. The included one-shot CLI
starts at a random nonzero sequence to make a collision after controller restart
unlikely. A state-command retry must repeat the entire command with the same
sequence; the service retains the 64 most recent state-command outcomes so
delayed retries remain idempotent. Reusing a cached state-command sequence for
different content returns `SEQUENCE_CONFLICT`. Polls do not alter that cache.

### Clean system shutdown

Send `SHUTDOWN` (code **8**) through the same 64-byte ET v1 protocol, with a
unique nonzero sequence, zero command flags, an all-zero payload, and the usual
CRC32C. Its timestamp should be zero (legacy nonzero values are ignored); it does
not change the clock. This is a binary command, not the text string `shutdown`.
It works from any Pi Tracker screen, including when no experiment is active.
Standalone `run`/`preview` reject it with `UNKNOWN_COMMAND`.

```sh
eye-tracker-controller --device /dev/ttyAMA10 shutdown
```

Python clients can call `controller.shutdown()`. On receipt, the launcher blocks
new activity, stops tracking and cameras, flushes recordings/metadata, and marks
the experiment closed. Unsaved calibration edits are discarded, as with Back.
It sends an `OK` response with `SHUTTING_DOWN` (status bit **7**) set and drains that
UART reply. When synchronization is configured, it attempts to upload all pending
experiments before requesting normal systemd power-off with
`/usr/bin/systemctl --no-ask-password --no-block poweroff`. There is no scheduled
minute-long delay and no force option; recording finalization and active transfer
take as long as needed. Without configured sync, power-off follows the reply
immediately. The operating system performs its normal service/filesystem
shutdown. Running the launcher as root supports this on Raspberry Pi OS;
`CAP_SYS_TIME` alone does not grant power-off permission.

An `OK` acknowledges recording finalization and an accepted request; it does
not confirm upload completion or that the Pi has powered off. During copying,
`POLL` remains responsive with `SHUTTING_DOWN` set. The link disappears when
power-off proceeds. A lost
reply or disconnected reader does not cancel an already accepted shutdown.
Retry a timeout with identical bytes/sequence; cached retries do not submit
power-off again. If recording finalization fails, return `INTERNAL_ERROR` and do not power off.
If data transfer fails after acknowledgement, keep local data and remain open:
`POLL` reports `INTERNAL_ERROR`, with `SHUTTING_DOWN` still set while the GUI offers
**Retry**, **Shut down anyway**, and **Cancel shutdown**. Only successful sync or
the explicit bypass proceeds to power-off. Cancelling sync or shutdown clears
`SHUTTING_DOWN`; identical retries of that cancelled shutdown return `INVALID_STATE`.
Use a new sequence to request a new shutdown. No GUI choice is needed for a
successful transfer.
If systemd rejects the request after the reply, the interface stays open and
shows the failure, `POLL` reports `INTERNAL_ERROR`, and `SHUTTING_DOWN` clears.
Retries of that failed sequence also return `INTERNAL_ERROR`; after fixing the
cause, use a new sequence to retry. Existing command-cache rules still apply.
Power-control work occurs only on shutdown, never in the tracking frame loop.

## Status payload

Status header bytes **16–23** contain `camera_time_ns`: the exact uint64
libcamera `SensorTimestamp` in nanoseconds, on the same sensor clock used by the
tracking JSONL and each video's `<recording>.frames.csv`. No recording-relative
offset, software timestamp, or wall-clock conversion is applied. The tracker
and recorder may use different frames/rates; compare their `camera_time_ns`
values to reconstruct ordering. The CSV's `video_frame_index` is zero-based
decoded video order, independent of `frame_sequence`.

This rename changes Python/JSON field names from `sensor_timestamp_ns` to
`camera_time_ns`; the 64-byte wire layout, field offset, units and version are
unchanged. Result age and processing time remain durations; the internal
software clock used to compute them is not transmitted or recorded.

Zero means no camera time is available. Missing sensor metadata during live
capture/recording is an explicit error, never replaced with `time.monotonic_ns()`.
Untimed video playback returns zero with `RESULT_STALE` and invalid eye slots.
Idle responses also use zero. `START_EXPERIMENT`'s incoming Unix UTC datetime
and the system-clock start/end times in `block.json` are lifecycle metadata,
separate from the shared frame clock.

Status packets use type `2`. For a state-command response, header `code` is the
outcome of that command; current service health remains visible in the status
flags. For a `POLL`, `code` reports current service health. This lets a cleanup
`STOP_TRACKING` succeed while the `ERROR` flag preserves an earlier capture or
tracking fault until `CLEAR_ERROR` succeeds. Header `code` values are:

| Code | Meaning |
|---:|---|
| 0 | OK |
| 1 | bad magic |
| 2 | incompatible version |
| 3 | wrong packet type/length |
| 4 | bad CRC |
| 5 | unknown command |
| 6 | command invalid in current state |
| 7 | internal/camera/tracker error |
| 8 | command sequence conflict |

Header status flags:

| Bit | Meaning |
|---:|---|
| 0 | tracking |
| 1 | recording |
| 2 | ROI configuration loaded |
| 3 | error |
| 4 | latest result stale or not produced yet |
| 5 | experiment active in Pi Tracker |
| 6 | calibration screen active in Pi Tracker |
| 7 | clean system shutdown in progress (`SHUTTING_DOWN`) |

The three Pi Tracker flags are clear in standalone `run`/`preview`. An active experiment
can be idle with both `TRACKING` and `RECORDING` clear. `CALIBRATING` describes
the configuration view, not valid pupil telemetry. Update clients that reject
unknown status flag bits to accept bits 5, 6, and 7 before connecting to Pi Tracker.

`RECORDING` indicates active video saving. The disk guard checks the recording
filesystem before starting video and every 10 seconds thereafter. Below 10 GB
free (or a higher `recording.minimum_free_gib` reserve), video stops, `RECORDING`
clears, and pupil tracking/UART/position logging continue. Low space alone does
not set `ERROR`. The Pi shows/logs a warning. After freeing space, an explicit
`START_RECORDING` opens a new video/log pair; it does not resume automatically.
No new command or packet field is needed for this behavior.

Two fixed eye slots begin at offsets 24 and 36. Only the first
`configured_eye_count` slots are configured.

| Eye-relative offset | Size | Field |
|---:|---:|---|
| 0 | 4 | signed Q16.16 crop-local x |
| 4 | 4 | signed Q16.16 crop-local y |
| 8 | 2 | unsigned Q12.4 equal-area pupil diameter |
| 10 | 1 | confidence `0..255` |
| 11 | 1 | eye flags |

### Coordinate orientation

X/y are crop-local **analysis-stream pixels**, with origin at the top-left of the
oriented eye crop, +x right and +y down. Calibration `camera_rotations` contains
four clockwise angles (0/90/180/270), indexed by the original left-to-right camera
order. The camera crop is rotated before detection; preview, recorded tracking
coordinates and UART all use that same orientation. Packet size/version and
Q16.16 precision are unchanged. Existing calibrations without rotations use 0°.

For raw crop width `W`, height `H`, and pixel center `(x, y)`, the corresponding
oriented coordinates are:

| Clockwise angle | Oriented position | Crop size |
|---:|---|---|
| 0° | `(x, y)` | `W × H` |
| 90° | `(H - 1 - y, x)` | `H × W` |
| 180° | `(W - 1 - x, H - 1 - y)` | `W × H` |
| 270° | `(y, W - 1 - x)` | `H × W` |

Do not apply the rotation a second time downstream. These are pixel positions,
not centered displacements or gaze angles. Use a gaze calibration matching the
saved orientation. Rotation angles are stored in calibration and JSONL session
metadata, not repeated in each UART packet. Video pixels may be at a different
resolution from analysis: JSONL schema 3 records both oriented crop sizes under
`eye_coordinate_systems` and reversible recording regions under `video_layout`.

Eye flags are `VALID=bit0`, `BLINK=bit1`, and `LOST=bit2`. Valid and blink
cannot both be set. Blink/lost diameter is exactly zero. X/y are last reliable
coordinates during invalid periods and must be ignored unless `VALID` is set.

Status tail:

| Offset | Size | Field |
|---:|---:|---|
| 48 | 4 | random service-start ID; changes after process restart |
| 52 | 2 | cumulative analysis frames dropped by the size-one queue, saturated |
| 54 | 2 | latest tracker processing time in microseconds, saturated |
| 56 | 2 | result age in milliseconds at packet construction, saturated |
| 58 | 1 | configured eye count, 1 or 2 (`0` before configuration) |
| 59 | 1 | reserved, zero |

## UART request/response timing

The preview's **UART valid unique** rate counts distinct fresh valid positions
per eye after a complete status packet has been written successfully on the Pi.
It uses two-second monotonic-time windows and excludes repeated frames, stale
results, blinks, and invalid detections. It is a local transmission measurement,
not a receiver acknowledgment. The metric adds no UART traffic or wire-format
changes. **results Hz** includes invalid detections; **compute ms/frame** is
processing duration, so 20 ms/frame is consistent with about 50 results/s.

The controller sends one complete command, then reads one complete status. The
status acknowledges that command in header offset 12 after the action has been
applied (or rejected):

```text
controller                         eye tracker
START(seq=41)  ----------------->  apply command
               <-----------------  status echo=41
POLL(seq=42)   ----------------->  sample latest result
               <-----------------  status echo=42
```

Controller rules:

1. Write and read complete 64-byte records; scan for `ET` to recover framing
   after a process restart or corrupt byte. Ignore a complete valid response
   whose echoed sequence does not match the current request and keep reading
   until the transaction deadline. A response with sequence zero and a protocol
   error code reports a request that could not be decoded; surface it immediately
   so the caller can retry rather than waiting for the full deadline.
2. Validate magic, version, packet type, reserved fields, state flags, and CRC
   before use.
3. Poll at 100 Hz. Deduplicate using service-start ID, frame sequence, and
   sensor timestamp so a restarted service cannot resemble an old sample.
4. Acquisition state changes are synchronous and can include camera/encoder startup or
   shutdown. Experiment upload runs after the acknowledgement and may take longer.
   Wait up to 10 seconds, then retry the identical state command if
   no matching response arrives; never use a later `POLL` as proof of success
   and never reuse its sequence for different content.
5. Treat `ERROR`, `RESULT_STALE`, invalid eye flags, a changed service-start ID,
   or an old result age as unusable data.
6. Do not interpret a response as an experimental sample unless `TRACKING` is
   set, its frame sequence is new, and its configured eye slot is `VALID`.

At 460800 baud, one 64-byte 8-N-1 record takes about 1.39 ms. A command/status
pair takes about 2.78 ms of sequential wire time, so polling at 100 Hz uses
about 28% and avoids phase-locking against a 30 Hz camera. The theoretical
request/response ceiling is `baud / 1280` polls/s before software or timing
margin. Thus 100 Hz cannot fit at 115200 baud; the included CLI rejects such a
combination. Keep the configured rate comfortably below the theoretical limit.

## Physical UART link

The default is `/dev/ttyAMA10` on each Pi 5's dedicated three-pin UART
connector. Connect tracking-Pi TX to controller-Pi RX, tracking-Pi RX to
controller-Pi TX, and ground to ground. Verify the connector's TX/RX/GND
labels; both boards use 3.3 V logic, and their 5 V rails must not be connected
for this link.

Enable serial hardware and disable the serial login console with
`raspi-config`. Ensure no console/getty owns `/dev/ttyAMA10`, reboot, and
confirm the device. See the
[Raspberry Pi UART documentation](https://www.raspberrypi.com/documentation/computers/configuration.html#configure-uarts).
The tracking service opens the port in raw 8-N-1 mode. The controller client
must use the same baud.

The optional GPIO14/TX (physical pin 8) and GPIO15/RX (pin 10) link uses
`dtoverlay=uart0-pi5` and `/dev/ttyAMA0` instead. To use that wiring, explicitly
set `transport.uart_device` on the tracker and `--device` on the controller.

`preview --video` and `run --video` expose the same request/response protocol
as live tracking. When an adjacent `.frames.csv` exists, they replay its original
`camera_time_ns` values (which repeat when the video loops). Without the index,
visual detection still runs, but camera time is zero and UART results are stale
with invalid eye slots. Container PTS and host time are never substituted.
The receiver must still poll; video playback does not send unsolicited packets.

This direct UART design avoids an intermediary. I2C address conflicts would be
solvable with a normal unused address such as `0x42`; address `0x00` is reserved
for the I2C general call, so first verify whether an existing device report means
address zero or bus zero. The pinned stock Pi 5 kernel must also be checked for
`CONFIG_I2C_SLAVE` (it is absent from the currently evaluated image); using I2C
target mode would require a separately qualified kernel and target backend.
