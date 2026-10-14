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
| 12 | 4 | zero for commands; echoed request sequence for status (`0` if undecodable) |
| 16 | 8 | controller timestamp for commands (see below); camera `SensorTimestamp` ns for status |

Bytes 60-63 contain the little-endian CRC-32C.

## Commands

Command packets use type `1`. Bytes 24-59 are zero except for
`START_EXPERIMENT`. Command flags retain their existing meaning; send zero
unless an extension explicitly specifies otherwise. Header bytes 12-15 must
always be zero. The timestamp at bytes 16-23 is controller monotonic
nanoseconds (or zero) for existing commands and `STOP_EXPERIMENT`. Only
`START_EXPERIMENT` uses **nonzero Unix UTC nanoseconds since 1970-01-01T00:00:00Z**.

| Code | Name | Meaning |
|---:|---|---|
| 0 | `POLL` | fetch telemetry; response echoes this poll's request sequence |
| 1 | `START_TRACKING` | start camera, optional configured recording, and analysis |
| 2 | `STOP_TRACKING` | stop pupil analysis and flush recording; keep live preview if enabled |
| 3 | `START_RECORDING` | start recording while tracking is active |
| 4 | `STOP_RECORDING` | flush recording while tracking continues |
| 5 | `CLEAR_ERROR` | clear an idle error state |
| 6 | `START_EXPERIMENT` | Pi Tracker: set system UTC time and open an experiment for the supplied subject |
| 7 | `STOP_EXPERIMENT` | Pi Tracker: stop activity, flush recording, close the experiment, and return to the opening screen |

### Pi Tracker experiment commands

These commands are supported by the `python pi_tracker.py CONFIG` interface.
The standalone `eye-tracker run` and `eye-tracker preview` services reject
them with `UNKNOWN_COMMAND`; their existing tracking commands are unchanged.
The interface owns the UART throughout all of its screens, so the controller
uses the same port and protocol during setup, calibration, and tracking.

`START_EXPERIMENT` has the following payload:

| Offset | Size | Field |
|---:|---:|---|
| 24 | 1 | subject length, 1–32 ASCII bytes |
| 25 | 32 | subject bytes followed by zero padding |
| 57 | 3 | reserved, zero |

Subject names must match `[A-Za-z0-9][A-Za-z0-9_.-]{0,31}`. They are case
sensitive, are not rewritten, and cannot contain whitespace, slashes, or
non-ASCII characters. All bytes after the subject, up to and including byte
59, must be zero. A missing subject, zero timestamp, invalid name, or nonzero
padding is rejected as `BAD_PACKET_TYPE` before any experiment or clock action.
`STOP_EXPERIMENT` requires the usual all-zero payload.

On a valid start, Pi Tracker sets the system wall clock to the supplied UTC
time and creates a datetime-stamped experiment directory for that subject.
The new experiment inherits the most recent prior experiment's calibration
when available. Experiment files, calibration, tracking data, and videos
remain under the experiment directory. Starting an experiment does not start
eye tracking. If an experiment is already active, a new start command stops
its activity, flushes recording, closes it, and opens the requested experiment.
A stop command when no experiment is active succeeds without creating files.

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
whether its **effective** capability set contains `CAP_SYS_TIME`. An ambient
capability configured on a service is not by itself proof that the GUI process
received it. If the effective capability is present, check service restrictions
such as `ProtectClock` (it must be `no`) and any container/security policy.
An unavailable Python clock API is reported separately from permission denial.
These diagnostics run only after failure and never alter the clock or permissions.

For an offline permission test, close the existing interface and launch from
the project directory with `sudo -E .venv/bin/python pi_tracker.py config/eye_tracker.json`
(adjust `.venv/bin/python` if your environment is elsewhere). A normal root
process on Raspberry Pi OS has clock-setting permission. Files produced during
that run are root-owned. When using a multiline `systemd-run` command instead,
each continuation backslash must be immediately followed by a newline; remove
the backslashes when entering the command on a single line.

Within an active experiment, `START_TRACKING` enters tracking mode and starts
analysis/recording according to the configuration. It is rejected with
`INVALID_STATE` during calibration or without an experiment. The GUI's
**Start tracking** button instead opens the tracking view with tracking idle,
ready for a manual start or `START_TRACKING`. `STOP_TRACKING` leaves the live
view and experiment open. `STOP_EXPERIMENT` also exits calibration or tracking
and returns to the opening screen.

The same sequence cache described below covers experiment commands. Retry a
timed-out start with **identical bytes**, including the original timestamp,
subject, and sequence. A cached duplicate returns its original outcome without
setting the clock again or creating another directory. A fresh sequence is a
new request, even if its subject and timestamp match a previous experiment.
The cache is in memory and resets when Pi Tracker restarts.

Controller examples (ISO 8601 dates require `Z` or an explicit UTC offset):

```sh
eye-tracker-controller start-experiment subject_01 --datetime 2026-10-14T14:30:00Z
eye-tracker-controller start-experiment subject_01 --datetime 2026-10-14T10:30:00-04:00
eye-tracker-controller start-tracking
eye-tracker-controller stop-tracking
eye-tracker-controller stop-experiment
```

Omitting `--datetime` sends the controller's current UTC wall-clock time.
For Python clients, use `controller.start_experiment("subject_01")` or pass
`timestamp_ns=...` explicitly, and use `controller.stop_experiment()` to end it.
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

## Status payload

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

The two new flags are clear in standalone `run`/`preview`. An active experiment
can be idle with both `TRACKING` and `RECORDING` clear. `CALIBRATING` describes
the configuration view, not valid pupil telemetry. Update clients that reject
unknown status flag bits to accept bits 5 and 6 before connecting to Pi Tracker.

Two fixed eye slots begin at offsets 24 and 36. Only the first
`configured_eye_count` slots are configured.

| Eye-relative offset | Size | Field |
|---:|---:|---|
| 0 | 4 | signed Q16.16 crop-local x |
| 4 | 4 | signed Q16.16 crop-local y |
| 8 | 2 | unsigned Q12.4 equal-area pupil diameter |
| 10 | 1 | confidence `0..255` |
| 11 | 1 | eye flags |

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
4. State actions are synchronous and can include camera/encoder startup or
   shutdown. Wait up to 10 seconds, then retry the identical state command if
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
as live tracking, including frame sequences, timestamps, and validity flags.
The receiver must still poll; video playback does not send unsolicited packets.

This direct UART design avoids an intermediary. I2C address conflicts would be
solvable with a normal unused address such as `0x42`; address `0x00` is reserved
for the I2C general call, so first verify whether an existing device report means
address zero or bus zero. The pinned stock Pi 5 kernel must also be checked for
`CONFIG_I2C_SLAVE` (it is absent from the currently evaluated image); using I2C
target mode would require a separately qualified kernel and target backend.
