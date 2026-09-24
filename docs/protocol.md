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
| 16 | 8 | controller monotonic ns for commands; camera `SensorTimestamp` ns for status |

Bytes 60-63 contain the little-endian CRC-32C.

## Commands

Command packets use type `1`. Bytes 24-59 are zero in version 1.

| Code | Name | Meaning |
|---:|---|---|
| 0 | `POLL` | fetch telemetry; response echoes this poll's request sequence |
| 1 | `START_TRACKING` | start camera, optional configured recording, and analysis |
| 2 | `STOP_TRACKING` | stop analysis/camera and flush any active recording |
| 3 | `START_RECORDING` | start recording while tracking is active |
| 4 | `STOP_RECORDING` | flush recording while tracking continues |
| 5 | `CLEAR_ERROR` | clear an idle error state |

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

Connect only three signals: tracking-Pi GPIO14/TX (physical pin 8) to
controller-Pi GPIO15/RX (pin 10), tracking-Pi GPIO15/RX (pin 10) to
controller-Pi GPIO14/TX (pin 8), and ground to ground. Both boards use 3.3 V
logic; never connect their 5 V rails for this link.

On each Pi 5, add `dtoverlay=uart0-pi5` to `/boot/firmware/config.txt`, ensure a
serial console/getty is not attached to `/dev/ttyAMA0`, reboot, and confirm the
device. The tracking service opens the port in raw 8-N-1 mode. The controller
client must use the same baud.

This direct UART design avoids an intermediary. I2C address conflicts would be
solvable with a normal unused address such as `0x42`; address `0x00` is reserved
for the I2C general call, so first verify whether an existing device report means
address zero or bus zero. The pinned stock Pi 5 kernel must also be checked for
`CONFIG_I2C_SLAVE` (it is absent from the currently evaluated image); using I2C
target mode would require a separately qualified kernel and target backend.
