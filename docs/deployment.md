# Raspberry Pi deployment checklist

## Pin the acquisition environment

Record at least:

```bash
cat /etc/os-release
uname -a
vcgencmd version
rpicam-hello --version
python3 -c 'import picamera2; print(picamera2.__version__)'
dpkg -l | grep -E 'arducam|libcamera|picamera|opencv|python3-av'
```

Arducam's Pivariety package is tied to kernel versions. Qualify upgrades on a
spare image before applying them to the experiment system.

## Storage and cooling

- Prefer an NVMe filesystem for long sessions.
- Keep several hours of bitrate plus margin free. Before video starts and every
  10 seconds while saving, the service checks the output filesystem and enforces
  a 10 GB floor or a higher `recording.minimum_free_gib` reserve. A cutoff finalizes
  video in the background and opens a warning popup; pupil tracking, UART, and
  position logs continue. Free space and explicitly restart recording to resume.
- Use the official active cooler or equivalent.
- During burn-in, log `vcgencmd measure_temp` and `vcgencmd get_throttled`.
- Verify that every MKV and JSONL sidecar closes cleanly after commanded stop.

## Experiment transfer storage

Pi Tracker's optional `sync.remote_data_url` names an already mounted NFS/NFS4
directory. Mount it before launching; the application does not create a mount
or prompt for credentials. Install the OS tools with
`sudo apt install rsync util-linux nfs-common`. The remote root must contain at
least one entry, matching the `neuro_tasks` mount check. The application creates
`pitracker/` below it and preserves the subject/experiment/block layout there.

Give the launcher's effective NFS identity write/create permission on this
directory. NFS root squashing can deny a root-run launcher even when local root
can access everything on the Pi. The rsync command omits owner/group preservation
to avoid unnecessary chown failures; it still preserves file modes and times.
Validate transfer permissions with a disposable experiment before acquisition.

Transfers occur after experiment closure, with acquisition held off until the
transfer finishes or is cancelled. Successful uploads remove their local
experiment folders; failures retain data and log the destination and error in
`<data-root>/interface_sync.log`. The small `<data-root>/.history/` cache preserves
calibration reuse and numbering. Keep that cache when moving the local data root.

Before deployment, exercise an unavailable mount and interrupted transfer to
confirm the GUI warning and shutdown Retry / Shut down anyway / Cancel choices
on the target Pi. A systemd service stop timeout must allow for recording
finalization and the desired upload duration; active transfers deliberately have
no total deadline. The standalone tracking service below does not perform
launcher experiment synchronization.

## 60 Hz acceptance test

Test the final camera mode, ROIs, illumination, recording target, and cooling
together; a development-machine benchmark cannot prove Pi performance.

- Run tracking and recording for longer than the intended warm-up period while
  polling UART at 100 Hz.
- Require `dropped_analysis_frames` to remain zero and verify consecutive
  `frame_sequence` values in the JSONL sidecar. Also verify the elapsed sensor
  timestamps: consecutive application sequences alone cannot detect a camera
  request loop slowed by encoding.
- Require tracker p99 processing time to stay below the 16.67 ms frame period,
  with thermal throttling absent.
- Compare the decoded video-frame count with the rows in its `.frames.csv` and
  the duration with that index's `camera_time_ns` span. Eye-trace JSONL rows can
  legitimately differ in number/rate; both use the same raw sensor clock.
  Inspect `session_end.encoder_stop_error` for encoder/mux/index failures.
- For the operational target of about 50 Hz valid pupil output, watch **UART
  valid unique** per eye. It counts unique valid positions written by the Pi,
  excluding duplicate polls and invalid/stale results, over two-second windows.
  This does not require validating the receiver. The stricter 60 Hz checks above
  are for qualifying full camera throughput.
- Treat preview `display` Hz as diagnostic only. **results** Hz and the analysis
  drop counter describe acquisition/detection even when display frames are
  skipped. **compute** is milliseconds per frame: 20 ms is consistent with 50 Hz.
- If encoding contends with tracking, reduce `recording.fps` or recording
  dimensions while leaving `camera.fps` and the analysis dimensions unchanged.
  Keep recording at 30 fps or higher. The default profile records the first two
  full camera views and eye crops at a 60 fps target, with 5120x720/60 Hz analysis.
  Its encoding worker has a bounded queue and reports overload as an error.
  Test 60 fps first; reduce only recording to 30 fps if needed.
  Rate reduction uses Picamera2's pre-encode frame skip; resolution reduction
  uses a libcamera ISP stream, so neither path performs Python resizing.

To isolate recording overhead, compare these commands sequentially, stopping
each before starting the next:

```bash
eye-tracker run --start-tracking --preview --no-recording
eye-tracker run --start-tracking --preview
```

Use the same scene and compare **results** Hz and **UART valid unique** with the
usual task polling. The first command disables
automatic recording but retains the same camera, detector, preview, and UART
configuration. Explicit downstream `START_RECORDING` commands still enable
recording, so exclude those from the first measurement. Confirm that the second
run's video timestamps sustain at least 30 fps as well as checking the tracking rate.

## Service account

The runtime user needs access to the camera, GPIO (when used), recording
directory, and `/dev/ttyAMA10`, the Pi 5 dedicated three-pin UART connector.
Enable the serial hardware and disable the serial login console using
`raspi-config`; make sure no serial console or getty owns that UART, then
reboot and verify the device. The `uart0-pi5` overlay is for the separate
GPIO14/15 UART and is not needed for this connector. Prefer a dedicated
group/udev rule over running the tracker as root. See the
[Raspberry Pi UART documentation](https://www.raspberrypi.com/documentation/computers/configuration.html#configure-uarts).

Example systemd unit (adjust paths and user):

```ini
[Unit]
Description=Head-fixed macaque eye tracking acquisition
After=local-fs.target
RequiresMountsFor=/mnt/nvme/eye-recordings

[Service]
Type=simple
User=eye-tracker
Group=eye-tracker
SupplementaryGroups=video gpio dialout
WorkingDirectory=/opt/macaque-eye-tracker
ExecStart=/opt/macaque-eye-tracker/.venv/bin/eye-tracker run --config /opt/macaque-eye-tracker/config/eye_tracker.uart.json
Restart=on-failure
RestartSec=2
TimeoutStopSec=30
Environment=PYTHONUNBUFFERED=1
UMask=0027
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

Do not set CPU affinity until on-target profiling shows that it improves p99
latency; Pi 5 software H.264 uses multiple threads and simplistic pinning can
make contention worse.

## Attached-display preview

First test the display, live cameras, ROIs, and pupil fit interactively from the
Pi's graphical login:

```bash
eye-tracker preview --config config/eye_tracker.uart.json
```

That command serves tracker results over the configured UART and does not
record. Add `--video session.mkv` to exercise the same downstream link using
a looping recording. During an actual command-service
run, add `--preview` to `ExecStart` or set `preview.enabled` to `true`. Closing
the preview window does not stop tracking or recording in `run`. An enabled
preview opens before a downstream `START_TRACKING` and continues showing live
crops after `STOP_TRACKING`; the idle camera/IR illumination remain active,
while pupil analysis, measurement output, and recording are off. The preview process is
lower priority and consumes only the latest queued frame, so it cannot build a
backlog or block acquisition.

`pi_tracker.py`, `eye-tracker run`, and `eye-tracker preview` hold temporary Linux
idle/sleep inhibitors while open, including idle preview, and release them after
cleanup. Pi OS labwc/swayidle uses the logind idle inhibitor; X11 also gets a
connection-scoped screensaver/DPMS suspension. Failures produce a warning instead
of stopping tracking. Launch from the graphical session with access to its
display; an X11 launch with `sudo` must retain `DISPLAY` and `XAUTHORITY`.
See [display troubleshooting](display-troubleshooting.md) for details.

The system service shown above normally starts outside the logged-in desktop's
graphical session. A monitor being physically attached is not sufficient: the
service user must also have access to the active X11 or Wayland session. Prefer
a systemd *user* service associated with the graphical login and order it after
`graphical-session.target`. If a system service is required, configure the
correct `DISPLAY` or `WAYLAND_DISPLAY` and `XDG_RUNTIME_DIR` for that Pi image
and authorize only the tracker user. Do not copy display environment values
from another account or image without verifying them. Preview startup failures
are printed to the journal but do not stop the acquisition service.

Install and inspect the adjusted unit with:

```bash
sudo install -m 0644 deploy/macaque-eye-tracker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now macaque-eye-tracker.service
systemctl status macaque-eye-tracker.service
journalctl -u macaque-eye-tracker.service -f
```

An internal capture/tracker failure is exposed over the protocol and stops the
camera/recording, but the command server deliberately remains alive for
inspection and restart commands. Consequently `Restart=on-failure` does not
restart the process for that condition. An external watchdog policy should be
qualified separately if automatic process restart is required.
