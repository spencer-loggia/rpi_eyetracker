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
- Keep several hours of bitrate plus margin free; the service also enforces
  `recording.minimum_free_gib`.
- Use the official active cooler or equivalent.
- During burn-in, log `vcgencmd measure_temp` and `vcgencmd get_throttled`.
- Verify that every MKV and JSONL sidecar closes cleanly after commanded stop.

## Service account

The runtime user needs access to the camera, GPIO (when used), recording
directory, and `/dev/ttyAMA0`. Add `dtoverlay=uart0-pi5` to
`/boot/firmware/config.txt` and make sure no serial console or getty owns that
UART. Prefer a dedicated group/udev rule over running the tracker as root.

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

That command intentionally does not record. During an actual command-service
run, add `--preview` to `ExecStart` or set `preview.enabled` to `true`. Closing
the preview window does not stop tracking or recording.

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
