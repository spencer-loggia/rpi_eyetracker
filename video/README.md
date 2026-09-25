# Arducam quad-camera recording

`recorder.py` sends the Arducam B0267 aggregate frame directly from
`libcamera`/`rpicam-vid` to H.264 and disk. Python never receives or copies a
frame.

The HAT's default four-in-one mode is one horizontal `5120x720` RAW8 image:
four synchronized `1280x720` camera images side by side. The default config
uses the Pi ISP to scale that to `3840x540`, retaining all four images at
`960x540` each. This keeps the software encode near the pixel count of 1080p.

## Use from Python

Start is non-blocking. Always stop in a `finally` block so the encoder flushes:

```python
from video import record

recording = record("/mnt/video/session_001.h264")
try:
    run_time_sensitive_operation()
finally:
    recording.stop()
```

Or use a context manager:

```python
with record("/mnt/video/session_001.h264"):
    run_time_sensitive_operation()
```

For a fixed duration:

```python
recording = record("/mnt/video/test.h264", duration_seconds=10)
saved_path = recording.wait()
```

For the intended eight-hour session, passing a duration also enables a free-space
check before the camera starts. The recording can still be stopped early:

```python
recording = record("/mnt/nvme/session_001.h264", duration_seconds=8 * 60 * 60)
try:
    run_time_sensitive_operation()
finally:
    recording.stop()
```

The recorder refuses to overwrite a file unless `overwrite=True` is passed.
Set `exposure_us` in `video/example_config.json` to the required shutter time in
microseconds. It must remain shorter than one frame period (for example, below
16,667 microseconds at 60 fps). The same setting is used by `live_preview.py`,
so its image matches the recorder.

## Live preview

Run the lightweight four-camera preview with:

```bash
python -m video.live_preview
```

It requests the configured camera mode and frame rate, displays the complete
four-camera strip in a `1280x180` native preview window, and shows the measured
FPS in the title bar. Close the window to exit. The script replaces itself with
`rpicam-hello`, so there is no Python frame loop, copying, or encoding. Override
the display size with `--width` and `--height`; this changes only the preview
stream, not the selected sensor mode.

## Command-line check

Inspect the command without touching the camera:

```bash
python -m video.recorder /mnt/video/test.h264 --duration 10 --dry-run
```

Record ten seconds:

```bash
python -m video.recorder /mnt/video/test.h264 --duration 10
```

The `.h264` elementary stream is intentional: it has no muxing process, it is
still usable if recording is interrupted, and it minimizes work during the
experiment. Remux afterward without re-encoding if MP4 is wanted:

```bash
ffmpeg -fflags +genpts -r 30 -i test.h264 -c:v copy test.mp4
```

## Raspberry Pi setup

This camera is an Arducam Pivariety device and needs Arducam's matching driver
and `rpicam-apps`. Follow the current Arducam CamArray quick-start guide. For a
Pi 5 connected to CAM0, the relevant `/boot/firmware/config.txt` overlay is:

```text
dtoverlay=arducam-pivariety,cam0
```

After rebooting, verify the camera and its aggregate mode before using this
module:

```bash
rpicam-hello --list-cameras
rpicam-vid --list-cameras
```

The HAT normally starts in synchronized four-in-one mode. If composition mode
was changed previously, Arducam documents value `0x00` as four-in-one; on a Pi
5 CAM0 their documented I2C bus is 6:

```bash
sudo i2cset -y 6 0x24 0x24 0x00
```

Only run that command on a kit whose `i2cdetect -l` output matches Arducam's
composition-mode instructions.

## Efficiency choices

- `--nopreview`: no display or GUI work.
- `--shutter`: applies `exposure_us` from `example_config.json` in microseconds.
- `--saturation 0`: fixes the camera output to grayscale before encoding.
- `--denoise cdn_off`: disables the extra colour-denoise pass.
- `--codec libav --libav-video-codec libx264`: explicitly selects software
  H.264 on Pi 5.
- `preset=ultrafast;crf=24`: prioritizes the least CPU-intensive x264 preset
  and uses constant-quality rate control.
- `maxrate=64000000;bufsize=128000000`: limits CRF excursions. Even if the
  64 Mbit/s ceiling were sustained for eight hours, video would be about
  230.4 GB, leaving roughly 281.6 GB of a nominal 512 GB disk before filesystem
  and other usage. Typical CRF 24 output should be smaller.
- `--low-latency`: also selects x264's `zerolatency` behavior on Pi 5.
- `taskset --cpu-list 1,2,3`: keeps capture and encode threads off the
  repository's timing-critical CPU 0.
- `nice -n 5`: makes the recorder yield to higher-priority experimental work.
- A fixed-duration call reserves the configured maximum bitrate plus 20%
  additional headroom rather than guessing the scene-dependent CRF average.

Pi 5 has no hardware H.264 encoder. The ISP scaling is hardware-assisted, but
H.264 necessarily consumes CPU. Raspberry Pi's published 1080p30 measurements
put low-latency encoding at roughly 0.7-0.8 of one CPU core at comparable
bitrates; scene content changes the exact number.

Skipping the encoder is not viable for an eight-hour session on a 512 GB drive.
The current 60 fps `3840x540` YUV420 stream would be about 5.37 TB uncompressed,
full `5120x720` RAW8 about 6.37 TB, and even the camera's lowest native
`2560x400` RAW8 four-feed mode about 1.77 TB. MJPEG is also software-encoded on
Pi 5 and normally trades substantially more disk space for no dependable CPU
advantage. A materially lower-CPU solution therefore requires either lower
resolution or different hardware with an encoder (for example, a supported
Jetson platform); there is no unused Pi 5 encoder that another API can expose.

To retain the native `1280x720` image from every camera, change
`output_width`/`output_height` in `example_config.json` to `5120`/`720`. That is 1.78x
as many pixels as the default and should be load-tested alongside the actual
experiment before use. For an even lighter `640x400` per-camera stream, use
sensor mode `2560:400:8` and output size `2560x400`.

The eight-hour intermediate is already H.264-compressed but intentionally uses
a fast encoder preset. After the experiment, it can be archived more tightly
with a slower encoder. This cannot restore detail discarded during acquisition;
it only reduces the stored size. It is a full re-encode and should not run
during the time-sensitive task:

```bash
ffmpeg -r 30 -i session_001.h264 -c:v libx265 -preset slow -crf 24 session_001_archive.mp4
```

Sources:

- [Arducam B0267 specifications](https://www.arducam.com/arducam-1mp4-quadrascopic-camera-bundle-kit-for-raspberry-pi-nvidia-jetson-nano-xavier-nx-four-ov9281-global-shutter-monochrome-camera-modules-and-camarray-camera-hat.html)
- [Arducam CamArray quick start](https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamArray/quick-start/)
- [Raspberry Pi camera software](https://www.raspberrypi.com/documentation/computers/camera_software.html)
