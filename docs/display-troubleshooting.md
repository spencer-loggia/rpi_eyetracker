# Pi display blackouts

First distinguish the camera pane from the whole monitor. The tracker and UART
workers run independently of the display, so continued telemetry does not identify
the cause of a blackout.

## Camera pane goes black; window controls remain visible

The launcher previously cleared its image whenever a status or error changed.
If the renderer had stopped producing frames, this left the pane blank. It now
retains the last frame across those updates and displays the error. Switching
experiments or image sources still clears the image to avoid showing a previous
subject's image under the new subject's name.

The retained image may be stale. Check the dashboard frame number and any
`Preview:` message. Display rendering uses a separate process and bounded queues;
this fix adds no tracking-frame processing. It does not restart tracking or
recording to recover the display.

## Entire monitor goes dark, including window controls

`pi_tracker.py`, `eye-tracker run`, and `eye-tracker preview` now hold temporary
power inhibitors until application cleanup finishes, including when tracking is
paused. A separate helper holds a logind **idle + sleep** block inhibitor; Pi OS
labwc's swayidle respects the idle inhibition. On X11, the helper also holds
`XScreenSaverSuspend` on its own display connection, covering screensaver and DPMS
timers. Both leases release on exit or parent-pipe closure, without rewriting
desktop settings. The inhibitor does not block the UART shutdown command.
These mechanisms are described by
[systemd's inhibitor documentation](https://github.com/systemd/systemd/blob/main/docs/INHIBITOR_LOCKS.md),
[swayidle's logind idle handling](https://github.com/swaywm/swayidle/blob/1.8.0/main.c),
and [the XScreenSaver API](https://xorg.freedesktop.org/archive/X11R7.5/doc/man/man3/Xss.3.html).

A warning popup/log reports failure to acquire protection or a helper that exits
unexpectedly. Keep the desktop session environment when running as root;
X11 needs `DISPLAY`, `XAUTHORITY`, and access to that display. X11 support needs
`libX11` and `libXss` (`libxss1` on Pi OS). Other Wayland desktops may use different
idle mechanisms. Automatic blanking protection does not prevent physical HDMI,
monitor-power, or display-driver failures.

Try moving the mouse. If that restores the desktop, check desktop screen
blanking first. When enabled, Raspberry Pi OS defaults to blanking after ten
minutes without user input. Camera updates and UART requests are not user input.
On the Pi desktop, open **Preferences → Control Centre → Display** and turn
**Screen Blanking** off. Older versions call the tool **Raspberry Pi
Configuration**. Alternatively, use `sudo raspi-config` → **Display Options →
Screen Blanking → No**. These settings are documented in the
[official Raspberry Pi screen-blanking guide](https://www.raspberrypi.com/documentation/computers/configuration.html#configure-screen-blanking).

The settings above are a persistent fallback if temporary inhibition is not
available. `consoleblank` configures text consoles and does not fix desktop
blanking.

If it happens with screen blanking disabled, note whether the monitor reports
**No signal**, whether its own menu still works, and whether moving the mouse has
any effect. Test another HDMI cable or monitor if available. Save the kernel log
after an incident for display-driver, HDMI, and power-related errors:

```bash
sudo journalctl -b -k -o short-monotonic --no-pager > display-kernel.log
```

Monotonic log timestamps remain useful when UART experiment commands change the
system wall clock. The recent clock-diagnostic and UART shutdown changes contain
no display blanking or monitor-mode commands. Shutdown requires an explicit valid
UART `SHUTDOWN` command and stops tracking before requesting power-off. A monitor
that blacks out while fresh telemetry continues is therefore not evidence that
shutdown ran.

Whole-monitor failures cannot be reproduced by desktop unit tests; a kernel log
and the observations above are needed before attributing them to the application,
desktop, power supply, or HDMI hardware.
