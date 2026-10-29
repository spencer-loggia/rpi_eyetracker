"""Tk experiment launcher. Camera and tracking workers never call into Tk."""

from __future__ import annotations

import argparse
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH
from .experiments import ExperimentStore
from .orientation import CameraCanvas
from .pi_runtime import ExperimentRuntime
from .power import KeepAwake


class PiTrackerWindow:
    def __init__(self, root, runtime: ExperimentRuntime) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.root, self.runtime = root, runtime
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi-interface")
        self.pending = None
        self.signature = None
        self._display_source = None
        self.photo = None
        self.display_scale = 1.0
        self._calibration_redraw_pending = False
        self.closing = False
        self._feed_failed = False
        self._shown_warning = None
        self._sync_popup = None
        self._sync_popup_kind = None
        self._transport_stop = threading.Event()
        self._transport_thread = None
        root.title("Pi tracker")
        root.geometry("1320x880")
        self.header = ttk.Label(root, font=("TkDefaultFont", 14, "bold"), padding=12)
        self.header.pack(fill="x")
        self.context = ttk.Label(root, padding=(12, 0, 12, 8))
        self.context.pack(fill="x")
        self.controls = ttk.Frame(root, padding=12)
        self.controls.pack(side="bottom", fill="x")
        self.message = ttk.Label(root, padding=8, wraplength=1200)
        self.message.pack(side="bottom", fill="x")
        self.options = ttk.Frame(root, padding=8)
        self.options.pack(side="bottom", fill="x")
        self.canvas = tk.Canvas(root, background="#181818", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<ButtonPress-1>", lambda e: self._mouse(e, 1))
        self.canvas.bind("<B1-Motion>", lambda e: self._mouse(e, 0))
        self.canvas.bind("<ButtonRelease-1>", lambda e: self._mouse(e, 4))
        self.canvas.bind("<ButtonPress-3>", lambda e: self._mouse(e, 2))
        self.canvas.bind("<B3-Motion>", lambda e: self._mouse(e, 0))
        self.canvas.bind("<ButtonRelease-3>", lambda e: self._mouse(e, 5))
        root.bind("<Escape>", lambda _event: self._back())
        root.bind("<Return>", lambda _event: self._calibration_next())
        root.bind("<Key-s>", lambda _event: self._calibration_next())
        root.bind("<Key-u>", lambda _event: self._edit_boxes("undo"))
        root.bind("<Key-c>", lambda _event: self._edit_boxes("clear"))
        for key, action in (("space", "play_video"), ("bracketleft", "frame_back"),
                            ("bracketright", "frame_forward")):
            root.bind(f"<Key-{key}>", lambda _event, a=action: self._video_action(a))
        for index in range(4):
            root.bind(f"<Key-{index + 1}>", lambda _event, i=index: self._rotate_camera(i))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._start_transport()
        root.after(0, self._tick)

    def _start_transport(self) -> None:
        from .transport import ProtocolEngine, UartServer, UnixSocketServer

        transport = self.runtime.base_config.transport
        engine = ProtocolEngine(self.runtime)
        if transport.backend == "none":
            return
        server = (
            UartServer(transport.uart_device, transport.uart_baud, engine)
            if transport.backend == "uart"
            else UnixSocketServer(transport.unix_socket, engine)
        )

        def serve():
            try:
                server.serve(self._transport_stop)
            except Exception as exc:  # noqa: BLE001 - report boundary failures and keep cleanup available
                self.runtime.report_error(f"Command interface unavailable: {exc}")

        self._transport_thread = threading.Thread(target=serve, name="pi-uart", daemon=True)
        self._transport_thread.start()

    def _submit(self, action) -> None:
        if self.pending is None and not self.closing and not self.runtime.view().busy:
            self.pending = self.executor.submit(action)
            self._enable_controls(False)

    def _enable_controls(self, enabled: bool) -> None:
        def visit(widget):
            for child in widget.winfo_children():
                if isinstance(
                    child, (self.ttk.Button, self.ttk.Entry, self.ttk.Combobox, self.tk.Scale)
                ):
                    child.configure(state=("normal" if enabled else "disabled"))
                visit(child)

        visit(self.controls)
        visit(self.options)

    def _button(self, text, action, *, right=False):
        button = self.ttk.Button(self.controls, text=text, command=action)
        button.pack(side="right" if right else "left", padx=5)
        return button

    def _back(self) -> None:
        view = self.runtime.view()
        if self.pending is not None or self.closing or view.busy:
            return
        if view.calibration is not None and view.calibration.phase == "tuning":
            view.calibration.previous()
            self.signature = None
        else:
            self._submit(self.runtime.back)

    def _calibration_next(self) -> None:
        view = self.runtime.view()
        if view.calibration is None or self.pending is not None or view.busy:
            return
        if view.calibration.phase == "boxes":
            try:
                view.calibration.next()
                self.signature = None
            except Exception as exc:  # noqa: BLE001 - report boundary failures and keep cleanup available
                self.message.configure(text=str(exc))
        else:
            self._submit(self.runtime.save_calibration)

    def _edit_boxes(self, action) -> None:
        view = self.runtime.view()
        if (
            self.pending is not None
            or view.busy
            or view.calibration is None
            or view.calibration.phase != "boxes"
        ):
            return
        boxes = view.calibration.roi.boxes
        if action == "clear":
            boxes.clear()
        elif boxes:
            boxes.pop()

    def _rotate_camera(self, index: int) -> None:
        """Apply a quarter turn in the shared editor; persist only when calibration saves."""
        view = self.runtime.view()
        if (self.pending is not None or view.busy or view.calibration is None
                or view.calibration.phase != "boxes"):
            return
        try:
            view.calibration.roi.rotate_camera(index)
            self.signature = None
        except ValueError as exc:
            self.message.configure(text=str(exc))

    def _mouse(self, event, code) -> None:
        view = self.runtime.view()
        if self.pending is None and not view.busy and view.calibration is not None:
            view.calibration.mouse(
                code, round(event.x / self.display_scale), round(event.y / self.display_scale)
            )
            if (view.calibration.phase == "tuning"
                    and view.calibration.tuning.controls.sample is not None):
                self._request_calibration_redraw()

    def _request_calibration_redraw(self) -> None:
        """Coalesce pointer events without waiting for the 67ms camera timer."""
        if not self._calibration_redraw_pending:
            self._calibration_redraw_pending = True
            self.root.after(1, self._redraw_calibration)

    def _redraw_calibration(self) -> None:
        self._calibration_redraw_pending = False
        if self.closing or self.pending is not None:
            return
        view = self.runtime.view()
        if (view.busy or view.calibration is None or view.calibration.phase != "tuning"
                or view.calibration.tuning.controls.sample is None):
            return
        try:
            # No camera reads and no widget/layout rebuilds in the paint path.
            self._display(view.calibration.render(refresh=False))
        except Exception as exc:  # noqa: BLE001 - keep cancellation available after render errors
            self.message.configure(text=str(exc))

    def _control_action(self, action: str) -> None:
        from .calibration_dialogs import confirm_reset_controls

        if action in ("load_video", "play_video", "frame_back", "frame_forward"):
            self._video_action(action)
            return
        view = self.runtime.view()
        if (self.pending is None and not view.busy and view.calibration is not None
                and view.calibration.phase == "tuning"):
            view.calibration.tuning.control_action(
                action, confirm_reset=lambda: confirm_reset_controls(self.root),
            )
            self.signature = None

    def _video_action(self, action="load_video") -> None:
        from .calibration_video import choose_video

        view = self.runtime.view()
        if self.pending is not None or view.busy or view.mode not in ("experiment", "calibration"):
            return
        if action != "load_video" and view.calibration is None:
            return
        path = choose_video(self.root) if action == "load_video" else None
        if action == "load_video" and not path:
            return
        if view.calibration is None:
            self._submit(lambda: self.runtime.begin_calibration(video_path=path))
        else:
            self._submit(lambda: self.runtime.calibration_video_action(action, path=path))

    def _build_controls(self, view) -> None:
        for parent in (self.controls, self.options):
            for child in parent.winfo_children():
                child.destroy()
        self.track_button = None
        self._button("Back", self._back)
        if view.mode == "home":
            self._button(
                "Start experiment", lambda: self._submit(self.runtime.show_subjects), right=True
            )
        elif view.mode == "subjects":
            self.ttk.Label(self.options, text="Select subject").pack(side="left", padx=5)
            names = self.runtime.store.subjects()
            subject = self.tk.StringVar(value=names[0] if names else "")
            selector = self.ttk.Combobox(
                self.options, textvariable=subject, values=names, width=32
            )
            selector.pack(side="left", padx=5)

            def open_subject():
                name = subject.get()  # Tk variables are read on the Tk thread.
                self._submit(lambda: self.runtime.start_experiment(name))

            self._button("Open experiment", open_subject, right=True)
            self.ttk.Label(self.options, text="New subject").pack(side="left", padx=5)
            new_subject = self.tk.StringVar()
            self.ttk.Entry(self.options, textvariable=new_subject, width=28).pack(
                side="left", padx=5
            )

            def add_subject():
                try:
                    self.runtime.store.add_subject(new_subject.get())
                    selector.configure(values=self.runtime.store.subjects())
                    subject.set(new_subject.get())
                    new_subject.set("")
                except Exception as exc:  # noqa: BLE001 - report boundary failures and keep cleanup available
                    self.message.configure(text=str(exc))

            self.ttk.Button(self.options, text="Add subject", command=add_subject).pack(
                side="left", padx=5
            )
        elif view.mode == "experiment":
            self._button("Calibrate", lambda: self._submit(self.runtime.begin_calibration))
            self._button("Calibrate video", self._video_action)
            self._button(
                "Start tracking", lambda: self._submit(self.runtime.open_tracking), right=True
            )
            self._button("End experiment", lambda: self._submit(self.runtime.end_experiment))
        elif view.mode == "tracking":
            self.track_button = self._button(
                "Start tracking / saving",
                lambda: self._submit(self.runtime.toggle_tracking),
                right=True,
            )
            self._button("End experiment", lambda: self._submit(self.runtime.end_experiment))
        elif view.mode == "calibration":
            calibration = view.calibration
            toolbar_rows = 1
            if calibration.phase == "boxes":
                self._button("Load video", self._video_action)
                self._button("Undo box", lambda: self._edit_boxes("undo"))
                self._button("Clear boxes", lambda: self._edit_boxes("clear"))
                self._button("Next: pupil tuning", self._calibration_next, right=True)
                for camera, angle in enumerate(calibration.roi.camera_rotations):
                    self.ttk.Button(
                        self.options, text=f"Cam {camera}: {angle}° ↻",
                        command=lambda i=camera: self._rotate_camera(i),
                    ).grid(row=0, column=camera, padx=4, sticky="ew")
                video = getattr(calibration, "video", None)
                if video is not None and video.camera is not None:
                    toolbar_rows = 2
                    for index, (label, action) in enumerate(video.buttons()[1:]):
                        self.ttk.Button(
                            self.options, text=label, command=lambda a=action: self._video_action(a),
                        ).grid(row=1, column=index, padx=4, sticky="ew")
            else:
                self._button("Save calibration", self._calibration_next, right=True)
                buttons = calibration.tuning.control_buttons()
                toolbar_rows = (len(buttons) + 3) // 4
                for index, (label, action) in enumerate(buttons):
                    self.ttk.Button(
                        self.options, text=label, command=lambda a=action: self._control_action(a),
                    ).grid(row=index // 4, column=index % 4, padx=3, pady=2, sticky="ew")
            self._button("Cancel calibration", lambda: self._submit(self.runtime.back))
            for index, (label, slider, value, callback) in enumerate(calibration.sliders()):
                control = self.tk.Scale(
                    self.options,
                    label=label,
                    from_=slider.minimum_position,
                    to=slider.maximum_position,
                    orient="horizontal",
                    length=260,
                )
                control.set(slider.position_for(value))
                control.configure(
                    command=lambda position, cb=callback: cb(round(float(position)))
                )
                row = index // 4 + toolbar_rows
                control.grid(row=row, column=index % 4, padx=4, sticky="ew")
        if view.error_message:
            self._button("Clear error / retry camera", self._retry)
        self._enable_controls(self.pending is None)

    def _retry(self):
        self._feed_failed = False
        self.signature = None
        self._submit(self.runtime.clear_error)

    def _display(self, frame) -> None:
        import cv2

        height, width = frame.shape[:2]
        scale = min(
            1.0,
            max(1, self.canvas.winfo_width()) / width,
            max(1, self.canvas.winfo_height()) / height,
        )
        self.display_scale = scale
        if scale < 1:
            frame = cv2.resize(
                frame, (max(1, round(width * scale)), max(1, round(height * scale)))
            )
        if frame.ndim == 2:
            # Tk understands binary PGM directly; no Pillow dependency needed.
            data = f"P5\n{frame.shape[1]} {frame.shape[0]}\n255\n".encode() + frame.tobytes()
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            data = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode() + rgb.tobytes()
        self.photo = self.tk.PhotoImage(data=data, format="PPM")
        self.canvas.itemconfigure(self.image_item, image=self.photo)

    def _show_warning(self, message):
        """Open a nonmodal warning so camera rendering and UART control stay live."""
        if not message:
            self._shown_warning = None
            return
        if message == getattr(self, "_shown_warning", None):
            return
        self._shown_warning = message
        popup = self.tk.Toplevel(self.root)
        popup.title("Pi Tracker warning")
        popup.transient(self.root)
        self.ttk.Label(popup, text=message, wraplength=560, padding=20).pack()
        self.ttk.Button(popup, text="OK", command=popup.destroy).pack(pady=12)
        popup.lift()

    def _sync_dialog(self, view) -> None:
        """Render progress and cleanup choices on Tk's thread; transfer workers use no Tk."""
        kind = ("progress" if getattr(view, "sync_active", False) else
                "decision" if getattr(view, "shutdown_decision", False) else None)
        if kind != getattr(self, "_sync_popup_kind", None):
            if getattr(self, "_sync_popup", None) is not None:
                self._sync_popup.destroy()
            self._sync_popup = None
            self._sync_popup_kind = kind
            if kind is not None:
                popup = self.tk.Toplevel(self.root)
                self._sync_popup = popup
                popup.title("Experiment data sync" if kind == "progress" else "Shutdown cleanup")
                popup.transient(self.root)
                popup.grab_set()
                self._sync_label = self.ttk.Label(popup, wraplength=640, padding=20)
                self._sync_label.pack()
                if kind == "progress":
                    action = self.runtime.cancel_sync
                    self._sync_cancel_button = self.ttk.Button(popup, text="Cancel Sync", command=action)
                    self._sync_cancel_button.pack(pady=12)
                else:
                    def choose(value):
                        self._submit(lambda: self.runtime.resolve_sync_failure(value))
                    for label, value in (("Retry", "retry"), ("Shut down anyway", "shutdown"),
                                         ("Cancel shutdown", "cancel")):
                        self.ttk.Button(popup, text=label,
                                        command=lambda v=value: choose(v)).pack(pady=8)
                    action = lambda: choose("cancel")
                popup.protocol("WM_DELETE_WINDOW", action)
                popup.bind("<Escape>", lambda _event: action())
        if kind == "progress":
            self._sync_label.configure(text=view.sync_message)
            self._sync_cancel_button.configure(
                state="disabled" if view.sync_message.startswith("Finalizing sync") else "normal",
            )
        elif kind == "decision":
            self._sync_label.configure(text=(
                f"{view.sync_report.summary}\n\n{view.sync_report.detail}\n\n"
                f"Details saved to {self.runtime.store.data_root / 'interface_sync.log'}."
            ))

    def _tick(self) -> None:
        if self.closing:
            self._finish_close()
            return
        if self.pending is not None and self.pending.done():
            try:
                self.pending.result()
                self._feed_failed = False
            except Exception as exc:  # noqa: BLE001 - report boundary failures and keep cleanup available
                self.runtime.report_error(str(exc))
                self._feed_failed = True
            self.pending = None
            self.signature = None
        view = self.runtime.view()
        self._sync_dialog(view)
        warning = getattr(view, "warning_message", None)
        self._show_warning(warning)
        if view.shutdown_submitted:
            self.close()
            self.root.after(67, self._tick)
            return
        if view.shutdown_requested:
            self._enable_controls(False)
            self.message.configure(text=(getattr(view, "sync_message", "") or
                                         "Shutting down: finalizing recordings and powering off…"))
            self.root.after(67, self._tick)
            return
        signature = (
            view.mode,
            view.experiment,
            getattr(view, "block", None),
            id(view.calibration),
            getattr(view.calibration, "phase", None),
            getattr(view.calibration, "control_signature", None),
            view.error_message,
        )
        try:
            source = (
                view.mode, view.experiment, id(view.feed), id(view.preview), id(view.calibration)
            )
            if source != self._display_source:
                # A status/error update must not erase the last camera frame:
                # a failed renderer may never supply a replacement. Clear only
                # when changing sources, so another experiment is never shown.
                self.canvas.itemconfigure(self.image_item, image="")
                self._display_source = source
            if signature != self.signature:
                self.signature = signature
                self._build_controls(view)
                title = {
                    "home": "Pi tracker",
                    "subjects": "Select subject",
                    "experiment": "Experiment",
                    "tracking": "Tracking",
                    "calibration": "Calibration",
                }[view.mode]
                self.header.configure(text=title)
                context = (
                    "Live view · all four cameras"
                    if view.experiment is None
                    else f"Subject: {view.experiment.subject}    Experiment: {view.experiment.name}"
                )
                if getattr(view, "block", None) is not None:
                    context += f"    Block: {view.block.number}"
                self.context.configure(text=context)
                self.root.title(f"Pi tracker — {context}")
                self.message.configure(text=view.error_message or "")
            self._enable_controls(self.pending is None and not view.busy)
            if (view.feed is None and view.preview is None and view.calibration is None
                    and not self._feed_failed):
                self._submit(self.runtime.ensure_feed)
            if self.pending is None and not view.busy:
                if view.calibration is not None:
                    self._display(view.calibration.render())
                    if view.calibration.phase == "tuning":
                        tuning = view.calibration.tuning
                        self.message.configure(text=view.error_message or tuning._status or tuning.controls.status)
                elif view.preview is not None:
                    frame = view.preview.read_rendered()
                    if frame is not None:
                        self._display(frame)
                    if view.preview.error_message:
                        self.message.configure(text=f"Preview: {view.preview.error_message}")
                elif view.feed is not None and view.feed.image is not None:
                    image = view.feed.image
                    if any(view.camera_rotations):
                        image = CameraCanvas(image.shape[1], image.shape[0],
                                             view.camera_rotations).compose(image)
                    self._display(image)
                    if view.feed.error_message:
                        self.message.configure(text=view.feed.error_message)
            if view.mode == "tracking" and self.track_button is not None and not view.busy:
                status = self.runtime.snapshot()
                self.track_button.configure(
                    text=(
                        "Stop tracking / saving"
                        if status.tracking
                        else "Start tracking / saving"
                    )
                )
                if warning:
                    self.message.configure(text=warning)
                elif not view.error_message and not (view.preview and view.preview.error_message):
                    self.message.configure(
                        text=(
                            f"Tracking {'ON' if status.tracking else 'OFF'}  ·  "
                            f"Recording {'ON' if status.recording else 'OFF'}  ·  UART commands enabled"
                        )
                    )
            if (getattr(view, "sync_message", "") and not view.error_message and not warning
                    and (view.sync_active or view.mode in {"home", "subjects"})):
                self.message.configure(text=view.sync_message)
        except Exception as exc:  # noqa: BLE001 - report boundary failures and keep cleanup available
            self.message.configure(text=str(exc))
        self.root.after(67, self._tick)

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self._transport_stop.set()
        self._enable_controls(False)
        self.message.configure(text="Closing experiment and finalizing recordings…")
        self.pending = self.executor.submit(self.runtime.close)

    def _finish_close(self) -> None:
        self._sync_dialog(self.runtime.view())
        if not self.pending.done():
            self.root.after(67, self._finish_close)
            return
        try:
            self.pending.result()
        except Exception as exc:  # noqa: BLE001 - report boundary failures and keep cleanup available
            # Keep the interface and error visible instead of silently claiming
            # that recordings were finalized successfully.
            self.closing = False
            self.pending = None
            self.runtime.report_error(f"Shutdown failed: {exc}")
            self.signature = None
            self.root.after(67, self._tick)
            return
        if self._transport_thread is not None:
            self._transport_thread.join(timeout=0.6)
        self.executor.shutdown(wait=False)
        self.root.destroy()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pi tracker experiment interface")
    parser.add_argument("config", type=Path, nargs="?", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--data", type=Path, default=Path("data"), help="subject data root (default: ./data)"
    )
    parser.add_argument(
        "--video", type=Path, help="loop a video for desktop testing; recording disabled"
    )
    args = parser.parse_args(argv)
    runtime = None
    try:
        import tkinter as tk

        runtime = ExperimentRuntime(
            ExperimentStore(args.data, args.config), video_path=args.video
        )
        root = tk.Tk()
        window = PiTrackerWindow(root, runtime)
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda _signum, _frame: root.after(0, window.close))
        with KeepAwake(runtime.report_warning):
            try:
                root.mainloop()
            finally:
                runtime.close()
                runtime = None
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"pi-tracker: {exc}\n")
    finally:
        if runtime is not None:
            runtime.close()
    return 0
