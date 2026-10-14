"""Tk experiment launcher. Camera and tracking workers never call into Tk."""

from __future__ import annotations

import argparse
import signal
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH
from .experiments import ExperimentStore
from .pi_runtime import ExperimentRuntime


class PiTrackerWindow:
    def __init__(self, root, runtime: ExperimentRuntime) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.root, self.runtime = root, runtime
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi-interface")
        self.pending = None
        self.signature = None
        self.photo = None
        self.display_scale = 1.0
        self.closing = False
        self._feed_failed = False
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
        root.bind("<Escape>", lambda _event: self._back())
        root.bind("<Return>", lambda _event: self._calibration_next())
        root.bind("<Key-s>", lambda _event: self._calibration_next())
        root.bind("<Key-u>", lambda _event: self._edit_boxes("undo"))
        root.bind("<Key-c>", lambda _event: self._edit_boxes("clear"))
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

    def _mouse(self, event, code) -> None:
        view = self.runtime.view()
        if self.pending is None and not view.busy and view.calibration is not None:
            view.calibration.mouse(
                code, round(event.x / self.display_scale), round(event.y / self.display_scale)
            )

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
            if calibration.phase == "boxes":
                self._button("Undo box", lambda: self._edit_boxes("undo"))
                self._button("Clear boxes", lambda: self._edit_boxes("clear"))
                self._button("Next: pupil tuning", self._calibration_next, right=True)
            else:
                self._button("Save calibration", self._calibration_next, right=True)
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
                control.grid(row=index // 4, column=index % 4, padx=4, sticky="ew")
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
        signature = (
            view.mode,
            view.experiment,
            id(view.calibration),
            getattr(view.calibration, "phase", None),
            view.error_message,
        )
        try:
            if signature != self.signature:
                self.signature = signature
                self.canvas.itemconfigure(self.image_item, image="")
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
                self.context.configure(text=context)
                self.root.title(f"Pi tracker — {context}")
                self.message.configure(text=view.error_message or "")
            self._enable_controls(self.pending is None and not view.busy)
            if view.feed is None and view.preview is None and not self._feed_failed:
                self._submit(self.runtime.ensure_feed)
            if self.pending is None and not view.busy:
                if view.calibration is not None:
                    self._display(view.calibration.render())
                elif view.preview is not None:
                    frame = view.preview.read_rendered()
                    if frame is not None:
                        self._display(frame)
                    if view.preview.error_message:
                        self.message.configure(text=f"Preview: {view.preview.error_message}")
                elif view.feed is not None and view.feed.image is not None:
                    self._display(view.feed.image)
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
                if not view.error_message and not (view.preview and view.preview.error_message):
                    self.message.configure(
                        text=(
                            f"Tracking {'ON' if status.tracking else 'OFF'}  ·  "
                            f"Recording {'ON' if status.recording else 'OFF'}  ·  UART commands enabled"
                        )
                    )
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
        root.mainloop()
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"pi-tracker: {exc}\n")
    finally:
        if runtime is not None:
            runtime.close()
    return 0
