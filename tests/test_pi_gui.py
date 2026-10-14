from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace

from macaque_tracker.pi_gui import PiTrackerWindow


class Widget:
    def __init__(self, parent=None, **options):
        self.parent = parent
        self.options = options
        self.children = []
        if parent is not None:
            parent.children.append(self)

    def pack(self, **_options):
        pass

    def configure(self, **options):
        self.options.update(options)

    def winfo_children(self):
        return list(self.children)

    def destroy(self):
        if self.parent is not None:
            self.parent.children.remove(self)


class ThreadBoundVariable:
    """Model Tk's requirement that variable operations run on its UI thread."""

    def __init__(self, value=""):
        self.value = value
        self.owner = threading.get_ident()

    def get(self):
        assert threading.get_ident() == self.owner, "Tk variable read outside the UI thread"
        return self.value

    def set(self, value):
        assert threading.get_ident() == self.owner, "Tk variable write outside the UI thread"
        self.value = value


class DeferredExecutor:
    def __init__(self):
        self.tasks = []
        self.shutdown_called = False

    def submit(self, action):
        future = Future()
        self.tasks.append((action, future))
        return future

    def run_next(self):
        action, future = self.tasks.pop(0)

        def run():
            try:
                future.set_result(action())
            except BaseException as exc:  # noqa: BLE001 - match Future's worker exception capture
                future.set_exception(exc)

        worker = threading.Thread(target=run)
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        return future.result()

    def shutdown(self, *, wait):
        assert not wait
        self.shutdown_called = True


class Root:
    def __init__(self):
        self.callbacks = []
        self.destroyed = False

    def after(self, _delay, callback):
        self.callbacks.append(callback)

    def destroy(self):
        self.destroyed = True


def window_for(runtime):
    if not hasattr(runtime, "view"):
        runtime.view = lambda: SimpleNamespace(busy=False)
    window = object.__new__(PiTrackerWindow)
    window.runtime = runtime
    window.root = Root()
    window.tk = SimpleNamespace(StringVar=ThreadBoundVariable, Scale=Widget)
    window.ttk = SimpleNamespace(Label=Widget, Button=Widget, Entry=Widget, Combobox=Widget)
    window.controls, window.options, window.message = Widget(), Widget(), Widget()
    window.executor = DeferredExecutor()
    window.pending = None
    window.closing = False
    window.signature = None
    window._transport_stop = threading.Event()
    window._transport_thread = None
    return window


def button(window, text):
    return next(child for child in window.controls.children if child.options.get("text") == text)


def test_subject_selection_reads_tk_variable_before_experiment_worker_runs():
    opened = []
    ui_thread = threading.get_ident()

    def open_experiment(name):
        assert threading.get_ident() != ui_thread
        opened.append(name)

    runtime = SimpleNamespace(
        store=SimpleNamespace(subjects=lambda: ["alpha", "beta"]),
        start_experiment=open_experiment,
    )
    window = window_for(runtime)
    window._build_controls(SimpleNamespace(mode="subjects", error_message=None))
    selector = next(widget for widget in window.options.children if "values" in widget.options)
    selector.options["textvariable"].set("beta")

    button(window, "Open experiment").options["command"]()
    window.executor.run_next()

    assert opened == ["beta"]


def test_repeated_tracking_clicks_queue_only_one_camera_transition():
    calls = []
    runtime = SimpleNamespace(
        open_tracking=lambda: calls.append("open idle tracking view"),
        begin_calibration=lambda: None,
        end_experiment=lambda: None,
    )
    window = window_for(runtime)
    window._build_controls(SimpleNamespace(mode="experiment", error_message=None))
    launch = button(window, "Start tracking")

    launch.options["command"]()
    launch.options["command"]()

    assert len(window.executor.tasks) == 1
    assert all(child.options["state"] == "disabled" for child in window.controls.children)
    window.executor.run_next()
    assert calls == ["open idle tracking view"]


def test_local_camera_transition_waits_for_active_uart_transition():
    window = window_for(SimpleNamespace(view=lambda: SimpleNamespace(busy=True)))

    window._submit(lambda: None)

    assert window.pending is None
    assert window.executor.tasks == []


def test_close_waits_for_pending_transition_and_recording_finalization():
    actions = []
    runtime = SimpleNamespace(close=lambda: actions.append("finalize recordings"))
    window = window_for(runtime)
    window._submit(lambda: actions.append("finish earlier transition"))

    window.close()
    window._finish_close()
    assert window._transport_stop.is_set()
    assert not window.root.destroyed
    assert len(window.executor.tasks) == 2

    window.executor.run_next()
    assert not window.root.destroyed
    window.executor.run_next()
    window._finish_close()

    assert actions == ["finish earlier transition", "finalize recordings"]
    assert window.root.destroyed
    assert window.executor.shutdown_called


def test_failed_finalization_keeps_window_open_and_exposes_failure():
    failures = []

    def close():
        raise OSError("recording flush failed")

    window = window_for(SimpleNamespace(close=close, report_error=failures.append))
    window.close()
    try:
        window.executor.run_next()
    except OSError:
        pass
    window._finish_close()

    assert not window.root.destroyed
    assert not window.closing
    assert window.pending is None
    assert failures == ["Shutdown failed: recording flush failed"]
