from __future__ import annotations

import threading
from concurrent.futures import Future
from types import SimpleNamespace

from macaque_tracker.data_sync import SyncReport, SyncStatus
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

    def grid(self, **options):
        self.grid_options = options

    def set(self, value):
        self.options["value"] = value

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
        self.delays = []
        self.destroyed = False

    def after(self, delay, callback):
        self.callbacks.append(callback)
        self.delays.append(delay)

    def destroy(self):
        self.destroyed = True

    def title(self, _text):
        pass


class Canvas:
    def __init__(self):
        self.image = ""

    def itemconfigure(self, _item, *, image):
        self.image = image


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
    window._display_source = None
    window.display_scale = 1.0
    window._calibration_redraw_pending = False
    window._transport_stop = threading.Event()
    window._transport_thread = None
    return window


def button(window, text):
    return next(child for child in window.controls.children if child.options.get("text") == text)


def test_pointer_events_get_coalesced_redraw_without_camera_read_or_waiting_for_tick():
    strokes, renders, frames = [], [], []
    calibration = SimpleNamespace(
        phase="tuning", mouse=lambda *event: strokes.append(event),
        render=lambda **options: renders.append(options) or "painted frame",
        tuning=SimpleNamespace(controls=SimpleNamespace(sample=object())),
    )
    view = SimpleNamespace(busy=False, calibration=calibration)
    window = window_for(SimpleNamespace(view=lambda: view))
    window._display = frames.append
    window.display_scale = 0.5

    for code, x in ((1, 10), (0, 20), (4, 30)):
        window._mouse(SimpleNamespace(x=x, y=15), code)
    assert strokes == [(1, 20, 30), (0, 40, 30), (4, 60, 30)]
    assert window.root.delays == [1]
    assert not renders
    window.root.callbacks.pop()()
    assert renders == [{"refresh": False}]
    assert frames == ["painted frame"]
    assert not window._calibration_redraw_pending


def test_queued_paint_redraw_is_discarded_after_leaving_calibration():
    view = SimpleNamespace(busy=False, calibration=SimpleNamespace(phase="tuning"))
    window = window_for(SimpleNamespace(view=lambda: view))
    window._request_calibration_redraw()
    view.calibration = None
    window.root.callbacks.pop()()
    assert not window._calibration_redraw_pending


def test_pointer_redraw_never_advances_a_live_preview_prior():
    controls = SimpleNamespace(sample=None)
    calibration = SimpleNamespace(
        phase="tuning", mouse=lambda *_args: None,
        tuning=SimpleNamespace(controls=controls),
    )
    window = window_for(SimpleNamespace(view=lambda: SimpleNamespace(busy=False, calibration=calibration)))
    window._mouse(SimpleNamespace(x=10, y=15), 1)
    assert window.root.callbacks == []
    controls.sample = object()
    window._mouse(SimpleNamespace(x=10, y=15), 1)
    assert len(window.root.callbacks) == 1
    # Switching back to live before the queued callback runs also cancels it.
    controls.sample = None
    window.root.callbacks.pop()()
    assert not window._calibration_redraw_pending


def test_sync_dialog_cancel_and_shutdown_choices_are_wired_on_ui_thread(tmp_path):
    calls, popups = [], []

    class Popup(Widget):
        def __init__(self, root):
            super().__init__()
            self.protocols = {}
            popups.append(self)

        def title(self, text):
            pass

        def transient(self, root):
            pass

        def grab_set(self):
            pass

        def protocol(self, name, callback):
            self.protocols[name] = callback

        def bind(self, event, callback):
            self.protocols[event] = callback

    view = SimpleNamespace(sync_active=True, sync_message="Syncing experiments",
                           shutdown_decision=False, busy=True)
    runtime = SimpleNamespace(view=lambda: view, cancel_sync=lambda: calls.append("cancel sync"),
                              resolve_sync_failure=calls.append,
                              store=SimpleNamespace(data_root=tmp_path))
    window = window_for(runtime)
    window.tk.Toplevel = Popup
    window._sync_dialog(view)
    window._sync_cancel_button.options["command"]()
    assert calls == ["cancel sync"]
    view.sync_message = "Finalizing sync; removing uploaded local experiments"
    window._sync_dialog(view)
    assert window._sync_cancel_button.options["state"] == "disabled"
    view.sync_active, view.busy, view.shutdown_decision = False, False, True
    view.sync_report = SyncReport(SyncStatus.FAILED, "rsync permission denied")
    window._sync_dialog(view)
    assert str(tmp_path / "interface_sync.log") in window._sync_label.options["text"]
    choices = {child.options["text"]: child.options["command"]
               for child in popups[-1].children if "command" in child.options}
    assert set(choices) == {"Retry", "Shut down anyway", "Cancel shutdown"}
    for label, value in (("Retry", "retry"), ("Shut down anyway", "shutdown"),
                         ("Cancel shutdown", "cancel")):
        choices[label]()
        window.executor.run_next()
        window.pending = None
        assert calls[-1] == value
    popups[-1].protocols["WM_DELETE_WINDOW"]()
    window.executor.run_next()
    assert calls[-1] == "cancel"
    view.shutdown_decision = False
    window._sync_dialog(view)
    assert window._sync_popup is None


def test_warning_popup_is_nonmodal_deduplicated_and_can_recur_after_recovery():
    popups = []

    class Popup(Widget):
        def __init__(self, root):
            super().__init__()
            popups.append(self)

        def title(self, text):
            assert "warning" in text

        def transient(self, root):
            pass

        def lift(self):
            pass

        # No wait_window/grab_set: either would interrupt normal GUI operation.

    window = window_for(SimpleNamespace())
    window.tk.Toplevel = Popup
    window._show_warning("Video saving disabled: low disk space")
    window._show_warning("Video saving disabled: low disk space")
    assert len(popups) == 1
    dismiss = next(child for child in popups[0].children if child.options.get("text") == "OK")
    dismiss.options["command"]()
    window._show_warning(None)
    window._show_warning("Video saving disabled: low disk space")
    assert len(popups) == 2


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


def test_pending_uart_shutdown_disables_controls_without_opening_camera():
    runtime = SimpleNamespace(view=lambda: SimpleNamespace(
        shutdown_submitted=False, shutdown_requested=True,
    ))
    window = window_for(runtime)
    Widget(window.controls)
    window._tick()
    assert not window.closing
    assert window.executor.tasks == []
    assert "Shutting down" in window.message.options["text"]
    assert all(child.options["state"] == "disabled" for child in window.controls.children)


def test_accepted_system_shutdown_closes_gui():
    runtime = SimpleNamespace(
        view=lambda: SimpleNamespace(shutdown_submitted=True, shutdown_requested=True),
        close=lambda: None,
    )
    window = window_for(runtime)
    window._tick()
    assert window.closing
    window.executor.run_next()
    window._finish_close()
    assert window.root.destroyed


def test_status_errors_preserve_last_frame_when_renderer_has_no_replacement():
    view = SimpleNamespace(
        mode="tracking", experiment=None, feed=None, calibration=None,
        preview=SimpleNamespace(read_rendered=lambda: None, error_message=None),
        error_message=None, busy=False, shutdown_requested=False, shutdown_submitted=False,
    )
    runtime = SimpleNamespace(
        view=lambda: view,
        snapshot=lambda: SimpleNamespace(tracking=True, recording=True),
    )
    window = window_for(runtime)
    window.canvas = Canvas()
    window.header, window.context = Widget(), Widget()
    window.image_item = 1
    window._feed_failed = False
    window._tick()
    window.canvas.image = "last received frame"

    view.error_message = "Preview stopped producing images"
    window._tick()
    assert window.canvas.image == "last received frame"
    assert window.message.options["text"] == view.error_message

    # Finishing a command rebuilds controls too, without changing the source.
    window.pending = Future()
    window.pending.set_result(None)
    window._tick()
    assert window.canvas.image == "last received frame"

    # Old frames must not appear under a different experiment's identity.
    view.experiment = SimpleNamespace(subject="other_subject", name="new_experiment")
    window._tick()
    assert window.canvas.image == ""

    window.canvas.image = "current preview"
    view.block = SimpleNamespace(number=7)
    window._tick()
    assert "Block: 7" in window.context.options["text"]
    assert window.canvas.image == "current preview"


def test_rotation_action_uses_shared_calibration_editor_and_rebuilds_controls():
    rotated = []
    view = SimpleNamespace(busy=False, calibration=SimpleNamespace(
        phase="boxes", roi=SimpleNamespace(rotate_camera=rotated.append),
    ))
    window = window_for(SimpleNamespace(view=lambda: view))
    window.signature = "old controls"
    window._rotate_camera(2)
    assert rotated == [2]
    assert window.signature is None
    view.calibration.phase = "tuning"
    window._rotate_camera(0)
    assert rotated == [2]


def test_annotation_buttons_use_shared_editor_actions_on_ui_thread():
    actions = []
    calibration = SimpleNamespace(
        phase="tuning", sliders=lambda: (),
        tuning=SimpleNamespace(
            control_buttons=lambda: [("Capture", "capture"), ("No pupil", "no_pupil"),
                                     ("Optimize eye", "optimize"), ("Live", "live")],
            control_action=actions.append,
        ),
    )
    view = SimpleNamespace(mode="calibration", busy=False, calibration=calibration, error_message=None)
    window = window_for(SimpleNamespace(view=lambda: view))
    window._build_controls(view)
    for name in ("Capture", "No pupil", "Optimize eye", "Live"):
        control = next(child for child in window.options.children if child.options.get("text") == name)
        control.options["command"]()
    assert actions == ["capture", "no_pupil", "optimize", "live"]
    assert window.signature is None
    view.busy = True
    window._control_action("capture")
    assert len(actions) == 4


def test_annotation_buttons_and_manual_sliders_share_four_columns_without_overlap():
    slider = SimpleNamespace(minimum_position=0, maximum_position=100, position_for=lambda value: value)
    controls = [(f"Button {index}", f"action{index}") for index in range(14)]
    calibration = SimpleNamespace(
        phase="tuning",
        sliders=lambda: ((f"Slider {index}", slider, 50, lambda _value: None) for index in range(8)),
        tuning=SimpleNamespace(control_buttons=lambda: controls),
    )
    view = SimpleNamespace(mode="calibration", busy=False, calibration=calibration, error_message=None)
    window = window_for(SimpleNamespace(view=lambda: view))
    window._build_controls(view)
    buttons = [child for child in window.options.children if "text" in child.options]
    sliders = [child for child in window.options.children if "label" in child.options]
    assert len(buttons) == 14 and len(sliders) == 8
    assert all(child.grid_options["column"] < 4 for child in buttons + sliders)
    assert max(child.grid_options["row"] for child in buttons) == 3
    assert min(child.grid_options["row"] for child in sliders) == 4


def test_video_picker_runs_on_ui_thread_and_loads_in_worker(monkeypatch):
    from macaque_tracker import calibration_video

    calls = []
    owner = threading.get_ident()
    view = SimpleNamespace(mode="experiment", busy=False, calibration=None)
    runtime = SimpleNamespace(view=lambda: view,
                              begin_calibration=lambda **kw: calls.append(kw))
    window = window_for(runtime)

    def pick(parent):
        assert threading.get_ident() == owner
        assert parent is window.root
        return "/tmp/blink.avi"

    monkeypatch.setattr(calibration_video, "choose_video", pick)
    window._video_action()
    assert calls == []
    window.executor.run_next()
    assert calls == [{"video_path": "/tmp/blink.avi"}]
    window.pending = None
    view.mode, view.calibration = "calibration", object()
    runtime.calibration_video_action = lambda action, **kw: calls.append((action, kw))
    window._video_action("frame_forward")
    window.executor.run_next()
    assert calls[-1] == ("frame_forward", {"path": None})
    window.pending = None
    monkeypatch.setattr(calibration_video, "choose_video", lambda _parent: None)
    window._video_action()
    assert not window.executor.tasks and window.pending is None
