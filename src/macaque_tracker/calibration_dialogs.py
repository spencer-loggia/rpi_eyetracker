"""UI-thread dialogs shared by native and embedded calibration."""

from contextlib import contextmanager


@contextmanager
def _dialog_parent(parent):
    import tkinter as tk

    owner = None
    if parent is None:
        owner = parent = tk.Tk()
        owner.withdraw()
    try:
        yield parent
    finally:
        if owner is not None:
            owner.destroy()


def choose_video(parent=None):
    from tkinter import filedialog

    with _dialog_parent(parent) as dialog_parent:
        return filedialog.askopenfilename(
            parent=dialog_parent, title="Load calibration video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.h264 *.m4v"), ("All files", "*")],
        ) or None


def confirm_reset_controls(parent=None):
    from tkinter import messagebox

    with _dialog_parent(parent) as dialog_parent:
        return messagebox.askyesno(
            "Reset all snapshots?",
            "Delete all captured snapshots and painted labels for both eyes in this calibration?\n\n"
            "Save calibration to keep the reset for future sessions. "
            "Earlier experiment archives will remain unchanged.",
            parent=dialog_parent, icon="warning", default="no",
        )
