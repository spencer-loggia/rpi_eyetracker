import sys
from types import SimpleNamespace

import pytest

from macaque_tracker.calibration_dialogs import confirm_reset_controls


@pytest.mark.parametrize("embedded", [False, True])
def test_reset_dialog_defaults_to_no_and_releases_only_its_own_root(monkeypatch, embedded):
    events, requests = [], []
    root = SimpleNamespace(withdraw=lambda: events.append("hide"),
                           destroy=lambda: events.append("destroy"))
    parent = object() if embedded else None

    def ask(*args, **kwargs):
        requests.append((args, kwargs))
        return False

    monkeypatch.setitem(sys.modules, "tkinter", SimpleNamespace(
        Tk=lambda: root, messagebox=SimpleNamespace(askyesno=ask),
    ))
    assert not confirm_reset_controls(parent)
    args, kwargs = requests[0]
    assert "both eyes" in args[1] and "Save calibration" in args[1]
    assert kwargs["default"] == "no"
    assert kwargs["parent"] is (parent if embedded else root)
    assert events == ([] if embedded else ["hide", "destroy"])
