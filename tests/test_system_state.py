from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from macaque_tracker import system_state


def _source_tree(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setattr(system_state, "__file__", str(root / "src/macaque_tracker/system_state.py"))
    return root


@pytest.mark.parametrize("worktree", [False, True])
@pytest.mark.parametrize("commit", ["e6f6775890359c6c0f7111d0dc31c5e8c59fbd4b", "a" * 64])
def test_git_lookup_uses_source_checkout_even_with_external_cwd_and_git_env(
    tmp_path, monkeypatch, worktree, commit,
):
    root = _source_tree(tmp_path, monkeypatch)
    if worktree:
        (root / ".git").write_text("gitdir: /some/worktree/metadata\n")
    else:
        (root / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_DIR", "/another/repository")
    monkeypatch.setenv("GIT_WORK_TREE", "/another/worktree")
    run = Mock(return_value=subprocess.CompletedProcess([], 0, commit + "\n"))
    monkeypatch.setattr(system_state.subprocess, "run", run)

    assert system_state._source_git_commit() == commit
    args, options = run.call_args
    command = args[0]
    assert command[command.index("-C") + 1] == str(root)
    assert f"safe.directory={root}" in command
    assert not any(key.startswith("GIT_") for key in options["env"])
    assert options["check"] is True
    assert 0 < options["timeout"] <= 2


@pytest.mark.parametrize("failure", [
    FileNotFoundError("git unavailable"),
    PermissionError("unreadable checkout"),
    subprocess.CalledProcessError(128, "git"),
    subprocess.TimeoutExpired("git", 2),
])
def test_unavailable_git_is_explicitly_unknown(tmp_path, monkeypatch, failure):
    root = _source_tree(tmp_path, monkeypatch)
    (root / ".git").mkdir()
    monkeypatch.setattr(system_state.subprocess, "run", Mock(side_effect=failure))
    state = system_state.capture_system_state()
    assert state["git_commit"] is None
    assert state["git_dirty"] is None


@pytest.mark.parametrize("changes, dirty", [("", False), (" M src/tracker.py\n", True),
                                            ("?? new_detector.py\n", True), (None, None)])
def test_provenance_marks_uncommitted_or_unknown_code_without_storing_filenames(
    tmp_path, monkeypatch, changes, dirty,
):
    root = _source_tree(tmp_path, monkeypatch)
    (root / ".git").mkdir()
    commit = "a" * 40
    status = (subprocess.TimeoutExpired("git", 2) if changes is None else
              subprocess.CompletedProcess([], 0, changes))
    run = Mock(side_effect=[subprocess.CompletedProcess([], 0, commit), status])
    monkeypatch.setattr(system_state.subprocess, "run", run)
    state = system_state.capture_system_state()
    assert state["git_commit"] == commit
    assert state["git_dirty"] is dirty
    assert "tracker.py" not in str(state) and "new_detector.py" not in str(state)
    assert run.call_args.args[0][-3:] == ["status", "--porcelain", "--untracked-files=normal"]


def test_installed_package_does_not_report_unrelated_parent_repository(tmp_path, monkeypatch):
    _source_tree(tmp_path, monkeypatch)
    (tmp_path / ".git").mkdir()
    run = Mock(side_effect=AssertionError("No source checkout; must not search ancestors"))
    monkeypatch.setattr(system_state.subprocess, "run", run)
    assert system_state._source_git_commit() is None
    run.assert_not_called()


@pytest.mark.parametrize("output", ["", "HEAD", "e6f6775", "z" * 40, "a" * 40 + "\nextra"])
def test_invalid_git_output_does_not_become_a_commit(tmp_path, monkeypatch, output):
    root = _source_tree(tmp_path, monkeypatch)
    (root / ".git").mkdir()
    monkeypatch.setattr(
        system_state.subprocess, "run",
        Mock(return_value=subprocess.CompletedProcess([], 0, output)),
    )
    assert system_state._source_git_commit() is None


def test_real_checkout_commit_survives_different_cwd_and_git_environment(tmp_path, monkeypatch):
    root = Path(system_state.__file__).resolve().parents[2]
    if shutil.which("git") is None or not (root / ".git").exists():
        pytest.skip("Requires the source checkout and Git")
    expected = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
        env={key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
        check=True, capture_output=True, text=True, timeout=2,
    ).stdout.strip()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "missing.git"))
    state = system_state.capture_system_state()
    assert state["git_commit"] == expected
    assert state["python_version"]
    assert state["os"]["system"]
    assert state["os"]["machine"]
