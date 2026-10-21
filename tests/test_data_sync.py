from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time

import pytest

from macaque_tracker import data_sync
from macaque_tracker.config import AppConfig, SyncConfig
from macaque_tracker.data_sync import ExperimentDataSync, SyncReport, SyncStatus
from macaque_tracker.experiment_history import read_history
from macaque_tracker.experiments import ExperimentStore

_TIME = 1_792_000_000_123_456_789


@pytest.fixture
def sync_case(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    AppConfig().save(config)
    store = ExperimentStore(tmp_path / "local data", config)
    experiments = tuple(store.create(subject, _TIME) for subject in ("alpha", "beta"))
    for experiment in experiments:
        store.finish(experiment)
        block = experiment.directory / "blocks" / "1"
        block.mkdir(parents=True)
        (block / "video.mkv").write_bytes(b"recording")
        (block / "eye.jsonl").write_text('{"x": 123, "y": 321}\n')
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "mounted-marker").touch()
    monkeypatch.setattr(data_sync, "mounted_filesystem_type", lambda path: "nfs4")
    return store, experiments, remote, ExperimentDataSync(store, SyncConfig(str(remote)))


def attempt(syncer, cancel=None):
    return syncer.run(cancel or threading.Event(), lambda message: None)


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is not installed")
def test_real_rsync_copies_only_experiments_then_removes_exact_plan(sync_case):
    store, experiments, remote, syncer = sync_case
    expected = {
        file.relative_to(store.data_root): file.read_bytes()
        for experiment in experiments for file in experiment.directory.rglob("*") if file.is_file()
    }
    (store.data_root / "notes.txt").write_text("keep local")
    active = store.create("alpha", _TIME + 1_000_000_000)
    report = syncer.run(threading.Event(), lambda message: None, exclude=(active.directory,))

    assert report.status is SyncStatus.SUCCESS, report.detail
    assert set(report.removed) == {experiment.directory for experiment in experiments}
    for path, content in expected.items():
        assert (remote / "pitracker" / path).read_bytes() == content
    assert all(not experiment.directory.exists() for experiment in experiments)
    assert active.directory.exists()
    assert not (remote / "pitracker" / active.directory.relative_to(store.data_root)).exists()
    assert (store.data_root / "notes.txt").exists()
    assert not (remote / "pitracker" / ".history").exists()
    assert not (remote / "pitracker" / "experiment_history.tsv").exists()
    assert not (remote / "pitracker" / ".experiment_history.lock").exists()
    rows = read_history(store.history_path)
    assert len(rows) == 3
    for experiment in experiments:
        row, = [row for row in rows if row["experiment_name"] == experiment.name]
        assert row["block_count"] == "1" and row["status"] == "closed"
    assert store.check_available("alpha", _TIME + 2_000_000_000) == 3
    assert "removed 2 locally" in (store.data_root / "interface_sync.log").read_text()


@pytest.mark.parametrize("failure", ["wrong_mount", "empty", "missing", "probe_timeout", "findmnt"])
def test_unavailable_remote_never_transfers_or_prunes(sync_case, monkeypatch, failure):
    store, experiments, remote, syncer = sync_case
    if failure == "wrong_mount":
        monkeypatch.setattr(data_sync, "mounted_filesystem_type", lambda path: "ext4")
    elif failure == "empty":
        (remote / "mounted-marker").unlink()
    elif failure == "missing":
        shutil.rmtree(remote)
    elif failure == "probe_timeout":
        monkeypatch.setattr(data_sync, "probe_remote_directory", lambda *args, **kwargs:
                            data_sync.DirectoryProbeResult(False, False, "probe timed out"))
    else:
        def failed_findmnt(path):
            raise RuntimeError("findmnt timed out")
        monkeypatch.setattr(data_sync, "mounted_filesystem_type", failed_findmnt)
    monkeypatch.setattr(syncer, "_transfer", lambda *args: pytest.fail("must not copy"))
    report = attempt(syncer)
    assert report.status is SyncStatus.FAILED
    assert all(experiment.directory.exists() for experiment in experiments)
    assert not (store.data_root / ".history").exists()


@pytest.mark.parametrize("status", [SyncStatus.FAILED, SyncStatus.CANCELLED])
def test_incomplete_copy_keeps_every_experiment_and_logs_error(sync_case, monkeypatch, status):
    store, experiments, _remote, syncer = sync_case
    monkeypatch.setattr(syncer, "_transfer", lambda *args: SyncReport(status, "server: permission denied"))
    report = attempt(syncer)
    assert report.status is status
    assert all(experiment.directory.exists() for experiment in experiments)
    assert "server: permission denied" in (store.data_root / "interface_sync.log").read_text()
    assert not (store.data_root / ".history").exists()


def test_history_write_failure_and_late_cancel_preserve_uploaded_experiments(sync_case, monkeypatch):
    store, experiments, _remote, syncer = sync_case
    monkeypatch.setattr(syncer, "_transfer", lambda *args: SyncReport(SyncStatus.SUCCESS, "copied"))

    def fail_history(paths):
        raise OSError("disk full while saving calibration cache")

    monkeypatch.setattr(store, "remember_synced", fail_history)
    report = attempt(syncer)
    assert report.status is SyncStatus.FAILED
    assert "disk full" in report.detail
    assert all(experiment.directory.exists() for experiment in experiments)
    cancelled = threading.Event()
    monkeypatch.setattr(store, "remember_synced", lambda paths: cancelled.set())
    assert attempt(syncer, cancelled).status is SyncStatus.CANCELLED
    assert all(experiment.directory.exists() for experiment in experiments)


def test_tsv_failure_prevents_pruning_after_successful_transfer(sync_case, monkeypatch):
    store, experiments, _remote, syncer = sync_case
    before = store.history_path.read_bytes()
    monkeypatch.setattr(syncer, "_transfer", lambda *args: SyncReport(SyncStatus.SUCCESS, "copied"))

    def fail_history(*args):
        raise OSError("experiment_history.tsv cannot be saved")

    monkeypatch.setattr("macaque_tracker.experiments.update_history", fail_history)
    report = attempt(syncer)
    assert report.status is SyncStatus.FAILED
    assert "experiment_history.tsv" in report.detail
    assert store.history_path.read_bytes() == before
    assert all(experiment.directory.exists() for experiment in experiments)


def test_new_directory_created_after_snapshot_is_never_pruned(sync_case, monkeypatch):
    store, experiments, _remote, syncer = sync_case
    new = []

    def transfer(*args):
        new.append(store.create("alpha", _TIME + 1_000_000_000))
        return SyncReport(SyncStatus.SUCCESS, "copied")

    monkeypatch.setattr(syncer, "_transfer", transfer)
    assert attempt(syncer).status is SyncStatus.SUCCESS
    assert new[0].directory.exists()
    assert all(not experiment.directory.exists() for experiment in experiments)


def test_destination_symlink_and_canonical_overlap_are_rejected(sync_case, monkeypatch):
    store, experiments, remote, syncer = sync_case
    (remote / "pitracker").symlink_to(store.data_root, target_is_directory=True)
    monkeypatch.setattr(syncer, "_transfer", lambda *args: pytest.fail("must not copy"))
    report = attempt(syncer)
    assert report.status is SyncStatus.FAILED
    assert "symbolic link" in report.detail
    assert all(experiment.directory.exists() for experiment in experiments)
    (remote / "pitracker").unlink()
    local_parent_alias = remote / "alias"
    local_parent_alias.symlink_to(store.data_root.parent, target_is_directory=True)
    probe = data_sync.probe_remote_directory(local_parent_alias, source=store.data_root.parent)
    assert "overlaps" in probe.error


def test_no_experiments_requires_no_remote_access(sync_case, monkeypatch):
    _store, experiments, _remote, syncer = sync_case
    for experiment in experiments:
        shutil.rmtree(experiment.directory)
    monkeypatch.setattr(data_sync, "mounted_filesystem_type", lambda path: pytest.fail("no probe"))
    assert attempt(syncer).status is SyncStatus.SUCCESS


def test_prune_error_is_reported_without_claiming_all_data_was_removed(sync_case, monkeypatch):
    _store, experiments, _remote, syncer = sync_case
    monkeypatch.setattr(syncer, "_transfer", lambda *args: SyncReport(SyncStatus.SUCCESS, "copied"))
    original = shutil.rmtree

    def remove(path, *args, **kwargs):
        if path == experiments[0].directory:
            raise PermissionError("local folder permission denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(data_sync.shutil, "rmtree", remove)
    report = attempt(syncer)
    assert report.status is SyncStatus.SUCCESS
    assert "local folder permission denied" in report.detail
    assert report.removed == (experiments[1].directory,)
    assert experiments[0].directory.exists()


@pytest.mark.parametrize("mode", ["exit", "cancel", "stall"])
def test_transfer_failures_include_stderr_and_stop_child(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(data_sync, "STALL_TIMEOUT_SECONDS", 0.2)
    cancelled = threading.Event()
    command = [sys.executable, "-c",
               "import sys,time; sys.stderr.write('remote error\\n'); sys.stderr.flush(); "
               + ("sys.exit(23)" if mode == "exit" else "time.sleep(30)")]
    timer = threading.Timer(0.15, cancelled.set) if mode == "cancel" else None
    if timer:
        timer.start()
    started = time.monotonic()
    try:
        report = ExperimentDataSync._transfer(command, tmp_path, cancelled)
    finally:
        if timer:
            timer.join()
    assert time.monotonic() - started < 3
    assert report.status is (SyncStatus.CANCELLED if mode == "cancel" else SyncStatus.FAILED)
    assert "remote error" in report.detail


def test_progress_resets_stall_timeout_without_total_transfer_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(data_sync, "STALL_TIMEOUT_SECONDS", 0.3)
    command = [sys.executable, "-c",
               "import time; [(print(i, flush=True), time.sleep(.1)) for i in range(8)]"]
    assert ExperimentDataSync._transfer(command, tmp_path, threading.Event()).status is SyncStatus.SUCCESS


def test_bounded_command_times_out_and_preserves_diagnostics():
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        data_sync.run_command_bounded(
            [sys.executable, "-c", "import time; print('probe started', flush=True); time.sleep(30)"],
            timeout_s=0.2,
        )
    assert b"probe started" in caught.value.output


def test_uninterruptible_process_is_left_to_background_reaper(monkeypatch):
    waits, signals, reaped = [], [], []

    class StuckProcess:
        pid = -1

        def poll(self):
            return None

        def wait(self, timeout=None):
            assert timeout is not None, "caller must never wait indefinitely"
            waits.append(timeout)
            raise subprocess.TimeoutExpired("stuck", timeout)

    process = StuckProcess()
    monkeypatch.setattr(data_sync, "_signal_process", lambda *args, **kwargs: signals.append(args[1]))
    monkeypatch.setattr(data_sync, "_reap_process_in_background", reaped.append)
    assert data_sync.terminate_process(process) is False
    assert len(waits) == len(signals) == 2
    assert reaped == [process]


def test_missing_rsync_retains_diagnostic(tmp_path):
    report = ExperimentDataSync._transfer([str(tmp_path / "missing")], tmp_path, threading.Event())
    assert report.status is SyncStatus.FAILED
    assert "Could not start rsync" in report.detail


def test_log_permission_failure_does_not_turn_a_completed_transfer_into_failure(sync_case, monkeypatch):
    _store, experiments, _remote, syncer = sync_case
    monkeypatch.setattr(syncer, "_transfer", lambda *args: SyncReport(SyncStatus.SUCCESS, "copied"))

    def refuse_log(path):
        raise ValueError("refusing to change hard-linked log permissions")

    monkeypatch.setattr(data_sync, "share_output", refuse_log)
    report = attempt(syncer)
    assert report.status is SyncStatus.SUCCESS
    assert all(not experiment.directory.exists() for experiment in experiments)
