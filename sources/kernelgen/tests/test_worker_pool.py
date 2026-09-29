"""Regression tests for the shared, workspace-local Coder lease primitive."""

import json
import os
import select
import subprocess
import sys

import pytest

from kernelgen.framework import local_state
from kernelgen.framework.worker_pool import LeaseRecord, WorkerLeasePool, set_max_workers


ENDPOINT = "http://127.0.0.1:8000"


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))


def lease(tmp_path, run_id="parent", weight=1):
    return LeaseRecord(
        run_id=run_id, pid=os.getpid(),
        process_start=local_state.process_start_identity(os.getpid()),
        workspace=tmp_path, worker_pool=ENDPOINT, weight=weight,
    )


def test_cli_imports_are_aliases_of_shared_implementation():
    from kernelgen.cli import main, models, runner, state
    from kernelgen.framework import worker_pool

    assert main.WorkerLeasePool is runner.WorkerLeasePool is state.WorkerLeasePool is WorkerLeasePool
    assert models.LeaseRecord is runner.LeaseRecord is LeaseRecord
    assert state.cli_home is local_state.state_home
    assert state.file_lock is local_state.file_lock
    assert state.atomic_write_json is local_state.atomic_write_json
    assert state.set_max_workers is worker_pool.set_max_workers
    assert state.normalize_worker_pool is worker_pool.normalize_worker_pool


def test_shared_pool_import_does_not_import_cli():
    code = """
import importlib.abc
import sys
class RejectCLI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'kernelgen.cli' or fullname.startswith('kernelgen.cli.'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectCLI())
from kernelgen.framework.worker_pool import WorkerLeasePool
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=10)


def test_default_home_and_override_keep_existing_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("KERNELGEN_CLI_HOME")
    monkeypatch.chdir(tmp_path)
    pool = WorkerLeasePool(ENDPOINT)
    assert pool.root == tmp_path / ".kernelgen"
    monkeypatch.setenv("KERNELGEN_CLI_HOME", "override")
    pool = WorkerLeasePool(ENDPOINT)
    assert pool.path == tmp_path / "override" / "worker-leases.json"
    assert pool.lock_path == tmp_path / "override" / "worker-leases.lock"
    set_max_workers(ENDPOINT, 3)
    assert (tmp_path / "override" / "config.json").is_file()
    assert pool.snapshot()["max_workers"] == 3


def test_existing_config_and_lease_schema_are_preserved(tmp_path):
    record = lease(tmp_path, weight=3)
    pool = WorkerLeasePool(ENDPOINT)
    local_state.atomic_write_json(pool.root / "config.json", {
        "unrelated": {"keep": True}, "run": {"max_workers": {ENDPOINT: 4}},
    })
    payload = {"schema_version": "1.0", "leases": [record.model_dump(mode="json")]}
    local_state.atomic_write_json(pool.path, payload)
    assert pool.snapshot()["available_workers"] == 1
    assert json.loads(pool.path.read_text()) == payload
    set_max_workers(ENDPOINT, 5)
    assert json.loads((pool.root / "config.json").read_text())["unrelated"] == {"keep": True}
    pool.release(record.run_id, record.pid, "not-the-owner")
    assert pool.snapshot()["used_workers"] == 3
    pool.release(record.run_id, record.pid, record.process_start)
    assert pool.snapshot()["used_workers"] == 0


def test_pid_reuse_record_is_reclaimed_without_removing_live_owner(tmp_path):
    live = lease(tmp_path)
    reused = live.model_copy(update={"run_id": "stale", "process_start": "old-process-start"})
    pool = WorkerLeasePool(ENDPOINT)
    local_state.atomic_write_json(pool.path, {
        "schema_version": "1.0", "leases": [r.model_dump(mode="json") for r in [live, reused]],
    })
    assert pool.snapshot()["leases"] == [live.model_dump(mode="json")]


CHILD = """
import os, sys
from pathlib import Path
from kernelgen.framework.local_state import process_start_identity
from kernelgen.framework.worker_pool import LeaseRecord, WorkerLeasePool
pool = WorkerLeasePool('http://127.0.0.1:8000')
record = LeaseRecord(run_id='child', pid=os.getpid(),
    process_start=process_start_identity(os.getpid()), workspace=Path.cwd(),
    worker_pool=pool.worker_pool, weight=1)
pool.acquire(record, cancelled=lambda: False)
print('acquired', flush=True)
if sys.argv[1] == 'crash':
    os._exit(9)
try:
    sys.stdin.readline()
finally:
    pool.release(record.run_id, record.pid, record.process_start)
"""


def test_weighted_capacity_and_cancel_across_processes(tmp_path):
    set_max_workers(ENDPOINT, 4)
    pool = WorkerLeasePool(ENDPOINT)
    parent = lease(tmp_path, weight=3)
    pool.acquire(parent, cancelled=lambda: False)
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, "normal"], cwd=tmp_path,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert select.select([child.stdout], [], [], 10)[0], "child did not acquire remaining slot"
        assert child.stdout.readline().strip() == "acquired"
        assert pool.snapshot()["used_workers"] == 4
        checks = iter([False, True])
        with pytest.raises(InterruptedError, match="cancelled"):
            pool.acquire(lease(tmp_path, "queued"), cancelled=lambda: next(checks))
        assert pool.snapshot()["used_workers"] == 4
        _, stderr = child.communicate(input="release\n", timeout=10)
        assert child.returncode == 0, stderr
        assert pool.snapshot()["used_workers"] == 3
    finally:
        if child.poll() is None:
            child.terminate()
            child.communicate(timeout=10)
        pool.release(parent.run_id, parent.pid, parent.process_start)
    assert pool.snapshot()["available_workers"] == 4


def test_abnormal_process_exit_reclaims_lease(tmp_path):
    child = subprocess.run(
        [sys.executable, "-c", CHILD, "crash"], cwd=tmp_path,
        capture_output=True, text=True, timeout=10,
    )
    assert child.returncode == 9, child.stderr
    pool = WorkerLeasePool(ENDPOINT)
    assert len(json.loads(pool.path.read_text())["leases"]) == 1
    assert pool.snapshot()["leases"] == []
    current = lease(tmp_path)
    pool.acquire(current, cancelled=lambda: False)
    pool.release(current.run_id, current.pid, current.process_start)
    assert pool.snapshot()["available_workers"] == 1


@pytest.mark.parametrize("mode,weight", [("simple_opt", 1), ("kernelgen", 3)])
@pytest.mark.parametrize("outcome,exit_code", [("success", 0), ("error", 2), ("cancel", 130)])
def test_runner_releases_exactly_one_allocation(tmp_path, monkeypatch, mode, weight, outcome, exit_code):
    from kernelgen.cli import runner
    from kernelgen.framework.run_control import WorkspaceRunControl

    set_max_workers(ENDPOINT, 4)
    request = runner.new_request(
        mode=mode, definition="identity", workspace=tmp_path / "run", workflow_args=[],
        n_parallel=3, target_hardware="A100", eval_server=ENDPOINT,
    )
    pool = WorkerLeasePool(ENDPOINT)

    def invoke(request, **kwargs):
        snapshot = pool.snapshot()
        assert snapshot["used_workers"] == weight
        assert len(snapshot["leases"]) == 1
        if outcome == "error":
            raise RuntimeError("synthetic workflow failure")
        if outcome == "cancel":
            control = WorkspaceRunControl(request.workspace)
            control.request_cancel("synthetic safe-point cancellation")
            control.checkpoint("AFTER_COMPLETE_OUTPUT")
        return 0

    monkeypatch.setattr(runner, "_invoke_workflow", invoke)
    assert runner.execute_request(request) == exit_code
    assert pool.snapshot()["used_workers"] == 0
    assert pool.snapshot()["leases"] == []
