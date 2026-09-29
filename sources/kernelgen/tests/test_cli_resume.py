"""Resume must not mutate another owner's workspace before acquiring its lock."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
from types import SimpleNamespace

import pytest

from kernelgen.cli import runner
from kernelgen.framework.run_control import RunState, WorkspaceRunControl


def test_concurrent_resume_prepares_only_after_owner_check(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    request = runner.new_request(
        mode="simple_opt", definition="demo", workspace=tmp_path / "run",
        workflow_args=[], n_parallel=1, target_hardware="H800",
        eval_server="http://localhost:8000",
    )
    control = WorkspaceRunControl(request.workspace)
    control.request_cancel("old cancellation")
    first_started = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    mutex = threading.Lock()
    calls = []
    active = []

    @contextmanager
    def submission_lock(workspace):
        if first_started.is_set():
            second_entered.set()
        with mutex:
            yield

    prepare = runner.prepare_resume

    def observed_prepare(value):
        assert mutex.locked()
        calls.append("prepare")
        return prepare(value)

    def execute(value, *, resume):
        active.append(SimpleNamespace(pid=123))
        control.update_progress(state=RunState.RUNNING, stage="CODING")
        first_started.set()
        assert release_first.wait(5)

    monkeypatch.setattr(runner, "workspace_submission_lock", submission_lock)
    monkeypatch.setattr(runner, "active_process", lambda _: active[0] if active else None)
    monkeypatch.setattr(runner, "prepare_resume", observed_prepare)
    monkeypatch.setattr(runner, "execute_request", execute)
    monkeypatch.setattr(runner, "load_process", lambda _: active[0])
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(runner.submit_request, request, foreground=True, resume=True)
        assert first_started.wait(5)
        cancellation = control.request_cancel("new cancellation while running")
        second = workers.submit(runner.submit_request, request, foreground=True, resume=True)
        try:
            assert second_entered.wait(5)
            assert calls == ["prepare"]
        finally:
            release_first.set()
        first.result(timeout=5)
        with pytest.raises(RuntimeError, match="already has active process"):
            second.result(timeout=5)
    assert calls == ["prepare"]
    assert control.cancellation_state() == cancellation
    assert control.progress().state == RunState.CANCEL_REQUESTED
