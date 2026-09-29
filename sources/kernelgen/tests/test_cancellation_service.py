"""Shared safe cancellation and durable remote operation ownership."""

import os
import signal

import pytest

from kernelgen.cli import runner
from kernelgen.framework.cancellation import request_run_cancellation, reconcile_server_operations
from kernelgen.framework.run_control import RunCancelled, RunState, WorkspaceRunControl
from kernelgen.framework.runtime.base import CLIRuntime, PumpState
from kernelgen.tools.kernelgen_server_adapter import tracked_server_operation
from kernelgen_client.http import ServerError


CAPABILITIES = {"capabilities": {"operation_cancel": {"enabled": True, "operations": ["evaluate", "profile"]}}}


def test_lost_response_preserves_operation_until_remote_terminal(tmp_path, monkeypatch):
    control = WorkspaceRunControl(tmp_path)
    state = {"state": "RUNNING"}
    monkeypatch.setattr("kernelgen_client.http.get_operation", lambda *a, **k: state)
    with pytest.raises(ServerError, match="lost response"):
        with tracked_server_operation(control, CAPABILITIES, "http://localhost:8000", "evaluate") as operation_id:
            raise ServerError("lost response")
    assert [item.operation_id for item in control.active_server_operations()] == [operation_id]
    assert reconcile_server_operations(control)
    state["state"] = "CANCELLED"
    assert reconcile_server_operations(control) == []
    assert control.active_server_operations() == []


@pytest.mark.parametrize("capabilities", [CAPABILITIES, {}])
def test_cancel_between_profile_workloads_prevents_next_submission(tmp_path, capabilities):
    control = WorkspaceRunControl(tmp_path)
    with tracked_server_operation(control, capabilities, "http://localhost:8000", "profile"):
        pass
    control.request_cancel("stop after first workload")
    with pytest.raises(RunCancelled):
        with tracked_server_operation(control, capabilities, "http://localhost:8000", "profile"):
            pytest.fail("cancelled operation must not be submitted")
    assert control.active_server_operations() == []


def test_cancel_retries_only_delete_when_registration_precedes_post(tmp_path, monkeypatch):
    control = WorkspaceRunControl(tmp_path)
    control.register_server_operation("evaluate", "http://localhost:8000")
    calls = []

    def cancel(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise ServerError("not registered yet", status_code=404)
        return {"state": "CANCEL_REQUESTED"}

    monkeypatch.setattr("kernelgen_client.http.cancel_operation", cancel)
    monkeypatch.setattr("kernelgen.framework.cancellation.time.sleep", lambda _: None)
    state, requested, failed = request_run_cancellation(control, "stop")
    assert state.requested and requested == 1 and failed == 0
    assert len(calls) == 2
    assert control.active_server_operations()


def test_failed_forward_preserves_intent_and_remote_registration(tmp_path, monkeypatch):
    control = WorkspaceRunControl(tmp_path)
    operation = control.register_server_operation("evaluate", "http://localhost:8000")

    def fail(*args, **kwargs):
        raise ServerError("connection unavailable")

    monkeypatch.setattr("kernelgen_client.http.cancel_operation", fail)
    state, requested, failed = request_run_cancellation(control, "stop")
    assert state.requested and requested == 0 and failed == 1
    assert control.active_server_operations() == [operation]


def test_rejected_request_does_not_leave_active_operation(tmp_path):
    control = WorkspaceRunControl(tmp_path)
    with pytest.raises(ServerError):
        with tracked_server_operation(control, CAPABILITIES, "http://localhost:8000", "evaluate"):
            raise ServerError("invalid payload", status_code=422)
    assert control.active_server_operations() == []


def test_resume_does_not_clear_cancel_before_remote_terminal(tmp_path, monkeypatch):
    control = WorkspaceRunControl(tmp_path)
    control.register_server_operation("evaluate", "http://localhost:8000")
    cancellation = control.request_cancel("stop")
    monkeypatch.setattr("kernelgen_client.http.get_operation", lambda *a, **k: {"state": "RUNNING"})
    from types import SimpleNamespace
    with pytest.raises(RuntimeError, match="previously registered"):
        runner.prepare_resume(SimpleNamespace(workspace=tmp_path))
    assert control.cancellation_state() == cancellation


def test_sigint_keeps_complete_model_output_and_session(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    workspace = tmp_path / "run"
    control = WorkspaceRunControl(workspace)
    completed = []

    class Runtime(CLIRuntime):
        def _command(self, *args, **kwargs): return []
        def _parse_line(self, *args): return None
        def _spawn(self, *args): return object()
        def _pump(self, proc):
            os.kill(os.getpid(), signal.SIGINT)
            completed.append("full model output")
            return PumpState(text_parts=["full model output"], done_ok=True, session_id="complete-session")
        def _kill_and_wait(self, proc):
            pytest.fail("SIGINT must not kill the provider")

    runtime = Runtime(workspace=workspace, run_control=control, verbose=False)
    def invoke(*args, **kwargs):
        runtime.invoke("test")
        return 0
    monkeypatch.setattr(runner, "_invoke_workflow", invoke)
    previous = signal.getsignal(signal.SIGINT)
    request = runner.new_request(
        mode="simple_opt", definition="demo", workspace=workspace, workflow_args=[],
        n_parallel=1, target_hardware="H800", eval_server="http://localhost:8000",
    )
    assert runner.execute_request(request) == 130
    assert completed == ["full model output"]
    assert runtime.last_session_id == "complete-session"
    assert control.progress().state == RunState.CANCELLED
    assert signal.getsignal(signal.SIGINT) == previous
