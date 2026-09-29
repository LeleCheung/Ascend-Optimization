"""Host-only tests for workspace-backed progress, events, and cancellation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys

import pytest

from kernelgen.framework.run_control import (
    RunCancelled,
    RunEvent,
    RunState,
    WorkspaceRunControl,
)
from kernelgen.framework.parallel import Directory, run_parallel
from kernelgen.framework.runtime.base import CLIRuntime, IdleTimeout, PumpState
from kernelgen.workflows.legacy.simple_opt import SimpleOptWorkflow


class _ObservedRuntime(CLIRuntime):
    def __init__(self, *args, cancel_while_parsing: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cancel_while_parsing = cancel_while_parsing
        self.spawned = False

    def _command(self, prompt, session_id, *, agent=None):
        return [sys.executable, "-c", "print('provider-result')"]

    def _spawn(self, cmd, prompt):
        self.spawned = True
        return super()._spawn(cmd, prompt)

    def _parse_line(self, line: str, state: PumpState):
        state.text_parts.append(line.strip())
        state.done_ok = True
        if self.cancel_while_parsing:
            self._get_run_control().request_cancel("stop after current call")
        return "provider completed"


class _NeverRuns:
    @classmethod
    def bind(cls, path, runtime_factory):
        raise AssertionError("cancelled task must not be bound")


def test_events_are_replayable_with_run_global_sequences(tmp_path):
    control = WorkspaceRunControl(tmp_path, source="workflow")

    first = control.record_event("RUN_STARTED", stage="PREPARING")
    second = control.emit(
        RunEvent(
            event_type="STAGE_STARTED",
            source="runtime",
            stage="CODING",
        )
    )

    assert first.sequence == 1
    assert second.sequence == 2
    assert [event.event_type for event in control.read_events()] == [
        "RUN_STARTED",
        "STAGE_STARTED",
    ]
    assert [event.sequence for event in control.read_events(after_sequence=1)] == [
        2
    ]


def test_concurrent_event_writers_do_not_duplicate_sequences(tmp_path):
    control = WorkspaceRunControl(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        events = list(
            executor.map(
                lambda index: control.record_event(
                    "ITEM", data={"index": index}
                ),
                range(40),
            )
        )

    assert sorted(event.sequence for event in events) == list(range(1, 41))
    persisted = control.read_events()
    assert len(persisted) == 40
    assert sorted(event.sequence for event in persisted) == list(range(1, 41))


def test_cancellation_is_idempotent_and_checked_only_at_safe_point(tmp_path):
    control = WorkspaceRunControl(tmp_path)

    first = control.request_cancel("operator requested stop")
    second = control.request_cancel("ignored duplicate")

    assert first == second
    assert first.generation == 1
    assert control.progress().state == RunState.CANCEL_REQUESTED
    assert control.progress().cancel_requested is True
    with pytest.raises(RunCancelled, match="operator requested stop") as captured:
        control.checkpoint("before_evaluation")
    assert captured.value.stage == "before_evaluation"
    assert [event.event_type for event in control.read_events()] == [
        "CANCEL_REQUESTED",
        "CANCEL_OBSERVED",
    ]


def test_cancellation_can_only_be_cleared_with_current_generation(tmp_path):
    control = WorkspaceRunControl(tmp_path)
    requested = control.request_cancel()

    with pytest.raises(ValueError, match="generation changed"):
        control.clear_cancellation(expected_generation=requested.generation + 1)

    cleared = control.clear_cancellation(
        expected_generation=requested.generation
    )
    assert cleared.requested is False
    assert cleared.generation == 2
    control.checkpoint("resume")


def test_nested_workspace_shares_control_and_tracks_its_scope(tmp_path):
    root = tmp_path / "run"
    child = root / "1R" / "agent0"
    root_control = WorkspaceRunControl(root, source="workflow")
    child.mkdir(parents=True)
    child_control = WorkspaceRunControl(child, source="runtime")

    child_control.update_progress(
        state=RunState.RUNNING,
        stage="EVALUATING",
        current_epoch=1,
    )
    child_control.record_event("ROUND_STARTED")

    snapshot = root_control.progress()
    assert child_control.root_workspace == root.resolve()
    assert child_control.scope == "1R/agent0"
    assert snapshot.state == RunState.PENDING
    assert snapshot.scopes["1R/agent0"].stage == "EVALUATING"
    assert snapshot.scopes["1R/agent0"].current_epoch == 1
    with pytest.raises(ValueError, match="unsupported progress field"):
        child_control.update_progress(best_geo_mean=1.25)
    assert root_control.read_events()[0].scope == "1R/agent0"

    root_control.request_cancel("stop all agents")
    assert child_control.is_cancellation_requested() is True


def test_explicit_link_supports_workspace_outside_run_root(tmp_path):
    root = tmp_path / "run"
    external = tmp_path / "allocated" / "worker0"
    control = WorkspaceRunControl(root)
    linked = control.link_workspace(external, scope="workers/worker0")

    linked.record_event("WORKER_READY")

    event = control.read_events()[0]
    assert linked.root_workspace == root.resolve()
    assert event.scope == "workers/worker0"


def test_active_server_operations_are_shared_and_removed_atomically(tmp_path):
    root = tmp_path / "run"
    child = root / "agent0"
    root_control = WorkspaceRunControl(root, source="cli")
    child.mkdir(parents=True)
    child_control = WorkspaceRunControl(child, source="mcp")

    operation = child_control.register_server_operation(
        "evaluate",
        "http://127.0.0.1:18000/",
        operation_id="evaluation-1",
    )

    assert operation.scope == "agent0"
    assert operation.server_url == "http://127.0.0.1:18000"
    assert root_control.active_server_operations() == [operation]
    child_control.unregister_server_operation(operation.operation_id)
    assert root_control.active_server_operations() == []
    assert [event.event_type for event in root_control.read_events()] == [
        "SERVER_OPERATION_STARTED",
        "SERVER_OPERATION_FINISHED",
    ]


def test_active_server_operation_rejects_credentials(tmp_path):
    control = WorkspaceRunControl(tmp_path)

    with pytest.raises(ValueError, match="credentials"):
        control.register_server_operation(
            "evaluate",
            "http://user:secret@server:8000",
        )


def test_progress_rejects_unversioned_extension_fields(tmp_path):
    control = WorkspaceRunControl(tmp_path)

    with pytest.raises(ValueError, match="unsupported progress field"):
        control.update_progress(queue_position=3)


def test_cli_runtime_checks_cancellation_before_spawning(tmp_path):
    control = WorkspaceRunControl(tmp_path)
    control.request_cancel("stop before provider call")
    runtime = _ObservedRuntime(
        workspace=tmp_path,
        run_control=control,
        verbose=False,
    )

    with pytest.raises(RunCancelled, match="stop before provider call"):
        runtime.invoke("ignored")

    assert runtime.spawned is False
    assert "MODEL_INVOCATION_STARTED" not in {
        event.event_type for event in control.read_events()
    }


def test_cli_runtime_emits_progress_and_stops_after_current_call(tmp_path):
    control = WorkspaceRunControl(tmp_path)
    runtime = _ObservedRuntime(
        workspace=tmp_path,
        run_control=control,
        cancel_while_parsing=True,
        verbose=False,
        timeout=5,
        idle_timeout=5,
    )

    with pytest.raises(RunCancelled, match="stop after current call"):
        runtime.invoke("request")

    assert runtime.spawned is True
    event_types = [event.event_type for event in control.read_events()]
    assert "MODEL_INVOCATION_STARTED" in event_types
    assert "RUNTIME_LOG" in event_types
    assert "MODEL_INVOCATION_COMPLETED" in event_types
    assert event_types.index("MODEL_INVOCATION_COMPLETED") < event_types.index(
        "CANCEL_OBSERVED"
    )


@pytest.mark.parametrize("failure", ["idle timeout", "hard timeout", "provider error"])
@pytest.mark.parametrize("cancel", [False, True])
def test_pending_cancel_prevents_provider_retry_only_after_process_ends(tmp_path, failure, cancel):
    control = WorkspaceRunControl(tmp_path)

    class Runtime(_ObservedRuntime):
        attempts = 0
        cleaned = False

        def _spawn(self, cmd, prompt):
            self.attempts += 1
            return super()._spawn(cmd, prompt)

        def _pump(self, proc):
            state = super()._pump(proc)
            assert proc.poll() is not None
            assert state.text == "provider-result"
            if self.attempts == 1:
                if cancel:
                    control.request_cancel("stop before retry")
                if failure != "provider error":
                    raise IdleTimeout(failure, session_id="retry-session")
                state.done_ok = False
                state.resumable = True
                state.session_id = "retry-session"
            return state

        def _cleanup(self):
            self.cleaned = True
            super()._cleanup()

    runtime = Runtime(workspace=tmp_path, run_control=control, verbose=False,
                      timeout=5, idle_timeout=5, max_resumes=1)
    if cancel:
        with pytest.raises(RunCancelled) as error:
            runtime.invoke("request")
        assert error.value.stage == "BEFORE_MODEL_RETRY"
        assert runtime.attempts == 1
    else:
        assert runtime.invoke("request") == "provider-result"
        assert runtime.attempts == 2
    assert runtime.cleaned


def test_parallel_queue_does_not_allocate_cancelled_tasks(tmp_path):
    control = WorkspaceRunControl(tmp_path)
    control.request_cancel("stop queued tasks")
    task_root = tmp_path / "tasks"

    with pytest.raises(RunCancelled, match="stop queued tasks"):
        run_parallel(
            _NeverRuns,
            [{"task": 1}, {"task": 2}],
            workspace=Directory(task_root),
            runtime_factory=lambda _: None,
            cancellation_token=control,
        )

    assert not task_root.exists()
    snapshot = control.progress()
    assert snapshot.total_tasks == 2
    assert snapshot.completed_tasks == 2
    assert snapshot.cancelled_tasks == 2


def test_simple_opt_acknowledges_cancel_without_loading_catalog(tmp_path):
    control = WorkspaceRunControl(tmp_path)
    control.request_cancel("stop workflow")
    workflow = SimpleOptWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: pytest.fail("runtime must not start"),
        run_control=control,
    )

    with pytest.raises(RunCancelled, match="stop workflow"):
        workflow.run({"definition_name": "not_loaded"})

    assert control.progress().state == RunState.CANCELLED
    assert [
        event.event_type for event in control.read_events()
    ].count("RUN_CANCELLED") == 1
