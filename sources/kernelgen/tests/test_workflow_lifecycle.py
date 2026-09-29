"""Lifecycle ownership is per invocation/control, not an extra durable state."""

import pytest
from pydantic import ValidationError

from kernelgen.framework.run_control import RunCancelled, RunState, WorkspaceRunControl
from kernelgen.workflows.lifecycle import run_workflow
from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationOutput


def _run(control, execute, *, name="simple_opt"):
    return run_workflow(
        control, name=name, mode="simple_opt", definition_name="demo", max_round=1,
        checkpoint="BEFORE_TEST", execute=execute, output_model=SingleCoderOptimizationOutput,
    )


def _result():
    return SingleCoderOptimizationOutput(definition_name="demo", status="PASSED", best_geo_mean=1.25)


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_nested_same_control_has_one_lifecycle_owner_and_resets(tmp_path, outcome):
    control = WorkspaceRunControl(tmp_path)

    def child():
        if outcome == "failure":
            raise RuntimeError("child failed")
        if outcome == "cancel":
            control.request_cancel("child cancelled")
            control.checkpoint("AFTER_COMPLETE_MODEL_OUTPUT")
        return _result()

    def parent():
        return _run(control, child, name="optimize_definition")

    if outcome == "success":
        assert _run(control, parent) == _result()
    else:
        with pytest.raises(RunCancelled if outcome == "cancel" else RuntimeError):
            _run(control, parent)
    events = control.read_events()
    assert len([e for e in events if e.event_type == "WORKFLOW_STARTED"]) == 1
    terminal_type = {"success": "WORKFLOW_COMPLETED", "failure": "WORKFLOW_FAILED", "cancel": "RUN_CANCELLED"}[outcome]
    assert len([e for e in events if e.event_type == terminal_type]) == 1
    if outcome == "cancel":
        control.clear_cancellation()
    # A later invocation must own its lifecycle again, even after exceptions.
    assert _run(control, _result) == _result()
    assert len([e for e in control.read_events() if e.event_type == "WORKFLOW_STARTED"]) == 2


def test_independent_coder_control_keeps_its_scope(tmp_path):
    root = WorkspaceRunControl(tmp_path)
    child = WorkspaceRunControl(tmp_path / "1R" / "agent0")
    _run(root, lambda: _run(child, _result, name="optimize_definition"))
    assert root.progress().state == RunState.SUCCEEDED
    assert root.progress().scopes["1R/agent0"].state == RunState.SUCCEEDED
    events = [e for e in root.read_events() if e.event_type == "WORKFLOW_COMPLETED"]
    assert {e.scope for e in events} == {"", "1R/agent0"}


def test_invalid_output_is_failed_before_success_event(tmp_path):
    control = WorkspaceRunControl(tmp_path)
    with pytest.raises(ValidationError):
        _run(control, lambda: {})
    assert control.progress().state == RunState.FAILED
    assert not [e for e in control.read_events() if e.event_type == "WORKFLOW_COMPLETED"]


def test_cancel_requested_by_last_stage_prevents_success(tmp_path):
    control = WorkspaceRunControl(tmp_path)

    def last_stage():
        control.request_cancel("last stage cancelled")
        return _result()

    with pytest.raises(RunCancelled, match="last stage cancelled"):
        _run(control, last_stage)
    assert control.progress().state == RunState.CANCELLED
    assert not [e for e in control.read_events() if e.event_type == "WORKFLOW_COMPLETED"]
