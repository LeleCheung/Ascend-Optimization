"""Real file-backed control plane, exclusively dummy business execution."""

import json
import os
from contextlib import contextmanager
import importlib
from pathlib import Path
import select
import subprocess
import sys

import pytest

from kernelgen.data._atomic import atomic_write_json
from kernelgen.framework.run_control import RunState, WorkspaceRunControl
from kernelgen.workflows.operator_development import (
    STAGES, DummyWorkflowCall, OperatorDevelopmentInput, OperatorDevelopmentWorkflow, WorkflowResult,
    campaign_status, lifecycle_status, run_campaign,
)


def run(root, *, resume=False, runner=None, operator="add", **options):
    return OperatorDevelopmentWorkflow(cwd=root, workflow_call=runner).run(
        {"operator": operator, "dummy": True, "resume": resume, **options}
    )


def control(root):
    return WorkspaceRunControl(root, root_workspace=root, scope="")


def reports(root):
    return sorted(root.glob("stages/*/attempts/*/result.json"))


def test_all_stages_share_one_operator_control_and_explicit_workspaces(tmp_path):
    seen = []

    def dummy(ctx):
        seen.append(ctx)
        assert ctx.control.root_workspace == tmp_path
        assert ctx.control.scope == f"stages/{ctx.stage}"
        assert ctx.workspace == tmp_path / "stages" / ctx.stage / "attempts/01"
        independent = WorkspaceRunControl(ctx.workspace)
        assert independent.root_workspace == tmp_path
        assert independent.scope == ctx.control.scope
        assert independent.progress().state == RunState.RUNNING
        assert len(ctx.inputs) == STAGES.index(ctx.stage)
        assert all(path.is_file() for path in ctx.inputs)
        return DummyWorkflowCall()(ctx)

    output = run(tmp_path, runner=dummy)
    assert output.state == "SUCCEEDED" and output.simulated
    assert [ctx.stage for ctx in seen] == list(STAGES)
    snapshot = control(tmp_path).progress()
    assert snapshot.state == RunState.SUCCEEDED
    assert snapshot.completed_tasks == snapshot.total_tasks == len(STAGES)
    assert all(record.state == RunState.SUCCEEDED for record in snapshot.scopes.values())
    assert len(list(tmp_path.rglob("run-progress.json"))) == 1
    assert not list(tmp_path.rglob(".ledger.json"))
    assert not list(tmp_path.rglob("run-request.json"))
    assert not (tmp_path / ".kernelgen/lifecycle-owner.json").exists()
    raw = json.loads(control(tmp_path).progress_path.read_text())
    assert "best_geo_mean" not in raw and "completed_rounds" not in raw
    first = json.loads(Path(output.output.reports[0]).read_text())
    assert first["result"]["data"]["adapter"] == "TestWriterAgent"
    assert first["result"]["data"]["called"] is False
    events = control(tmp_path).read_events()
    assert [e.sequence for e in events] == list(range(1, len(events) + 1))
    assert [e.stage for e in events if e.event_type == "LIFECYCLE_STAGE_COMPLETED"] == list(STAGES)
    assert len([e for e in events if e.event_type == "LIFECYCLE_COMPLETED"]) == 1
    assert control(tmp_path).read_events(after_sequence=events[-2].sequence) == events[-1:]


@pytest.mark.parametrize("stop_stage", STAGES)
def test_failure_stops_downstream_then_resume_reuses_successes(tmp_path, stop_stage):
    first = run(tmp_path, runner=DummyWorkflowCall(fail_at=stop_stage))
    assert first.state == "FAILED"
    index = STAGES.index(stop_stage)
    assert len(reports(tmp_path)) == index + 1
    assert control(tmp_path).progress().failed_tasks == 1
    frozen = {path: path.read_bytes() for path in reports(tmp_path)}
    seen = []

    def dummy(ctx):
        seen.append(ctx.stage)
        return DummyWorkflowCall()(ctx)

    output = run(tmp_path, resume=True, runner=dummy)
    assert output.state == "SUCCEEDED"
    assert seen == list(STAGES[index:])
    assert all(path.read_bytes() == value for path, value in frozen.items())
    assert (tmp_path / "stages" / stop_stage / "attempts/02/result.json").is_file()
    assert control(tmp_path).progress().failed_tasks == 0


def test_exception_records_stage_and_root_failure_and_releases_owner(tmp_path):
    def crash(ctx):
        raise RuntimeError("dummy stage exception")

    with pytest.raises(RuntimeError, match="dummy stage exception"):
        run(tmp_path, runner=crash)
    snapshot = control(tmp_path).progress()
    assert snapshot.state == RunState.FAILED
    assert snapshot.scopes["stages/pytest_generate"].state == RunState.FAILED
    assert not (tmp_path / ".kernelgen/lifecycle-owner.json").exists()
    assert run(tmp_path, resume=True).state == "SUCCEEDED"


def test_cancel_waits_for_complete_stage_output_and_preserves_generation(tmp_path):
    completed = []

    def dummy(ctx):
        if ctx.stage == "code_review":
            state = ctx.control.request_cancel("cancel in stage")
            assert ctx.control.request_cancel("duplicate").generation == state.generation
            completed.append("complete stage output")
        return WorkflowResult(simulated=True, output={"output": "complete stage output"})

    out = run(tmp_path, runner=dummy)
    assert out.state == "CANCELLED" and completed == ["complete stage output"]
    snapshot = control(tmp_path).progress()
    assert snapshot.state == RunState.CANCELLED
    assert snapshot.scopes["stages/code_review"].state == RunState.CANCELLED
    assert snapshot.scopes["stages/local_ci"].state == RunState.PENDING
    saved = json.loads((tmp_path / "stages/code_review/attempts/01/result.json").read_text())
    assert saved["result"]["data"]["output"] == "complete stage output"
    generation = control(tmp_path).cancellation_state().generation
    assert run(tmp_path, resume=True).state == "SUCCEEDED"
    assert control(tmp_path).cancellation_state().generation == generation + 1
    assert not control(tmp_path).is_cancellation_requested()
    assert len(reports(tmp_path)) == len(STAGES)  # committed complete output reused


def test_waiting_releases_process_and_resumes_without_resetting_prior_stages(tmp_path):
    result = run(tmp_path, runner=DummyWorkflowCall(wait_at="pr_followup"))
    assert result.state == "WAITING"
    status = lifecycle_status(tmp_path)
    assert status["state"] == "PENDING" and not status["process_alive"]
    assert status["progress"]["stage"] == "WAITING_PR_FOLLOWUP"
    assert status["progress"]["progress"]["completed_tasks"] == len(STAGES) - 1
    assert run(tmp_path, resume=True).state == "SUCCEEDED"
    assert len(reports(tmp_path)) == len(STAGES) + 1


def test_reusing_complete_run_does_not_execute_stages_or_rewrite_receipts(tmp_path):
    run(tmp_path)
    before = {path: path.read_bytes() for path in reports(tmp_path)}
    def forbidden(ctx):
        pytest.fail("completed stage executed again")
    run(tmp_path, resume=True, runner=forbidden)
    assert before == {path: path.read_bytes() for path in reports(tmp_path)}


@pytest.mark.parametrize("stage", ["pytest_generate", "pr_followup"])
def test_modified_receipt_is_rejected_even_for_final_stage(tmp_path, stage):
    run(tmp_path)
    path = tmp_path / "stages" / stage / "attempts/01/result.json"
    value = json.loads(path.read_text())
    value["result"]["message"] = "edited result"
    atomic_write_json(path, value)
    with pytest.raises(ValueError, match="receipt content changed"):
        run(tmp_path, resume=True)
    assert control(tmp_path).progress().state == RunState.FAILED


def test_successful_receipt_missing_is_not_silently_reexecuted(tmp_path):
    run(tmp_path)
    (tmp_path / "stages/pr_followup/attempts/01/result.json").unlink()
    with pytest.raises(ValueError, match="receipt is missing"):
        run(tmp_path, resume=True)


def test_missing_retry_success_does_not_fall_back_to_an_older_failed_attempt(tmp_path):
    run(tmp_path, runner=DummyWorkflowCall(fail_at="pr_followup"))
    run(tmp_path, resume=True)
    (tmp_path / "stages/pr_followup/attempts/02/result.json").unlink()
    with pytest.raises(ValueError, match="receipt is missing"):
        run(tmp_path, resume=True)


def test_attempt_numbers_grow_past_99_and_resume_uses_numeric_order(tmp_path):
    run(tmp_path, runner=DummyWorkflowCall(fail_at="pr_followup"))
    attempts = tmp_path / "stages/pr_followup/attempts"
    (attempts / "98").mkdir()  # An interrupted attempt without a committed result.
    run(tmp_path, resume=True, runner=DummyWorkflowCall(fail_at="pr_followup"))
    assert (attempts / "99/result.json").is_file()
    assert run(tmp_path, resume=True).state == "SUCCEEDED"
    assert (attempts / "100/result.json").is_file()

    def forbidden(ctx):
        pytest.fail("numeric latest successful attempt was not reused")

    assert run(tmp_path, resume=True, runner=forbidden).state == "SUCCEEDED"
    assert not (attempts / "101").exists()


def test_owner_exclusion_is_reentrant_and_pid_reuse_safe(tmp_path):
    def dummy(ctx):
        with pytest.raises(RuntimeError, match="active owner"):
            run(tmp_path, resume=True)
        return WorkflowResult(simulated=True, output={})
    run(tmp_path, runner=dummy)
    atomic_write_json(tmp_path / ".kernelgen/lifecycle-owner.json",
                      {"pid": os.getpid(), "process_start": "stale-reused-pid"})
    assert run(tmp_path, resume=True).state == "SUCCEEDED"


def test_actual_worker_exit_is_interrupted_and_resume_uses_new_attempt(tmp_path):
    script = '''
import os, sys
from kernelgen.workflows.operator_development import OperatorDevelopmentWorkflow, DummyWorkflowCall
def stage(ctx):
    if ctx.stage == "optimize":
        os._exit(17)
    return DummyWorkflowCall()(ctx)
OperatorDevelopmentWorkflow(cwd=sys.argv[1], workflow_call=stage).run({"operator":"add", "dummy":True})
'''
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path)], timeout=30)
    assert process.returncode == 17
    assert lifecycle_status(tmp_path)["state"] == "INTERRUPTED"
    assert control(tmp_path).progress().state == RunState.RUNNING  # projection is not persisted
    assert run(tmp_path, resume=True).state == "SUCCEEDED"
    assert (tmp_path / "stages/optimize/attempts/02/result.json").exists()


def test_campaign_allocates_isolated_operators_and_preserves_index(tmp_path):
    result = run_campaign(tmp_path, ["add", "mul"], dummy=True)
    index = tmp_path / ".kernelgen/operator-lifecycle-campaign.json"
    frozen = index.read_bytes()
    assert all(op["state"] == "SUCCEEDED" for op in result["operators"])
    assert not (tmp_path / ".kernelgen/run-progress.json").exists()
    a, b = tmp_path / "operators/add", tmp_path / "operators/mul"
    control(a).request_cancel("only add")
    assert not control(b).is_cancellation_requested()
    run_campaign(tmp_path, ["add", "mul"], dummy=True, resume=True)
    assert index.read_bytes() == frozen
    with pytest.raises(ValueError, match="operator list changed"):
        run_campaign(tmp_path, ["add", "sub"], dummy=True, resume=True)
    with pytest.raises(ValueError, match="operator list changed"):
        run_campaign(tmp_path, ["mul", "add"], dummy=True, resume=True)


def test_campaign_cancellation_stops_new_operators_but_resume_can_start_them(tmp_path):
    result = run_campaign(tmp_path, ["add", "mul"], dummy=True,
                          workflow_call=DummyWorkflowCall(cancel_at="pytest_review"))
    assert [op["state"] for op in result["operators"]] == ["CANCELLED", "PENDING"]
    assert not (tmp_path / "operators/mul").exists()
    resumed = run_campaign(tmp_path, ["add", "mul"], dummy=True, resume=True)
    assert all(op["state"] == "SUCCEEDED" for op in resumed["operators"])


@pytest.mark.parametrize("operators", [[], ["add", "add"], ["../escape"], ["a/b"], ["."]])
def test_invalid_operator_list_is_rejected_before_state_creation(tmp_path, operators):
    with pytest.raises(ValueError):
        run_campaign(tmp_path, operators, dummy=True)
    assert not list(tmp_path.iterdir())


def test_explicit_dummy_required_and_existing_data_preserved(tmp_path):
    with pytest.raises(NotImplementedError):
        OperatorDevelopmentWorkflow(cwd=tmp_path).run({"operator": "add"})
    with pytest.raises(NotImplementedError):
        run_campaign(tmp_path, ["add"])
    assert not list(tmp_path.iterdir())
    (tmp_path / "user.pytest").write_text("keep")
    with pytest.raises(ValueError, match="empty workspace"):
        run(tmp_path)
    assert (tmp_path / "user.pytest").read_text() == "keep"


def test_plan_change_requires_new_workspace_and_resume_is_explicit(tmp_path):
    run(tmp_path)
    with pytest.raises(ValueError, match="explicit resume"):
        run(tmp_path)
    with pytest.raises(ValueError, match="plan changed"):
        run(tmp_path, resume=True, operator="mul")


def test_fresh_process_status_and_example_resume(tmp_path):
    prefix = [sys.executable, "-m", "kernelgen.examples.operator_lifecycle.run_example"]
    args = ["run", "--dummy", "--workspace", str(tmp_path), "--operators", "add", "mul"]
    failed = subprocess.run(prefix + args + ["--fail-stage", "code_review"], capture_output=True, text=True, timeout=30)
    assert failed.returncode == 1, failed.stderr
    resumed = subprocess.run(prefix + args + ["--resume"], capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stderr
    queried = subprocess.run(prefix + ["status", str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert queried.returncode == 0, queried.stderr
    assert json.loads(queried.stdout) == campaign_status(tmp_path)


def test_uncommitted_receipt_after_exception_is_preserved_but_retried(tmp_path, monkeypatch):
    original = WorkspaceRunControl.record_event

    def fail_commit(self, event_type, **kwargs):
        if event_type == "LIFECYCLE_STAGE_RESULT":
            raise OSError("dummy interrupted receipt commit")
        return original(self, event_type, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(WorkspaceRunControl, "record_event", fail_commit)
        with pytest.raises(OSError, match="interrupted receipt"):
            run(tmp_path)
    orphan = tmp_path / "stages/pytest_generate/attempts/01/result.json"
    frozen = orphan.read_bytes()
    assert run(tmp_path, resume=True).state == "SUCCEEDED"
    assert orphan.read_bytes() == frozen
    assert (tmp_path / "stages/pytest_generate/attempts/02/result.json").exists()


def test_late_sigint_notification_does_not_return_success_with_cancel_requested(tmp_path, monkeypatch):
    module = importlib.import_module("kernelgen.framework.workflow_execution")

    @contextmanager
    def late_notice(controller):
        yield
        controller.request_cancel("late simulated SIGINT notification")

    monkeypatch.setattr(module, "cooperative_sigint", late_notice)
    assert run(tmp_path).state == "CANCELLED"
    assert control(tmp_path).progress().state == RunState.CANCELLED


def test_separate_process_can_cancel_but_does_not_interrupt_dummy_output(tmp_path):
    script = '''
import sys
from kernelgen.workflows.operator_development import OperatorDevelopmentWorkflow, WorkflowResult
def stage(ctx):
    print("ready", flush=True)
    sys.stdin.readline()
    return WorkflowResult(simulated=True, output={"output":"fully completed"})
out = OperatorDevelopmentWorkflow(cwd=sys.argv[1], workflow_call=stage).run({"operator":"add", "dummy":True})
print(out.model_dump_json(), flush=True)
'''
    proc = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([proc.stdout], [], [], 15)[0], "dummy worker did not become ready"
        assert proc.stdout.readline().strip() == "ready"
        command = [sys.executable, "-m", "kernelgen.examples.operator_lifecycle.run_example", "cancel", str(tmp_path)]
        cancelled = subprocess.run(command, capture_output=True, text=True, timeout=15)
        assert cancelled.returncode == 0, cancelled.stderr
        assert proc.poll() is None
        stdout, stderr = proc.communicate("finish output\n", timeout=15)
        assert proc.returncode == 0, stderr
        assert json.loads(stdout)["state"] == "CANCELLED"
        assert json.loads(reports(tmp_path)[0].read_text())["result"]["data"]["output"] == "fully completed"
    finally:
        if proc.poll() is None:
            proc.terminate()  # Only this test-owned dummy subprocess, never an Agent.
            proc.communicate(timeout=15)


def test_example_rejects_terminal_cancel_and_can_cancel_waiting(tmp_path, capsys):
    from kernelgen.examples.operator_lifecycle.run_example import main

    run(tmp_path, runner=DummyWorkflowCall(wait_at="pr_followup"))
    assert main(["cancel", str(tmp_path)]) == 0
    assert control(tmp_path).progress().state == RunState.CANCELLED
    assert control(tmp_path).progress().scopes["stages/pr_followup"].state == RunState.CANCELLED
    run(tmp_path, resume=True)
    with pytest.raises(SystemExit) as caught:
        main(["cancel", str(tmp_path)])
    assert caught.value.code == 2
    assert control(tmp_path).progress().state == RunState.SUCCEEDED


def test_real_runtime_is_not_constructed_even_when_factory_is_supplied(tmp_path):
    def forbidden(_):
        pytest.fail("dummy attempted to construct a real Runtime")
    workflow = OperatorDevelopmentWorkflow.bind(str(tmp_path), forbidden)
    assert workflow.run({"operator": "add", "dummy": True}).state == "SUCCEEDED"


def test_selected_stages_use_canonical_order_without_inventing_skipped_results(tmp_path):
    seen = []

    def dummy(ctx):
        seen.append(ctx)
        return DummyWorkflowCall()(ctx)

    out = run(tmp_path, stages=["code_review", "pytest_generate"], runner=dummy)
    assert out.state == "SUCCEEDED"
    assert [ctx.stage for ctx in seen] == ["pytest_generate", "code_review"]
    assert seen[0].inputs == ()
    assert seen[1].inputs == (Path(out.output.reports[0]),)
    assert all(ctx.optimize is None for ctx in seen)
    snapshot = control(tmp_path).progress()
    assert snapshot.total_tasks == snapshot.completed_tasks == 2
    assert set(snapshot.scopes) == {"stages/pytest_generate", "stages/code_review"}
    assert {path.name for path in (tmp_path / "stages").iterdir()} == {"pytest_generate", "code_review"}
    plan = json.loads((tmp_path / ".kernelgen/operator-lifecycle.json").read_text())
    assert plan["stages"] == ["pytest_generate", "code_review"]
    assert plan["optimize"] is None
    assert len(out.output.reports) == len(reports(tmp_path)) == 2
    assert run(tmp_path, stages=["pytest_generate", "code_review"], resume=True).state == "SUCCEEDED"
    assert len(reports(tmp_path)) == 2


@pytest.mark.parametrize("stage", STAGES)
def test_each_stage_can_run_alone_in_dummy_mode(tmp_path, stage):
    result = run(tmp_path, stages=[stage])
    assert result.state == "SUCCEEDED"
    assert control(tmp_path).progress().completed_tasks == control(tmp_path).progress().total_tasks == 1
    assert set(control(tmp_path).progress().scopes) == {f"stages/{stage}"}
    assert len(result.output.reports) == 1


@pytest.mark.parametrize("options,mode", [({}, "simple_opt"),
                                         ({"optimize": {"mode": "simple_opt"}}, "simple_opt"),
                                         ({"optimize": {"mode": "kernelgen"}}, "kernelgen")])
def test_optimize_mode_is_bound_to_plan_and_only_passed_to_optimize(tmp_path, options, mode):
    seen = []

    def dummy(ctx):
        seen.append(ctx)
        assert (ctx.optimize is not None) == (ctx.stage == "optimize")
        return DummyWorkflowCall()(ctx)

    out = run(tmp_path, stages=["optimize", "code_review"], runner=dummy, **options)
    assert out.state == "SUCCEEDED"
    assert seen[0].optimize.mode.value == mode
    assert seen[0].workspace == tmp_path / "stages/optimize/attempts/01"
    plan = json.loads((tmp_path / ".kernelgen/operator-lifecycle.json").read_text())
    assert plan["optimize"] == {"mode": mode}
    result = json.loads(Path(out.output.reports[0]).read_text())["result"]
    assert result["data"]["mode"] == mode and result["data"]["called"] is False
    assert result["simulated"] is True
    assert "mode" not in json.loads(Path(out.output.reports[1]).read_text())["result"]["data"]


@pytest.mark.parametrize("options", [
    {"stages": []}, {"stages": ["optimize", "optimize"]}, {"stages": ["unknown"]},
    {"stages": ["simple_opt"]}, {"stages": ["kernelgen"]},
    {"stages": ["pytest_review"], "optimize": {}},
    {"stages": ["pytest_review"], "optimize": {"mode": "kernelgen"}},
    {"optimize": {"mode": "unknown"}}, {"optimize": {"n_parallel": 2}},
])
def test_invalid_selection_or_config_rejected_before_state_creation(tmp_path, options):
    with pytest.raises(ValueError):
        run(tmp_path, **options)
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        run_campaign(tmp_path, ["add", "mul"], dummy=True, **options)
    assert not list(tmp_path.iterdir())


def test_default_plan_matches_explicit_full_selection_and_simple_opt():
    default = OperatorDevelopmentInput(operator="add")
    explicit = OperatorDevelopmentInput(operator="add", stages=list(reversed(STAGES)), optimize={"mode": "simple_opt"})
    assert default.stage_plan() == explicit.stage_plan()


@pytest.mark.parametrize("injection,status", [("fail_at", "FAILED"),
                                             ("wait_at", "WAITING"),
                                             ("cancel_at", "CANCELLED")])
def test_selected_plan_resumes_failure_wait_or_cancel_without_running_omitted_stages(tmp_path, injection, status):
    options = {"stages": ["optimize", "code_review"], "optimize": {"mode": "kernelgen"}}
    assert run(tmp_path, runner=DummyWorkflowCall(**{injection: "code_review"}), **options).state == status
    old = {path: path.read_bytes() for path in reports(tmp_path)}
    seen = []

    def dummy(ctx):
        seen.append(ctx.stage)
        return DummyWorkflowCall()(ctx)

    assert run(tmp_path, resume=True, runner=dummy, **options).state == "SUCCEEDED"
    assert seen == ([] if injection == "cancel_at" else ["code_review"])
    assert all(path.read_bytes() == content for path, content in old.items())
    snapshot = control(tmp_path).progress()
    assert snapshot.total_tasks == snapshot.completed_tasks == 2
    assert set(snapshot.scopes) == {"stages/optimize", "stages/code_review"}


@pytest.mark.parametrize("changed", [
    {"stages": ["optimize"]},
    {"stages": ["optimize", "code_review", "local_ci"]},
    {"stages": ["optimize", "code_review"], "optimize": {"mode": "kernelgen"}},
])
def test_resume_rejects_changed_selection_or_mode_before_resetting_control(tmp_path, changed):
    run(tmp_path, stages=["optimize", "code_review"], runner=DummyWorkflowCall(cancel_at="optimize"))
    frozen = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="plan changed"):
        run(tmp_path, resume=True, **changed)
    assert frozen == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


def test_campaign_freezes_selection_for_unstarted_operators_and_passes_mode(tmp_path):
    options = {"stages": ["code_review", "optimize"], "optimize": {"mode": "kernelgen"}, "dummy": True}
    first = run_campaign(tmp_path, ["add", "mul"], workflow_call=DummyWorkflowCall(cancel_at="optimize"), **options)
    assert [op["state"] for op in first["operators"]] == ["CANCELLED", "PENDING"]
    frozen = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    for changed in ({"stages": ["optimize"], "optimize": {"mode": "kernelgen"}},
                    {"stages": ["optimize", "code_review"], "optimize": {"mode": "simple_opt"}}):
        with pytest.raises(ValueError, match="execution plan changed"):
            run_campaign(tmp_path, ["add", "mul"], dummy=True, resume=True, **changed)
        assert frozen == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = run_campaign(tmp_path, ["add", "mul"], resume=True, **options)
    assert all(op["state"] == "SUCCEEDED" for op in result["operators"])
    assert all(op["progress"]["progress"]["total_tasks"] == 2 for op in result["operators"])
    for operator in ("add", "mul"):
        report = tmp_path / "operators" / operator / "stages/optimize/attempts/01/result.json"
        assert json.loads(report.read_text())["result"]["data"]["mode"] == "kernelgen"
    assert (tmp_path / ".kernelgen/operator-lifecycle-campaign.json").read_bytes() == frozen[tmp_path / ".kernelgen/operator-lifecycle-campaign.json"]


def test_example_selected_stages_mode_failure_and_resume_in_new_process(tmp_path):
    prefix = [sys.executable, "-m", "kernelgen.examples.operator_lifecycle.run_example"]
    args = ["run", "--dummy", "--workspace", str(tmp_path), "--operators", "add", "mul",
            "--stages", "optimize", "code_review", "--optimize-mode", "kernelgen"]
    failed = subprocess.run(prefix + args + ["--fail-stage", "code_review"], capture_output=True, text=True, timeout=30)
    assert failed.returncode == 1, failed.stderr
    resumed = subprocess.run(prefix + args + ["--resume"], capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout) == campaign_status(tmp_path)
    assert all(op["progress"]["progress"]["total_tasks"] == 2 for op in json.loads(resumed.stdout)["operators"])


def test_example_rejects_optimize_config_without_optimize_stage(tmp_path, capsys):
    from kernelgen.examples.operator_lifecycle.run_example import main

    with pytest.raises(SystemExit) as caught:
        main(["run", "--dummy", "--workspace", str(tmp_path), "--operators", "add",
              "--stages", "pytest_review", "--optimize-mode", "simple_opt"])
    assert caught.value.code == 2
    assert "optimize config requires" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
