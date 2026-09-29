import json

import pytest

from kernelgen.framework.progress_schema import progress_v2
from kernelgen.framework.run_control import WorkspaceRunControl, RunState


def test_each_scope_owns_its_schema_and_common_control(tmp_path):
    root = WorkspaceRunControl(tmp_path)
    root.update_progress(progress_kind="tasks", total_tasks=2)
    review = WorkspaceRunControl(tmp_path, root_workspace=tmp_path, scope="review")
    review.update_progress(progress_kind="basic", state="RUNNING", message="Reading evidence")
    coder = WorkspaceRunControl(tmp_path, root_workspace=tmp_path, scope="coder")
    coder.update_progress(progress_kind="rounds", max_round=2)
    result = progress_v2(WorkspaceRunControl(tmp_path).progress().model_dump(mode="json"))
    assert result["progress"] == {"total_tasks": 2, "completed_tasks": 0, "failed_tasks": 0, "cancelled_tasks": 0}
    assert result["scopes"]["review"]["progress"] == {}
    assert result["scopes"]["review"]["state"] == "RUNNING"
    assert result["scopes"]["review"]["message"] == "Reading evidence"
    assert result["scopes"]["coder"]["progress"]["max_round"] == 2
    assert "current_epoch" not in result["scopes"]["coder"]["progress"]
    assert "current_round" not in result["scopes"]["review"]


def test_schema_does_not_create_another_measured_source(tmp_path):
    root = WorkspaceRunControl(tmp_path)
    root.update_progress(progress_kind="rounds", max_round=2)
    with pytest.raises(ValueError, match="unsupported progress"):
        root.update_progress(completed_rounds=7)
    payload = json.loads(root.progress_path.read_text())
    assert "completed_rounds" not in payload
    assert "best_geo_mean" not in payload
    snapshot = root.progress().model_dump(mode="json")
    # Simulate the existing ledger projection performed by cli.history.
    snapshot.update(completed_rounds=2, current_round=2, best_geo_mean=1.2)
    assert progress_v2(snapshot)["progress"]["completed_rounds"] == 2
    assert progress_v2(snapshot)["progress"]["best_geo_mean"] == 1.2
    assert "completed_rounds" not in json.loads(root.progress_path.read_text())


def test_invalid_schema_is_rejected_without_writing(tmp_path):
    root = WorkspaceRunControl(tmp_path)
    root.update_progress(progress_kind="basic")
    before = root.progress_path.read_bytes()
    with pytest.raises(ValueError):
        root.update_progress(progress_kind="arbitrary-untyped-dict")
    assert root.progress_path.read_bytes() == before


def test_workflow_declares_its_stage_type_without_cli_mapping(tmp_path):
    from kernelgen.workflows.operator_development import OperatorDevelopmentWorkflow, WorkflowResult
    class ReviewPipeline(OperatorDevelopmentWorkflow):
        call_progress_kinds = {"pytest_review": "tasks"}
    def review(context):
        context.control.update_progress(total_tasks=3, completed_tasks=3)
        return WorkflowResult(simulated=True, output={})
    ReviewPipeline(cwd=tmp_path, workflow_call=review).run({
        "operator": "add", "dummy": True, "stages": ["pytest_review"]})
    snapshot = progress_v2(WorkspaceRunControl(tmp_path).progress().model_dump(mode="json"))
    details = snapshot["scopes"]["stages/pytest_review"]["progress"]
    assert details["completed_tasks"] == 3
    assert "current_round" not in details and "current_epoch" not in details


def test_batch_v2_versions_missing_child_without_guessing(tmp_path):
    from kernelgen.cli.api import status
    from kernelgen.cli.batch import save_batch_request
    from kernelgen.cli.models import BatchRequestRecord, BatchChildRecord
    save_batch_request(BatchRequestRecord(batch_id="batch", workspace=tmp_path,
        source_path=tmp_path / "batch.yaml", children=[BatchChildRecord(definition="add",
            mode="simple_opt", workspace=tmp_path / "missing-child", run_id="child")]))
    result = status(tmp_path)
    assert result["schema_version"] == result["runs"][0]["schema_version"] == "2.0"
    assert result["runs"][0]["state"] == "SUBMISSION_FAILED"
    assert result["runs"][0]["progress"] is None


def test_cli_uses_latest_schema_only(tmp_path, monkeypatch):
    from kernelgen.cli.main import main
    from kernelgen.cli.api import run_status
    from kernelgen.cli.runner import new_request
    from kernelgen.cli.state import save_request
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    request = new_request(mode="simple_opt", definition="add", workspace=tmp_path / "run",
        workflow_args=[], n_parallel=1, target_hardware="test", eval_server="http://localhost:8000")
    save_request(request)
    control = WorkspaceRunControl(request.workspace)
    control.update_progress(progress_kind="rounds", state=RunState.SUCCEEDED, max_round=2)
    new = run_status(request.workspace)
    assert new["schema_version"] == "2.0"
    assert "current_epoch" not in new["progress"]["progress"]
    assert new["progress"]["progress"]["max_round"] == 2
    assert new["state"] == "SUCCEEDED"
    assert main(["status", str(request.workspace), "--detail"]) == 0
    with pytest.raises(SystemExit):
        main(["status", str(request.workspace), "--json"])
    assert main(["status", str(request.workspace)]) == 0


def test_catalog_status_projects_only_optimizer_ledger_fields(tmp_path, monkeypatch, capsys):
    from kernelgen.cli.api import run_status
    from kernelgen.cli.main import main
    from kernelgen.cli.runner import new_request
    from kernelgen.cli.state import save_request
    from kernelgen.data.ledger import Ledger
    from kernelgen.tests.test_cli_history import _record_round
    from kernelgen.workflows.optimization import OperatorOptimizeWorkflow
    from kernelgen.cli.api import build_request
    from kernelgen.framework.run_options import resolve_run_options

    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    workspace = tmp_path / "run"
    inp = dict(operator="relu", catalog_path=tmp_path,
               optimization={"definition_name": "relu", "mode": "simple_opt"}, dummy=True)
    workflow = OperatorOptimizeWorkflow(cwd=workspace)
    workflow.run(inp)
    receipts = {p: p.read_bytes() for p in workspace.glob("stages/*/attempts/*/result.json")}
    save_request(build_request(resolve_run_options({"mode": "simple_opt", "catalog_path": tmp_path}),
                               definition="relu", workspace=workspace))
    _record_round(Ledger(workspace / "stages/optimize/work"), 1, 1.2)
    status = run_status(workspace)
    assert status["progress"]["progress"]["completed_tasks"] == 4
    scopes = status["progress"]["scopes"]
    for name in ("prepare_catalog", "review_tests", "code_review"):
        assert scopes[f"stages/{name}"]["progress"] == {}
        assert "current_round" not in scopes[f"stages/{name}"]
    assert scopes["stages/optimize"]["progress"]["best_geo_mean"] == 1.2
    assert scopes["stages/optimize"]["progress"]["completed_rounds"] == 1
    assert "current_epoch" not in scopes["stages/optimize"]["progress"]
    assert main(["status", str(workspace), "--detail"]) == 0
    assert json.loads(capsys.readouterr().out) == status
    assert main(["status", str(workspace)]) == 0
    text = capsys.readouterr().out
    assert "tasks: 4/4" in text and "best_geo_mean: 1.2" in text
    workflow.run({**inp, "resume": True})
    assert all(p.read_bytes() == content for p, content in receipts.items())


def test_status_rename_does_not_change_other_json_options():
    from kernelgen.cli.main import build_parser
    parser = build_parser()
    for args in (["history", "run", "--json"], ["list", "--json"], ["server", "status", "example", "--json"]):
        assert parser.parse_args(args).json is True


def test_completed_function_keeps_its_declared_progress_on_resume(tmp_path):
    from kernelgen.workflows.operator_development import OperatorDevelopmentWorkflow, WorkflowResult
    calls = []
    def review(context):
        calls.append(context.stage)
        context.control.update_progress(progress_kind="tasks", total_tasks=3, completed_tasks=3)
        return WorkflowResult(simulated=True, output={})
    workflow = OperatorDevelopmentWorkflow(cwd=tmp_path, workflow_call=review)
    inp = {"operator": "add", "dummy": True, "stages": ["pytest_review"]}
    workflow.run(inp)
    workflow.run({**inp, "resume": True})
    assert calls == ["pytest_review"]
    record = progress_v2(WorkspaceRunControl(tmp_path).progress().model_dump(mode="json"))["scopes"]["stages/pytest_review"]
    assert record["progress_kind"] == "tasks"
    assert record["progress"]["completed_tasks"] == 3
