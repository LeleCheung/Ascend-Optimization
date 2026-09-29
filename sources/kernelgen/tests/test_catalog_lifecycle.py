from pathlib import Path

import pytest

from kernelgen.workflows.optimization import OperatorOptimizeInput, OperatorOptimizeWorkflow
from kernelgen.workflows.operator_development import WorkflowResult
from kernelgen.workflows.operator_development.workflow import lifecycle_status
from kernelgen.workflows.optimization.operations import workflow_result

EXPECTED_CALLS = ("prepare_catalog", "review_tests", "optimize", "code_review")


def request(tmp_path, **kwargs):
    return dict(operator="add", catalog_path=str(tmp_path / "catalog"), optimization={"definition_name": "add"},
                dummy=True, **kwargs)


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
@pytest.mark.parametrize("dps", [None, True])
def test_optimize_uses_shared_inputs_and_releases_one_lease(tmp_path, monkeypatch, outcome, dps):
    from types import SimpleNamespace
    from kernelgen.data._atomic import atomic_write_json
    from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot, snapshot_optimization_context
    from kernelgen.framework.run_control import RunCancelled, WorkspaceRunControl
    from kernelgen.framework.worker_pool import WorkerLeasePool
    from kernelgen.workflows.optimization import operations as stages
    from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationInput, SingleCoderOptimizationOutput, SingleCoderOptimizationWorkflow
    from kernelgen.data import catalog

    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(catalog, "resolve_builtin_catalog_path", lambda *_: pytest.fail("must use bound snapshot"))
    snapshot = CatalogEvaluationSnapshot(
        catalog_name="uploaded-fixture", catalog_api_version="v6.2", definition_name="add",
        definition={"api_version": "v6.2", "name": "add", "parameters": [{"name": "x", "required": True}], "outputs": ["out"]},
        correctness_workloads=[{"name": "correctness", "inputs": {"x": {"type": "scalar", "value": 1}}}],
        timing_workloads=[{"name": "timing", "inputs": {"x": {"type": "scalar", "value": 1}}}],
        bundle_id="sha256:" + "a" * 64, benchmark_fingerprint="frozen-fingerprint",
    )
    snapshot_path = tmp_path / "evaluation-contract.json"
    atomic_write_json(snapshot_path, snapshot.model_dump(mode="json"))
    monkeypatch.setattr(stages, "receipt_data", lambda *_: {"snapshot": str(snapshot_path)})
    inp = OperatorOptimizeInput.model_validate({**request(tmp_path), "optimization": {
        "definition_name": "add", "mode": "simple_opt", "eval_server_url": "http://kgs:8000", "target_hardware": "Ascend",
        "eval_timeout_seconds": 91, "warmup_ms": 0, "benchmark_ms": 23, "num_trials": 2,
        "profile_enabled": True, "max_round": 4, "min_rounds": 1, "max_coder_sessions": 2,
        "early_stop_rounds": 0, "destination_passing_style": dps,
    }})
    definition, workloads = snapshot_optimization_context(snapshot)
    # Preserve the stage's original mapping, including Bundle and timeout overrides.
    expected = inp.optimization.model_dump(include=set(SingleCoderOptimizationInput.model_fields), mode="python")
    expected.update(definition=definition, workloads=workloads, catalog_name=snapshot.catalog_name,
                    evaluation_snapshot=snapshot.model_dump(mode="json"), destination_passing_style=dps if dps is not None else False,
                    eval_transport_timeout_seconds=391)
    root = tmp_path / "campaign"
    control = WorkspaceRunControl(root)
    pool = WorkerLeasePool(inp.optimization.eval_server_url)
    seen = []

    def run(optimizer, args):
        assert args == expected
        assert optimizer._cwd == root / "stages" / "optimize" / "work"
        assert optimizer._run_mode == "simple_opt"
        assert pool.snapshot()["used_workers"] == 1
        assert len(pool.snapshot()["leases"]) == 1
        seen.append(True)
        if outcome == "error":
            raise RuntimeError("synthetic optimization error")
        if outcome == "cancel":
            control.request_cancel("synthetic safe point")
            optimizer._run_control.checkpoint("AFTER_COMPLETE_OUTPUT")
        candidate = "def run(x): return x"
        (optimizer._cwd / ".best_kernel.py").write_text(candidate)
        atomic_write_json(optimizer._cwd / ".ledger.json", {})
        result = SingleCoderOptimizationOutput(definition_name="add", status="PASSED", best_code=candidate)
        atomic_write_json(optimizer._cwd / "optimize_definition_output.json", result.model_dump(mode="json"))
        return result

    monkeypatch.setattr(SingleCoderOptimizationWorkflow, "run", run)
    context = SimpleNamespace(control=control)
    if outcome == "success":
        assert stages.optimize(inp, root, lambda _: None, context).state == "SUCCEEDED"
    else:
        with pytest.raises(RuntimeError if outcome == "error" else RunCancelled):
            stages.optimize(inp, root, lambda _: None, context)
    assert seen == [True]
    assert pool.snapshot()["leases"] == []


def test_catalog_plan_uses_one_optimization_config(tmp_path):
    inp = OperatorOptimizeInput.model_validate(request(tmp_path))
    assert OperatorOptimizeWorkflow.name == "catalog_optimize"
    assert tuple(name for name, _ in OperatorOptimizeWorkflow().build(inp)) == EXPECTED_CALLS
    assert "optimize" not in inp.execution_plan()
    assert inp.optimization.max_round == 10


def test_catalog_dummy_reuses_control_and_completed_receipts(tmp_path):
    root = tmp_path / "workspace"
    output = OperatorOptimizeWorkflow(cwd=root).run(request(tmp_path))
    assert output.state == "SUCCEEDED"
    assert len(output.output.reports) == 4
    calls = []
    resumed = OperatorOptimizeWorkflow(cwd=root, workflow_call=lambda ctx: calls.append(ctx.stage)).run(request(tmp_path, resume=True))
    assert resumed.output.reports == output.output.reports and calls == []
    assert lifecycle_status(root)["progress"]["progress"]["completed_tasks"] == 4


def test_dummy_cannot_claim_real_stage_acceptance(tmp_path):
    with pytest.raises(ValueError, match="simulation mode"):
        OperatorOptimizeWorkflow(cwd=tmp_path / "work", workflow_call=lambda ctx: WorkflowResult(simulated=False, output={})).run(request(tmp_path))


def test_review_wait_blocks_downstream_and_resumes(tmp_path):
    calls = []
    def runner(ctx):
        calls.append(ctx.stage)
        return WorkflowResult(simulated=True, output={}, state="WAITING" if ctx.stage == "review_tests" else "SUCCEEDED")
    root = tmp_path / "work"
    output = OperatorOptimizeWorkflow(cwd=root, workflow_call=runner).run(request(tmp_path))
    assert output.state == "WAITING"
    assert calls == ["prepare_catalog", "review_tests"]
    output = OperatorOptimizeWorkflow(cwd=root).run(request(tmp_path, resume=True))
    assert output.state == "SUCCEEDED"
    assert Path(output.output.reports[1]).parent.name == "02"


def test_artifact_hash_is_required_for_reuse(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    path = root / "candidate.py"
    path.write_text("original")
    result = workflow_result({}, [path])
    workflow = OperatorOptimizeWorkflow(cwd=root)
    workflow.validate_artifacts(result)
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        workflow.validate_artifacts(result)


def test_foreign_artifact_is_rejected(tmp_path):
    path = tmp_path / "outside.py"
    path.write_text("outside")
    with pytest.raises(ValueError, match="foreign"):
        OperatorOptimizeWorkflow(cwd=tmp_path / "work").validate_artifacts(workflow_result({}, [path]))


def test_bundle_snapshot_never_resolves_builtin_catalog(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot, freeze_evaluation_snapshot
    from kernelgen.data.tool_context import ToolContext
    from kernelgen.tools import kernelgen_server_adapter as adapter
    from kernelgen_client import http as client

    snapshot = CatalogEvaluationSnapshot(
        catalog_name="uploaded-fixture", catalog_api_version="v6.2", definition_name="add",
        definition={"api_version": "v6.2", "name": "add", "parameters": [{"name": "x", "required": True}], "outputs": ["out"]},
        correctness_workloads=[{"name": "correctness", "inputs": {"x": {"type": "scalar", "value": 1}}}],
        timing_workloads=[{"name": "timing", "inputs": {"x": {"type": "scalar", "value": 1}}}],
        bundle_id="sha256:" + "a" * 64, benchmark_fingerprint="frozen-fingerprint")
    _, path = freeze_evaluation_snapshot(tmp_path, snapshot)
    context = ToolContext(definition="add", target_hardware="Ascend", catalog_name=snapshot.catalog_name,
                          evaluation_snapshot_path=str(path))
    kernel = tmp_path / "main.py"
    kernel.write_text("def run(x): return x")
    monkeypatch.setattr(adapter, "resolve_catalog_path", lambda **kwargs: pytest.fail("resolved a built-in catalog for a bundle"))
    monkeypatch.setattr(adapter, "get_service_status", lambda *args: {"capabilities": {"operator_bundle_upload": {"enabled": True, "evaluation_binding": True}}})
    observed = []
    def inspect(request, server):
        observed.append(request.binding)
        return SimpleNamespace(kind="native", benchmark_fingerprint=snapshot.benchmark_fingerprint, case_list=SimpleNamespace(cases=[]))
    monkeypatch.setattr(client, "inspect", inspect)
    bundle = adapter.prepare_evaluation_bundle(kernel, context)
    assert bundle.binding.bundle_id == snapshot.bundle_id
    assert observed[0].catalog_name is None
    assert adapter.build_evaluate_request(bundle, context).binding == observed[0]
    monkeypatch.setattr(client, "inspect", lambda *args: SimpleNamespace(kind="flaggems", benchmark_fingerprint=snapshot.benchmark_fingerprint))
    with pytest.raises(RuntimeError, match="evaluator kind"):
        adapter.prepare_evaluation_bundle(kernel, context)
    monkeypatch.setattr(client, "inspect", lambda *args: SimpleNamespace(kind="native", benchmark_fingerprint="changed"))
    with pytest.raises(RuntimeError, match="fingerprint"):
        adapter.prepare_evaluation_bundle(kernel, context)
    with pytest.raises(ValueError, match="different request"):
        freeze_evaluation_snapshot(tmp_path, snapshot.model_copy(update={"bundle_id": "sha256:" + "b" * 64}))


@pytest.mark.parametrize("options", [
    {"catalog_name": "another-catalog"},
])
def test_unimplemented_options_are_not_silently_dropped(tmp_path, options):
    inp = request(tmp_path)
    inp["optimization"].update(options)
    with pytest.raises(ValueError):
        OperatorOptimizeInput.model_validate(inp)


@pytest.mark.parametrize("blocked", ["missing_file", "p0", "none"])
def test_review_gate_binds_report_to_content(tmp_path, monkeypatch, blocked):
    import json
    from types import SimpleNamespace
    from kernelgen.agents.artifact_reviewer import ArtifactReviewerAgent, ReviewOutput
    from kernelgen.workflows.optimization.operations import _review
    evidence = tmp_path / "oracle.py"
    evidence.write_text("def run(x): return x")
    findings = [{"priority": "P0", "evidence": "oracle.py:1", "reason": "wrong oracle", "requested_change": "correct semantics"}] if blocked == "p0" else []
    report = ReviewOutput(summary="review complete", findings=findings,
                          reviewed_files=[str(tmp_path / "other.py") if blocked == "missing_file" else str(evidence)])
    monkeypatch.setattr(ArtifactReviewerAgent, "run", lambda *args: report)
    result = _review(lambda path: object(), SimpleNamespace(operator="add", workspace=tmp_path), "catalog", "a" * 64, [evidence])
    assert result.state == ("SUCCEEDED" if blocked == "none" else "WAITING")
    assert result.simulated is False
    assert json.loads((tmp_path / "review.json").read_text())["subject_sha256"] == "a" * 64
    OperatorOptimizeWorkflow(cwd=tmp_path).validate_artifacts(result)


def test_real_control_requires_bundle_capability_and_preserves_resume_plan(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import kernelgen_client
    import kernelgen.workflows.optimization.workflow as module
    from kernelgen.tools import kernelgen_server_adapter as adapter
    from kernelgen.framework.run_control import WorkspaceRunControl
    inp = request(tmp_path)
    inp["dummy"] = False
    monkeypatch.setattr(module, "catalog_identity", lambda *args: "a" * 64)
    monkeypatch.setattr(kernelgen_client, "Catalog", lambda path: SimpleNamespace(evaluator="native"))
    status = {"api_version": "v6.2", "backend": "npu", "timing": "profiler",
              "target": {"device": "Ascend910B"}, "software": {},
              "capabilities": {"operator_bundle_upload": {"enabled": True, "evaluation_binding": False}}}
    monkeypatch.setattr(adapter, "get_service_status", lambda *args: status)
    monkeypatch.setattr(adapter, "require_target_context", lambda *args: SimpleNamespace(device="Ascend910B"))
    calls = []
    def runner(ctx):
        calls.append(ctx.stage)
        if ctx.stage == "review_tests":
            ctx.control.request_cancel("test safe point")
        return WorkflowResult(simulated=False, output={})
    root = tmp_path / "work"
    workflow = OperatorOptimizeWorkflow(cwd=root, workflow_call=runner)
    with pytest.raises(RuntimeError, match="bundle execution"):
        workflow.run(inp)
    assert calls == [] and not root.exists()
    status["capabilities"]["operator_bundle_upload"]["evaluation_binding"] = True
    output = workflow.run(inp)
    assert output.state == "CANCELLED" and output.simulated is False
    assert calls == list(EXPECTED_CALLS[:2])
    control = WorkspaceRunControl(root)
    generation = control.cancellation_state().generation
    inp["resume"] = True
    output = workflow.run(inp)
    assert output.state == "SUCCEEDED"
    assert calls == list(EXPECTED_CALLS)
    assert control.cancellation_state().generation == generation + 1
    status["software"] = {"runtime_version": "changed"}
    with pytest.raises(ValueError, match="plan changed"):
        workflow.run(inp)


@pytest.mark.parametrize("provider", ["claude", "codex"])
def test_launcher_materializes_roles_for_readonly_and_optimization_stages(tmp_path, monkeypatch, provider):
    from kernelgen.cli import optimization as example
    from kernelgen.workflows.operator_development.contracts import OperatorDevelopmentOutput
    root = tmp_path / "work"
    monkeypatch.setattr(example, "resolve_cli_runtime_options", lambda *a, **k: {"model": "inherit", "auth_token": None, "base_url": None})
    class Workflow:
        def __init__(self, *, cwd, runtime_factory):
            self.factory = runtime_factory
        def run(self, inp):
            for stage in ("review_tests/attempts/01", "optimize/work"):
                path = root / "stages" / stage
                runtime = self.factory(str(path))
                assert (path / ".claude/agents/kernel-artifact-reviewer.md").is_file()
                assert (path / ".mcp.json").is_file() == stage.startswith("optimize")
                if stage.startswith("review"):
                    assert runtime.allowed_tools == "Read" if provider == "claude" else runtime.sandbox_mode == "read-only"
            return OperatorDevelopmentOutput(output={"operator": "add", "workspace": str(root)}, simulated=True)
    monkeypatch.setattr(example, "OperatorOptimizeWorkflow", Workflow)
    assert example.run_operator_optimization(request(tmp_path), workspace=root, runtime=provider) == 0


@pytest.mark.parametrize("cancelled", [False, True])
def test_target_validation_accepts_typed_response_and_releases_operations(tmp_path, monkeypatch, cancelled):
    import json
    from types import SimpleNamespace
    import kernelgen_client
    from kernelgen_client import http as client, Definition, Workload, EvaluateResponse
    from kernelgen.framework.run_control import WorkspaceRunControl
    from kernelgen.tools import kernelgen_server_adapter as adapter
    from kernelgen.workflows.optimization import operations as module

    inp = OperatorOptimizeInput.model_validate(request(tmp_path))
    control = WorkspaceRunControl(tmp_path)
    context = SimpleNamespace(operator="add", workspace=tmp_path, control=control)
    identity = "sha256:" + "a" * 64
    monkeypatch.setattr(module, "receipt_data", lambda ctx, stage: {"bundle_id":identity,"subject_sha256":identity,"operator_dir":str(tmp_path),"catalog":str(tmp_path)})
    definition = Definition(api_version="v6.2",name="add",parameters=[{"name":"x","required":True}],outputs=["out"],reference="def run(x): return x")
    operator = SimpleNamespace(definition=definition,
        correctness_workloads=[Workload(name="correctness",inputs={"x":{"type":"scalar","value":1}})],
        timing_workloads=[Workload(name="timing",inputs={"x":{"type":"scalar","value":1}})])
    monkeypatch.setattr(kernelgen_client, "Catalog", lambda path: SimpleNamespace(evaluator="native", load=lambda name: operator))
    monkeypatch.setattr(adapter, "get_service_status", lambda url: {"capabilities":{
        "operator_bundle_upload":{"enabled":True,"evaluation_binding":True},
        "operation_cancel":{"enabled":True,"operations":["preflight","evaluate"]}}})
    monkeypatch.setattr(client, "upload_operator_bundle", lambda *a: SimpleNamespace(bundle_id=identity,sha256="a"*64))
    monkeypatch.setattr(client, "inspect", lambda *a: SimpleNamespace(kind="native", benchmark_fingerprint="fixed"))
    seen=[]
    def execute(kind, req, server, *, timeout, operation_id):
        assert operation_id and control.active_server_operations()
        assert req.binding.bundle_id == identity
        seen.append(kind)
        if kind == "preflight":
            return {"status":"PASSED","stage":"complete"}
        if cancelled:
            control.request_cancel("cancel evaluation")
            raise client.OperationCancelledError(operation_id, kind)
        return EvaluateResponse(status="PASSED",device="npu:0",server_backend="npu",num_workloads=2,num_passed=2,geo_mean=1.0)
    monkeypatch.setattr(client, "preflight", lambda *a, **kw: execute("preflight",*a,**kw))
    monkeypatch.setattr(client, "evaluate", lambda *a, **kw: execute("evaluate",*a,**kw))
    from kernelgen.workflows.optimization.sources import upload_catalog_snapshot
    from kernelgen.data._atomic import atomic_write_json
    prepared = module.receipt_data(context, "prepare_catalog")
    snapshot = upload_catalog_snapshot(inp, prepared, context)
    path = tmp_path / "evaluation-contract.json"
    atomic_write_json(path, snapshot.model_dump(mode="json"))
    prepared["snapshot"] = str(path)
    if cancelled:
        from kernelgen.framework.run_control import RunCancelled
        with pytest.raises(RunCancelled):
            module.validate_target(inp, context, prepared)
        assert not control.active_server_operations()
        assert path.exists(), "cancellation must preserve the prepared input"
        return
    result = module.validate_target(inp, context, prepared)
    assert result.state == "SUCCEEDED" and seen == ["preflight","evaluate"]
    assert not control.active_server_operations()
    assert json.loads((tmp_path/"evaluate.json").read_text())["geo_mean"] == 1.0
    assert json.loads((tmp_path/"evaluation-contract.json").read_text())["bundle_id"] == identity


def test_lifecycle_cancel_forwards_registered_operations(tmp_path, monkeypatch):
    from kernelgen_client import http as client
    from kernelgen.framework.run_control import WorkspaceRunControl
    from kernelgen.workflows.operator_development.workflow import cancel_lifecycle
    root = tmp_path/"work"
    OperatorOptimizeWorkflow(cwd=root,workflow_call=lambda ctx: WorkflowResult(simulated=True, output={}, state="WAITING")).run(request(tmp_path))
    control = WorkspaceRunControl(root)
    operation = control.register_server_operation("evaluate","http://127.0.0.1:1234")
    called=[]
    def cancel(operation_id, url, **kwargs):
        called.append(operation_id)
        return {"state":"CANCELLED"}
    monkeypatch.setattr(client,"cancel_operation",cancel)
    result=cancel_lifecycle(root)
    assert result["requested"] and called == [operation.operation_id]
    assert not control.active_server_operations()
    assert lifecycle_status(root)["state"] == "CANCELLED"
