"""Existing Catalog input shares lifecycle gates without source extraction."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelgen.workflows.optimization import OperatorOptimizeInput, OperatorOptimizeWorkflow
from kernelgen.workflows.optimization.artifacts import catalog_identity, snapshot_catalog
from kernelgen.workflows.operator_development import WorkflowResult
from kernelgen.workflows.operator_development.workflow import lifecycle_status


@pytest.mark.parametrize('capability,accepted', [
    ({}, False),
    ({'enabled': True, 'workload_conditions': True, 'policy_id': 'pinned', 'flags': None}, False),
    ({'enabled': True, 'workload_conditions': True, 'policy_id': 'other', 'flags': {}}, False),
    ({'enabled': True, 'workload_conditions': True, 'policy_id': 'pinned', 'flags': {'support_fp64': False}}, True),
])
def test_source_policy_requires_target_capability_not_release(monkeypatch, capability, accepted):
    from kernelgen.workflows.optimization.sources import require_source_policy
    import kernelgen.tools.kernelgen_server_adapter as adapter
    monkeypatch.setattr(adapter, 'get_service_status', lambda _: {
        'server_version': 'v999.0.0', 'capabilities': {'native_source_policy': capability}})
    definition = SimpleNamespace(source_policy_id='pinned')
    if accepted:
        require_source_policy(definition, 'server')
    else:
        with pytest.raises(RuntimeError, match='source policy'):
            require_source_policy(definition, 'server')


def test_ordinary_catalog_does_not_require_source_policy(monkeypatch):
    from kernelgen.workflows.optimization.sources import require_source_policy
    import kernelgen.tools.kernelgen_server_adapter as adapter
    monkeypatch.setattr(adapter, 'get_service_status', lambda _: pytest.fail('unnecessary status call'))
    require_source_policy(SimpleNamespace(source_policy_id=None), 'server')


@pytest.fixture
def catalog(tmp_path):
    root = tmp_path / "catalog"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"api_version": "v6.2", "evaluator": "native", "layout": "per-operator"}))
    for name in ("add", "other"):
        op = root / "ops/group" / name
        op.mkdir(parents=True)
        (op / "definition.json").write_text(json.dumps({"api_version": "v6.2", "name": name,
            "parameters": [{"name": "x", "required": True}], "outputs": ["out"]}))
        (op / "oracle.py").write_text("REFERENCE_DEVICE = 'cpu'\ndef run(x): return x\n")
        for workload in ("correctness", "timing"):
            (op / f"{workload}.jsonl").write_text(json.dumps({"name": workload, "inputs": {"x": {"type": "scalar", "value": 1}}}) + "\n")
    return root


def request(catalog, **kwargs):
    return {"operator": "add", "catalog_path": str(catalog), "optimization": {"definition_name": "add", "mode": "simple_opt"}, **kwargs}


def test_old_reference_plan_resumes_without_rewriting_or_relaxing_contract(catalog, tmp_path, monkeypatch):
    source = tmp_path / "reference.cu"
    source.write_text("__global__ void example() {}")
    root = tmp_path / "legacy-run"
    inp = request(catalog, dummy=True)
    inp["optimization"]["reference_code_path"] = str(source)
    workflow = OperatorOptimizeWorkflow(cwd=root)
    original_plan = OperatorOptimizeInput.execution_plan
    def legacy_plan(model):
        serialized = json.dumps(original_plan(model)).replace("reference_code_path", "reference_triton_path").replace(
            "reference_code_prompt_path", "reference_triton_prompt_path")
        return json.loads(serialized)
    # Produce both plan and receipts in the old format, rather than tampering
    # with a new plan whose existing receipts correctly bind its original hash.
    with monkeypatch.context() as patch:
        patch.setattr(OperatorOptimizeInput, "execution_plan", legacy_plan)
        assert workflow.run(inp).state == "SUCCEEDED"
    plan_path = root / ".kernelgen/operator-lifecycle.json"
    saved = plan_path.read_text()
    assert workflow.run({**inp, "resume": True}).state == "SUCCEEDED"
    assert plan_path.read_text() == saved
    source.write_text("different CUDA source")
    with pytest.raises(ValueError, match="plan changed"):
        workflow.run({**inp, "resume": True})
    assert plan_path.read_text() == saved


def test_one_catalog_workflow_has_no_source_specific_inputs(catalog):
    from kernelgen.framework.workflow import Workflow
    assert OperatorOptimizeWorkflow.__bases__ == (Workflow,)
    inp = OperatorOptimizeInput.model_validate(request(catalog))
    calls = OperatorOptimizeWorkflow(runtime_factory=lambda _: None).build(inp)
    assert tuple(name for name, _ in calls) == ("prepare_catalog", "review_tests", "optimize", "code_review")
    assert "stages" not in OperatorOptimizeInput.model_fields
    assert calls[0][1].func.__name__ == "prepare_catalog"
    for key in ("flaggems_repo", "revision", "case_list_path"):
        with pytest.raises(ValueError, match="Extra inputs"):
            OperatorOptimizeInput.model_validate(request(catalog, **{key: "unused"}))


def test_snapshot_selects_one_operator_without_changing_input(catalog, tmp_path):
    original = {p: p.read_bytes() for p in catalog.rglob("*") if p.is_file()}
    workflow = OperatorOptimizeWorkflow(cwd=tmp_path / "run")
    inp = workflow.prepare_input(OperatorOptimizeInput.model_validate(request(catalog)))
    workspace = tmp_path / "run/stages/prepare_catalog/attempts/01"
    workspace.mkdir(parents=True)
    result = snapshot_catalog(inp, workspace)
    assert result.output["source_evidence"] == []
    assert not (workspace / "catalog/ops/group/other").exists()
    assert catalog_identity(Path(result.output["catalog"]), "add") == inp.catalog_sha256
    assert original == {p: p.read_bytes() for p in catalog.rglob("*") if p.is_file()}
    workflow.validate_artifacts(result)
    (Path(result.output["operator_dir"]) / "oracle.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        workflow.validate_artifacts(result)


@pytest.mark.parametrize("change", ["oracle", "manifest", "empty", "symlink"])
def test_changed_or_invalid_catalog_is_rejected(catalog, tmp_path, change):
    workflow = OperatorOptimizeWorkflow(cwd=tmp_path / "run")
    inp = workflow.prepare_input(OperatorOptimizeInput.model_validate(request(catalog)))
    op = catalog / "ops/group/add"
    if change == "oracle":
        with (op / "oracle.py").open("a") as handle:
            handle.write("# changed\n")
    elif change == "manifest":
        with (catalog / "manifest.json").open("a") as handle:
            handle.write("\n")
    elif change == "empty":
        (op / "correctness.jsonl").write_text("")
    else:
        (op / "link").symlink_to(op / "oracle.py")
    with pytest.raises(ValueError):
        workflow.prepare_input(inp)


def test_real_lifecycle_skips_extractor_cancels_and_resumes(catalog, tmp_path, monkeypatch):
    from kernelgen.tools import kernelgen_server_adapter as adapter
    from kernelgen.workflows.catalog_extract import CatalogExtractWorkflow

    monkeypatch.setattr(CatalogExtractWorkflow, "run", lambda *_: pytest.fail("must not extract"))
    status = {"api_version": "v6.2", "backend": "npu", "timing": "profiler", "target": {"device": "Ascend"}, "software": {},
              "capabilities": {"operator_bundle_upload": {"enabled": True, "evaluation_binding": True}}}
    monkeypatch.setattr(adapter, "get_service_status", lambda *_: status)
    monkeypatch.setattr(adapter, "require_target_context", lambda *_: SimpleNamespace(device="Ascend"))
    root = tmp_path / "run"
    calls = []
    def runner(context):
        calls.append(context.stage)
        if context.stage == "prepare_catalog":
            inp = workflow.prepare_input(OperatorOptimizeInput.model_validate(request(catalog)))
            return snapshot_catalog(inp, context.workspace)
        if context.stage == "review_tests":
            context.control.request_cancel("complete review output")
        return WorkflowResult(simulated=False, output={})
    workflow = OperatorOptimizeWorkflow(cwd=root, workflow_call=runner)
    output = workflow.run(request(catalog))
    assert output.state == "CANCELLED"
    output = workflow.run(request(catalog, resume=True))
    assert output.state == "SUCCEEDED"
    assert calls == ["prepare_catalog", "review_tests", "optimize", "code_review"]
    assert lifecycle_status(root)["progress"]["progress"]["completed_tasks"] == 4
    with (catalog / "ops/group/add/oracle.py").open("a") as handle:
        handle.write("# source changed\n")
    with pytest.raises(ValueError, match="plan changed"):
        workflow.run(request(catalog, resume=True))


def test_dummy_needs_neither_catalog_nor_runtime(tmp_path):
    output = OperatorOptimizeWorkflow(cwd=tmp_path / "run").run(request(tmp_path / "missing", dummy=True))
    assert output.state == "SUCCEEDED"
    assert len(output.output.reports) == 4


@pytest.mark.parametrize("interruption", [None, "code_p0", "optimize_failed", "optimize_cancelled", "review_waiting", "catalog_waiting", "preflight_failed", "readiness_cancelled"])
def test_existing_catalog_full_pipeline_with_fake_model_and_kgs(catalog, tmp_path, monkeypatch, interruption):
    from kernelgen.agents.artifact_reviewer import ArtifactReviewerAgent, ReviewOutput
    from kernelgen.workflows.catalog_extract import CatalogExtractWorkflow
    from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationWorkflow, SingleCoderOptimizationOutput
    from kernelgen.framework.run_control import WorkspaceRunControl
    from kernelgen.tools import kernelgen_server_adapter as adapter
    from kernelgen_client import EvaluateResponse, http as client
    from kernelgen_client.operator_bundles import pack_operator_bundle
    import tempfile

    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(CatalogExtractWorkflow, "run", lambda *_: pytest.fail("must not extract"))
    status = {"api_version": "v6.2", "backend": "npu", "timing": "profiler", "target": {"device": "Ascend"}, "software": {},
              "capabilities": {"operator_contract": {"enabled": True, "test_sources": True},
                               "operator_bundle_upload": {"enabled": True, "evaluation_binding": True},
                               "operation_cancel": {"enabled": True, "operations": ["preflight", "evaluate"]}}}
    monkeypatch.setattr(adapter, "get_service_status", lambda *_: status)
    monkeypatch.setattr(adapter, "require_target_context", lambda *_: SimpleNamespace(device="Ascend"))
    calls = []
    catalog_reviews = []
    target_calls = []
    review_workspaces = []
    optimizer_calls = []
    def review(agent, inp, runtime):
        calls.append(inp["kind"] + "_review")
        if inp["kind"] == "tests":
            catalog_reviews.append(runtime)
            if interruption == "catalog_waiting" and len(catalog_reviews) == 1:
                return ReviewOutput(summary="requires source fix", reviewed_files=inp["evidence_paths"], findings=[{
                    "priority": "P0", "evidence": "oracle.py:1", "reason": "fixture blocker", "requested_change": "review source"}])
        if inp["kind"] == "code":
            review_workspaces.append(Path(runtime))
            assert not (Path(runtime) / "review.json").exists()
            if interruption == "code_p0":
                return ReviewOutput(summary="Recorded ABI issue", reviewed_files=inp["evidence_paths"], findings=[{
                    "priority": "P0", "evidence": "candidate.py:1", "reason": "unsupported arity",
                    "requested_change": "generalize the next epoch candidate"}])
            if interruption == "review_waiting" and len(review_workspaces) == 1:
                return ReviewOutput(summary="missing evidence", reviewed_files=inp["evidence_paths"][:1], findings=[])
        return ReviewOutput(summary="fixture review", reviewed_files=inp["evidence_paths"], findings=[])
    monkeypatch.setattr(ArtifactReviewerAgent, "run", review)
    def upload(path, server):
        calls.append("upload")
        with tempfile.TemporaryFile() as output:
            digest, _ = pack_operator_bundle(path, output)
        return SimpleNamespace(bundle_id="sha256:" + digest, sha256=digest)
    monkeypatch.setattr(client, "upload_operator_bundle", upload)
    monkeypatch.setattr(client, "inspect", lambda *_: SimpleNamespace(kind="native", benchmark_fingerprint="fixed"))
    from kernelgen_server import Catalog, OperatorContract
    from kernelgen_server.evaluation.test_review_sources import export_test_sources
    def read_contract(req, server, *, include_test_sources):
        source = Catalog(catalog)
        op = source.load("add")
        assert include_test_sources
        return OperatorContract(binding=req.binding, kind="native", catalog_api_version="v6.2",
                                definition=op.definition, correctness_workloads=op.correctness_workloads,
                                timing_workloads=op.timing_workloads, test_sources=export_test_sources(source, op))
    monkeypatch.setattr(client, "get_operator_contract", read_contract)
    root = tmp_path / "run"
    def execute(kind, request, server, **kwargs):
        calls.append(kind)
        target_calls.append(kind)
        assert kwargs["operation_id"]
        assert WorkspaceRunControl(root).active_server_operations()
        if kind == "preflight":
            if interruption == "preflight_failed" and len(target_calls) == 1:
                return {"status": "FAILED"}
            return {"status": "PASSED"}
        if interruption == "readiness_cancelled" and target_calls.count("evaluate") == 1:
            context_control = WorkspaceRunControl(root)
            context_control.request_cancel("cancel during reference eval")
            raise client.OperationCancelledError(kwargs["operation_id"], "evaluate")
        return EvaluateResponse(status="PASSED", device="npu:0", server_backend="npu", num_workloads=2, num_passed=2, geo_mean=1.0)
    monkeypatch.setattr(client, "preflight", lambda *args, **kwargs: execute("preflight", *args, **kwargs))
    monkeypatch.setattr(client, "evaluate", lambda *args, **kwargs: execute("evaluate", *args, **kwargs))
    def optimize(workflow, inp):
        calls.append("optimize")
        assert inp["evaluation_snapshot"]["bundle_id"].startswith("sha256:")
        assert workflow._cwd == root / "stages/optimize/work"
        optimizer_calls.append(workflow._cwd)
        ledger = workflow._cwd / ".ledger.json"
        if len(optimizer_calls) == 1:
            ledger.write_text('{"rounds": [{"round": 1}]}')
        else:
            assert ledger.read_text() == '{"rounds": [{"round": 1}]}'
        if interruption == "optimize_cancelled" and len(optimizer_calls) == 1:
            workflow._run_control.request_cancel("interrupt optimizer")
            workflow._run_control.checkpoint("AFTER_MODEL_OUTPUT")
        code = "def run(x): return x"
        (workflow._cwd / ".best_kernel.py").write_text(code)
        failed = interruption == "optimize_failed" and len(optimizer_calls) == 1
        output = SingleCoderOptimizationOutput(definition_name="add", status="FAILED" if failed else "PASSED", best_code=code)
        (workflow._cwd / "optimize_definition_output.json").write_text(output.model_dump_json())
        return output
    monkeypatch.setattr(SingleCoderOptimizationWorkflow, "run", optimize)
    workflow = OperatorOptimizeWorkflow(cwd=root, runtime_factory=lambda path: path)
    output = workflow.run(request(catalog))
    if interruption and interruption != "code_p0":
        assert output.state == {"optimize_failed": "FAILED", "optimize_cancelled": "CANCELLED", "review_waiting": "WAITING",
                               "catalog_waiting": "WAITING", "preflight_failed": "FAILED", "readiness_cancelled": "CANCELLED"}[interruption]
        if interruption == "catalog_waiting":
            assert calls == ["upload", "tests_review"], "upload must not execute unreviewed content"
        original = {p: p.read_bytes() for p in root.glob("stages/*/attempts/*/result.json")}
        calls.clear()
        output = workflow.run(request(catalog, resume=True))
        if interruption in {"catalog_waiting", "preflight_failed", "readiness_cancelled"}:
            assert calls == ["tests_review", "preflight", "evaluate", "optimize", "code_review"]
        else:
            assert calls == (["code_review"] if interruption == "review_waiting" else ["optimize", "code_review"])
        assert all(p.read_bytes() == content for p, content in original.items())
        stage = ("review_tests" if interruption in {"catalog_waiting", "preflight_failed", "readiness_cancelled"}
                 else "code_review" if interruption == "review_waiting" else "optimize")
        assert (root / "stages" / stage / "attempts/02/result.json").is_file()
    else:
        assert calls == ["upload", "tests_review", "preflight", "evaluate", "optimize", "code_review"]
    assert output.state == "SUCCEEDED"
    if interruption == "code_p0":
        report = json.loads((review_workspaces[0] / "review.json").read_text())
        assert report["policy"] == "advisory"
        assert report["findings"][0]["priority"] == "P0"
        assert report["findings"][0]["requested_change"] == "generalize the next epoch candidate"
    assert len(optimizer_calls) == (2 if interruption in {"optimize_failed", "optimize_cancelled"} else 1)
    if interruption == "review_waiting":
        assert [p.name for p in review_workspaces] == ["01", "02"]
    assert not WorkspaceRunControl(root).active_server_operations()
    calls.clear()
    resumed = OperatorOptimizeWorkflow(cwd=root, runtime_factory=lambda _: None).run(request(catalog, resume=True))
    assert resumed.output.reports == output.output.reports and calls == []
    original = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with (catalog / "ops/group/add/oracle.py").open("a") as handle:
        handle.write("# changed input\n")
    with pytest.raises(ValueError, match="plan changed"):
        workflow.run(request(catalog, resume=True))
    assert not calls
    assert all(p.read_bytes() == content for p, content in original.items())
