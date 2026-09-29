"""Installed sources use target contracts, never a local Catalog fallback."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelgen_client import EvaluateResponse, http as client
from kernelgen_server import schema as kgs_schema

# The new installed-source path requires KGS !55, but the unchanged CLI still
# supports the currently released/locked client. Validate both combinations.
if not hasattr(kgs_schema, "OperatorContract"):
    pytest.skip("installed-source tests require the KGS !55 client", allow_module_level=True)
OperatorContract = kgs_schema.OperatorContract
from kernelgen.data._atomic import atomic_write_json
from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot, freeze_evaluation_snapshot
from kernelgen.framework.run_control import WorkspaceRunControl
from kernelgen.tools import kernelgen_server_adapter as adapter
from kernelgen.workflows.optimization import OperatorOptimizeInput, OperatorOptimizeWorkflow


def contract(kind="native"):
    return OperatorContract.model_validate({
        "binding": {"catalog_name": "remote-only", "definition": "add"},
        "kind": kind, "catalog_api_version": "v6.2" if kind == "native" else "v6.0",
        "definition": {"name": "add", "api_version": "v6.2" if kind == "native" else "v6.0",
            "parameters": [{"name": "x", "required": True}, {"name": "bias", "required": False, "default": None}],
            "outputs": ["out"], "reference": "def run(x, bias=None): return x" if kind == "native" else None},
        "correctness_workloads": [{"name": "c", "inputs": {"x": {"type": "scalar", "value": 1}}}] if kind == "native" else [],
        "timing_workloads": [{"name": "t", "inputs": {"x": {"type": "scalar", "value": 2}}}] if kind == "native" else [],
    })


@pytest.fixture
def target(monkeypatch):
    state = {"api_version": "v6.2", "backend": "npu", "timing": "profiler",
        "target": {"device": "Ascend"}, "software": {},
        "capabilities": {"operator_contract": {"enabled": True, "test_sources": True},
            "benchmark_reference": {"enabled": True, "level": "core", "evaluators": ["flaggems"]},
            "operation_cancel": {"enabled": True, "operations": ["preflight", "evaluate", "reference"]}}}
    monkeypatch.setattr(adapter, "get_service_status", lambda *_: state)
    monkeypatch.setattr(adapter, "require_target_context", lambda *_: SimpleNamespace(device="Ascend"))
    monkeypatch.setattr(adapter, "resolve_catalog_path", lambda **_: pytest.fail("must not load local Catalog"))
    monkeypatch.setattr(client, "upload_operator_bundle", lambda *_: pytest.fail("installed input must not upload"))
    return state


def request(mode="simple_opt"):
    return {"operator": "add", "catalog_name": "remote-only", "optimization": {"definition_name": "add", "mode": mode}}


@pytest.mark.parametrize("kind,mode", [("native", "simple_opt"), ("native", "kernelgen"), ("flaggems", "simple_opt"), ("flaggems", "kernelgen")])
def test_installed_workflow_routes_and_resumes_without_extraction_or_local_catalog(tmp_path, monkeypatch, target, kind, mode):
    from kernelgen.agents.artifact_reviewer import ArtifactReviewerAgent, ReviewOutput
    from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationWorkflow, SingleCoderOptimizationOutput
    from kernelgen.workflows.optimization.kernelgen import KernelGenWorkflow, KernelGenOutput, epoch

    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    from kernelgen.framework.worker_pool import set_max_workers
    options = OperatorOptimizeInput.model_validate(request(mode)).optimization
    set_max_workers(options.eval_server_url, 2)
    calls = []
    def read(*_, include_test_sources=False):
        calls.append("sources" if include_test_sources else "contract")
        value = contract(kind)
        if include_test_sources:
            from kernelgen_server.protocol.schema import TestReviewSources
            value.test_sources = TestReviewSources(files=[{"path": "tests.py", "content": "original test source"}])
        return value
    def inspect(*_):
        calls.append("inspect")
        return SimpleNamespace(kind=kind, benchmark_fingerprint="frozen")
    monkeypatch.setattr(client, "get_operator_contract", read)
    monkeypatch.setattr(client, "inspect", inspect)
    def execute(operation, req, *_args, **kwargs):
        calls.append(operation)
        assert req.binding.catalog_name == "remote-only" and req.binding.bundle_id is None
        assert kwargs["operation_id"]
        assert WorkspaceRunControl(root).active_server_operations()
        return {"status": "PASSED"} if operation == "preflight" else EvaluateResponse(
            status="PASSED", device="npu:0", server_backend="npu", num_workloads=2, num_passed=2, geo_mean=1.0)
    monkeypatch.setattr(client, "preflight", lambda *a, **k: execute("preflight", *a, **k))
    monkeypatch.setattr(client, "evaluate", lambda *a, **k: execute("evaluate", *a, **k))
    def reference(req, *_args, **kwargs):
        from kernelgen_server import ReferenceResult
        calls.append("reference")
        assert req.benchmark_fingerprint == "frozen" and kwargs["operation_id"]
        assert WorkspaceRunControl(root).active_server_operations()
        assert "implementation" not in req.wire_payload()
        return ReferenceResult(status="PASSED", benchmark_fingerprint="frozen")
    monkeypatch.setattr(client, "reference", reference)
    code = "def run(x, bias=None): return x"
    def optimize(workflow, args):
        calls.append(mode)
        snapshot = args["evaluation_snapshot"]
        assert snapshot["catalog_name"] == "remote-only" and snapshot["bundle_id"] is None
        assert snapshot["evaluator_kind"] == kind
        assert snapshot["definition"]["parameters"][1]["default"] is None
        winner = workflow._cwd / "1R/agent0" if mode == "kernelgen" else workflow._cwd
        winner.mkdir(parents=True, exist_ok=True)
        (winner / ".best_kernel.py").write_text(code)
        for name in (".ledger.json", "optimize_definition_output.json", ".kernelgen/final-verification.json"):
            atomic_write_json(winner / name, {})
        output_type = KernelGenOutput if mode == "kernelgen" else SingleCoderOptimizationOutput
        return output_type(definition_name="add", status="PASSED", best_code=code, best_geo_mean=1.2)
    monkeypatch.setattr(SingleCoderOptimizationWorkflow, "run", optimize)
    monkeypatch.setattr(KernelGenWorkflow, "run", optimize)
    monkeypatch.setattr(epoch, "confirmed_workspace_best", lambda _: ("PASSED", {"code": code, "geo_mean": 1.2}))
    def review(agent, inp, runtime):
        assert inp["kind"] in {"tests", "code"}
        calls.append(inp["kind"] + "_review")
        return ReviewOutput(summary="test", reviewed_files=inp["evidence_paths"], findings=[])
    monkeypatch.setattr(ArtifactReviewerAgent, "run", review)
    root = tmp_path / "run"
    workflow = OperatorOptimizeWorkflow(cwd=root, runtime_factory=lambda _: None)
    result = workflow.run(request(mode))
    assert result.state == "SUCCEEDED"
    assert calls == ["contract", "inspect", "sources", "tests_review", *(["preflight", "evaluate"] if kind == "native" else ["reference"]), mode, "code_review"]
    assert (root / "stages/review_tests/attempts/01/result.json").is_file()
    assert len(list(root.rglob("evaluation-contract.json"))) == 1
    assert not WorkspaceRunControl(root).active_server_operations()
    calls.clear()
    assert workflow.run({**request(mode), "resume": True}).state == "SUCCEEDED"
    assert calls == []


def test_kernelgen_freezes_gems_contract_without_native_oracle(tmp_path):
    from kernelgen.workflows.optimization.kernelgen import KernelGenInput
    from kernelgen.workflows.optimization.kernelgen.preparation import freeze_run_evaluation_contract
    snapshot = CatalogEvaluationSnapshot.from_server_contract(contract("flaggems"), "frozen")
    inp = KernelGenInput(definition={"name": "add"}, catalog_name="remote-only", target_hardware="Ascend",
                        evaluation_snapshot=snapshot.model_dump(mode="json"))
    prepared, frozen = freeze_run_evaluation_contract(tmp_path, inp)
    assert frozen["evaluator_kind"] == "flaggems"
    assert frozen["benchmark_fingerprint"] == "frozen"
    assert frozen["correctness_workloads"] == frozen["timing_workloads"] == []
    assert prepared.definition.name == "add"
    from kernelgen.workflows.optimization.kernelgen.epoch import resolve_optimization_context
    _, workloads, coder_snapshot = resolve_optimization_context(prepared, frozen)
    assert workloads == []
    assert coder_snapshot == frozen
    changed = snapshot.model_copy(update={"benchmark_fingerprint": "different"})
    with pytest.raises(ValueError):
        freeze_run_evaluation_contract(tmp_path, inp.model_copy(update={"evaluation_snapshot": changed.model_dump(mode="json")}))


def test_installed_input_requires_only_export_capability(tmp_path, target):
    target["capabilities"]["operator_contract"]["enabled"] = False
    with pytest.raises(RuntimeError, match="contract export"):
        OperatorOptimizeWorkflow(cwd=tmp_path / "run").run(request())
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("fields", [{"catalog_name": "x", "catalog_path": "/tmp/catalog"},
    {"catalog_name": "../x"}, {"bundle_id": "sha256:" + "a" * 64}])
def test_public_source_is_one_name_or_path_not_a_digest(fields):
    with pytest.raises(ValueError):
        OperatorOptimizeInput(operator="add", optimization={"definition_name": "add"}, **fields)


@pytest.mark.parametrize("kind", ["native", "flaggems"])
def test_eval_uses_same_remote_snapshot_and_rejects_drift(tmp_path, monkeypatch, target, kind):
    remote = contract(kind)
    snapshot = CatalogEvaluationSnapshot.from_server_contract(remote, "frozen")
    _, path = freeze_evaluation_snapshot(tmp_path, snapshot)
    monkeypatch.setattr(client, "get_operator_contract", lambda *_: remote)
    case = SimpleNamespace(model_dump=lambda **_: {"case_id": "adapter-timing"})
    manifest = SimpleNamespace(kind=kind, benchmark_fingerprint="frozen", case_list=SimpleNamespace(cases=[case]))
    monkeypatch.setattr(client, "inspect", lambda *_: manifest)
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def run(x, bias=None): return x")
    context = SimpleNamespace(catalog_name="remote-only", definition="add", evaluation_snapshot_path=str(path),
        eval_server_url="http://target", implementation_language=SimpleNamespace(value="python"))
    bundle = adapter.prepare_evaluation_bundle(candidate, context)
    assert bundle.adapter_kind == kind and bundle.binding.catalog_name == "remote-only"
    assert bundle.timing_workloads[0].name == ("t" if kind == "native" else "adapter-timing")
    manifest.benchmark_fingerprint = "changed"
    with pytest.raises(RuntimeError, match="fingerprint"):
        adapter.prepare_evaluation_bundle(candidate, context)
    manifest.benchmark_fingerprint = "frozen"
    remote.definition.description = "changed contract"
    with pytest.raises(RuntimeError, match="Catalog changed"):
        adapter.prepare_evaluation_bundle(candidate, context)
    changed = CatalogEvaluationSnapshot.from_server_contract(remote, "frozen")
    with pytest.raises(ValueError, match="different request"):
        freeze_evaluation_snapshot(tmp_path, changed)


def test_paired_http_contract_readiness_and_candidate_binding(tmp_path, monkeypatch):
    """Real KGS routing/SDK/schema; hardware execution is explicitly simulated."""
    pytest.importorskip("torch")
    from fastapi.testclient import TestClient
    import kernelgen_server.api.app as server
    from kernelgen_server.protocol import client as transport
    from kernelgen_server.schema import PreflightResult
    from kernelgen.data.tool_context import ToolContext
    from kernelgen.workflows.optimization import operations

    monkeypatch.setattr(server, "_make_device", lambda _: SimpleNamespace(count_devices_safe=lambda: 1))
    monkeypatch.setattr(server, "configure_device", lambda *a, **k: None)
    monkeypatch.setattr(server, "probe_device", lambda *a, **k: None)
    monkeypatch.setattr(server, "environment_info", lambda *a: {})
    seen = []
    def execute(kind, request, backend, device, timing, **kwargs):
        seen.append((kind, request.binding.catalog_name))
        if kind == "preflight":
            return PreflightResult(status="PASSED", stage="complete")
        return EvaluateResponse(status="PASSED", device=device, server_backend=backend, num_workloads=2, num_passed=2, geo_mean=1.0)
    monkeypatch.setattr(server, "run_isolated", execute)
    monkeypatch.setattr(adapter, "resolve_catalog_path", lambda **_: pytest.fail("KG local Catalog lookup"))
    app = server.create_app(backend="cuda", enable_debug_jobs=False, operator_bundle_root=tmp_path / "bundles",
                            profile_artifact_root=tmp_path / "profiles")
    with TestClient(app) as http:
        def post(url, endpoint, payload, timeout, operation_id=None):
            response = http.post(endpoint, json=payload, headers={"X-KernelGen-Operation-Id": operation_id} if operation_id else {})
            assert response.status_code == 200, response.text
            return response.json()
        monkeypatch.setattr(transport, "_post", post)
        monkeypatch.setattr(adapter, "get_service_status", lambda *_: http.get("/status").json())
        inp = OperatorOptimizeInput(operator="identity", catalog_name="simple-v6-test", skip_review=True,
                                    optimization={"definition_name": "identity"})
        root = tmp_path / "run"
        workspace = root / "prepare"
        workspace.mkdir(parents=True)
        context = SimpleNamespace(operator="identity", workspace=workspace, control=WorkspaceRunControl(root), outputs={})
        prepared = operations.prepare_catalog(inp, root, None, context)
        context.outputs["prepare_catalog"] = prepared.output
        context.workspace = root / "readiness"
        context.workspace.mkdir()
        ready = operations.review_tests(inp, root, None, context)
        assert ready.state == "SUCCEEDED"
        candidate = root / "candidate.py"
        candidate.write_text("def run(x): return x")
        tool = ToolContext(definition="identity", catalog_name="simple-v6-test", target_hardware="fixture",
            eval_server_url="http://test", evaluation_snapshot_path=ready.output["snapshot"])
        bundle = adapter.prepare_evaluation_bundle(candidate, tool)
        request_body = adapter.build_evaluate_request(bundle, tool)
        assert client.evaluate(request_body, tool.eval_server_url).status == "PASSED"
        assert seen == [("preflight", "simple-v6-test"), ("evaluate", "simple-v6-test"), ("evaluate", "simple-v6-test")]
        assert not context.control.active_server_operations()
        state = http.get("/status").json()["scheduler"]
        assert state["active"] == state["waiting"] == state["broken"] == 0
