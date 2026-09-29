"""Real source export and Bundle installation, with mocked remote GPU/model calls."""

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot
from kernelgen.workflows.gems_adapter_definition import GemsAdapterDefinitionWorkflow
from kernelgen.workflows.optimization import OperatorOptimizeInput, OperatorOptimizeWorkflow
from kernelgen.workflows.optimization.artifacts import catalog_identity, snapshot_catalog
from kernelgen.workflows.optimization.sources import upload_catalog_snapshot
from kernelgen_client import Catalog, http as client
from kernelgen_client.operator_bundles import pack_operator_bundle
from kernelgen_client.protocol.schema import OperatorContract
from kernelgen_server.operator_bundles import OperatorBundleStore


@pytest.fixture
def source(tmp_path):
    from kernelgen.tests.test_flaggems_adapter import _repo
    repo = _repo(tmp_path / "gems", "def addmm_(self, mat1, mat2, *, beta=1, alpha=1): return self\n")
    # All supplementary pytest suites must remain in the source contract.
    for suite in ("tests", "benchmark"):
        (repo / suite / "test_addmm_.py").write_text("import pytest\npytestmark = pytest.mark.addmm_\n")
        (repo / suite / "test_addmm_extra.py").write_text("import pytest\npytestmark = pytest.mark.addmm_\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "--", "src", "tests", "benchmark", "conf"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    return repo


def extract(source, tmp_path, **kwargs):
    return GemsAdapterDefinitionWorkflow(cwd=tmp_path / "extract").run({
        "flaggems_repo": source, "pytest_path": "tests/test_addmm_.py", **kwargs})


def test_source_export_is_pure_complete_and_repeatable(source, tmp_path):
    output = extract(source, tmp_path)
    assert output == extract(source, tmp_path)
    assert output.operator == "addmm_"
    assert len(output.source_files) == 5
    assert output.definition_sha256 == hashlib.sha256(output.definition_path.read_bytes()).hexdigest()
    catalog = Catalog(output.catalog_path)
    assert catalog.evaluator == "flaggems"
    assert catalog.load("addmm_").correctness_workloads == []
    assert not (output.catalog_path / "ops").exists()
    assert [p.name for p in catalog.load("addmm_").definition.parameters] == ["self", "mat1", "mat2", "beta", "alpha"]
    assert "reference" not in json.loads(output.definition_path.read_text())
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"])


def test_changed_checkout_and_foreign_pytest_are_rejected(source, tmp_path):
    with pytest.raises(ValueError, match="correctness test"):
        extract(source, tmp_path, pytest_path=tmp_path / "test_foreign.py")
    with pytest.raises(ValueError, match="matching correctness"):
        extract(source, tmp_path, operator="unknown")
    (source / "tests/test_addmm_.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="clean, committed"):
        extract(source, tmp_path)


def test_no_writes_inside_source_checkout(source):
    with pytest.raises(ValueError, match="outside"):
        GemsAdapterDefinitionWorkflow(cwd=source / "output").run({
            "flaggems_repo": source, "pytest_path": "tests/test_addmm_.py"})
    assert not (source / "output").exists()


def test_new_commit_does_not_reuse_old_source_discovery(source, tmp_path):
    first = extract(source, tmp_path)
    ops = source / "src/flag_gems/ops"
    (ops / "new_addmm.py").write_text("def addmm_(self, mat1, mat2, *, beta=1, alpha=1, extra=None): return self\n")
    (ops / "__init__.py").write_text("from flag_gems.ops.new_addmm import addmm_\n__all__ = ['addmm_']\n")
    (source / "tests/test_additional.py").write_text("import pytest\npytestmark = pytest.mark.addmm_\n")
    subprocess.run(["git", "-C", str(source), "add", "--", "src", "tests"], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "new source"], check=True)
    second = extract(source, tmp_path / "next")
    assert second.source_revision != first.source_revision
    assert "tests/test_additional.py" in second.source_files
    assert "src/flag_gems/ops/new_addmm.py" in second.source_files
    assert json.loads(second.definition_path.read_text())["parameters"][-1]["name"] == "extra"
    with pytest.raises(ValueError, match="different input"):
        extract(source, tmp_path)


def test_generated_definition_snapshot_upload_and_optimizer_binding(source, tmp_path, monkeypatch):
    output = extract(source, tmp_path)
    workflow = OperatorOptimizeWorkflow(cwd=tmp_path / "optimize")
    inp = workflow.prepare_input(OperatorOptimizeInput(operator=output.operator, catalog_path=output.catalog_path))
    assert inp.optimization.mode == "kernelgen"
    assert (inp.optimization.n_parallel, inp.optimization.n_epoch, inp.optimization.max_round) == (1, 1, 10)
    workspace = workflow.root / "stages/prepare_catalog/attempts/01"
    workspace.mkdir(parents=True)
    prepared = snapshot_catalog(inp, workspace)
    workflow.validate_artifacts(prepared)
    assert catalog_identity(Path(prepared.output["catalog"]), output.operator) == inp.catalog_sha256
    store = OperatorBundleStore(tmp_path / "server-bundles")
    def upload(directory, server):
        archive = tmp_path / "upload.tar"
        with archive.open("w+b") as handle:
            digest, _ = pack_operator_bundle(directory, handle)
        return store.install(archive, digest)[0]
    monkeypatch.setattr(client, "upload_operator_bundle", upload)
    def contract(request, server):
        assert request.binding.catalog_name is None
        catalog = Catalog(store.catalog_path(request.binding.bundle_id))
        return OperatorContract(binding=request.binding, kind="flaggems", catalog_api_version="v6.0",
                                definition=catalog.load(output.operator).definition,
                                correctness_workloads=[], timing_workloads=[])
    monkeypatch.setattr(client, "get_operator_contract", contract)
    manifest = SimpleNamespace(kind="flaggems", benchmark_fingerprint="pytest-cases-fingerprint",
                               case_list=SimpleNamespace(cases=[]))
    monkeypatch.setattr(client, "inspect", lambda *_: manifest)
    context = SimpleNamespace(operator=output.operator, workspace=workspace)
    frozen = upload_catalog_snapshot(inp, prepared.output, context)
    assert frozen.bundle_id == prepared.output["bundle_id"]
    assert frozen.evaluator_kind == "flaggems"
    assert frozen.definition == Catalog(output.catalog_path).load(output.operator).definition.model_dump(mode="json", exclude_unset=True)
    assert frozen.correctness_workloads == frozen.timing_workloads == []
    # Exercise the actual evaluation request builder, not just the input model.
    from kernelgen.data.evaluation_snapshot import freeze_evaluation_snapshot
    from kernelgen.tools import kernelgen_server_adapter as adapter
    _, path = freeze_evaluation_snapshot(workflow.root, frozen)
    monkeypatch.setattr(adapter, "get_service_status", lambda *_: {"capabilities": {"operator_bundle_upload": {
        "enabled": True, "evaluation_binding": True, "evaluators": ["native", "flaggems"]}}})
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def run(self, mat1, mat2, *, beta=1, alpha=1): return self\n")
    bundle = adapter.prepare_evaluation_bundle(candidate, SimpleNamespace(
        catalog_name=frozen.catalog_name, definition=output.operator, evaluation_snapshot_path=str(path), eval_server_url="http://target",
        implementation_language=SimpleNamespace(value="python")))
    assert bundle.binding.bundle_id == frozen.bundle_id
    assert bundle.adapter_kind == "flaggems"
    manifest.benchmark_fingerprint = "changed"
    with pytest.raises(RuntimeError, match="fingerprint"):
        adapter.prepare_evaluation_bundle(candidate, SimpleNamespace(
            catalog_name=frozen.catalog_name, definition=output.operator, evaluation_snapshot_path=str(path), eval_server_url="http://target",
            implementation_language=SimpleNamespace(value="python")))


def test_gems_snapshot_requires_inspection():
    with pytest.raises(ValueError, match="inspected"):
        CatalogEvaluationSnapshot(catalog_name="x", catalog_api_version="v6.0", definition_name="x",
            definition={"name": "x", "api_version": "v6.0", "parameters": [], "outputs": []},
            correctness_workloads=[], timing_workloads=[], evaluator_kind="flaggems")


def test_old_server_cannot_silently_use_its_builtin_definition(source, tmp_path, monkeypatch):
    from kernelgen.tools import kernelgen_server_adapter as adapter
    output = extract(source, tmp_path)
    monkeypatch.setattr(adapter, "get_service_status", lambda *_: {"capabilities": {
        "operator_bundle_upload": {"enabled": True, "evaluation_binding": True}}})
    monkeypatch.setattr(adapter, "require_target_context", lambda *_: SimpleNamespace(device="H20"))
    with pytest.raises(RuntimeError, match="uploaded Gems Definitions"):
        OperatorOptimizeWorkflow(cwd=tmp_path / "run").run({"operator": output.operator, "catalog_path": output.catalog_path})
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("interrupt_optimizer", [False, True])
def test_definition_then_kernelgen_reuses_completed_calls_on_resume(source, tmp_path, monkeypatch, interrupt_optimizer):
    from kernelgen.agents.artifact_reviewer import ArtifactReviewerAgent, ReviewOutput
    from kernelgen.data._atomic import atomic_write_json
    from kernelgen.tools import kernelgen_server_adapter as adapter
    from kernelgen.workflows.optimization.kernelgen import KernelGenWorkflow, KernelGenOutput, epoch

    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "cli-state"))
    exported = extract(source, tmp_path)
    store = OperatorBundleStore(tmp_path / "server-bundles")
    calls = []
    target = {"api_version": "v6.2", "backend": "cuda", "timing": "triton",
              "target": {"device": "H20"}, "software": {}, "capabilities": {
                  "operator_contract": {"enabled": True, "test_sources": True},
                  "benchmark_reference": {"enabled": True, "level": "core", "evaluators": ["flaggems"]},
                  "operator_bundle_upload": {"enabled": True, "evaluation_binding": True,
                                             "evaluators": ["native", "flaggems"]}}}
    monkeypatch.setattr(adapter, "get_service_status", lambda *_: target)
    monkeypatch.setattr(adapter, "require_target_context", lambda *_: SimpleNamespace(device="H20"))

    def upload(directory, server):
        calls.append("upload")
        archive = tmp_path / "request.tar"
        with archive.open("w+b") as handle:
            digest, _ = pack_operator_bundle(directory, handle)
        return store.install(archive, digest)[0]

    def inspect(request, server):
        calls.append("inspect")
        assert request.binding.bundle_id and request.binding.catalog_name is None
        return SimpleNamespace(kind="flaggems", benchmark_fingerprint="original-pytest")

    def contract(request, server, *, include_test_sources=False):
        calls.append("contract")
        catalog = Catalog(store.catalog_path(request.binding.bundle_id))
        return OperatorContract(binding=request.binding, kind="flaggems", catalog_api_version="v6.0",
            definition=catalog.load(exported.operator).definition, correctness_workloads=[], timing_workloads=[],
            test_sources={"framework_revision": exported.source_revision, "files": [
                {"path": name, "content": (source / name).read_text()} for name in exported.source_files
            ]} if include_test_sources else None)

    monkeypatch.setattr(client, "upload_operator_bundle", upload)
    monkeypatch.setattr(client, "inspect", inspect)
    monkeypatch.setattr(client, "get_operator_contract", contract)
    def reference(req, server, **kwargs):
        from kernelgen_client import ReferenceResult
        calls.append("reference")
        assert req.binding.bundle_id and req.benchmark_fingerprint == "original-pytest"
        return ReferenceResult(status="PASSED", benchmark_fingerprint="original-pytest")
    monkeypatch.setattr(client, "reference", reference)
    code = "def run(self, mat1, mat2, *, beta=1, alpha=1): return self\n"

    def optimize(workflow, args):
        calls.append("kernelgen")
        assert args["evaluation_snapshot"]["evaluator_kind"] == "flaggems"
        assert args["evaluation_snapshot"]["bundle_id"]
        assert (args["n_parallel"], args["n_epoch"], args["max_round"]) == (1, 1, 10)
        if interrupt_optimizer and calls.count("kernelgen") == 1:
            raise RuntimeError("test optimizer interruption")
        if interrupt_optimizer:
            assert args["start_mode"] == "resume"
        winner = workflow._cwd / "1R/agent0"
        winner.mkdir(parents=True, exist_ok=True)
        (winner / ".best_kernel.py").write_text(code)
        for name in (".ledger.json", "optimize_definition_output.json", ".kernelgen/final-verification.json"):
            atomic_write_json(winner / name, {})
        return KernelGenOutput(definition_name=exported.operator, status="PASSED", best_code=code, best_geo_mean=1.2)

    monkeypatch.setattr(KernelGenWorkflow, "run", optimize)
    monkeypatch.setattr(epoch, "confirmed_workspace_best", lambda _: ("PASSED", {"code": code, "geo_mean": 1.2}))

    def review(agent, inp, runtime):
        assert inp["kind"] in {"tests", "code"}
        calls.append(inp["kind"] + "_review")
        return ReviewOutput(summary="fixture", reviewed_files=inp["evidence_paths"], findings=[])

    monkeypatch.setattr(ArtifactReviewerAgent, "run", review)
    root = tmp_path / "optimize"
    request = {"operator": exported.operator, "catalog_path": exported.catalog_path}
    workflow = OperatorOptimizeWorkflow(cwd=root, runtime_factory=lambda _: None)
    if interrupt_optimizer:
        with pytest.raises(RuntimeError, match="test optimizer interruption"):
            workflow.run(request)
        assert calls == ["upload", "inspect", "contract", "contract", "tests_review", "reference", "kernelgen"]
        result = workflow.run({**request, "resume": True})
    else:
        result = workflow.run(request)
    assert result.state == "SUCCEEDED"
    assert calls == ["upload", "inspect", "contract", "contract", "tests_review", "reference", "kernelgen", *(["kernelgen"] if interrupt_optimizer else []), "code_review"]
    receipt = json.loads((root / "stages/review_tests/attempts/01/result.json").read_text())
    assert receipt["result"]["data"]["test_review"] == "ACCEPTED"
    calls.clear()
    assert workflow.run({**request, "resume": True}).state == "SUCCEEDED"
    assert calls == []
