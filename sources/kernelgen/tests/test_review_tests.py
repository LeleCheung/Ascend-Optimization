"""One review policy for all sources; only explicit skip omits model review."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelgen.agents.artifact_reviewer import ArtifactReviewerAgent, ReviewOutput
from kernelgen.cli.api import build_request
from kernelgen.cli.batch import load_batch_file
from kernelgen.cli.main import build_parser
from kernelgen.framework.run_options import resolve_run_options
from kernelgen.workflows.optimization import operations
from kernelgen.workflows.optimization.artifacts import workflow_result


def test_review_sources_are_lossless_readable_copies_with_bound_identity(tmp_path):
    from kernelgen.workflows.optimization.sources import review_source_files
    from kernelgen.workflows.optimization.artifacts import file_digest

    source = tmp_path / "test-sources.json"
    content = "# source comment\nx = 'line with quote'\n" * 2000
    source.write_text(json.dumps({"framework_revision": "a" * 40,
        "files": [{"path": "benchmark/base.py", "content": content},
                  {"path": "tests/test_op.py", "content": "raise RuntimeError('must not execute')\n"}]}))
    before = source.read_bytes()
    paths = review_source_files(source, tmp_path / "attempt")
    index = json.loads(paths[0].read_text())
    assert index["source_sha256"] == file_digest(source)
    assert index["framework_revision"] == "a" * 40
    assert index["files"][0]["source_path"] == "benchmark/base.py"
    assert paths[1].read_text() == content
    assert len(paths[1].read_text().splitlines()) == 4000
    assert source.read_bytes() == before
    for row, path in zip(index["files"], paths[1:]):
        assert Path(row["read_path"]) == path
        assert row["sha256"] == file_digest(path)
    with pytest.raises(FileExistsError):
        review_source_files(source, tmp_path / "attempt")
    assert paths[1].read_text() == content


def test_code_review_reuses_readable_evidence_without_raw_source_json(tmp_path, monkeypatch):
    candidate = tmp_path / "best.py"
    candidate.write_text("def run(x): return x\n")
    contract, raw, readable, index = [str(tmp_path / name) for name in (
        "contract.json", "test-sources.json", "review-sources/01.txt", "review-sources/index.json")]
    context = SimpleNamespace(outputs={
        "optimize": {"candidate": str(candidate), "artifacts": {str(candidate): "hash", contract: "hash"}},
        "prepare_catalog": {"test_evidence": raw, "artifacts": {raw: "hash", contract: "hash", "bundle.tar": "hash"}},
        "review_tests": {"artifacts": {contract: "hash", index: "hash", readable: "hash"}},
    })
    seen = []
    monkeypatch.setattr(operations, "_review", lambda runtime, context, kind, digest, evidence: seen.append((kind, evidence)))
    operations.code_review(None, None, None, context)
    assert seen == [("code", [str(candidate), contract, index, readable])]


@pytest.mark.parametrize("source", ["installed_native", "installed_gems", "local_native", "local_gems"])
@pytest.mark.parametrize("skip", [False, True])
def test_same_review_and_target_validation_policy(tmp_path, monkeypatch, source, skip):
    snapshot = tmp_path / "evaluation-contract.json"
    evidence = tmp_path / "test-sources.json"
    snapshot.write_text(json.dumps({"evaluator_kind": "flaggems" if "gems" in source else "native"}))
    evidence.write_text('{"files": [{"path": "test.py", "content": "original tests"}]}')
    prepared = {"snapshot": str(snapshot), "test_evidence": str(evidence)}
    if source.startswith("installed"):
        prepared["source"] = "installed_catalog"
    context = SimpleNamespace(operator="op", workspace=tmp_path,
                              outputs={"prepare_catalog": prepared})
    calls = []
    def review(agent, inp, runtime):
        calls.append("review")
        assert inp["kind"] == "tests"
        assert inp["evidence_paths"] == [str(snapshot), str(tmp_path / "review-sources/index.json"),
                                         str(tmp_path / "review-sources/01.txt")]
        assert (tmp_path / "review-sources/01.txt").read_text() == "original tests"
        return ReviewOutput(summary="reviewed original tests", reviewed_files=inp["evidence_paths"], findings=[])
    monkeypatch.setattr(ArtifactReviewerAgent, "run", review)
    def validate(*args):
        calls.append("target")
        return workflow_result({"readiness": "FAILED"}, state="FAILED")
    monkeypatch.setattr(operations, "validate_target", validate)
    result = operations.review_tests(SimpleNamespace(skip_review=skip), tmp_path, lambda _: None, context)
    assert calls == (["target"] if skip else ["review", "target"])
    assert result.state == "FAILED", "skip must not turn target failure into success"
    assert result.output["test_review"] == ("SKIPPED_EXPLICIT" if skip else "ACCEPTED")
    report = json.loads((tmp_path / "review.json").read_text())
    assert report.get("status") == ("SKIPPED_EXPLICIT" if skip else None)


@pytest.mark.parametrize("failure", ["p0", "missing_read", "missing_dependency"])
def test_review_failure_blocks_execution(tmp_path, monkeypatch, failure):
    evidence = tmp_path / "source.json"
    evidence.write_text('{"files": [{"path": "test.py", "content": "original tests"}]}')
    context = SimpleNamespace(operator="op", workspace=tmp_path, outputs={"prepare_catalog": {
        "snapshot": str(evidence), "test_evidence": str(evidence)}})
    def review(agent, inp, runtime):
        return ReviewOutput(summary="blocked", reviewed_files=["unread"] if failure == "missing_read" else inp["evidence_paths"],
            findings=[{"priority": "P0", "category": "test_contract", "evidence": "source.json:1",
                       "reason": "candidate never called", "requested_change": "test the candidate"}] if failure == "p0" else [],
            missing_dependencies=["tests/helper.py"] if failure == "missing_dependency" else [])
    monkeypatch.setattr(ArtifactReviewerAgent, "run", review)
    monkeypatch.setattr(operations, "validate_target", lambda *a: pytest.fail("review must block execution"))
    result = operations.review_tests(SimpleNamespace(skip_review=False), tmp_path, lambda _: None, context)
    assert result.state == "WAITING"
    assert (tmp_path / "review.json").is_file()


def test_skip_is_one_top_level_option_for_cli_python_and_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    args = build_parser().parse_args(["run", "--definition", "op", "--skip-review"])
    assert args.skip_review is True
    assert build_parser().parse_args(["run", "--definition", "op", "--no-skip-review"]).skip_review is False
    source = tmp_path / "batch.yaml"
    source.write_text("version: 1\ndefaults:\n  skip_review: true\noperators:\n  - definition: op\n")
    _, defaults, _, _ = load_batch_file(source)
    for mode in ("simple_opt", "kernelgen"):
        req = build_request(resolve_run_options(defaults, {"mode": mode}), definition="op", workspace=tmp_path / mode)
        assert req.workflow_input["skip_review"] is True
        assert "skip_review" not in req.workflow_input["optimization"]


def test_missing_source_capability_never_silently_skips(tmp_path, monkeypatch):
    from kernelgen.workflows.optimization.sources import prepare_test_evidence
    from kernelgen.tools import kernelgen_server_adapter as adapter
    monkeypatch.setattr(adapter, "get_service_status", lambda _: {"capabilities": {"operator_contract": {"enabled": True}}})
    inp = SimpleNamespace(optimization=SimpleNamespace(eval_server_url="http://local"))
    with pytest.raises(RuntimeError, match="test-source export"):
        prepare_test_evidence(inp, {}, SimpleNamespace(workspace=tmp_path))


def test_missing_source_is_not_replaced_by_local_files(tmp_path, monkeypatch):
    from kernelgen.workflows.optimization.sources import prepare_test_evidence
    from kernelgen.tools import kernelgen_server_adapter as adapter
    from kernelgen.tests.test_catalog_input_sources import contract
    from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot
    from kernelgen_client import http as client
    value = contract("flaggems")
    snapshot = tmp_path / "contract.json"
    snapshot.write_text(CatalogEvaluationSnapshot.from_server_contract(value, "fixed").model_dump_json())
    monkeypatch.setattr(adapter, "get_service_status", lambda _: {"capabilities": {"operator_contract": {"enabled": True, "test_sources": True}}})
    monkeypatch.setattr(client, "get_operator_contract", lambda *a, **k: value)
    inp = SimpleNamespace(optimization=SimpleNamespace(eval_server_url="http://local"))
    with pytest.raises(ValueError, match="actual source evidence"):
        prepare_test_evidence(inp, {"snapshot": str(snapshot)}, SimpleNamespace(workspace=tmp_path, operator="add"))


@pytest.mark.parametrize("status", ["PASSED", "FAILED", "ALL_SKIP", "UNSUPPORTED", "TIMEOUT", "SUSPECTED_DEVICE_ERROR"])
def test_gems_target_validation_runs_core_reference_without_candidate(tmp_path, monkeypatch, status):
    from kernelgen.tests.test_catalog_input_sources import contract
    from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot
    from kernelgen.framework.run_control import WorkspaceRunControl
    from kernelgen.tools import kernelgen_server_adapter as adapter
    from kernelgen_client import ReferenceResult, http as client
    path = tmp_path / "snapshot.json"
    path.write_text(CatalogEvaluationSnapshot.from_server_contract(contract("flaggems"), "fixed").model_dump_json())
    control = WorkspaceRunControl(tmp_path)
    context = SimpleNamespace(operator="add", workspace=tmp_path, control=control)
    monkeypatch.setattr(adapter, "get_service_status", lambda _: {"capabilities": {
        "benchmark_reference": {"enabled": True, "evaluators": ["flaggems"], "level": "core"},
        "operation_cancel": {"enabled": True, "operations": ["reference"]}}})
    case_report = {"records": [{"case_id": "core-half", "dtype": "torch.float16", "stage": "invoke",
        "status": "FAILED", "failure": {"category": "DTYPE_UNSUPPORTED", "type": "RuntimeError",
        "message": "original dtype error", "traceback": "original traceback"}}]} if status == "FAILED" else {}
    def reference(request, server, *, timeout, operation_id):
        assert request.binding.catalog_name == "remote-only" and request.benchmark_fingerprint == "fixed"
        assert operation_id and control.active_server_operations()[0].kind == "reference"
        assert "implementation" not in request.wire_payload()
        return ReferenceResult(status=status, benchmark_fingerprint="fixed", report=case_report)
    monkeypatch.setattr(client, "reference", reference)
    inp = SimpleNamespace(skip_review=True, optimization=SimpleNamespace(eval_server_url="http://localhost:8000", eval_timeout_seconds=5))
    output = operations.validate_target(inp, context, {"snapshot": str(path)})
    assert output.state == ("SUCCEEDED" if status == "PASSED" else "FAILED")
    assert output.output["validation_scope"] == "benchmark_core"
    assert output.output["correctness"] == "candidate_preflight_pending"
    assert json.loads((tmp_path / "benchmark-reference.json").read_text())["status"] == status
    assert json.loads((tmp_path / "benchmark-reference.json").read_text())["report"] == case_report
    assert not control.active_server_operations()


def test_reference_cancellation_preserves_receipt_and_clears_operation(tmp_path, monkeypatch):
    from kernelgen.tests.test_catalog_input_sources import contract
    from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot
    from kernelgen.framework.run_control import WorkspaceRunControl, RunCancelled
    from kernelgen.tools import kernelgen_server_adapter as adapter
    from kernelgen_client import http as client
    path = tmp_path / "snapshot.json"
    path.write_text(CatalogEvaluationSnapshot.from_server_contract(contract("flaggems"), "fixed").model_dump_json())
    control = WorkspaceRunControl(tmp_path)
    monkeypatch.setattr(adapter, "get_service_status", lambda _: {"capabilities": {
        "benchmark_reference": {"enabled": True, "evaluators": ["flaggems"], "level": "core"},
        "operation_cancel": {"enabled": True, "operations": ["reference"]}}})
    def cancel(req, server, *, timeout, operation_id):
        control.request_cancel("user request")
        raise client.OperationCancelledError(operation_id, "reference")
    monkeypatch.setattr(client, "reference", cancel)
    inp = SimpleNamespace(optimization=SimpleNamespace(eval_server_url="http://localhost:8000", eval_timeout_seconds=5))
    with pytest.raises(RunCancelled):
        operations.validate_target(inp, SimpleNamespace(operator="add", workspace=tmp_path, control=control), {"snapshot": str(path)})
    assert path.is_file() and not control.active_server_operations()


def test_resume_cannot_reuse_receipt_from_before_core_validation(tmp_path):
    from kernelgen.workflows.optimization import OperatorOptimizeWorkflow
    workflow = OperatorOptimizeWorkflow(cwd=tmp_path / "run")
    inp = {"operator": "negative", "dummy": True}
    workflow.run(inp)
    plan = tmp_path / "run/.kernelgen/operator-lifecycle.json"
    old = json.loads(plan.read_text())
    old.pop("test_validation")
    plan.write_text(json.dumps(old))
    evidence = plan.read_bytes()
    with pytest.raises(ValueError, match="plan changed"):
        workflow.run({**inp, "resume": True})
    assert plan.read_bytes() == evidence
