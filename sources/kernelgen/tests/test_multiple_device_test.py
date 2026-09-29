"""A frozen Gems candidate is tested only after each target's core baseline."""

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelgen.framework.run_control import RunState, WorkspaceRunControl
from kernelgen.framework.workflow_result import WorkflowResult
from kernelgen.tests.test_gems_adapter_definition import source as source
from kernelgen.workflows.gems_adapter_definition import GemsAdapterDefinitionWorkflow
from kernelgen.workflows.multiple_device_test import MultipleDeviceTestWorkflow
from kernelgen_server import EvaluateResponse, ReferenceResult
from kernelgen_server.operator_bundles import OperatorBundleInfo, pack_operator_bundle
from kernelgen_server.protocol import client


@pytest.fixture
def prepared(source, tmp_path):
    exported = GemsAdapterDefinitionWorkflow(cwd=tmp_path / "definition").run({
        "flaggems_repo": source, "pytest_path": "tests/test_addmm_.py"})
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def run(self, mat1, mat2, beta=1, alpha=1): return mat1\n")
    return exported, candidate


def status():
    return {"status": "ok", "api_version": "v6.2", "backend": "cuda", "timing": "triton",
            "target": {"device": "TEST GPU"}, "software": {"language": "triton"},
            "scheduler": {"healthy": 1, "broken": 0}, "capabilities": {
                "operator_bundle_upload": {"enabled": True, "evaluation_binding": True,
                                           "evaluators": ["flaggems"]},
                "benchmark_reference": {"enabled": True, "level": "core",
                                        "evaluators": ["flaggems"]},
                "operation_cancel": {"enabled": True,
                                     "operations": ["reference", "preflight", "evaluate"]}}}


def test_reference_gate_preserves_case_failures_and_continues_next_target(prepared, tmp_path, monkeypatch):
    from kernelgen.workflows import multiple_device_test as module

    exported, candidate = prepared
    urls = ["http://127.0.0.1:19001", "http://127.0.0.1:19002"]
    events = []
    monkeypatch.setattr(module, "get_service_status", lambda url: status())

    def upload(path, url):
        events.append((url, "upload"))
        with tempfile.TemporaryFile(mode="w+b") as stream:
            digest, size = pack_operator_bundle(path, stream)
        return OperatorBundleInfo(bundle_id="sha256:" + digest, sha256=digest, definition="addmm_",
                                  size_bytes=size, created_at="2026-09-25T00:00:00Z")

    def inspect(request, url):
        events.append((url, "inspect"))
        assert request.binding.bundle_id.startswith("sha256:")
        return SimpleNamespace(kind="flaggems", benchmark_fingerprint="frozen",
            case_list=SimpleNamespace(cases=[SimpleNamespace(case_id="case-1")]),
            model_dump=lambda **kw: {"kind": "flaggems", "benchmark_fingerprint": "frozen"})

    def reference(request, url, *, operation_id):
        events.append((url, "reference"))
        assert operation_id and request.benchmark_fingerprint == "frozen"
        if url == urls[0]:
            candidate.write_text("def run(*args): raise RuntimeError('source changed after freeze')\n")
            return ReferenceResult(status="FAILED", benchmark_fingerprint="frozen",
                report={"schema_version": "flaggems.reference/v1", "status": "FAILED", "records": [
                    {"case_id": "case-1", "dtype": "torch.float16", "status": "FAILED",
                     "failure": {"category": "DTYPE_UNSUPPORTED", "message": "original dtype error"}}]})
        return ReferenceResult(status="PASSED", benchmark_fingerprint="frozen",
            report={"schema_version": "flaggems.reference/v1", "status": "PASSED", "records": [
                {"case_id": "case-1", "status": "PASSED", "count": 1}]})

    def preflight(request, url, *, operation_id):
        events.append((url, "preflight"))
        assert operation_id and request.implementation.sources[0].content == frozen_code
        return {"status": "PASSED", "stage": "candidate"}

    def evaluate(request, url, *, operation_id):
        events.append((url, "evaluate"))
        assert operation_id and request.implementation.sources[0].content == frozen_code
        return EvaluateResponse(status="PASSED", device="cuda:0", server_backend="cuda",
                                num_workloads=3, num_passed=3, geo_mean=1.2)

    frozen_code = candidate.read_text()
    monkeypatch.setattr(client, "upload_operator_bundle", upload)
    monkeypatch.setattr(client, "inspect", inspect)
    monkeypatch.setattr(client, "reference", reference)
    monkeypatch.setattr(client, "preflight", preflight)
    monkeypatch.setattr(client, "evaluate", evaluate)

    output = MultipleDeviceTestWorkflow(cwd=tmp_path / "run").run({
        "operator": "addmm_", "catalog_path": exported.catalog_path, "candidate_path": candidate,
        "targets": [{"name": "unsupported", "server_url": urls[0]},
                    {"name": "working", "server_url": urls[1]}]})
    assert isinstance(output, WorkflowResult)
    assert output.state == "FAILED" and output.output.completed_targets == 2 and output.output.passed_targets == 1
    assert [result.state for result in output.output.results] == ["REFERENCE_FAILED", "PASSED"]
    assert events == [(urls[0], "upload"), (urls[0], "inspect"), (urls[0], "reference"),
                      (urls[1], "upload"), (urls[1], "inspect"), (urls[1], "reference"),
                      (urls[1], "preflight"), (urls[1], "evaluate")]
    first = json.loads((tmp_path / "run/targets/unsupported/reference.json").read_text())
    assert first["report"]["records"][0]["failure"]["category"] == "DTYPE_UNSUPPORTED"
    stored = (tmp_path / "run/workflow_result.json").read_text()
    assert json.loads(stored) == output.model_dump(mode="json")
    assert module.MultipleDeviceTestWorkflow.OutputModel.model_validate_json(stored) == output
    assert WorkspaceRunControl(tmp_path / "run").progress().state == RunState.FAILED
    assert not WorkspaceRunControl(tmp_path / "run").active_server_operations()
    assert output.output.candidate_sha256 == hashlib.sha256(frozen_code.encode()).hexdigest()


@pytest.mark.parametrize("url", ["http://10.0.0.9:18306", "https://127.0.0.1:18306",
                                 "http://user@127.0.0.1:18306"])
def test_direct_or_credentialed_target_url_is_rejected(prepared, tmp_path, url):
    exported, candidate = prepared
    with pytest.raises(ValueError, match="local loopback"):
        MultipleDeviceTestWorkflow(cwd=tmp_path / "run").run({
            "operator": "addmm_", "catalog_path": exported.catalog_path, "candidate_path": candidate,
            "targets": [{"name": "remote", "server_url": url}]})
    assert not (tmp_path / "run").exists()


def test_protocol_failure_is_recorded_without_upload(prepared, tmp_path, monkeypatch):
    from kernelgen.workflows import multiple_device_test as module

    exported, candidate = prepared
    monkeypatch.setattr(module, "get_service_status", lambda url: {**status(), "api_version": "v5.1"})
    monkeypatch.setattr(client, "upload_operator_bundle", lambda *args: pytest.fail("must not upload"))
    result = MultipleDeviceTestWorkflow(cwd=tmp_path / "run").run({
        "operator": "addmm_", "catalog_path": exported.catalog_path, "candidate_path": candidate,
        "targets": [{"name": "old", "server_url": "http://127.0.0.1:18001"}]})
    assert result.state == "FAILED" and result.output.results[0].phase == "status"
    assert "incompatible KGS protocol" in result.output.results[0].error
    assert not (tmp_path / "run/targets/old/reference.json").exists()


def test_reference_skip_does_not_admit_candidate(prepared, tmp_path, monkeypatch):
    from kernelgen.workflows import multiple_device_test as module

    exported, candidate = prepared
    monkeypatch.setattr(module, "get_service_status", lambda url: status())

    def upload(path, url):
        with tempfile.TemporaryFile(mode="w+b") as stream:
            digest, size = pack_operator_bundle(path, stream)
        return OperatorBundleInfo(bundle_id="sha256:" + digest, sha256=digest, definition="addmm_",
                                  size_bytes=size, created_at="2026-09-25T00:00:00Z")

    monkeypatch.setattr(client, "upload_operator_bundle", upload)
    monkeypatch.setattr(client, "inspect", lambda request, url: SimpleNamespace(
        kind="flaggems", benchmark_fingerprint="frozen",
        case_list=SimpleNamespace(cases=[SimpleNamespace(case_id="case-1"),
                                         SimpleNamespace(case_id="skipped-dtype::case-2")]),
        model_dump=lambda **kw: {"kind": "flaggems", "benchmark_fingerprint": "frozen"}))
    monkeypatch.setattr(client, "reference", lambda request, url, *, operation_id: ReferenceResult(
        status="PASSED", benchmark_fingerprint="frozen",
        report={"schema_version": "flaggems.reference/v1", "phase": "timing", "status": "PASSED", "records": [
            {"case_id": "case-1", "status": "PASSED", "count": 1},
            {"nodeid": "skipped-dtype", "status": "SKIP", "pytest_phase": "setup"}]}))
    monkeypatch.setattr(client, "preflight", lambda *args, **kwargs: pytest.fail("must not preflight"))
    monkeypatch.setattr(client, "evaluate", lambda *args, **kwargs: pytest.fail("must not evaluate"))

    result = MultipleDeviceTestWorkflow(cwd=tmp_path / "run").run({
        "operator": "addmm_", "catalog_path": exported.catalog_path, "candidate_path": candidate,
        "targets": [{"name": "skip", "server_url": "http://127.0.0.1:18001"}]})
    assert result.output.results[0].state == "REFERENCE_FAILED"
    assert result.output.results[0].reference_status == "PASSED"
    assert "skipped 1 core record" in result.output.results[0].error


def test_success_returns_and_persists_workflow_result(prepared, tmp_path, monkeypatch):
    from kernelgen.workflows import multiple_device_test as module

    exported, candidate = prepared
    monkeypatch.setattr(module, "get_service_status", lambda url: status())

    def upload(path, url):
        with tempfile.TemporaryFile(mode="w+b") as stream:
            digest, size = pack_operator_bundle(path, stream)
        return OperatorBundleInfo(bundle_id="sha256:" + digest, sha256=digest, definition="addmm_",
                                  size_bytes=size, created_at="2026-09-25T00:00:00Z")

    monkeypatch.setattr(client, "upload_operator_bundle", upload)
    monkeypatch.setattr(client, "inspect", lambda request, url: SimpleNamespace(
        kind="flaggems", benchmark_fingerprint="frozen",
        case_list=SimpleNamespace(cases=[SimpleNamespace(case_id="case-1")]),
        model_dump=lambda **kw: {"kind": "flaggems", "benchmark_fingerprint": "frozen"}))
    monkeypatch.setattr(client, "reference", lambda request, url, *, operation_id: ReferenceResult(
        status="PASSED", benchmark_fingerprint="frozen",
        report={"schema_version": "flaggems.reference/v1", "phase": "timing", "status": "PASSED",
                "records": [{"case_id": "case-1", "status": "PASSED", "count": 1}]}))
    monkeypatch.setattr(client, "preflight", lambda request, url, *, operation_id: {
        "status": "PASSED", "stage": "candidate"})
    monkeypatch.setattr(client, "evaluate", lambda request, url, *, operation_id: EvaluateResponse(
        status="PASSED", device="cuda:0", server_backend="cuda", num_workloads=1,
        num_passed=1, geo_mean=1.1))

    result = MultipleDeviceTestWorkflow(cwd=tmp_path / "run").run({
        "operator": "addmm_", "catalog_path": exported.catalog_path, "candidate_path": candidate,
        "targets": [{"name": "working", "server_url": "http://127.0.0.1:18001"}]})
    assert result.state == "SUCCEEDED" and result.output.results[0].state == "PASSED"
    assert json.loads((tmp_path / "run/workflow_result.json").read_text()) == result.model_dump(mode="json")
    assert WorkspaceRunControl(tmp_path / "run").progress().state == RunState.SUCCEEDED


def test_cancellation_between_targets_keeps_completed_receipt(prepared, tmp_path, monkeypatch):
    from kernelgen.workflows import multiple_device_test as module

    exported, candidate = prepared
    monkeypatch.setattr(module, "get_service_status", lambda url: status())
    def target(self, inp, target, plan, code, child):
        if target.name == "first":
            WorkspaceRunControl(self.root).request_cancel("stop before second target")
        return module.DeviceTestResult(name=target.name, server_url=target.server_url, state="PASSED",
                                       phase="evaluate", result_path=str(self.root / "targets" / target.name / "result.json"))
    monkeypatch.setattr(module.MultipleDeviceTestWorkflow, "_test_target", target)
    result = module.MultipleDeviceTestWorkflow(cwd=tmp_path / "run").run({
        "operator": "addmm_", "catalog_path": exported.catalog_path, "candidate_path": candidate,
        "targets": [{"name": "first", "server_url": "http://127.0.0.1:18001"},
                    {"name": "second", "server_url": "http://127.0.0.1:18002"}]})
    assert result.state == "CANCELLED" and result.output.completed_targets == 1
    assert json.loads((tmp_path / "run/workflow_result.json").read_text()) == result.model_dump(mode="json")
    assert WorkspaceRunControl(tmp_path / "run").progress().state == RunState.CANCELLED
