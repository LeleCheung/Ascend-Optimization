"""Original core baseline only; reuse normal request isolation and cancellation."""

import asyncio
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelgen_server import ReferenceRequest, ReferenceResult
from kernelgen_server.evaluation.adapters.flaggems import adapter as module
from kernelgen_server.evaluation.audit import RequestAudit
from kernelgen_server.evaluation.executor import EvaluationExecutor
from kernelgen_server.protocol import client
from kernelgen_server.runtime.device_pool import DevicePool, DeviceSlot
from kernelgen_server.runtime.operations import OperationCancelled, OperationRegistry


def request():
    return ReferenceRequest(binding={"catalog_name": "flaggems-adapter-definitions", "definition": "negative"},
                            benchmark_fingerprint="frozen")


@pytest.fixture
def runner(tmp_path, monkeypatch):
    assets = SimpleNamespace(root=tmp_path, performance=(tmp_path / "benchmark/test_negative.py",),
                             revision="a" * 40, native_operator="negative")
    adapter = module.FlagGemsEvaluationAdapter(catalog=SimpleNamespace(manifest={"benchmark_level": "core"}),
        operator=SimpleNamespace(definition=SimpleNamespace(name="negative")), device=object())
    monkeypatch.setattr(adapter, "_assets", lambda: assets)
    monkeypatch.setattr(adapter, "_base_env", lambda root: {"fixture": "environment"})
    ids = ["benchmark/test_negative.py::test_negative::core::float32::" + str(i) for i in range(2)]
    monkeypatch.setattr(adapter, "inspect", lambda: SimpleNamespace(benchmark_fingerprint="frozen",
        case_list=SimpleNamespace(cases=[SimpleNamespace(case_id=case) for case in ids])))
    report = {"schema_version": "flaggems.reference/v1", "phase": "timing", "status": "PASSED",
              "records": [{"case_id": case, "nodeid": "benchmark/test_negative.py::test_negative",
                           "status": "PASSED", "count": 1} for case in ids]}
    def run(args, *, root, env, timeout):
        assert "--reference-only" in args and "--level" not in args
        assert not {"--override", "--preflight-only", "--profile-only"}.intersection(args)
        assert args[1] == "benchmark/test_negative.py"
        assert env == {"fixture": "environment"} and timeout == 1500
        Path(args[args.index("--output") + 1]).write_text(json.dumps(report))
        return subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(module, "_run_pytest", run)
    return adapter, report


def test_reference_has_no_candidate_and_validates_exact_core_cases(runner):
    adapter, report = runner
    output = adapter.reference(request())
    assert output.status == "PASSED" and output.scope == "benchmark_core"
    assert output.report == report
    assert "implementation" not in request().wire_payload()


@pytest.mark.parametrize("status", ["FAILED", "ALL_SKIP", "UNSUPPORTED", "NO_CASES"])
def test_nonpassed_reference_is_preserved(runner, status):
    adapter, report = runner
    report.update(status=status, records=[])
    assert adapter.reference(request()).status == status


def test_per_case_failure_details_survive_reference_result(runner):
    adapter, report = runner
    report["status"] = "FAILED"
    report["records"][0].update(status="FAILED", dtype="torch.float16", shape={"N": 16},
        params={}, stage="invoke", failure={"category": "DTYPE_UNSUPPORTED", "type": "RuntimeError",
        "message": "original dtype error", "traceback": "original traceback"})
    report["records"][1].update(status="NOT_RUN", count=0, reason="execution stopped")
    result = adapter.reference(request())
    assert result.status == "FAILED"
    assert ReferenceResult.model_validate_json(result.model_dump_json()).report == report


@pytest.mark.parametrize("skipped", [False, True])
def test_not_run_cannot_contribute_to_pass_except_source_skipped_node(runner, skipped, monkeypatch):
    adapter, report = runner
    record = report["records"][1]
    record.update(status="NOT_RUN", count=0, nodeid="benchmark/test_negative.py::test_skipped")
    record["case_id"] = record["nodeid"] + "::core::float32::0"
    monkeypatch.setattr(adapter, "inspect", lambda: SimpleNamespace(benchmark_fingerprint="frozen",
        case_list=SimpleNamespace(cases=[SimpleNamespace(case_id=r["case_id"])
            for r in report["records"] if "case_id" in r])))
    if skipped:
        report["records"].append({"nodeid": record["nodeid"], "pytest_phase": "call", "status": "SKIP"})
    result = adapter.reference(request())
    assert result.status == ("PASSED" if skipped else "RUNTIME_ERROR")
    assert result.report == report


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "unknown", "no_call", "phase", "failure"])
def test_false_pass_is_rejected(runner, invalid):
    adapter, report = runner
    if invalid == "missing": report["records"].pop()
    elif invalid == "duplicate": report["records"].append(report["records"][0])
    elif invalid == "unknown": report["records"][0]["case_id"] = "foreign"
    elif invalid == "no_call": report["records"][0]["count"] = 0
    elif invalid == "phase": report["phase"] = "correctness"
    else: report["records"][0]["status"] = "FAILED"
    assert adapter.reference(request()).status == "RUNTIME_ERROR"


def test_fingerprint_drift_stops_before_execution(runner, monkeypatch):
    adapter, _ = runner
    monkeypatch.setattr(module, "_run_pytest", lambda *a, **k: pytest.fail("must not execute changed contract"))
    assert adapter.reference(request().model_copy(update={"benchmark_fingerprint": "changed"})).status == "FAILED"


@pytest.mark.parametrize("cancel", [False, True])
def test_reference_uses_probe_and_releases_slot(tmp_path, cancel):
    registry = OperationRegistry()
    control = registry.create("reference", "reference-test")
    probes = []
    audit = RequestAudit(tmp_path / "audit", manifest={})
    def run(*args):
        if cancel: raise OperationCancelled(control.operation_id, "reference")
        raise TimeoutError("test timeout")
    with ThreadPoolExecutor(max_workers=1) as executor:
        pool = DevicePool([DeviceSlot("cuda:0", "available")], worker_count=1, executor=executor,
                          probe=lambda device: probes.append(device))
        service = EvaluationExecutor(backend="cuda", timing="triton", device_pool=pool,
                                     executor=executor, runner=run, request_audit=audit)
        if cancel:
            with pytest.raises(OperationCancelled): asyncio.run(service.run("reference", request(), control))
        else:
            result = asyncio.run(service.run("reference", request(), control))
            assert isinstance(result, ReferenceResult) and result.status == "TIMEOUT"
        state = pool.snapshot(probe_timeout_seconds=30)
        assert state["available"] == 1 and state["active"] == state["checking"] == state["broken"] == 0
    assert probes == ["cuda:0"]
    metadata = json.loads(next(audit.requests_root.glob("*/request.json")).read_text())["metadata"]
    assert "candidate_sha256" not in metadata and "implementation" not in metadata


def test_sdk_keeps_operation_id_and_rejects_changed_fingerprint(monkeypatch):
    response = ReferenceResult(status="PASSED", benchmark_fingerprint="frozen").model_dump(mode="json")
    def post(url, endpoint, payload, timeout, operation_id):
        assert endpoint == "/reference" and operation_id == "tracked-reference"
        assert "implementation" not in payload
        return response
    monkeypatch.setattr(client, "_post", post)
    assert client.reference(request(), "http://local", operation_id="tracked-reference").status == "PASSED"
    response["benchmark_fingerprint"] = "wrong"
    with pytest.raises(client.ServerError, match="fingerprint"):
        client.reference(request(), "http://local", operation_id="tracked-reference")


def test_reference_http_route_uses_shared_scheduler(tmp_path, monkeypatch):
    from test_operation_cancel import _patch_server, server
    from fastapi.testclient import TestClient
    if server is None:
        pytest.skip("HTTP application requires Torch")
    _patch_server(monkeypatch)
    seen = []
    def execute(operation, req, backend, device, timing):
        assert isinstance(req, ReferenceRequest) and "implementation" not in req.wire_payload()
        seen.append((operation, device))
        return ReferenceResult(status="PASSED", benchmark_fingerprint=req.benchmark_fingerprint)
    monkeypatch.setattr(server, "run_isolated", execute)
    app = server.create_app(backend="cuda", enable_debug_jobs=False,
                            operator_bundle_root=tmp_path / "bundles", profile_artifact_root=tmp_path / "profiles")
    with TestClient(app) as http:
        response = http.post("/reference", json=request().wire_payload(), headers={"X-KernelGen-Operation-Id": "ref-http"})
        assert response.status_code == 200 and response.json()["scope"] == "benchmark_core"
        operation = http.get("/operations/ref-http").json()
        assert operation["kind"] == "reference" and operation["state"] == "SUCCEEDED"
        state = http.get("/status").json()
        assert state["capabilities"]["benchmark_reference"]["level"] == "core"
        assert state["scheduler"]["active"] == 0 and state["scheduler"]["available"] == 1
        assert http.post("/reference", json={**request().wire_payload(), "implementation": {}}).status_code == 422
    assert seen == [("reference", "cuda:0")]


def test_isolated_worker_decodes_candidate_free_reference_request(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import tempfile
    from kernelgen_server.runtime import isolated, device
    from kernelgen_server.evaluation import adapters
    monkeypatch.setattr(isolated, "_enter_isolated_process_group", lambda: None)
    monkeypatch.setattr(isolated, "_install_graceful_termination_handler", lambda: None)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    monkeypatch.setattr(device, "configure_device", lambda *a, **k: None)
    monkeypatch.setattr(device, "get_device", lambda *a: object())
    def reference(req):
        assert isinstance(req, ReferenceRequest) and req.benchmark_fingerprint == "frozen"
        return ReferenceResult(status="PASSED", benchmark_fingerprint="frozen")
    monkeypatch.setattr(adapters, "create_adapter", lambda *a, **k: SimpleNamespace(reference=reference))
    messages = []
    connection = SimpleNamespace(send=messages.append, close=lambda: None)
    isolated._worker(connection, "reference", request().wire_payload(), "cuda", "cuda:0", "triton", str(tmp_path))
    assert messages[0]["ok"] is True
    assert messages[0]["payload"]["value"]["status"] == "PASSED"
