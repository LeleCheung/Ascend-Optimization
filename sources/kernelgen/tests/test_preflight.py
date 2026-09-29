"""Host-side tests for autonomous preflight and content-bound receipts."""

from __future__ import annotations

from types import SimpleNamespace

from kernelgen.data.ledger import Ledger
from kernelgen.data.stop_policy import StopConfig
from kernelgen.data.tool_context import ToolContext
from kernelgen.framework.run_control import WorkspaceRunControl
from kernelgen.tests.helpers import experiment_plan, round_conclusion
from kernelgen.tools.preflight import (
    PREFLIGHT_RELATIVE_DIR,
    consume_preflight_receipt,
    preflight_candidate,
    run_preflight,
    validate_preflight_receipt,
)
from kernelgen.tools.profile_round import EvaluationBundle


def test_preflight_honors_cancellation_before_configuration(tmp_path):
    WorkspaceRunControl(tmp_path).request_cancel("stop before preflight")

    result = preflight_candidate(tmp_path)

    assert result["status"] == "RUN_CANCELLED"
    assert result["stage"] == "BEFORE_PREFLIGHT"
    assert result["exit_code"] == 0


def test_preflight_returns_server_cancellation_without_receipt(tmp_path, monkeypatch):
    from kernelgen_client.http import OperationCancelledError
    from kernelgen.tools import kernelgen_server_adapter
    from kernelgen.tools import preflight as preflight_module

    context = _context()
    context.write(tmp_path)
    candidate = tmp_path / "tmp" / "main.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("candidate", encoding="utf-8")
    monkeypatch.setattr(
        preflight_module,
        "prepare_evaluation_bundle",
        lambda kernel, loaded_context: _bundle("candidate"),
    )

    def cancelled(bundle, loaded_context, *, run_control):
        run_control.request_cancel("stop active preflight")
        raise OperationCancelledError("preflight-1", "preflight")

    monkeypatch.setattr(
        kernelgen_server_adapter,
        "preflight_bundle",
        cancelled,
    )

    result = preflight_candidate(tmp_path)

    assert result["status"] == "RUN_CANCELLED"
    assert result["stage"] == "DURING_PREFLIGHT"
    assert result["exit_code"] == 0
    assert not (tmp_path / PREFLIGHT_RELATIVE_DIR / "receipt.json").exists()


class _DumpModel(SimpleNamespace):
    def model_dump(self, mode=None):
        del mode
        return self.data


def _bundle(code: str = "def run(x, out):\n    out.copy_(x)\n") -> EvaluationBundle:
    definition_data = {
        "name": "copy",
        "inputs": ["x"],
        "outputs": ["out"],
    }
    definition = _DumpModel(
        data=definition_data,
        name="copy",
        inputs=definition_data["inputs"],
        outputs=definition_data["outputs"],
    )
    solution = _DumpModel(data={"name": "candidate", "sources": [{"content": code}]})
    workload = _DumpModel(
        data={"name": "w0", "inputs": {"x": {"type": "scalar", "value": 1}}}
    )
    workload.name = "w0"
    return EvaluationBundle(
        kernel_code=code,
        solution=solution,
        definition=definition,
        workloads=[workload],
        timing_workloads=[workload],
        profile_workload_uuids=["w0"],
        benchmark_fingerprint="benchmark-fingerprint",
    )


def _context() -> ToolContext:
    return ToolContext(
        definition="copy",
        target_hardware="Ascend910B",
        eval_server_url="http://eval:8000",
        catalog_name="fixture",
        destination_passing_style=False,
    )


def _patch_service(monkeypatch, *, target_status="PASSED", calls=None):
    from kernelgen.tools import kernelgen_server_adapter

    service_status = {
        "status": "ok",
        "capabilities": {"candidate_admission": {"version": 1, "policy_sha256": "a" * 64, "stages": ["preflight"]}},
        "api_version": "v6.2",
        "backend": "npu",
        "timing": "profiler",
        "target": {
            "backend": "ascend",
            "vendor": "huawei",
            "architecture": "DAV_2201",
            "device": "Ascend 910B",
        },
    }

    def preflight_bundle(bundle, context):
        if calls is not None:
            calls.append((bundle, context))
        return ({
            "api_version": "v6.2",
            "status": target_status,
            "stage": "complete" if target_status == "PASSED" else "timing_candidate_smoke",
            "is_hack": False,
            "hack_reason": "",
            "benchmark_fingerprint": bundle.benchmark_fingerprint,
            "num_cases": len(bundle.timing_workloads),
            "log": "" if target_status == "PASSED" else "compile failed",
        }, service_status)

    monkeypatch.setattr(
        kernelgen_server_adapter, "preflight_bundle", preflight_bundle
    )
    monkeypatch.setattr(
        kernelgen_server_adapter,
        "get_service_status",
        lambda *args, **kwargs: service_status,
    )


def test_successful_preflight_receipt_is_exact_and_single_use(tmp_path, monkeypatch):
    _patch_service(monkeypatch)
    bundle = _bundle()
    context = _context()

    result = run_preflight(tmp_path, bundle, context)
    assert result["status"] == "PASSED"
    assert result["target"]["num_cases"] == 1
    assert result["target"]["device"] == "Ascend910B"
    assert "server_version" not in result
    assert "server_version" not in result["service_signature"]
    assert result["api_version"] == "v6.2"
    assert result["service_signature"]["timing"] == "profiler"
    assert result["service_signature"]["target_device"] == "Ascend910B"
    receipt, error = validate_preflight_receipt(tmp_path, bundle, context)
    assert error == ""
    assert receipt["kernel_sha256"] == result["kernel_sha256"]
    assert receipt["workload_mode"] == "phased"
    assert receipt["profile_workload_uuids"] == ["w0"]

    candidate = consume_preflight_receipt(tmp_path, receipt)
    assert candidate.read_text() == bundle.kernel_code
    missing, error = validate_preflight_receipt(tmp_path, bundle, context)
    assert missing is None
    assert "no successful" in error


def test_edit_after_preflight_makes_receipt_stale(tmp_path, monkeypatch):
    _patch_service(monkeypatch)
    context = _context()
    run_preflight(tmp_path, _bundle(), context)

    receipt, error = validate_preflight_receipt(
        tmp_path,
        _bundle("def run(x, out):\n    out.zero_()\n"),
        context,
    )
    assert receipt is None
    assert "kernel_sha256 changed" in error


def test_failed_preflight_is_audited_without_receipt(tmp_path, monkeypatch):
    _patch_service(monkeypatch, target_status="FAILED")
    result = run_preflight(tmp_path, _bundle(), _context())
    assert result["status"] == "FAILED"
    assert result["stage"] == "timing_candidate_smoke"
    assert not (tmp_path / PREFLIGHT_RELATIVE_DIR / "receipt.json").exists()
    attempts = (tmp_path / PREFLIGHT_RELATIVE_DIR / "attempts.jsonl").read_text()
    assert result["attempt_id"] in attempts


def test_timed_out_preflight_preserves_structured_status(tmp_path, monkeypatch):
    _patch_service(monkeypatch, target_status="TIMEOUT")

    result = run_preflight(tmp_path, _bundle(), _context())

    assert result["status"] == "TIMEOUT"
    assert result["target"]["status"] == "TIMEOUT"
    assert result["passed"] is False
    assert "execution budget" in result["instruction"]
    assert not (tmp_path / PREFLIGHT_RELATIVE_DIR / "receipt.json").exists()


def test_suspected_device_error_preflight_requires_scheduler_inspection(
    tmp_path,
    monkeypatch,
):
    _patch_service(monkeypatch, target_status="SUSPECTED_DEVICE_ERROR")

    result = run_preflight(tmp_path, _bundle(), _context())

    assert result["status"] == "SUSPECTED_DEVICE_ERROR"
    assert result["target"]["status"] == "SUSPECTED_DEVICE_ERROR"
    assert result["passed"] is False
    assert "Do not edit the candidate" in result["instruction"]
    assert "only broken>0 confirms" in result["instruction"]
    assert "larger configured timeout" in result["instruction"]
    assert not (tmp_path / PREFLIGHT_RELATIVE_DIR / "receipt.json").exists()


def test_phased_preflight_preserves_workload_groups(tmp_path, monkeypatch):
    calls = []
    _patch_service(monkeypatch, calls=calls)
    correctness = [_DumpModel(
        data={"name": "c0", "inputs": {"x": {"type": "scalar", "value": 1}}},
    )]
    correctness[0].name = "c0"
    timing = [_DumpModel(
        data={"name": "t0", "inputs": {"x": {"type": "scalar", "value": 2}}},
    )]
    timing[0].name = "t0"
    bundle = _bundle()
    bundle = EvaluationBundle(
        kernel_code=bundle.kernel_code,
        solution=bundle.solution,
        definition=bundle.definition,
        workloads=correctness + timing,
        correctness_workloads=correctness,
        timing_workloads=timing,
        is_phased=True,
        profile_workload_uuids=["t0"],
    )

    result = run_preflight(tmp_path, bundle, _context())

    assert result["status"] == "PASSED"
    submitted_bundle, submitted_context = calls[0]
    assert submitted_bundle.correctness_workloads == correctness
    assert submitted_bundle.timing_workloads == timing
    assert submitted_context.eval_server_url == "http://eval:8000"


def test_complete_preflight_use_case_enforces_pending_round_gate(tmp_path):
    Ledger(tmp_path).record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.0,
            "num_workloads": 0,
            "num_passed": 0,
            "per_workload": [],
        },
        "code",
        experiment_plan(1),
    )

    result = preflight_candidate(tmp_path)

    assert result["status"] == "ROUND_CONCLUSION_REQUIRED"
    assert result["round_num"] == 1


def test_preflight_rejects_after_persisted_max_round_stop(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.set_stop_config(StopConfig(max_round=1))
    ledger.record_eval(
        {"status": "RUNTIME_ERROR", "per_workload": []},
        "bad",
        experiment_plan(1),
    )
    verdict = ledger.finalize_round(1, round_conclusion(1))
    assert verdict.code == "max_round_reached"

    result = preflight_candidate(tmp_path)

    assert result["status"] == "RUN_STOPPED"
    assert result["reason_code"] == "max_round_reached"
    assert result["round_count"] == 1
    assert result["max_round"] == 1


def test_passed_with_hack_never_issues_receipt_and_revokes_previous(tmp_path, monkeypatch):
    from kernelgen.tools import kernelgen_server_adapter as adapter
    _patch_service(monkeypatch)
    run_preflight(tmp_path, _bundle(), _context())
    original = adapter.preflight_bundle
    def rejected(bundle, context):
        result, service = original(bundle, context)
        return {**result, 'is_hack': True, 'hack_reason': 'protected Torch state'}, service
    monkeypatch.setattr(adapter, 'preflight_bundle', rejected)
    result = run_preflight(tmp_path, _bundle(), _context())
    assert result['status'] == 'FAILED'
    assert result['stage'] == 'candidate_admission'
    assert 'protected Torch state' in result['instruction']
    assert not (tmp_path / PREFLIGHT_RELATIVE_DIR / 'receipt.json').exists()


def test_review_warning_is_preserved_without_becoming_confirmed_hack(tmp_path, monkeypatch):
    from kernelgen.tools import kernelgen_server_adapter as adapter
    _patch_service(monkeypatch)
    original = adapter.preflight_bundle
    def reviewed(bundle, context):
        result, service = original(bundle, context)
        return {**result, 'log': 'ADMISSION_REVIEW: unknown receiver .softmax'}, service
    monkeypatch.setattr(adapter, 'preflight_bundle', reviewed)
    result = run_preflight(tmp_path, _bundle(), _context())
    assert result['passed']
    assert 'ADMISSION_REVIEW' in result['target']['log']


def test_policy_change_invalidates_receipt(tmp_path, monkeypatch):
    from kernelgen.tools import kernelgen_server_adapter as adapter
    _patch_service(monkeypatch)
    run_preflight(tmp_path, _bundle(), _context())
    service = adapter.get_service_status(None)
    service['capabilities']['candidate_admission']['policy_sha256'] = 'b' * 64
    receipt, error = validate_preflight_receipt(tmp_path, _bundle(), _context())
    assert receipt is None and 'changed' in error


def test_old_service_is_rejected_before_submission(monkeypatch):
    import pytest
    from kernelgen.tools import kernelgen_server_adapter as adapter
    from kernelgen_client import http as client
    real_preflight_bundle = adapter.preflight_bundle
    _patch_service(monkeypatch)
    service = adapter.get_service_status(None)
    service.pop('capabilities')
    def unexpected(*args, **kwargs):
        raise AssertionError('must not submit to a service without admission')
    monkeypatch.setattr(client, 'preflight', unexpected)
    with pytest.raises(RuntimeError, match='candidate_admission'):
        real_preflight_bundle(_bundle(), _context())



def test_receipt_with_rejected_target_cannot_be_replayed(tmp_path, monkeypatch):
    import json
    _patch_service(monkeypatch)
    run_preflight(tmp_path, _bundle(), _context())
    path = tmp_path / PREFLIGHT_RELATIVE_DIR / 'receipt.json'
    value = json.loads(path.read_text())
    value['target']['is_hack'] = True
    path.write_text(json.dumps(value))
    receipt, error = validate_preflight_receipt(tmp_path, _bundle(), _context())
    assert receipt is None and 'admission was rejected' in error
