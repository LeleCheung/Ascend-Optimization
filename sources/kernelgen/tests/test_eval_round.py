"""Unit tests for eval_round's atomic measurement-recording seam."""

import sys
import types
from types import SimpleNamespace

from kernelgen.data.ledger import Ledger
from kernelgen.data.stop_policy import StopConfig
from kernelgen.data.tool_context import ToolContext
from kernelgen.framework.run_control import WorkspaceRunControl
from kernelgen.tests.helpers import experiment_plan, round_conclusion
from kernelgen.tools import preflight, profile_round
from kernelgen.tools.eval_round import (
    evaluate_round,
    evaluate_kernel_and_record,
    record_and_augment,
)
from kernelgen.tools.profile_round import EvaluationBundle


def test_eval_round_honors_cancellation_before_configuration(tmp_path):
    WorkspaceRunControl(tmp_path).request_cancel("stop before evaluation")

    result = evaluate_round(tmp_path, "tmp/main.py", {})

    assert result["status"] == "RUN_CANCELLED"
    assert result["stage"] == "BEFORE_EVALUATION"
    assert result["exit_code"] == 0


def _result(status="PASSED", geo=1.2):
    return {
        "api_version": "v5.1",
        "status": status,
        "geo_mean": geo,
        "min_speedup": geo,
        "worst_workload_uuid": "u0",
        "latency_ms": 0.123456789,
        "abs_err": 0.0,
        "rel_err": 0.0,
        "num_workloads": 1,
        "num_passed": 1 if status == "PASSED" else 0,
        "per_workload": [
            {
                "uuid": "u0",
                "axes": {},
                "status": status,
                "speedup": geo,
                "latency_ms": 0.123456789,
                "reference_latency_ms": 0.1481481468,
            }
        ],
    }


def test_records_passed_plan_solution_and_measurement(tmp_path):
    out = record_and_augment(tmp_path, _result(), "code_v1", experiment_plan(1))
    assert out["round_num"] == 1
    assert out["is_new_best"] is True
    assert out["conclusion_recorded"] is False
    record = Ledger(tmp_path).get_round(1)
    assert record.plan.kind == "baseline"
    assert record.solution.code == "code_v1"
    assert record.evaluation.workloads[0].reference_latency_ms == 0.1481481468


def test_records_failed_round(tmp_path):
    out = record_and_augment(
        tmp_path,
        _result(status="COMPILE_ERROR", geo=None),
        "bad",
        experiment_plan(1),
    )
    assert out["round_num"] == 1
    assert out["is_new_best"] is False
    assert Ledger(tmp_path).get_round(1).evaluation.status == "COMPILE_ERROR"


def test_records_structured_timeout_round(tmp_path):
    result = _result(status="TIMEOUT", geo=None)
    result["per_workload"] = []
    result["log"] = "evaluate timed out after 600s on cuda:0"

    out = record_and_augment(
        tmp_path,
        result,
        "slow",
        experiment_plan(1),
    )

    assert out["status"] == "TIMEOUT"
    assert out["round_num"] == 1
    assert out["is_new_best"] is False
    record = Ledger(tmp_path).get_round(1)
    assert record.evaluation.status == "TIMEOUT"
    assert "timed out after 600s" in record.evaluation.log


def test_records_suspected_device_error_round(tmp_path):
    result = _result(status="SUSPECTED_DEVICE_ERROR", geo=None)
    result["per_workload"] = []
    result["log"] = "same request failed on two different device slots"

    out = record_and_augment(
        tmp_path,
        result,
        "candidate",
        experiment_plan(1),
    )

    assert out["status"] == "SUSPECTED_DEVICE_ERROR"
    assert out["round_num"] == 1
    assert out["is_new_best"] is False
    record = Ledger(tmp_path).get_round(1)
    assert record.evaluation.status == "SUSPECTED_DEVICE_ERROR"
    assert "two different device slots" in record.evaluation.log


def test_round_num_increments_after_conclusion(tmp_path):
    record_and_augment(tmp_path, _result(), "v1", experiment_plan(1))
    Ledger(tmp_path).finalize_round(1, round_conclusion(1))
    out = record_and_augment(tmp_path, _result(geo=1.4), "v2", experiment_plan(2))
    assert out["round_num"] == 2
    assert out["is_new_best"] is True


def test_complete_eval_rejects_after_persisted_max_round_stop(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.set_stop_config(StopConfig(max_round=1))
    ledger.record_eval(
        _result(status="RUNTIME_ERROR", geo=None),
        "bad",
        experiment_plan(1),
    )
    verdict = ledger.finalize_round(1, round_conclusion(1))
    assert verdict.code == "max_round_reached"

    result = evaluate_round(tmp_path, "tmp/main.py", experiment_plan(2))

    assert result["status"] == "RUN_STOPPED"
    assert result["reason_code"] == "max_round_reached"
    assert result["round_count"] == 1
    assert result["max_round"] == 1


def test_hack_result_is_recorded_but_not_promoted_to_best(tmp_path):
    result = _result(geo=100.0) | {
        "is_hack": True,
        "hack_reason": "forbidden Torch API: main.py:1 torch.mm",
    }

    out = record_and_augment(
        tmp_path,
        result,
        "def run(x, y): return torch.mm(x, y)",
        experiment_plan(1),
        profile_enabled=True,
    )

    assert out["status"] == "PASSED"
    assert out["is_hack"] is True
    assert out["is_new_best"] is False
    assert out["profile_required"] is False
    ledger = Ledger(tmp_path)
    record = ledger.get_round(1)
    assert record.evaluation.is_hack is True
    assert record.evaluation.hack_reason == result["hack_reason"]
    assert ledger.history.best_round == 0
    assert not ledger.best_kernel_path.exists()


def test_records_context_metadata(tmp_path):
    out = record_and_augment(
        tmp_path,
        _result(),
        "code",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
        implementation_language="triton",
    )
    assert out["conclusion_recorded"] is False
    history = Ledger(tmp_path).history
    assert history.definition_name == "abl_t1_gelu"
    assert history.target_hardware == "A100"
    assert history.implementation_language == "triton"


def test_unmeasured_not_recorded(tmp_path):
    for status in (
        "ERROR",
        "HARDWARE_MISMATCH",
        "TARGET_BACKEND_MISSING",
        "TARGET_DEVICE_MISSING",
        "TARGET_HARDWARE_MISMATCH",
        "TRANSPORT_ERROR",
        "RUN_CANCELLED",
    ):
        out = record_and_augment(
            tmp_path,
            {"status": status, "error": "x"},
            "code",
            experiment_plan(1),
        )
        assert "round_num" not in out
    assert Ledger(tmp_path).history.rounds == []


def test_cancelled_server_evaluation_is_not_recorded(tmp_path, monkeypatch):
    from kernelgen_client.http import OperationCancelledError
    from kernelgen.tools import kernelgen_server_adapter

    def cancelled(*args, **kwargs):
        raise OperationCancelledError("evaluation-1", "evaluate")

    monkeypatch.setattr(kernelgen_server_adapter, "evaluate_bundle", cancelled)
    bundle = EvaluationBundle(
        kernel_code="code",
        solution=SimpleNamespace(),
        definition=SimpleNamespace(name="op"),
        workloads=[],
    )
    context = ToolContext(
        definition="op",
        target_hardware="H100",
        implementation_language="triton",
        eval_server_url="http://eval:8000",
        catalog_name="fixture",
        profile_enabled=False,
    )
    kernel = tmp_path / "main.py"
    kernel.write_text("code", encoding="utf-8")
    control = WorkspaceRunControl(tmp_path)
    control.request_cancel("stop active evaluation")

    result, exit_code = evaluate_kernel_and_record(
        tmp_path,
        kernel,
        bundle,
        context,
        experiment_plan=experiment_plan(1),
        server_backend="cuda",
        run_control=control,
    )

    assert exit_code == 0
    assert result["status"] == "RUN_CANCELLED"
    assert result["stage"] == "DURING_EVALUATION"
    assert Ledger(tmp_path).history.rounds == []


def test_client_eval_preserves_phased_workloads(tmp_path, monkeypatch):
    calls = []
    from kernelgen.tools import kernelgen_server_adapter

    monkeypatch.setattr(
        kernelgen_server_adapter,
        "evaluate_bundle",
        lambda bundle, context: (
            calls.append((bundle, context)) or _result(),
            {
                "backend": "cuda",
                "target": {"backend": "cuda", "device": "NVIDIA H100"},
            },
        ),
    )

    correctness = [SimpleNamespace(name="c0")]
    timing = [SimpleNamespace(name="t0")]
    bundle = EvaluationBundle(
        kernel_code="code",
        solution=SimpleNamespace(),
        definition=SimpleNamespace(name="op"),
        workloads=correctness + timing,
        correctness_workloads=correctness,
        timing_workloads=timing,
        is_phased=True,
        profile_workload_uuids=["t0"],
    )
    context = ToolContext(
        definition="op",
        target_hardware="H100",
        implementation_language="triton",
        eval_server_url="http://eval:8000",
        catalog_name="flaggems-v5",
        profile_enabled=False,
    )
    kernel = tmp_path / "main.py"
    kernel.write_text("code", encoding="utf-8")

    result, exit_code = evaluate_kernel_and_record(
        tmp_path,
        kernel,
        bundle,
        context,
        experiment_plan=experiment_plan(1),
        server_backend="cuda",
    )

    assert exit_code == 0
    assert result["round_num"] == 1
    assert len(calls) == 1
    assert calls[0][0].correctness_workloads == correctness
    assert calls[0][0].timing_workloads == timing
    assert calls[0][1].eval_server_url == "http://eval:8000"


def test_client_eval_keeps_hardware_guard(tmp_path):
    bundle = EvaluationBundle(
        kernel_code="code",
        solution=SimpleNamespace(),
        definition=SimpleNamespace(name="op"),
        workloads=[SimpleNamespace(name="w0")],
    )
    context = ToolContext(
        definition="op",
        target_hardware="Ascend910B",
        profile_enabled=False,
    )
    kernel = tmp_path / "main.py"
    kernel.write_text("code", encoding="utf-8")

    result, exit_code = evaluate_kernel_and_record(
        tmp_path,
        kernel,
        bundle,
        context,
        experiment_plan=experiment_plan(1),
        server_backend="cuda",
    )

    assert exit_code == 2
    assert result["status"] == "TARGET_HARDWARE_MISMATCH"
    assert Ledger(tmp_path).history.rounds == []


def test_complete_eval_use_case_owns_receipt_snapshot_and_ledger(
    tmp_path,
    monkeypatch,
):
    calls = []
    from kernelgen.tools import kernelgen_server_adapter
    monkeypatch.setattr(
        kernelgen_server_adapter,
        "evaluate_bundle",
        lambda bundle, loaded_context, **kwargs: (
            calls.append((bundle, loaded_context)) or _result(),
            {
                "backend": "cuda",
                "target": {"backend": "cuda", "device": "NVIDIA H100"},
            },
        ),
    )

    candidate = tmp_path / "tmp" / "main.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("code", encoding="utf-8")
    context = ToolContext(
        definition="op",
        target_hardware="H100",
        eval_server_url="http://eval:8000",
        profile_enabled=True,
    )
    context.write(tmp_path)
    workload = SimpleNamespace(name="u0")
    bundle = EvaluationBundle(
        kernel_code="code",
        solution=SimpleNamespace(),
        definition=SimpleNamespace(name="op"),
        workloads=[workload],
    )
    monkeypatch.setattr(
        profile_round,
        "prepare_evaluation_bundle",
        lambda kernel, loaded_context: bundle,
    )
    monkeypatch.setattr(
        preflight,
        "validate_preflight_receipt",
        lambda workspace, loaded_bundle, loaded_context: (
            {"target": {"backend": "cuda"}},
            "",
        ),
    )
    monkeypatch.setattr(
        preflight,
        "consume_preflight_receipt",
        lambda workspace, receipt: candidate,
    )
    monkeypatch.setattr(
        profile_round,
        "write_evaluation_snapshot",
        lambda *args: {
            "evaluation_fingerprint": "fingerprint-1",
            "solution_sha256": "solution-1",
            "snapshot_path": ".kernelgen/evals/round-0001",
        },
    )

    result = evaluate_round(tmp_path, "tmp/main.py", experiment_plan(1))

    assert result["status"] == "PASSED"
    assert result["exit_code"] == 0
    assert result["round_num"] == 1
    assert result["evaluation_fingerprint"] == "fingerprint-1"
    assert result["profile_required"] is True
    assert result["profile_task"] == {
        "recommended_agent": "kernel-profile-analyzer",
        "round_num": 1,
        "instruction": "Profile this new-best round when the measurements need diagnosis.",
    }
    assert len(calls) == 1
    record = Ledger(tmp_path).get_round(1)
    assert record.evaluation.fingerprint == "fingerprint-1"
    assert record.solution.snapshot_path == ".kernelgen/evals/round-0001"
    assert record.solution.candidate_path == "tmp/main.py"
