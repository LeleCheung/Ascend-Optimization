"""Host-only tests for the versioned, phase-separated ledger."""

import json
import math
import sys
from pathlib import Path

from kernelgen.data.ledger import BEST_KERNEL_FILENAME, LEDGER_FILENAME, Ledger
from kernelgen.data.stop_policy import StopConfig
from kernelgen.tests.helpers import experiment_plan, round_conclusion


def _eval(status="PASSED", geo=1.0, minimum=None, per_workload=None, latency=0.123456789, worst="u0"):
    workloads = per_workload if per_workload is not None else [
        {
            "uuid": "u0",
            "axes": {"batch": 8},
            "status": status,
            "speedup": geo,
            "latency_ms": latency,
            "reference_latency_ms": latency * (geo or 1.0),
            "abs_err": 0.0,
            "rel_err": 0.0,
        }
    ]
    return {
        "api_version": "v5.1",
        "status": status,
        "geo_mean": geo,
        "min_speedup": minimum if minimum is not None else geo,
        "worst_workload_uuid": worst,
        "latency_ms": latency,
        "abs_err": 0.0,
        "rel_err": 0.0,
        "num_workloads": len(workloads),
        "num_passed": sum(item["status"] == "PASSED" for item in workloads),
        "requested_hardware": "A100",
        "server_backend": "cuda",
        "log": "",
        "per_workload": workloads,
    }


def _complete(ledger, round_num, **overrides):
    ledger.finalize_round(
        round_num,
        round_conclusion(round_num, **overrides),
        StopConfig(early_stop_rounds=0, max_round=1000),
    )


def test_empty_ledger_load(tmp_path):
    ledger = Ledger(tmp_path)
    assert ledger.history.schema_version == "3.0"
    assert ledger.history.rounds == []
    assert ledger.snapshot()["round_count"] == 0
    assert not (tmp_path / LEDGER_FILENAME).exists()


def test_record_eval_new_best_and_counter(tmp_path):
    ledger = Ledger(tmp_path)
    first = ledger.record_eval(_eval(geo=1.2), "code_v1", experiment_plan(1))
    assert first.is_new_best is True
    assert first.round_num == 1
    assert first.conclusion_recorded is False
    assert (tmp_path / BEST_KERNEL_FILENAME).read_text() == "code_v1"

    _complete(ledger, 1)
    second = ledger.record_eval(_eval(geo=1.05), "code_v2", experiment_plan(2))
    assert second.is_new_best is False
    assert ledger.history.rounds_without_improvement == 1
    assert Ledger(tmp_path).history.rounds_without_improvement == 1
    assert (tmp_path / BEST_KERNEL_FILENAME).read_text() == "code_v1"

    _complete(ledger, 2, expectation_status="not_met")
    third = ledger.record_eval(_eval(geo=1.34), "code_v3", experiment_plan(3))
    assert third.is_new_best is True
    assert ledger.history.rounds_without_improvement == 0
    assert (tmp_path / BEST_KERNEL_FILENAME).read_text() == "code_v3"


def test_user_cancelled_supervisor_stop_can_be_restored_for_resume(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.set_stop_config(StopConfig(early_stop_rounds=0, max_round=10))
    ledger.record_eval(_eval(), "code_v1", experiment_plan(1))
    ledger.finalize_round(1, round_conclusion(1))
    stopped = ledger.record_supervisor_stop(
        1,
        code="user_cancelled",
        reason="cancelled by user",
    )

    assert stopped.should_continue is False
    restored = Ledger(tmp_path).clear_supervisor_stop()

    assert restored.should_continue is True
    assert Ledger(tmp_path).stopped_round() is None


def test_policy_stop_cannot_be_cleared_as_user_cancellation(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.set_stop_config(StopConfig(early_stop_rounds=0, max_round=1))
    ledger.record_eval(_eval(), "code_v1", experiment_plan(1))
    stopped = ledger.finalize_round(1, round_conclusion(1))

    assert stopped.code == "max_round_reached"
    try:
        ledger.clear_supervisor_stop()
    except ValueError as exc:
        assert "not resumable" in str(exc)
    else:
        raise AssertionError("a genuine policy stop must remain immutable")


def test_hack_round_is_persisted_but_cannot_replace_best(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(_eval(geo=1.2), "real_kernel", experiment_plan(1))
    _complete(ledger, 1)

    hacked = _eval(geo=100.0) | {
        "is_hack": True,
        "hack_reason": "forbidden Torch API: main.py:1 torch.mm",
    }
    result = ledger.record_eval(hacked, "torch_fallback", experiment_plan(2))

    assert result.is_new_best is False
    assert ledger.history.best_round == 1
    assert ledger.history.best_geo_mean == 1.2
    assert (tmp_path / BEST_KERNEL_FILENAME).read_text() == "real_kernel"
    evaluation = Ledger(tmp_path).get_round(2).evaluation
    assert evaluation.is_hack is True
    assert evaluation.hack_reason == hacked["hack_reason"]
    assert "Anti-hack: REJECTED" in Ledger(tmp_path).get_round(2).format_for_prompt()


def test_load_repairs_failed_rounds_in_legacy_plateau_counter(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(status="INCORRECT_NUMERICAL", geo=None),
        "wrong",
        experiment_plan(1),
    )
    raw = json.loads((tmp_path / LEDGER_FILENAME).read_text())
    raw["rounds_without_improvement"] = 7
    (tmp_path / LEDGER_FILENAME).write_text(json.dumps(raw))

    reloaded = Ledger(tmp_path)
    assert reloaded.snapshot()["round_count"] == 1
    assert reloaded.history.rounds_without_improvement == 0


def test_first_round_must_be_baseline_and_later_round_cannot_be_baseline(tmp_path):
    ledger = Ledger(tmp_path)
    try:
        ledger.record_eval(_eval(), "code", experiment_plan(2))
    except ValueError as exc:
        assert "first measured round" in str(exc)
    else:
        raise AssertionError("first round must be a baseline")
    ledger.record_eval(_eval(), "code", experiment_plan(1))
    _complete(ledger, 1)
    try:
        ledger.record_eval(_eval(), "code", experiment_plan(1))
    except ValueError as exc:
        assert "only valid for the first" in str(exc)
    else:
        raise AssertionError("later round cannot be a baseline")


def test_persistence_is_nested_and_preserves_full_precision(tmp_path):
    workloads = [
        {
            "uuid": "u0",
            "axes": {"batch": 8},
            "status": "PASSED",
            "speedup": 1.111111111111,
            "latency_ms": 0.123456789123,
            "reference_latency_ms": 0.137174209025,
            "abs_err": 0.0,
            "rel_err": 0.0,
        },
        {
            "uuid": "u1",
            "axes": {"batch": 64},
            "status": "PASSED",
            "speedup": 1.333333333333,
            "latency_ms": 0.234567891234,
            "reference_latency_ms": 0.312757188312,
            "abs_err": 0.0,
            "rel_err": 0.0,
        },
    ]
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(geo=1.2171612389, minimum=1.111111111111, per_workload=workloads),
        "code_v1",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    raw = json.loads((tmp_path / LEDGER_FILENAME).read_text())
    assert raw["schema_version"] == "3.0"
    assert raw["target_hardware"] == "A100"
    assert raw["implementation_language"] == "triton"
    round_data = raw["rounds"][0]
    assert set(round_data) == {
        "round_num",
        "experiment_parent_round_num",
        "plan",
        "solution",
        "evaluation",
        "profile",
        "conclusion",
        "next_verdict",
    }
    assert round_data["evaluation"]["workloads"][0]["latency_ms"] == 0.123456789123
    assert round_data["evaluation"]["workloads"][0]["reference_latency_ms"] == 0.137174209025
    assert round_data["solution"]["code"] == "code_v1"
    assert round_data["solution"]["candidate_path"] == "tmp/main.py"
    assert len(round_data["solution"]["sha256"]) == 64
    assert set(round_data["plan"]) == {
        "kind",
        "strategy",
        "code_changes",
        "hypothesis",
        "expected_effect",
        "source",
        "key_params",
        "knowledge_uses",
    }
    assert set(round_data["plan"]["expected_effect"]) == {
        "metric",
        "direction",
        "mechanism",
    }
    assert round_data["plan"]["source"] == {
        "origin": "baseline",
        "parent_round_num": None,
    }


def test_persistence_preserves_correctness_gated_timing(tmp_path):
    workloads = [
        {
            "uuid": "correctness-failed",
            "axes": {"case": 0},
            "phase": "correctness",
            "status": "INCORRECT_NUMERICAL",
            "abs_err": 0.1,
            "rel_err": 0.2,
        },
        {
            "uuid": "timing-skipped",
            "axes": {"case": 1},
            "phase": "timing",
            "status": "SKIPPED",
            "skip_reason": "CORRECTNESS_FAILED",
        },
    ]
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(
            status="PARTIAL_PASS",
            geo=None,
            minimum=None,
            per_workload=workloads,
            latency=None,
            worst="correctness-failed",
        )
        | {"timing_skipped": True},
        "code",
        experiment_plan(1),
    )

    evaluation = Ledger(tmp_path).get_round(1).evaluation
    assert evaluation.timing_skipped is True
    assert evaluation.workloads[0].phase == "correctness"
    assert evaluation.workloads[1].phase == "timing"
    assert evaluation.workloads[1].status == "SKIPPED"
    assert evaluation.workloads[1].skip_reason == "CORRECTNESS_FAILED"
    assert "Timing: SKIPPED" in Ledger(tmp_path).get_round(1).format_for_prompt()


def test_rejects_unversioned_legacy_ledger(tmp_path):
    (tmp_path / LEDGER_FILENAME).write_text(json.dumps({"definition_name": "old", "rounds": []}))
    try:
        Ledger(tmp_path)
    except ValueError as exc:
        assert "unsupported ledger schema" in str(exc)
        assert "fresh agent workspace" in str(exc)
    else:
        raise AssertionError("legacy ledgers must not be migrated implicitly")


def test_older_schema_v2_defaults_added_metadata(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(_eval(), "code", experiment_plan(1))
    raw = json.loads((tmp_path / LEDGER_FILENAME).read_text())
    raw.pop("implementation_language")
    raw["rounds"][0]["solution"].pop("candidate_path")
    (tmp_path / LEDGER_FILENAME).write_text(json.dumps(raw))

    history = Ledger(tmp_path).history
    assert history.implementation_language == "triton"
    assert history.rounds[0].solution.candidate_path == "tmp/main.py"


def test_comparison_uses_previous_authoritative_best(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(_eval(geo=1.2, latency=0.2), "best", experiment_plan(1))
    _complete(ledger, 1)
    ledger.record_eval(_eval(geo=1.1, latency=0.25), "worse", experiment_plan(2))
    comparison = ledger.get_round(2).evaluation.comparison
    assert comparison.baseline_round_num == 1
    assert comparison.geo_mean_before == 1.2
    assert comparison.geo_mean_after == 1.1
    assert math.isclose(comparison.geo_mean_delta_pct, (1.1 - 1.2) / 1.2 * 100.0)
    assert comparison.workloads[0].before_latency_ms == 0.2
    assert comparison.workloads[0].after_latency_ms == 0.25
    assert comparison.workloads[0].latency_delta_ms == 0.04999999999999999


def test_conclusion_is_exactly_once_and_requires_perf_gap_after_baseline(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(_eval(), "baseline", experiment_plan(1))
    _complete(ledger, 1)
    assert ledger.get_round(1).conclusion.expectation_status == "baseline"
    ledger.record_eval(_eval(geo=1.1), "second", experiment_plan(2))
    bad = round_conclusion(2, perf_gap_analysis="")
    try:
        ledger.finalize_round(2, bad)
    except ValueError as exc:
        assert "perf_gap_analysis" in str(exc)
    else:
        raise AssertionError("non-baseline conclusion requires perf gap analysis")
    _complete(ledger, 2, expectation_status="met")
    try:
        _complete(ledger, 2)
    except ValueError as exc:
        assert "already recorded" in str(exc)
    else:
        raise AssertionError("conclusion must be exactly once")


def test_eval_requires_previous_round_conclusion(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(_eval(), "first", experiment_plan(1))
    assert ledger.pending_conclusion_round().round_num == 1
    try:
        ledger.record_eval(_eval(), "second", experiment_plan(2))
    except ValueError as exc:
        assert "finalize_round" in str(exc)
    else:
        raise AssertionError("a second eval must not bypass the conclusion gate")


def test_profile_plan_source_must_link_to_recorded_next_experiment(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(),
        "first",
        experiment_plan(1),
        profile_enabled=True,
    )
    analysis_path = tmp_path / ".kernelgen" / "profile-analysis" / "round-0001.json"
    analysis_path.parent.mkdir(parents=True)
    analysis_path.write_text(json.dumps({"next_experiment": {"action_description": "tile differently"}}))
    relative = str(analysis_path.relative_to(tmp_path))
    ledger.attach_profile_analysis(
        1,
        status="completed",
        analysis_path=relative,
        summary="memory: redundant loads",
    )
    _complete(ledger, 1)
    linked = experiment_plan(
        2,
        source={
            "origin": "profile_next_experiment",
            "parent_round_num": 1,
        },
    )
    assert (
        ledger.validate_experiment_plan(linked).source.parent_round_num == 1
    )
    analysis_path.unlink()
    try:
        ledger.validate_experiment_plan(linked)
    except ValueError as exc:
        assert "file is missing" in str(exc)
    else:
        raise AssertionError("profile plan source must resolve a recorded analysis")


def test_metadata_stays_stable(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _eval(),
        "code",
        experiment_plan(1),
        definition_name="abl_t1_gelu",
        target_hardware="A100",
    )
    assert ledger.history.target_hardware == "A100"
    _complete(ledger, 1)
    try:
        ledger.record_eval(
            _eval(),
            "other",
            experiment_plan(2),
            definition_name="different_op",
            target_hardware="A100",
        )
    except ValueError as exc:
        assert "definition_name" in str(exc)
    else:
        raise AssertionError("ledger metadata must remain stable")


def test_partial_pass_retains_all_workload_measurements_without_fake_headline(tmp_path):
    workloads = [
        {"uuid": "u0", "axes": {}, "status": "PASSED", "speedup": 1.4, "latency_ms": 0.1, "reference_latency_ms": 0.14},
        {"uuid": "u1", "axes": {}, "status": "RUNTIME_ERROR"},
    ]
    ledger = Ledger(tmp_path)
    result = ledger.record_eval(
        _eval(status="PARTIAL_PASS", geo=None, minimum=None, per_workload=workloads),
        "partial",
        experiment_plan(1),
    )
    assert result.is_new_best is False
    assert result.geo_mean is None
    assert len(ledger.get_round(1).evaluation.workloads) == 2
    assert ledger.get_round(1).evaluation.workloads[0].reference_latency_ms == 0.14
    assert not (tmp_path / BEST_KERNEL_FILENAME).exists()


def test_revert_and_external_best(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(_eval(geo=1.3), "best_code", experiment_plan(1))
    kernel = tmp_path / "kernel.py"
    kernel.write_text("broken experiment")
    assert ledger.revert_to_best() is True
    assert kernel.read_text() == "best_code"
    ledger.update_best(1.5, "winner_from_other_agent", round_num=3)
    assert ledger.best["code"] == "winner_from_other_agent"


if __name__ == "__main__":
    import inspect
    import tempfile
    import traceback

    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            if "tmp_path" in inspect.signature(test).parameters:
                with tempfile.TemporaryDirectory() as directory:
                    test(Path(directory))
            else:
                test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
