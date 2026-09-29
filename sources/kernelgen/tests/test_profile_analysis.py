"""Host-only tests for profile state gates and validated analysis persistence."""

import json
import tempfile
from pathlib import Path

from kernelgen.data.ledger import Ledger
from kernelgen.data.profile_analysis import ProfileAnalysis
from kernelgen.tools.profile_round import record_profile_analysis
from kernelgen.tools.finalize_round import finalize_round
from kernelgen.tests.helpers import experiment_plan, round_conclusion


def _eval():
    return {
        "status": "PASSED",
        "geo_mean": 1.2,
        "min_speedup": 1.0,
        "worst_workload_uuid": "u0",
        "latency_ms": 0.1,
        "abs_err": 0.0,
        "rel_err": 0.0,
        "num_workloads": 1,
        "num_passed": 1,
        "per_workload": [
            {
                "uuid": "u0",
                "axes": {"M": 128},
                "status": "PASSED",
                "speedup": 1.2,
                "latency_ms": 0.1,
                "reference_latency_ms": 0.12,
            }
        ],
    }


def _prepare_snapshot(root: Path) -> Path:
    snapshot = root / ".kernelgen" / "evals" / "round-0001"
    snapshot.mkdir(parents=True)
    payloads = {
        "result.json": _eval(),
        "solution.json": {},
        "definition.json": {},
        "workloads.json": [{"uuid": "u0", "axes": {"M": 128}}],
        "identity.json": {
            "round_num": 1,
            "evaluation_fingerprint": "eval-fingerprint",
            "solution_sha256": "solution-sha",
            "definition_name": "op",
            "definition_sha256": "definition-sha",
            "workload_uuids": ["u0"],
            "workload_sha256": ["workload-sha"],
            "catalog_name": "fixture",
            "target_hardware": "H100",
            "server_backend": "cuda",
        },
    }
    (snapshot / "main.py").write_text("def run(): pass", encoding="utf-8")
    for name, payload in payloads.items():
        (snapshot / name).write_text(json.dumps(payload), encoding="utf-8")
    return snapshot


def _prepare_profile(root: Path) -> tuple[Path, Path]:
    profile_dir = root / ".kernelgen" / "profiles" / "round-0001" / "u0" / "request-1"
    profile_dir.mkdir(parents=True)
    report = profile_dir / "ncu-details.txt"
    report.write_text("SM throughput: 50%", encoding="utf-8")
    manifest = profile_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "result": {
                    "profile_id": "profile-1",
                    "artifacts": [{"local_path": str(report)}],
                }
            }
        ),
        encoding="utf-8",
    )
    return manifest, report


def _completed_analysis(manifest: Path, report: Path):
    root = manifest.parents[5]
    return {
        "schema_version": "1.0",
        "round_num": 1,
        "evaluation_fingerprint": "eval-fingerprint",
        "status": "completed",
        "solution_sha256": "solution-sha",
        "backend": "cuda",
        "profiler": "ncu",
        "capabilities": ["performance_counters"],
        "dominant_bound": "memory",
        "profiled_workloads": [
            {
                "uuid": "u0",
                "axes": {"M": 128},
                "eval_speedup": 1.2,
                "eval_latency_ms": 0.1,
                "selection_reason": "worst workload",
                "profile_ids": ["profile-1"],
                "manifest_paths": [str(manifest.relative_to(root))],
                "status": "completed",
            }
        ],
        "findings": [
            {
                "category": "memory_bandwidth",
                "label": "limited memory throughput",
                "backend_detail": "ncu counters",
                "workload_uuids": ["u0"],
                "confidence": "medium",
                "evidence": [
                    {
                        "metric": "SM throughput",
                        "value": 50,
                        "unit": "%",
                        "artifact_path": str(report.relative_to(root)),
                        "artifact_kind": "agent_report",
                        "locator": {},
                    }
                ],
                "inference": "the current tile leaves memory throughput on the table",
            }
        ],
        "performance_interpretation": "The selected workload is limited by memory behavior.",
        "next_experiment": {
            "action_category": "memory_pipeline",
            "action_description": "increase independent coalesced loads",
            "expected_impact": "improve memory issue efficiency",
            "risks_and_rollback": "revert if register pressure or geo_mean regresses",
            "validation_workloads": ["u0"],
            "success_criteria": ["full eval remains PASSED", "geo_mean improves"],
        },
        "warnings": [],
        "open_questions": [],
    }


def test_profile_is_optional_during_loop_and_can_be_recorded_later(tmp_path):
    ledger = Ledger(tmp_path)
    measured = ledger.record_eval(
        _eval(),
        "def run(): pass",
        experiment_plan(1),
        profile_enabled=True,
    )
    assert measured.profile_required is True
    snapshot = _prepare_snapshot(tmp_path)
    ledger.attach_evaluation_snapshot(
        1,
        evaluation_fingerprint="eval-fingerprint",
        snapshot_path=str(snapshot.relative_to(tmp_path)),
    )

    finalized = finalize_round(tmp_path, round_conclusion(1))
    assert finalized["recorded"] is True
    assert "profile best round 1" in finalized["recommendations"][0]

    manifest, report = _prepare_profile(tmp_path)
    recorded = record_profile_analysis(
        tmp_path,
        1,
        _completed_analysis(manifest, report),
    )
    assert recorded["recorded"] is True
    assert recorded["status"] == "completed"

    reloaded = Ledger(tmp_path).get_round(1)
    assert reloaded.profile.status == "completed"
    assert reloaded.profile.analysis_path == ".kernelgen/profile-analysis/round-0001.json"
    analysis = json.loads((tmp_path / reloaded.profile.analysis_path).read_text())
    profiled = analysis["profiled_workloads"][0]
    assert profiled["eval_latency_ms"] == 0.1
    assert profiled["eval_reference_latency_ms"] == 0.12


def test_nonpassing_eval_does_not_require_profile(tmp_path):
    result = _eval()
    result["status"] = "COMPILE_ERROR"
    result["geo_mean"] = None
    measured = Ledger(tmp_path).record_eval(
        result,
        "broken",
        experiment_plan(1),
        profile_enabled=True,
    )
    assert measured.profile_required is False
    assert measured.profile_status == "not_required"


def test_passing_non_best_does_not_request_profile(tmp_path):
    ledger = Ledger(tmp_path)
    first = ledger.record_eval(
        _eval(),
        "best",
        experiment_plan(1),
        profile_enabled=True,
    )
    assert finalize_round(tmp_path, round_conclusion(1))["recorded"] is True
    ledger = Ledger(tmp_path)
    slower = _eval()
    slower["geo_mean"] = 1.1
    second = ledger.record_eval(
        slower,
        "slower",
        experiment_plan(2),
        profile_enabled=True,
    )

    assert first.profile_required is True
    assert second.profile_required is False
    assert second.profile_status == "not_required"
    assert ledger.pending_profile_round().round_num == 1


def test_terminal_analysis_is_reused_for_identical_fingerprint(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(_eval(), "same code", experiment_plan(1), profile_enabled=True)
    ledger.attach_evaluation_snapshot(
        1,
        evaluation_fingerprint="same-fingerprint",
        snapshot_path=".kernelgen/evals/round-0001",
    )
    ledger.attach_profile_analysis(
        1,
        status="completed",
        analysis_path=".kernelgen/profile-analysis/round-0001.json",
        summary="memory: coalescing",
    )
    ledger.finalize_round(1, round_conclusion(1))

    improved = _eval()
    improved["geo_mean"] = 1.3
    second = ledger.record_eval(
        improved,
        "same code",
        experiment_plan(2),
        profile_enabled=True,
    )
    assert second.profile_required is True
    reuse = ledger.attach_evaluation_snapshot(
        2,
        evaluation_fingerprint="same-fingerprint",
        snapshot_path=".kernelgen/evals/round-0002",
    )
    assert reuse["profile_reused"] is True
    assert reuse["profile_required"] is False
    assert reuse["profile_status"] == "completed"
    assert ledger.pending_profile_round() is None
    assert ledger.pending_conclusion_round().round_num == 2


def test_failed_analysis_is_terminal_and_clears_gate(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(_eval(), "code", experiment_plan(1), profile_enabled=True)
    snapshot = _prepare_snapshot(tmp_path)
    ledger.attach_evaluation_snapshot(
        1,
        evaluation_fingerprint="eval-fingerprint",
        snapshot_path=str(snapshot.relative_to(tmp_path)),
    )
    recorded = record_profile_analysis(
        tmp_path,
        1,
        {
            "schema_version": "1.0",
            "round_num": 1,
            "evaluation_fingerprint": "eval-fingerprint",
            "status": "failed",
            "solution_sha256": "solution-sha",
            "backend": "cuda",
            "dominant_bound": "unknown",
            "error": "profiler transport failed",
        },
    )
    assert recorded["status"] == "failed"
    assert Ledger(tmp_path).pending_profile_round() is None


def test_completed_schema_rejects_missing_evidence_and_experiment(tmp_path):
    try:
        ProfileAnalysis.model_validate(
            {
                "round_num": 1,
                "evaluation_fingerprint": "fingerprint",
                "status": "completed",
                "solution_sha256": "solution-sha",
                "backend": "cuda",
                "performance_interpretation": "unsupported assertion",
            }
        )
    except ValueError as exc:
        assert "profiled workload" in str(exc)
    else:
        raise AssertionError("completed analysis without evidence should fail validation")


def test_agent_supplied_absolute_evidence_path_is_rejected(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(_eval(), "code", experiment_plan(1), profile_enabled=True)
    snapshot = _prepare_snapshot(tmp_path)
    ledger.attach_evaluation_snapshot(
        1,
        evaluation_fingerprint="eval-fingerprint",
        snapshot_path=str(snapshot.relative_to(tmp_path)),
    )
    manifest, report = _prepare_profile(tmp_path)
    analysis = _completed_analysis(manifest, report)
    analysis["profiled_workloads"][0]["manifest_paths"] = [str(manifest)]
    try:
        record_profile_analysis(tmp_path, 1, analysis)
    except ValueError as exc:
        assert "absolute paths" in str(exc)
    else:
        raise AssertionError("absolute agent path should be rejected")


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            with tempfile.TemporaryDirectory() as directory:
                test(Path(directory))
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception:
            import traceback

            print(f"  ✗ {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    raise SystemExit(0 if passed == len(tests) else 1)
