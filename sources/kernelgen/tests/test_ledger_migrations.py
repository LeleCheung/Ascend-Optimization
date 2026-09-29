"""Cross-branch, side-effect-free ledger migration tests."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from kernelgen.data.ledger import Ledger
from kernelgen.data.ledger_migrations import (
    AmbiguousLedgerOrigin,
    migrate_ledger,
)
from kernelgen.data.optimization_history import OptimizationHistory
from kernelgen.tests.helpers import experiment_plan, round_conclusion


def _passed_eval() -> dict:
    return {
        "api_version": "5.1",
        "status": "PASSED",
        "geo_mean": 1.25,
        "min_speedup": 1.1,
        "worst_workload_uuid": "wl-1",
        "latency_ms": 0.2,
        "num_workloads": 1,
        "num_passed": 1,
        "requested_hardware": "Ascend910B",
        "server_backend": "npu",
        "per_workload": [
            {
                "uuid": "wl-1",
                "phase": "timing",
                "status": "PASSED",
                "latency_ms": 0.2,
                "reference_latency_ms": 0.25,
                "speedup": 1.25,
            }
        ],
    }


def test_current_v2_migration_preserves_lifecycle_and_execution_facts(tmp_path):
    ledger = Ledger(tmp_path)
    ledger.record_eval(
        _passed_eval(),
        "def run():\n    return None\n",
        experiment_plan(1),
        definition_name="test_definition",
        target_hardware="Ascend910B",
    )
    ledger.attach_evaluation_snapshot(
        1,
        evaluation_fingerprint="fingerprint-v51",
        snapshot_path=".kernelgen/rounds/round-0001.py",
    )
    ledger.finalize_round(1, round_conclusion(1))
    raw = json.loads(ledger.path.read_text(encoding="utf-8"))
    raw["schema_version"] = "2.0"
    record = raw["rounds"][0]
    record.pop("experiment_parent_round_num")
    record["plan"].pop("knowledge_uses")
    record["plan"]["target_workloads"] = ["wl-1"]
    record["plan"]["risks"] = ["legacy placeholder"]
    record["plan"]["expected_effect"]["estimated_change_pct_min"] = 1.0
    record["plan"]["expected_effect"]["estimated_change_pct_max"] = 2.0
    source = record["plan"]["source"]
    source["kind"] = source.pop("origin")
    source["round_num"] = source.pop("parent_round_num")
    source["analysis_path"] = ".kernelgen/legacy-analysis.json"
    source["detail"] = "legacy detail"
    record["conclusion"]["architecture_tag"] = "legacy-architecture"
    record["conclusion"]["suggestion_followed"] = "yes"
    record["conclusion"]["key_numbers"] = {"tile": 128}
    record["conclusion"]["composable"] = True
    record["conclusion"]["composition_group"] = "tiling"
    record["conclusion"]["diagnostic_results"] = "legacy diagnostic"
    record["evaluation"].pop("evaluated_at")
    comparison = record["evaluation"]["comparison"]
    comparison["baseline_round_num"] = comparison.pop(
        "performance_baseline_round_num"
    )
    before = deepcopy(raw)

    migrated = migrate_ledger(raw)

    assert raw == before
    assert migrated["schema_version"] == "3.0"
    migrated_round = migrated["rounds"][0]
    assert migrated_round["solution"]["candidate_path"] == "tmp/main.py"
    assert migrated_round["evaluation"]["api_version"] == "5.1"
    assert migrated_round["evaluation"]["fingerprint"] == "fingerprint-v51"
    assert migrated_round["evaluation"]["is_hack"] is False
    assert migrated_round["evaluation"]["workloads"][0]["phase"] == "timing"
    assert migrated_round["next_verdict"]["code"] == "continue"
    assert migrated_round["plan"]["source"] == {
        "origin": "baseline",
        "parent_round_num": None,
    }
    assert "target_workloads" not in migrated_round["plan"]
    assert "risks" not in migrated_round["plan"]
    assert "estimated_change_pct_min" not in migrated_round["plan"]["expected_effect"]
    assert "analysis_path" not in migrated_round["plan"]["source"]
    assert "architecture_tag" not in migrated_round["conclusion"]
    assert "key_numbers" not in migrated_round["conclusion"]


def _jiabei_v26_ledger() -> dict:
    return {
        "schema_version": "2.6",
        "definition_name": "test_definition",
        "target_hardware": "Ascend910B",
        "implementation_language": "triton",
        "best_geo_mean": 1.4,
        "best_round": 2,
        "best_code": "def run():\n    return None\n",
        "rounds_without_improvement": 0,
        "rounds": [
            {
                "round_num": 2,
                "experiment_parent_round_num": 1,
                "plan": {
                    "kind": "performance",
                    "strategy": "tile the reduction",
                    "code_changes": "use a two-stage reduction",
                    "hypothesis": "fewer global reads improve throughput",
                    "expected_effect": {
                        "metric": "geo_mean",
                        "direction": "increase",
                        "mechanism": "reuse partial sums",
                    },
                    "source": {
                        "origin": "knowledge_base",
                        "parent_round_num": 1,
                    },
                    "key_params": {"BLOCK_SIZE": 256},
                    "knowledge_uses": [
                        {
                            "concept_ref": "kg:method:reduction-test",
                            "query_event_id": "query:round-2",
                            "role": "implementation",
                            "disposition": "adopted",
                            "application_note": "selected a two-stage reduction",
                            "affected_parts": ["strategy", "code_changes"],
                        }
                    ],
                },
                "solution": {
                    "sha256": "a" * 64,
                    "snapshot_path": ".kernelgen/rounds/round-0002.py",
                    "code": "def run():\n    return None\n",
                },
                "evaluation": {
                    "evaluated_at": "2026-08-03T12:00:00Z",
                    "schema_version": "1.0",
                    "status": "PASSED",
                    "geo_mean": 1.4,
                    "min_speedup": 1.2,
                    "worst_workload_uuid": "wl-1",
                    "latency_ms": 0.18,
                    "max_abs_error": 0.0,
                    "max_rel_error": 0.0,
                    "num_workloads": 1,
                    "num_passed": 1,
                    "requested_hardware": "Ascend910B",
                    "server_backend": "npu",
                    "log": "",
                    "workloads": [
                        {
                            "uuid": "wl-1",
                            "axes": {},
                            "status": "PASSED",
                            "latency_ms": 0.18,
                            "reference_latency_ms": 0.25,
                            "speedup": 1.4,
                            "abs_err": 0.0,
                            "rel_err": 0.0,
                        }
                    ],
                    "comparison": {
                        "performance_baseline_round_num": 1,
                        "geo_mean_before": 1.0,
                        "geo_mean_after": 1.4,
                        "geo_mean_delta_pct": 40.0,
                        "workloads": [],
                    },
                },
                "profile": {
                    "required": False,
                    "status": "not_required",
                    "analysis_path": "",
                    "summary": "",
                },
                "conclusion": {
                    "round_num": 2,
                    "expectation_status": "met",
                    "root_cause": "the reduction was memory bound",
                    "perf_gap_analysis": "global reads decreased",
                    "next_suggestion": "test a wider tile",
                    "optimization_level": "L2_memory",
                    "debug_lesson": "",
                    "strategy_evolution": "",
                },
            }
        ],
    }


def test_jiabei_v26_migration_preserves_knowledge_and_evaluation_lineage(
    tmp_path,
):
    raw = _jiabei_v26_ledger()
    before = deepcopy(raw)
    path = tmp_path / ".ledger.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    history = OptimizationHistory.load(path)

    assert json.loads(path.read_text(encoding="utf-8")) == before
    record = history.rounds[0]
    assert history.schema_version == "3.0"
    assert record.experiment_parent_round_num == 1
    assert record.evaluation.evaluated_at.isoformat().startswith(
        "2026-08-03T12:00:00"
    )
    assert record.evaluation.api_version == "1.0"
    assert (
        record.evaluation.comparison.performance_baseline_round_num == 1
    )
    assert record.plan.source.origin == "knowledge_base"
    assert record.plan.source.parent_round_num == 1
    assert record.plan.knowledge_uses[0].query_event_id == "query:round-2"
    assert record.solution.candidate_path == "tmp/main.py"
    assert record.next_verdict is None
    assert "architecture_tag" not in record.conclusion.model_fields_set


def test_same_version_empty_v2_requires_explicit_origin():
    raw = {
        "schema_version": "2.0",
        "definition_name": "",
        "rounds": [],
    }

    with pytest.raises(AmbiguousLedgerOrigin):
        migrate_ledger(raw)

    assert migrate_ledger(raw, origin="dev-xy")["schema_version"] == "3.0"
    assert migrate_ledger(raw, origin="jiabei")["schema_version"] == "3.0"
