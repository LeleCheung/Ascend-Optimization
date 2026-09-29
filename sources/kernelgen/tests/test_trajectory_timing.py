"""Summaries must not confuse oracle drift with candidate improvement."""
from copy import deepcopy

import pytest

from kernelgen.agents.distiller import DistillerAgent
from kernelgen.agents.knowledge_distiller import KnowledgeDistillerAgent
from kernelgen.data.trajectory import build_synthesis_trajectory, build_trajectory
from kernelgen.framework import FakeRuntime


def measured(number, candidate, reference, *, baseline=None, parent=None):
    return {
        "round_num": number,
        "experiment_parent_round_num": parent,
        "solution": {"code": "def run():\n    return 1\n"},
        "evaluation": {
            "status": "PASSED",
            "geo_mean": reference / candidate,
            "comparison": {"performance_baseline_round_num": baseline},
            "workloads": [{
                "uuid": "timing-a", "phase": "timing", "status": "PASSED",
                "latency_ms": candidate, "reference_latency_ms": reference,
                "speedup": reference / candidate,
            }, {
                "uuid": "correctness-a", "phase": "correctness", "status": "PASSED",
                "latency_ms": None, "reference_latency_ms": None,
            }],
        },
    }


@pytest.mark.parametrize("builder", [build_trajectory, build_synthesis_trajectory])
def test_oracle_drift_is_not_candidate_improvement_and_records_stay_unchanged(builder):
    records = [measured(1, 1, 1), measured(2, 1, 2, baseline=1, parent=1)]
    original = deepcopy(records)
    text = builder(records)
    assert "paired_timing_vs_best+parent_R1: cases=1" in text
    assert "candidate_latency_delta=+0.000000%" in text
    assert "reference_latency_delta=+100.000000%" in text
    assert "observational, not causal or a noise estimate" in text
    assert text.count("paired_timing_vs_best+parent_R1") == 1
    assert records == original


@pytest.mark.parametrize("builder", [build_trajectory, build_synthesis_trajectory])
def test_parent_and_performance_best_are_separate_controls(builder):
    text = builder([
        measured(1, 1, 2),
        measured(2, 2, 2, baseline=1, parent=1),
        measured(3, 1.5, 3, baseline=1, parent=2),
    ])
    assert "paired_timing_vs_best_R1: cases=1, candidate_latency_delta=+50.000000%" in text
    assert "paired_timing_vs_parent_R2: cases=1, candidate_latency_delta=-25.000000%" in text


def test_geometric_mean_uses_matching_uuid_not_order_or_correctness_rows():
    before, after = measured(1, 1, 2), measured(2, 0.5, 2, baseline=1)
    before["evaluation"]["workloads"].append({
        "uuid": "timing-b", "phase": "timing", "status": "PASSED",
        "latency_ms": 2, "reference_latency_ms": 8,
    })
    after["evaluation"]["workloads"].insert(0, {
        "uuid": "timing-b", "phase": "timing", "status": "PASSED",
        "latency_ms": 4, "reference_latency_ms": 8,
    })
    text = build_trajectory([before, after])
    assert "cases=2, candidate_latency_delta=+0.000000%, reference_latency_delta=+0.000000%" in text


@pytest.mark.parametrize("invalid", [None, 0, -1, True, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["latency_ms", "reference_latency_ms"])
def test_invalid_timing_is_not_silently_dropped(invalid, field):
    before, after = measured(1, 1, 2), measured(2, 1, 2, baseline=1)
    after["evaluation"]["workloads"][0][field] = invalid
    text = build_trajectory([before, after])
    assert "paired_timing_vs_best_R1: unavailable (invalid or failed timing)" in text
    assert "candidate_latency_delta=" not in text


@pytest.mark.parametrize("failure", ["different", "duplicate", "empty", "axes_changed", "row_failed", "round_failed", "missing_baseline"])
def test_incomplete_comparisons_are_explicitly_unavailable(failure):
    before, after = measured(1, 1, 2), measured(2, 1, 2, baseline=1)
    rows = after["evaluation"]["workloads"]
    if failure == "different":
        rows[0]["uuid"] = "different"
    elif failure == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif failure == "empty":
        rows.clear()
    elif failure == "axes_changed":
        rows[0]["axes"] = {"N": 64}
    elif failure == "row_failed":
        rows[0]["status"] = "SKIPPED"
    elif failure == "round_failed":
        after["evaluation"]["status"] = "PARTIAL_PASS"
    else:
        after["evaluation"]["comparison"]["performance_baseline_round_num"] = 9
    text = build_synthesis_trajectory([before, after])
    assert "unavailable (" in text
    assert "candidate_latency_delta=" not in text


def test_legacy_baseline_name_keeps_its_actual_identity():
    before, after = measured(1, 1, 1), measured(2, 2, 1)
    after["evaluation"]["comparison"] = {"baseline_round_num": 1}
    text = build_trajectory([before, after])
    assert "paired_timing_vs_best_R1: cases=1, candidate_latency_delta=+100.000000%" in text


def test_overflow_cannot_crash_distillation_or_produce_infinite_gain():
    before, after = measured(1, 1e-300, 1), measured(2, 1e300, 1, baseline=1)
    text = build_trajectory([before, after])
    assert "unavailable (non-finite comparison)" in text
    assert "candidate_latency_delta=" not in text


@pytest.mark.parametrize("agent_type", [DistillerAgent, KnowledgeDistillerAgent])
def test_both_distiller_prompts_receive_paired_timing(agent_type):
    agent = agent_type()
    records = [measured(1, 1, 1), measured(2, 1, 2, baseline=1)]
    prompt = agent.preprocess(agent.InputModel.model_validate({
        "definition_name": "attention", "op_type": "attention",
        "target_hardware": "Ascend910B", "rounds": records,
        "best_geo_mean": 2, "best_round": 2,
    }), FakeRuntime([]))
    assert "candidate_latency_delta=+0.000000%" in prompt
    assert "reference_latency_delta=+100.000000%" in prompt


def test_bounded_synthesis_keeps_best_and_final_paired_evidence():
    records = [measured(1, 1, 1)] + [
        measured(n, 1 + n / 100, 1, baseline=1, parent=1) for n in range(2, 60)
    ]
    text = build_synthesis_trajectory(records, best_round=1, max_chars=4096)
    assert len(text) <= 4096
    assert "FINAL R59:" in text
    assert "paired_timing_vs_best+parent_R1: cases=1, candidate_latency_delta=+59.000000%" in text
