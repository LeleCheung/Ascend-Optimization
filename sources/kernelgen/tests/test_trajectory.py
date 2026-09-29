import pytest
from types import SimpleNamespace

from kernelgen.agents.distiller import DistillerAgent
from kernelgen.agents.knowledge_distiller import KnowledgeDistillerAgent
from kernelgen.data.trajectory import (
    build_synthesis_trajectory,
    build_trajectory,
)
from kernelgen.tests.helpers import experiment_plan, round_conclusion
from kernelgen.framework import Directory, FakeRuntime
from kernelgen.data.implementation import ImplementationLanguage
from kernelgen.workflows.optimization.kernelgen import epoch
from kernelgen.workflows.optimization.kernelgen.contracts import EpochResult


_COMPLETE_CODE = """import triton
import triton.language as tl

BLOCK_M = 128

@triton.jit
def kernel(X, Y, BLOCK_N: tl.constexpr):
    values = tl.load(X + tl.arange(0, BLOCK_N))

    # Blank lines must not erase the measured computation.
    weights = tl.exp(values - tl.max(values, 0))
    tl.store(Y + tl.arange(0, BLOCK_N), weights / tl.sum(weights, 0))

def pick_tiles():
    block_n = 256
    if block_n > BLOCK_M:
        block_n = BLOCK_M
    return BLOCK_M, block_n

def run(x, y):
    block_m, block_n = pick_tiles()
    kernel[(1,)](x, y, BLOCK_N=block_n)
"""


def _round(round_num, code, geo, latency, expectation_status=None):
    conclusion = round_conclusion(round_num)
    if expectation_status:
        conclusion["expectation_status"] = expectation_status
    return {
        "round_num": round_num,
        "plan": experiment_plan(round_num),
        "solution": {"code": code},
        "evaluation": {
            "status": "PASSED",
            "geo_mean": geo,
            "workloads": [
                {
                    "uuid": "u0",
                    "latency_ms": latency,
                    "reference_latency_ms": 0.2,
                    "speedup": 0.2 / latency,
                }
            ],
            "comparison": {
                "baseline_round_num": 1 if round_num > 1 else None,
                "geo_mean_delta_pct": (geo - 1.0) * 100.0 if round_num > 1 else None,
            },
        },
        "profile": {"status": "not_required"},
        "conclusion": conclusion,
    }


def test_trajectory_exposes_frozen_expectation_and_observed_gap():
    trajectory = build_trajectory(
        [
            _round(1, "def run():\n    pass\n", 1.0, 0.2),
            _round(2, "def run():\n    return 1\n", 1.1, 0.18, "met"),
        ]
    )
    assert "hypothesis: hypothesis 2" in trajectory
    assert "expectation_status: met" in trajectory
    assert "measured_vs_best_R1: geo_delta=+10.000000%" in trajectory
    assert "perf_gap: observed result matched plan 2" in trajectory


@pytest.mark.parametrize("agent_type", [DistillerAgent, KnowledgeDistillerAgent])
def test_distillers_receive_complete_measured_source_not_only_jit_fragments(agent_type):
    record = _round(1, _COMPLETE_CODE, 1.0, 0.2)
    agent = agent_type()
    prompt = agent.preprocess(agent.InputModel.model_validate({
        "definition_name": "attention", "op_type": "attention", "target_hardware": "Ascend910B",
        "rounds": [record], "best_geo_mean": 1.0, "best_round": 1,
    }), FakeRuntime([]))

    assert f"```python\n{_COMPLETE_CODE}\n```" in prompt


def test_rewrite_preserves_host_dispatch_and_complete_kernel_body():
    initial = "def run(x, y):\n    return x\n"
    trajectory = build_trajectory([
        _round(1, initial, 1.0, 0.2),
        _round(2, _COMPLETE_CODE, 1.1, 0.18),
    ])

    assert "[REWRITE" in trajectory
    assert f"```python\n{_COMPLETE_CODE}\n```" in trajectory


def test_small_dispatch_diff_retains_unchanged_clamp_in_initial_source():
    updated = _COMPLETE_CODE.replace("block_n = 256", "block_n = 512")
    trajectory = build_trajectory([
        _round(1, _COMPLETE_CODE, 1.0, 0.2),
        _round(2, updated, 1.1, 0.18),
    ])

    assert "[REWRITE" not in trajectory
    assert f"```python\n{_COMPLETE_CODE}\n```" in trajectory
    assert "-    block_n = 256" in trajectory
    assert "+    block_n = 512" in trajectory


def test_epoch_summary_keeps_complete_authoritative_best_source(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(epoch.EpochSummaryAgent, "run", lambda self, inp, runtime: captured.append(inp))
    inp = SimpleNamespace(
        definition=SimpleNamespace(name="attention", op_type="attention"),
        target_hardware="Ascend910B", implementation_language=ImplementationLanguage.TRITON,
    )
    best = EpochResult("attention", best_code=_COMPLETE_CODE, best_round=2,
                       best_geo_mean=1.1, best_workspace_path=tmp_path/"1R/agent0")

    # No new report is needed to preserve a winner from a completed epoch.
    epoch.summarize_epoch(tmp_path, inp, [], best, FakeRuntime([]),
                          epoch_workspace=Directory(base=tmp_path/"2R"))

    assert captured[0]["fixed_best_kernel"] == _COMPLETE_CODE
    assert captured[0]["fixed_best_agent"] == "1R/agent0"


def test_synthesis_trajectory_is_bounded_and_code_free():
    rounds = []
    for round_num in range(1, 60):
        geo = 1.0 + min(round_num, 37) / 100.0
        record = _round(
            round_num,
            f"SECRET_KERNEL_R{round_num}\n" + ("kernel body\n" * 100),
            geo,
            0.2 / geo,
        )
        record["plan"]["strategy"] = f"strategy {round_num} " + ("detail " * 80)
        record["plan"]["hypothesis"] = f"hypothesis {round_num} " + ("reason " * 80)
        if round_num == 40:
            record["evaluation"]["status"] = "COMPILE_FAILED"
            record["evaluation"]["geo_mean"] = None
            record["conclusion"]["root_cause"] = "unique compilation cause"
        rounds.append(record)

    trajectory = build_synthesis_trajectory(
        rounds,
        best_round=37,
        max_chars=8000,
    )

    assert len(trajectory) <= 8000
    assert "SECRET_KERNEL" not in trajectory
    assert "```" not in trajectory
    assert "[REWRITE" not in trajectory
    assert "BEST R37:" in trajectory
    assert "FINAL R59:" in trajectory
    assert "R40 | COMPILE_FAILED" in trajectory
    assert "unique compilation cause" in trajectory
