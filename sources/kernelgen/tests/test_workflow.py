"""Unit tests for Runnable + Workflow (ADR-3 #9 core).

Proves: a Workflow satisfies the SAME contract as an agent (run + I/O validation),
orchestrates inner Runnables (agents) + plain Python (pick_best called directly,
NOT wrapped), and nests inside another Workflow.

    cd /data/akg_kernel_bench_lite
    python -m pytest tests/test_workflow.py -v
    # or:
    python tests/test_workflow.py
"""

import sys
from pathlib import Path


from pydantic import BaseModel  # noqa: E402

from kernelgen.framework import (  # noqa: E402
    Runnable,
    Workflow,
    BaseAgent,
    FakeRuntime,
)
from kernelgen.data.selection import pick_best  # noqa: E402


# --- a trivial agent (Runnable) used inside workflows ---------------------

class _ScoreIn(BaseModel):
    x: int


class _ScoreOut(BaseModel):
    score: float


class _ScoreAgent(BaseAgent):
    name = "scorer"
    md_path = ""
    InputModel = _ScoreIn
    OutputModel = _ScoreOut


def test_agent_is_runnable():
    assert issubclass(BaseAgent, Runnable)
    assert issubclass(Workflow, Runnable)


# --- a workflow orchestrating agents + plain-Python pick_best -------------

class _WfIn(BaseModel):
    candidates: list  # list of ints


class _WfOut(BaseModel):
    best_score: float
    n: int


class _ScoreThenPickWorkflow(Workflow):
    """Run _ScoreAgent on each candidate (heavy step), then pick_best in PLAIN
    Python (light step, called directly — not wrapped as a node)."""
    name = "score_then_pick"
    InputModel = _WfIn
    OutputModel = _WfOut

    def _execute(self, inp):
        scored = []
        for x in inp.candidates:
            out = _ScoreAgent().run({"x": x}, self._runtime)   # inner Runnable
            scored.append((x, out.score))
        best = pick_best(scored, key=lambda t: t[1])  # plain Python, direct call
        return {"best_score": best[1], "n": len(scored)}


def _ctx_for(scores):
    # FakeRuntime replies one score per agent call
    replies = [f'{{"score": {s}}}' for s in scores]
    return FakeRuntime(replies)


def test_workflow_runs_and_validates_output():
    rt = _ctx_for([1.1, 1.5, 0.9])
    out = _ScoreThenPickWorkflow().run({"candidates": [10, 20, 30]}, rt)
    assert isinstance(out, _WfOut)
    assert out.best_score == 1.5      # pick_best chose the max
    assert out.n == 3


def test_workflow_validates_input():
    rt = _ctx_for([1.0])
    try:
        _ScoreThenPickWorkflow().run({"wrong_field": 1}, rt)  # missing 'candidates'
    except Exception:
        pass
    else:
        raise AssertionError("workflow should validate input and raise on bad input")


def test_workflow_validates_output():
    # agent returns a valid _ScoreOut, but if a workflow returns a bad dict it must fail
    class _BadWorkflow(Workflow):
        InputModel = _WfIn
        OutputModel = _WfOut

        def _execute(self, inp):
            return {"best_score": "not-a-number"}   # invalid + missing n

    rt = _ctx_for([])
    try:
        _BadWorkflow().run({"candidates": []}, rt)
    except Exception:
        pass
    else:
        raise AssertionError("workflow should validate output")


# --- nesting: a workflow inside another workflow --------------------------

class _OuterIn(BaseModel):
    groups: list  # list of candidate-lists


class _OuterOut(BaseModel):
    overall_best: float


class _OuterWorkflow(Workflow):
    """Nests _ScoreThenPickWorkflow (a Runnable) and combines with plain Python."""
    name = "outer"
    InputModel = _OuterIn
    OutputModel = _OuterOut

    def _execute(self, inp):
        inner = _ScoreThenPickWorkflow()
        bests = [inner.run({"candidates": g}, self._runtime).best_score for g in inp.groups]
        return {"overall_best": max(bests)}


def test_workflow_nesting():
    # 2 groups: group1 scores [1.1,1.5], group2 scores [2.0,0.5] -> bests 1.5, 2.0
    rt = FakeRuntime(
        ['{"score": 1.1}', '{"score": 1.5}', '{"score": 2.0}', '{"score": 0.5}'])
    out = _OuterWorkflow().run({"groups": [[1, 2], [3, 4]]}, rt)
    assert isinstance(out, _OuterOut)
    assert out.overall_best == 2.0


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
