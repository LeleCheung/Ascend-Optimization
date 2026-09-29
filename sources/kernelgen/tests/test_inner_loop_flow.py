"""Integration test: the CoderAgent inner-loop DATA FLOW
(ADR-3 #7). No LLM, no GPU, no server — we drive the tools directly with real
data (not FakeRuntime scripts) to catch wiring bugs the unit tests can't:
round_num agreement between eval_round/finalize_round, atomic round finalization,
plateau counter, and Python-owned KEEP/REVERT via .best_kernel.py.

This is the layer BETWEEN unit tests (schema/wiring) and the docker end-to-end
(real LLM + GPU, deferred to #11).

    cd /data/akg_kernel_bench_lite
    python -m pytest tests/test_inner_loop_flow.py -v
    # or:
    python tests/test_inner_loop_flow.py
"""

import contextlib
import io
import json
import sys
from pathlib import Path


from kernelgen.tools.eval_round import record_and_augment  # noqa: E402
from kernelgen.tools import finalize_round as finalize_tool  # noqa: E402
from kernelgen.data.ledger import Ledger  # noqa: E402
from kernelgen.data.stop_policy import StopConfig  # noqa: E402
from kernelgen.tests.helpers import experiment_plan, round_conclusion


def _eval_result(status="PASSED", geo=1.0):
    """What eval_cli would print for a single-workload eval."""
    return {"status": status, "geo_mean": geo, "min_speedup": geo,
            "worst_workload_uuid": "", "latency_ms": 0.1, "abs_err": 0.0,
            "rel_err": 0.0, "num_workloads": 1, "num_passed": 1,
            "per_workload": [{"uuid": "u0", "axes": {}, "status": status, "speedup": geo}]}


def _conclusion(round_num, level="L2_memory"):
    return round_conclusion(round_num, optimization_level=level)


def _run_cli(mod, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = mod.main(argv)
    return code, json.loads(buf.getvalue())


def _agent_round(wt, round_num, kernel_code, eval_result, conclusion):
    """Simulate ONE agent iteration the way code_eval_profile.md prescribes:
    eval_round (measure+record) -> write summary.json -> finalize_round (finalize).
    Returns the eval result augmented with the finalization verdict."""
    candidate = wt / "tmp" / "main.py"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(kernel_code)
    # 3. eval_round: authoritative measure + record (the seam the agent can't split)
    aug = record_and_augment(wt, eval_result, kernel_code, experiment_plan(round_num))
    # 5. finalize_round: atomically record conclusion and the stop verdict
    sfile = wt / "tmp" / "conclusion.json"
    sfile.parent.mkdir(parents=True, exist_ok=True)
    s = dict(conclusion, round_num=aug["round_num"])
    sfile.write_text(json.dumps(s))
    code, out = _run_cli(finalize_tool, ["--ledger-dir", str(wt), str(sfile)])
    assert code == 0 and out["recorded"], out
    assert out["round_finalized"] is True
    aug["finalization"] = out
    return aug


def test_round_num_flows_across_tools(tmp_path):
    """eval_round assigns round_num; finalize_round must attach to the SAME round;
    its returned verdict must reflect the finalized snapshot."""
    aug = _agent_round(tmp_path, 1, "code_v1", _eval_result(geo=1.2), _conclusion(1))
    assert aug["round_num"] == 1 and aug["is_new_best"] is True

    # ledger now has round 1 with BOTH measurement and conclusion
    led = Ledger(tmp_path)
    rec = led.history.rounds[0]
    assert rec.round_num == 1
    assert abs(rec.evaluation.geo_mean - 1.2) < 1e-9
    assert rec.plan.strategy == "establish measured baseline"
    assert rec.conclusion.optimization_level == "L2_memory"

    out = aug["finalization"]
    assert out["snapshot"]["round_count"] == 1
    assert out["snapshot"]["best_geo_mean"] == 1.2
    assert out["snapshot"]["levels_covered"] == ["L2_memory"]
    assert out["continue"] is True                   # 1 round, no plateau yet


def test_plateau_drives_stop(tmp_path):
    """finalize_round must stop after two non-improving rounds."""
    Ledger(tmp_path).set_stop_config(StopConfig(early_stop_rounds=2, min_rounds=1))
    out1 = _agent_round(
        tmp_path, 1, "v1", _eval_result(geo=1.30), _conclusion(1, "L1_architecture")
    )["finalization"]
    assert out1["continue"] is True
    assert out1["snapshot"]["rounds_without_improvement"] == 0

    out2 = _agent_round(
        tmp_path, 2, "v2", _eval_result(geo=1.10), _conclusion(2)
    )["finalization"]
    assert out2["continue"] is True                  # 1 < 2
    assert out2["snapshot"]["rounds_without_improvement"] == 1

    out3 = _agent_round(
        tmp_path, 3, "v3", _eval_result(geo=1.05), _conclusion(3)
    )["finalization"]
    assert out3["continue"] is False                 # plateau hit -> STOP
    assert "no improvement" in out3["reason"]


def test_revert_uses_best_kernel_file(tmp_path):
    """finalize_round must deterministically KEEP or restore the authoritative best."""
    first = _agent_round(
        tmp_path, 1, "BEST_CODE", _eval_result(geo=1.5), _conclusion(1)
    )
    assert first["finalization"]["candidate_action"] == "KEEP"
    assert (tmp_path / ".best_kernel.py").read_text() == "BEST_CODE"

    second = _agent_round(
        tmp_path, 2, "WORSE_CODE", _eval_result(geo=1.2), _conclusion(2)
    )
    assert second["finalization"]["candidate_action"] == "REVERT"
    assert (tmp_path / ".best_kernel.py").read_text() == "BEST_CODE"   # not overwritten
    assert (tmp_path / "tmp" / "main.py").read_text() == "BEST_CODE"


def test_failed_round_records_but_not_best(tmp_path):
    """A COMPILE_ERROR round is recorded but is neither a KEEP nor a plateau."""
    _agent_round(tmp_path, 1, "ok", _eval_result(geo=1.3), _conclusion(1))
    aug = _agent_round(tmp_path, 2, "broken",
                       _eval_result(status="COMPILE_ERROR", geo=0.0), _conclusion(2))
    assert aug["is_new_best"] is False
    assert aug["finalization"]["candidate_action"] == "REVERT"
    assert (tmp_path / "tmp" / "main.py").read_text() == "ok"
    led = Ledger(tmp_path)
    assert led.history.rounds[1].evaluation.status == "COMPILE_ERROR"
    assert abs(led.history.best_geo_mean - 1.3) < 1e-9          # best unchanged
    # Failed measurements do not consume the performance-plateau budget.
    out = aug["finalization"]
    assert out["snapshot"]["round_count"] == 2
    assert out["snapshot"]["rounds_without_improvement"] == 0


def test_numerical_failures_do_not_trigger_performance_plateau(tmp_path):
    """Correctness repair must continue even when early-stop-rounds is small."""
    Ledger(tmp_path).set_stop_config(StopConfig(early_stop_rounds=1, min_rounds=1))
    failed = _eval_result(status="INCORRECT_NUMERICAL", geo=None)
    failed["num_passed"] = 0
    first = _agent_round(tmp_path, 1, "wrong_v1", failed, _conclusion(1))
    aug = _agent_round(tmp_path, 2, "wrong_v2", failed, _conclusion(2))

    assert first["finalization"]["candidate_action"] == "REPAIR"
    assert aug["finalization"]["candidate_action"] == "REPAIR"
    assert (tmp_path / "tmp" / "main.py").read_text() == "wrong_v2"
    out = aug["finalization"]
    assert out["continue"] is True
    assert out["snapshot"]["round_count"] == 2
    assert out["snapshot"]["rounds_without_improvement"] == 0


def test_min_rounds_floor_blocks_early_stop(tmp_path):
    """Even at a plateau, finalization must NOT stop before min_rounds."""
    Ledger(tmp_path).set_stop_config(StopConfig(early_stop_rounds=1, min_rounds=5))
    _agent_round(tmp_path, 1, "v1", _eval_result(geo=1.3), _conclusion(1))
    out = _agent_round(
        tmp_path, 2, "v2", _eval_result(geo=1.1), _conclusion(2)
    )["finalization"]
    assert out["continue"] is True                   # trigger fires but floor defers
    assert "deferred" in out["reason"]


if __name__ == "__main__":
    import inspect
    import tempfile
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            if "tmp_path" in inspect.signature(t).parameters:
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
                t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
