"""Unit tests for the stop policy (ADR-3 task #2).

Pure Python, no torch/GPU:

    cd /data/akg_kernel_bench_lite
    python -m pytest tests/test_stop_policy.py -v
    # or:
    python tests/test_stop_policy.py
"""

import sys
from pathlib import Path


from kernelgen.data.stop_policy import (  # noqa: E402
    StopConfig,
    Verdict,
    max_round_reached,
    next_verdict,
    no_improvement,
    min_rounds_floor,
)
from kernelgen.data.ledger import Ledger  # noqa: E402
from kernelgen.tests.helpers import experiment_plan, round_conclusion


def _snap(rwi=0, rounds=0):
    return {
        "rounds_without_improvement": rwi,
        "round_count": rounds,
    }


# --- pure hooks -----------------------------------------------------------

def test_no_improvement_trigger():
    cfg = StopConfig(early_stop_rounds=3)
    assert no_improvement(_snap(rwi=2), cfg) is None          # below threshold
    assert no_improvement(_snap(rwi=3), cfg) is not None       # at threshold -> wants stop
    assert no_improvement(_snap(rwi=5), cfg) is not None       # above


def test_no_improvement_disabled_when_zero():
    cfg = StopConfig(early_stop_rounds=0)                       # disabled
    assert no_improvement(_snap(rwi=99), cfg) is None


def test_max_round_defaults_to_fifteen():
    cfg = StopConfig()
    assert cfg.early_stop_rounds == 3
    assert cfg.max_round == 15
    assert max_round_reached(_snap(rounds=14), cfg) is None
    assert max_round_reached(_snap(rounds=15), cfg) is not None


def test_max_round_is_hard_and_bypasses_soft_stop_and_min_rounds():
    cfg = StopConfig(
        early_stop_rounds=0,
        min_rounds=100,
        max_round=2,
        soft_stop_disabled=True,
    )
    verdict = next_verdict(_snap(rounds=2), cfg)
    assert verdict.should_continue is False
    assert verdict.code == "max_round_reached"


def test_min_rounds_floor_veto():
    cfg = StopConfig(min_rounds=2)
    assert min_rounds_floor(_snap(rounds=1), cfg) is not None   # below floor -> veto stop
    assert min_rounds_floor(_snap(rounds=2), cfg) is None       # at floor -> no veto
    assert min_rounds_floor(_snap(rounds=5), cfg) is None


# --- next_verdict composition --------------------------------------------

def test_no_trigger_continues():
    cfg = StopConfig(early_stop_rounds=3, min_rounds=1)
    v = next_verdict(_snap(rwi=1, rounds=5), cfg)
    assert v.should_continue is True
    assert v.reason == "no stop trigger"


def test_trigger_fires_and_stops():
    cfg = StopConfig(early_stop_rounds=3, min_rounds=2)
    v = next_verdict(_snap(rwi=3, rounds=5), cfg)              # plateaued, past floor
    assert v.should_continue is False
    assert "no improvement" in v.reason


def test_trigger_deferred_by_min_rounds():
    cfg = StopConfig(early_stop_rounds=3, min_rounds=5)
    v = next_verdict(_snap(rwi=3, rounds=2), cfg)             # plateaued but too few rounds
    assert v.should_continue is True
    assert "deferred" in v.reason


def test_soft_stop_disabled_always_continues():
    cfg = StopConfig(
        early_stop_rounds=1,
        min_rounds=1,
        max_round=100,
        soft_stop_disabled=True,
    )
    v = next_verdict(_snap(rwi=99, rounds=99), cfg)           # would otherwise stop
    assert v.should_continue is True
    assert "paper" in v.reason.lower() or "disabled" in v.reason.lower()


def test_default_config_stops_at_max_round():
    # Default early_stop_rounds=0 disables the plateau trigger, not max_round.
    v = next_verdict(_snap(rwi=100, rounds=15), StopConfig())
    assert v.should_continue is False
    assert v.code == "max_round_reached"


# --- wiring through Ledger.should_stop ------------------------------------

def _eval(status="PASSED", geo=1.0):
    return {"status": status, "geo_mean": geo, "min_speedup": geo,
            "worst_workload_uuid": "", "latency_ms": 0.1, "abs_err": 0.0,
            "rel_err": 0.0, "num_workloads": 1, "num_passed": 1,
            "per_workload": [{"uuid": "u0", "axes": {}, "status": status,
                              "speedup": geo}]}


def _record_conclusion(ledger, round_num):
    ledger.finalize_round(
        round_num,
        round_conclusion(round_num),
        StopConfig(early_stop_rounds=0, max_round=1000),
    )


def test_ledger_should_stop_wiring(tmp_path):
    led = Ledger(tmp_path)
    cfg = StopConfig(early_stop_rounds=2, min_rounds=1)

    led.record_eval(_eval(geo=1.30), "best", experiment_plan(1))   # round 1: new best, rwi=0
    v = led.should_stop(cfg)
    assert isinstance(v, Verdict) and v.should_continue is True   # no plateau

    _record_conclusion(led, 1)
    led.record_eval(_eval(geo=1.10), "w1", experiment_plan(2))     # round 2: no improve, rwi=1
    assert led.should_stop(cfg).should_continue is True           # 1 < 2

    _record_conclusion(led, 2)
    led.record_eval(_eval(geo=1.05), "w2", experiment_plan(3))     # round 3: no improve, rwi=2
    v = led.should_stop(cfg)
    assert v.should_continue is False                             # plateau hit, past floor
    assert "no improvement" in v.reason


def test_ledger_should_stop_default_cfg(tmp_path):
    led = Ledger(tmp_path)
    led.record_eval(_eval(geo=1.0), "c", experiment_plan(1))
    # One measured round is below the default three-round plateau threshold.
    assert led.should_stop().should_continue is True


def test_ledger_max_round_counts_passed_and_failed_rounds(tmp_path):
    led = Ledger(tmp_path)
    cfg = StopConfig(max_round=3)
    failed = _eval(status="RUNTIME_ERROR", geo=None)
    led.record_eval(failed, "bad1", experiment_plan(1))
    _record_conclusion(led, 1)
    led.record_eval(failed, "bad2", experiment_plan(2))
    assert led.should_stop(cfg).should_continue is True

    _record_conclusion(led, 2)
    led.record_eval(_eval(status="PASSED", geo=1.0), "good", experiment_plan(3))
    verdict = led.should_stop(cfg)
    assert verdict.should_continue is False
    assert verdict.code == "max_round_reached"


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
