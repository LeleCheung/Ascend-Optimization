"""Normal execution and epoch recovery share the measured completion contract."""

import json

import pytest

from kernelgen.data.implementation import ImplementationLanguage
from kernelgen.data.ledger import Ledger
from kernelgen.data.stop_policy import StopConfig
from kernelgen.tests.helpers import experiment_plan, round_conclusion
from kernelgen.workflows.optimization.kernelgen.recovery import load_completed_epoch


IDENTITY = {
    "definition_name": "test_definition",
    "target_hardware": "A100",
    "implementation_language": "triton",
}


def _ledger(root, *, status="PASSED"):
    ledger = Ledger(root / "1R" / "agent0")
    ledger.record_eval(
        {"status": status, "geo_mean": 1.25 if status == "PASSED" else None},
        "candidate_code",
        experiment_plan(1),
        profile_enabled=True,
        **IDENTITY,
    )
    return ledger


def _manifest(root):
    (root / "1R" / "epoch-completion.json").write_text(json.dumps({
        "attempted_agents": ["agent0", "agent1"],
        "successful_agents": ["agent0"],
        "failed_agents": [{"name": "agent1", "error_type": "RuntimeError", "error": "startup failed"}],
    }), encoding="utf-8")


def _recover(root):
    return load_completed_epoch(
        cwd=root, definition_name=IDENTITY["definition_name"],
        target_hardware=IDENTITY["target_hardware"],
        implementation_language=ImplementationLanguage.TRITON, epoch_num=1,
    )


@pytest.mark.parametrize("manifest", [False, True])
@pytest.mark.parametrize("phase", ["pending_conclusion", "missing_verdict", "continue", "cancelled", "stopped"])
def test_recovery_requires_same_completion_boundary(tmp_path, manifest, phase):
    ledger = _ledger(tmp_path)
    if phase != "pending_conclusion":
        ledger.finalize_round(1, round_conclusion(1), StopConfig(max_round=1 if phase == "stopped" else 10))
    if phase == "cancelled":
        ledger.record_supervisor_stop(1, code="user_cancelled", reason="user cancellation")
    elif phase == "missing_verdict":
        raw = json.loads(ledger.path.read_text())
        raw["rounds"][0]["next_verdict"] = None
        ledger.path.write_text(json.dumps(raw), encoding="utf-8")
    if manifest:
        _manifest(tmp_path)
    reloaded = Ledger(ledger.dir)
    assert reloaded.coder_completed is (phase == "stopped")
    if phase == "stopped":
        # Optional final-best profiling is not a Coder completion gate.
        assert reloaded.pending_profile_round() is not None
        results, _ = _recover(tmp_path)
        # Coder completion remains recoverable, but an old measured success
        # without independent confirmation cannot become an accepted winner.
        assert results[0][0].status == "NEEDS_RETEST"
    else:
        with pytest.raises(ValueError, match="Coder is not complete"):
            _recover(tmp_path)


@pytest.mark.parametrize("field", list(IDENTITY))
@pytest.mark.parametrize("value", ["", "wrong"])
def test_measured_identity_is_strict_in_both_paths(tmp_path, field, value):
    ledger = _ledger(tmp_path)
    ledger.finalize_round(1, round_conclusion(1), StopConfig(max_round=1))
    raw = json.loads(ledger.path.read_text())
    raw[field] = value
    ledger.path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{field} mismatch"):
        Ledger(ledger.dir).validate_identity(**IDENTITY)
    with pytest.raises(ValueError, match=f"{field} mismatch"):
        _recover(tmp_path)


def test_failed_measured_stop_remains_completed(tmp_path):
    ledger = _ledger(tmp_path, status="RUNTIME_ERROR")
    ledger.finalize_round(1, round_conclusion(1), StopConfig(max_round=1))
    assert ledger.coder_completed
    results, _ = _recover(tmp_path)
    assert results[0][0].status == "RUNTIME_ERROR"


def test_empty_ledger_needs_completion_manifest_and_cannot_claim_best(tmp_path):
    ledger = Ledger(tmp_path / "1R" / "agent0")
    ledger.history.save(ledger.path)
    assert not ledger.coder_completed
    with pytest.raises(ValueError, match="Coder is not complete"):
        _recover(tmp_path)
    _manifest(tmp_path)
    results, _ = _recover(tmp_path)
    assert results[0][0].status == "FAILED"
    ledger.update_best(1.25, "unmeasured_code", round_num=1)
    with pytest.raises(ValueError, match="Coder is not complete"):
        _recover(tmp_path)


def test_cancellation_cannot_become_complete_until_real_stop(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.finalize_round(1, round_conclusion(1), StopConfig(max_round=10))
    ledger.record_supervisor_stop(1, code="user_cancelled", reason="user cancellation")
    assert not ledger.coder_completed
    assert ledger.clear_supervisor_stop().should_continue
    assert not ledger.coder_completed
    ledger.record_supervisor_stop(1, code="coder_session_limit_reached", reason="invocation budget exhausted")
    assert Ledger(ledger.dir).coder_completed
