"""The exported best-kernel file must not affect measured optimization facts."""

import pytest

from kernelgen.agents.coder import CoderReport
from kernelgen.data.ledger import Ledger
from kernelgen.data.stop_policy import StopConfig
from kernelgen.framework import Directory
from kernelgen.tests.helpers import experiment_plan, round_conclusion
from kernelgen.workflows.optimization.kernelgen.epoch import collect_epoch_result
from kernelgen.workflows.optimization.single_coder.coder_loop import _add_fresh_context


def _record(ledger, code, geo, round_num=1):
    ledger.record_eval(
        {"status": "PASSED", "geo_mean": geo, "per_workload": [
            {"uuid": "w0", "status": "PASSED", "speedup": geo},
        ]},
        code,
        experiment_plan(round_num),
    )


@pytest.mark.parametrize("export", [None, "", "old_code", "uncommitted_code", b"\xff"])
def test_best_revert_and_resume_seed_ignore_export(tmp_path, export):
    ledger = Ledger(tmp_path)
    _record(ledger, "measured_code", 1.25)
    if export is None:
        ledger.best_kernel_path.unlink()
    elif isinstance(export, bytes):
        ledger.best_kernel_path.write_bytes(export)
    else:
        ledger.best_kernel_path.write_text(export, encoding="utf-8")

    reloaded = Ledger(tmp_path)
    assert reloaded.best == {"geo_mean": 1.25, "round": 1, "code": "measured_code"}
    assert reloaded.revert_to_best()
    assert (tmp_path / "kernel.py").read_text() == "measured_code"
    coder_input = {}
    _add_fresh_context(coder_input, reloaded)
    assert coder_input["seed_code"] == "measured_code"


def test_interrupted_ledger_commit_cannot_publish_uncommitted_best(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path)
    _record(ledger, "round_1_code", 1.25)
    ledger.finalize_round(1, round_conclusion(1), StopConfig(max_round=10))

    def fail_save():
        raise OSError("interrupted before ledger replacement")

    monkeypatch.setattr(ledger, "_atomic_save", fail_save)
    with pytest.raises(OSError, match="interrupted"):
        _record(ledger, "round_2_code", 1.5, 2)

    assert ledger.best_kernel_path.read_text() == "round_2_code"
    committed = Ledger(tmp_path)
    assert committed.best == {"geo_mean": 1.25, "round": 1, "code": "round_1_code"}
    assert len(committed.history.rounds) == 1


@pytest.mark.parametrize("missing_export", [False, True])
def test_kernelgen_collect_uses_winners_ledger_code(tmp_path, missing_export):
    workspace = Directory(base=tmp_path)
    for name, geo in [("agent0", 1.1), ("agent1", 1.5)]:
        workspace.allocate(name)
        ledger = Ledger(tmp_path / name)
        _record(ledger, f"{name}_code", geo)
        if missing_export:
            ledger.best_kernel_path.unlink()
        else:
            ledger.best_kernel_path.write_text("unmeasured_code", encoding="utf-8")
    result = collect_epoch_result(
        "test_definition",
        [(CoderReport(status="PASSED", summary=name), name) for name in ("agent0", "agent1")],
        workspace,
    )
    assert result.best_geo_mean == 1.5
    assert result.best_code == "agent1_code"
    assert result.best_workspace_path == tmp_path / "agent1"
