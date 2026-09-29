"""Winner provenance stays attached across epoch selection, recovery and synthesis."""

from pathlib import Path

import pytest

from kernelgen.agents.coder import CoderReport
from kernelgen.agents.epoch_summary import EpochSummaryOutput
from kernelgen.data.implementation import ImplementationLanguage
from kernelgen.data.ledger import Ledger
from kernelgen.data.stop_policy import StopConfig
from kernelgen.framework import Directory, FakeRuntime
from kernelgen.tests.helpers import experiment_plan, round_conclusion
from kernelgen.workflows.optimization.kernelgen import KernelGenInput, KernelGenOutput
from kernelgen.workflows.optimization.kernelgen import epoch, recovery
from kernelgen.workflows.optimization.kernelgen.contracts import EpochResult


def _record(path, geo, code, *, rounds=1):
    ledger = Ledger(path)
    for number in range(1, rounds + 1):
        score = geo if number == rounds else geo / 2
        ledger.record_eval(
            {"status": "PASSED", "geo_mean": score},
            code if number == rounds else "BASELINE_CODE",
            experiment_plan(number), definition_name="identity", target_hardware="A100",
        )
        ledger.finalize_round(number, round_conclusion(number), StopConfig(max_round=rounds))
    return ledger


def _results(workspace, names):
    for name in names:
        workspace.allocate(name)
    return [(CoderReport(status="PASSED", summary=name), name) for name in names]


@pytest.mark.parametrize("current_geo", [1.4, 1.5, 1.6])
def test_winner_identity_survives_ties_older_epochs_and_recovery(tmp_path, monkeypatch, current_geo):
    prior_workspace = Directory(base=tmp_path / "1R")
    current_workspace = Directory(base=tmp_path / "2R")
    prior_reports = _results(prior_workspace, ["agent1"])
    current_reports = _results(current_workspace, ["agent0", "agent1"])
    _record(prior_workspace.path_of("agent1"), 1.5, "PRIOR_BEST_CODE", rounds=2)
    _record(current_workspace.path_of("agent0"), current_geo, "CURRENT_BEST_CODE")
    _record(current_workspace.path_of("agent1"), 1.1, "OTHER_CODE")
    prior = epoch.collect_epoch_result("identity", prior_reports, prior_workspace)
    current = epoch.collect_epoch_result("identity", current_reports, current_workspace)
    best = epoch.select_best_result(prior, current)
    assert best is (current if current_geo > 1.5 else prior)
    restored = recovery.load_completed_result(
        cwd=tmp_path, definition_name="identity", target_hardware="A100",
        implementation_language=ImplementationLanguage.TRITON, through_epoch=2,
    )
    assert restored == best

    captured = []

    def capture(self, inp, runtime):
        captured.append(inp)
        return EpochSummaryOutput(next_directions=[], synthesis_report="checked")

    monkeypatch.setattr(epoch.EpochSummaryAgent, "run", capture)
    inp = KernelGenInput.model_validate({
        "definition": {
            "name": "identity", "op_type": "elementwise", "axes": {},
            "inputs": {"x": {"shape": [16], "dtype": "float32"}},
            "outputs": {"y": {"shape": [16], "dtype": "float32"}},
            "reference": "def run(x): return x",
        },
        "target_hardware": "A100", "n_parallel": 2, "n_epoch": 2,
    })
    epoch.summarize_epoch(tmp_path, inp, current_reports, best, FakeRuntime([]), epoch_workspace=current_workspace)
    synthesis = captured[0]
    assert synthesis["fixed_best_agent"] == ("2R/agent0" if current_geo > 1.5 else "1R/agent1")
    assert synthesis["fixed_best_round"] == (1 if current_geo > 1.5 else 2)
    assert synthesis["fixed_best_geo"] == best.best_geo_mean
    assert synthesis["fixed_best_kernel"] == best.best_code
    assert [item["agent_id"] for item in synthesis["agent_results"]] == ["2R/agent0", "2R/agent1"]
    # Internal provenance does not expand the stable public JSON contract.
    assert set(best.as_output().model_dump()) == set(KernelGenOutput.model_fields)


@pytest.mark.parametrize("order", [("agent0", "agent1"), ("agent1", "agent0")])
def test_collect_reads_each_ledger_once_and_keeps_first_tie(tmp_path, monkeypatch, order):
    workspace = Directory(base=tmp_path)
    reports = _results(workspace, order)
    for name in order:
        _record(workspace.path_of(name), 1.5, name)
    reads = []

    def load(path):
        reads.append(Path(path))
        return Ledger(path)

    monkeypatch.setattr(epoch, "Ledger", load)
    result = epoch.collect_epoch_result("identity", reports, workspace)
    assert reads == [tmp_path / name for name in order]
    assert result.best_workspace_path == tmp_path / order[0]
    assert result.best_round == 1
    assert result.best_code == order[0]


def test_empty_selection_and_failed_epochs_keep_public_metadata(tmp_path):
    empty = EpochResult("identity")
    workspace = Directory(base=tmp_path)
    reports = _results(workspace, ["agent0"])
    failed = epoch.collect_epoch_result("identity", reports, workspace)
    selected = epoch.select_best_result(empty, failed)
    assert selected is failed
    assert selected.status == "FAILED"
    assert selected.best_workspace_path is None
    assert selected.as_output().num_agents == 1
    assert epoch.select_best_result(selected, empty) is selected


def test_corrupt_ledger_is_not_silently_reclassified_as_no_best(tmp_path):
    workspace = Directory(base=tmp_path)
    reports = _results(workspace, ["agent0"])
    (tmp_path / "agent0" / ".ledger.json").write_text('{"schema_version":', encoding="utf-8")
    with pytest.raises(ValueError):
        epoch.collect_epoch_result("identity", reports, workspace)
