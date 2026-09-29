"""Host-only tests for the compact ``kg history`` interface."""

from __future__ import annotations

import json

import pytest

import kernelgen.cli.main as cli_main
from kernelgen.cli.batch import save_batch_request
from kernelgen.cli.history import load_run_history
from kernelgen.cli.models import BatchChildRecord, BatchRequestRecord
from kernelgen.cli.runner import new_request
from kernelgen.cli.state import save_request, set_max_workers
from kernelgen.data.ledger import Ledger
from kernelgen.data.stop_policy import StopConfig
from kernelgen.tests.helpers import experiment_plan, round_conclusion
from kernelgen.framework.run_control import LEDGER_PROGRESS_FIELDS, RunState, WorkspaceRunControl


def _new_managed_run(workspace, *, mode="simple_opt"):
    request = new_request(
        mode=mode,
        definition="demo",
        workspace=workspace,
        workflow_args=[],
        n_parallel=2 if mode == "kernelgen" else 1,
        target_hardware="H800",
        eval_server="http://cuda-kgs:8000",
    )
    workspace.mkdir(parents=True)
    save_request(request)
    return request


def _evaluation(geo_mean, *, is_hack=False):
    return {
        "api_version": "v6.2",
        "status": "PASSED",
        "is_hack": is_hack,
        "hack_reason": "forbidden fallback" if is_hack else "",
        "geo_mean": geo_mean,
        "min_speedup": geo_mean - 0.1,
        "num_workloads": 1,
        "num_passed": 1,
        "requested_hardware": "H800",
        "server_backend": "cuda",
        "per_workload": [],
    }


def test_status_and_history_use_same_ledger_despite_stale_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    workspace = tmp_path / "run"
    _new_managed_run(workspace)
    control = WorkspaceRunControl(workspace)
    control.update_progress(state=RunState.RUNNING, stage="CODING", progress_kind="rounds")
    # Simulate a workspace written by the previous CLI revision.
    raw = json.loads(control.progress_path.read_text())
    raw.update(best_round=1, best_geo_mean=0.8, current_round=1)
    control.progress_path.write_text(json.dumps(raw))
    _record_round(Ledger(workspace), 1, 1.2)
    status = cli_main._status(workspace)
    history = load_run_history(workspace)
    assert status["progress"]["progress"]["best_geo_mean"] == history["best_geo_mean"] == 1.2
    assert status["progress"]["progress"]["best_round"] == history["best_round"] == 1
    assert status["progress"]["progress"]["completed_rounds"] == history["round_count"] == 1
    control.update_progress(stage="STOPPING")
    persisted = json.loads(control.progress_path.read_text())
    assert not LEDGER_PROGRESS_FIELDS.intersection(persisted)
    assert "cancel_requested" not in persisted


def test_dead_code_review_waiting_run_is_generated(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    workspace = tmp_path / "run"
    _new_managed_run(workspace)
    control = WorkspaceRunControl(workspace)
    control.update_progress(
        state=RunState.PENDING,
        stage="WAITING_CODE_REVIEW",
        message="Review requires changes or missing evidence",
    )

    status = cli_main._status(workspace)

    assert status["state"] == "GENERATED"
    assert status["process_alive"] is False
    assert status["progress"]["stage"] == "WAITING_CODE_REVIEW"
    assert status["progress"]["message"] == "Review requires changes or missing evidence"


def test_batch_with_generated_child_is_generated_not_queued(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    batch = tmp_path / "batch"
    workspace = batch / "run"
    request = _new_managed_run(workspace)
    WorkspaceRunControl(workspace).update_progress(
        state=RunState.PENDING,
        stage="WAITING_CODE_REVIEW",
    )
    save_batch_request(
        BatchRequestRecord(
            batch_id="batch",
            workspace=batch,
            source_path=tmp_path / "batch.yaml",
            children=[
                BatchChildRecord(
                    definition=request.definition,
                    mode=request.mode,
                    workspace=workspace,
                    run_id=request.run_id,
                )
            ],
        )
    )

    status = cli_main._status(batch)

    assert status["state"] == "GENERATED"
    assert status["counts"] == {"GENERATED": 1}
    assert status["runs"][0]["progress"]["stage"] == "WAITING_CODE_REVIEW"


def test_dead_cancel_requested_run_is_interrupted(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    workspace = tmp_path / "run"
    _new_managed_run(workspace)
    control = WorkspaceRunControl(workspace)
    cancellation = control.request_cancel("cancel before process disappeared")
    status = cli_main._status(workspace)
    assert status["state"] == "INTERRUPTED"
    assert status["progress"]["cancel_requested"] is True
    assert control.cancellation_state() == cancellation


def _record_round(ledger, round_num, geo_mean, *, is_hack=False, finalize=True):
    ledger.record_eval(
        _evaluation(geo_mean, is_hack=is_hack),
        f"code_{round_num}",
        experiment_plan(round_num),
        definition_name="demo",
        target_hardware="H800",
    )
    if finalize:
        ledger.finalize_round(
            round_num,
            round_conclusion(round_num),
            StopConfig(early_stop_rounds=0, max_round=100),
        )


def test_history_reports_running_best_and_excludes_hack(tmp_path):
    workspace = tmp_path / "run"
    request = _new_managed_run(workspace)
    ledger = Ledger(workspace)
    _record_round(ledger, 1, 0.9)
    _record_round(ledger, 2, 0.8)
    _record_round(ledger, 3, 1.2)
    _record_round(ledger, 4, 10.0, is_hack=True, finalize=False)
    _record_round(
        Ledger(workspace / "knowledge-archive"),
        1,
        20.0,
        finalize=False,
    )

    history = load_run_history(workspace)

    assert history["run_id"] == request.run_id
    assert history["kind"] == "run_history"
    assert history["best_scope"] == "."
    assert history["best_round"] == 3
    assert history["best_geo_mean"] == 1.2
    assert history["series_count"] == 1
    assert history["round_count"] == 4
    rounds = history["series"][0]["rounds"]
    assert [item["best_geo_mean_so_far"] for item in rounds] == [
        0.9,
        0.9,
        1.2,
        1.2,
    ]
    assert [item["is_new_best"] for item in rounds] == [True, False, True, False]
    assert [item["is_final_best"] for item in rounds] == [False, False, True, False]
    assert rounds[-1]["is_hack"] is True


def test_history_keeps_kernelgen_agent_series_separate(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    set_max_workers("http://cuda-kgs:8000", 2)
    workspace = tmp_path / "kernelgen-run"
    _new_managed_run(workspace, mode="kernelgen")
    first = Ledger(workspace / "1R" / "agent0")
    second = Ledger(workspace / "1R" / "agent1")
    _record_round(first, 1, 1.1, finalize=False)
    _record_round(second, 1, 1.3, finalize=False)

    history = load_run_history(workspace)

    assert [item["scope"] for item in history["series"]] == [
        "1R/agent0",
        "1R/agent1",
    ]
    assert history["best_scope"] == "1R/agent1"
    assert history["best_round"] == 1
    assert history["best_geo_mean"] == 1.3
    assert history["round_count"] == 2


def test_history_allows_managed_run_before_first_evaluation(tmp_path):
    workspace = tmp_path / "queued"
    _new_managed_run(workspace)

    history = load_run_history(workspace)

    assert history["best_scope"] is None
    assert history["best_round"] is None
    assert history["best_geo_mean"] is None
    assert history["series"] == []
    assert history["round_count"] == 0


def test_history_cli_supports_json_and_text(tmp_path, capsys):
    workspace = tmp_path / "run"
    _new_managed_run(workspace)
    _record_round(Ledger(workspace), 1, 1.125, finalize=False)

    assert cli_main.main(["history", str(workspace), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["best_geo_mean"] == 1.125

    assert cli_main.main(["history", str(workspace)]) == 0
    output = capsys.readouterr().out
    assert "best_geo_mean: 1.125" in output
    assert ". R1 PASSED geo_mean=1.125 best_so_far=1.125" in output


def test_history_rejects_batch_workspace(tmp_path):
    child = _new_managed_run(tmp_path / "batch" / "definitions" / "demo")
    batch_workspace = tmp_path / "batch"
    save_batch_request(
        BatchRequestRecord(
            batch_id="batch-1",
            workspace=batch_workspace,
            source_path=tmp_path / "batch.yaml",
            children=[
                BatchChildRecord(
                    definition=child.definition,
                    mode=child.mode,
                    workspace=child.workspace,
                    run_id=child.run_id,
                )
            ],
        )
    )

    with pytest.raises(ValueError, match="requires a single-run workspace"):
        load_run_history(batch_workspace)
