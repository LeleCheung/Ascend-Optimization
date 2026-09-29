"""Host-only tests for YAML Batch submission and aggregate control."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import kernelgen.cli.main as cli_main
from kernelgen.cli.batch import load_batch_file, save_batch_request
from kernelgen.cli.models import (
    BatchChildRecord,
    BatchRequestRecord,
    ProcessState,
    RunProcessRecord,
    utc_now,
)
from kernelgen.cli.runner import new_request
from kernelgen.cli.state import (
    process_start_identity,
    runner_log_path,
    save_process,
    save_request,
    set_max_workers,
)
from kernelgen.framework.run_control import RunState, WorkspaceRunControl


def _write_batch(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_batch_yaml_resolves_relative_paths_and_rejects_duplicates(tmp_path):
    reference = tmp_path / "refs" / "square.py"
    reference.parent.mkdir()
    reference.write_text("def run(x): return x * x\n", encoding="utf-8")
    source = _write_batch(
        tmp_path / "batch.yaml",
        """
version: 1
workspace: runs/demo
defaults:
  mode: simple_opt
  runtime: codex
operators:
  - definition: square
    reference_code_path: refs/square.py
  - definition: neg
""",
    )

    loaded_source, defaults, operators, workspace = load_batch_file(source)

    assert loaded_source == source
    assert defaults == {"mode": "simple_opt", "runtime": "codex"}
    assert operators[0]["reference_code_path"] == reference
    assert workspace == tmp_path / "runs" / "demo"

    duplicate = _write_batch(
        tmp_path / "duplicate.yaml",
        """
version: 1
operators:
  - definition: square
  - definition: square
""",
    )
    with pytest.raises(ValueError, match="duplicate Batch definitions"):
        load_batch_file(duplicate)


def test_one_batch_command_builds_independent_requests_with_item_overrides(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    endpoint = "http://cuda-kgs:8000"
    set_max_workers(endpoint, 4)
    reference = tmp_path / "square.py"
    reference.write_text("def run(x): return x * x\n", encoding="utf-8")
    seed = tmp_path / "flash.py"
    seed.write_text("def run(x): return x\n", encoding="utf-8")
    source = _write_batch(
        tmp_path / "batch.yaml",
        f"""
version: 1
workspace: batch-output
defaults:
  mode: simple_opt
  runtime: codex
  eval_server: {endpoint}
  target_hardware: H800
  min_rounds: 1
  max_round: 1
operators:
  - definition: square
    warmup_ms: 25
    benchmark_ms: 40
    num_trials: 3
    reference_code_path: {reference.name}
  - definition: flash
    mode: kernelgen
    n_parallel: 2
    seed_code_path: {seed.name}
""",
    )
    captured = []

    def fake_submit(request, *, foreground):
        captured.append(request)
        return SimpleNamespace(pid=1000 + len(captured), state=ProcessState.QUEUED)

    monkeypatch.setattr(cli_main, "submit_request", fake_submit)

    assert cli_main.main(["run", "--batch-file", str(source)]) == 0

    assert [item.definition for item in captured] == ["square", "flash"]
    assert all(item.batch_workspace == tmp_path / "batch-output" for item in captured)
    assert captured[0].workspace == tmp_path / "batch-output/definitions/square"
    assert captured[0].workflow_args == []
    first = captured[0].workflow_input["optimization"]
    assert first["reference_code_path"] == str(reference)
    assert (first["warmup_ms"], first["benchmark_ms"], first["num_trials"]) == (25, 40, 3)
    assert captured[1].worker_weight == 2
    assert captured[1].workflow_input["optimization"]["seed_code_path"] == str(seed)
    output = capsys.readouterr().out
    assert "tasks: 2" in output
    assert "status: SUBMITTED" in output


def _persist_child(request, *, state: RunState, process_state: ProcessState) -> None:
    request.workspace.mkdir(parents=True)
    save_request(request)
    identity = process_start_identity(os.getpid())
    assert identity is not None
    save_process(
        RunProcessRecord(
            run_id=request.run_id,
            pid=os.getpid(),
            process_start=identity,
            state=process_state,
            workspace=request.workspace,
            log_path=runner_log_path(request.workspace),
            worker_weight=request.worker_weight,
            submitted_at=utc_now(),
            exit_code=0 if process_state == ProcessState.EXITED else None,
        )
    )
    WorkspaceRunControl(request.workspace).update_progress(
        state=state,
        stage="COMPLETED" if state == RunState.SUCCEEDED else "MODEL_INVOCATION",
    )


def test_batch_status_aggregates_children_and_cancel_targets_active_runs(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    endpoint = "http://cuda-kgs:8000"
    set_max_workers(endpoint, 2)
    batch_workspace = tmp_path / "batch"
    completed = new_request(
        mode="simple_opt",
        definition="square",
        workspace=batch_workspace / "definitions/square",
        workflow_args=[],
        n_parallel=1,
        target_hardware="H800",
        eval_server=endpoint,
        batch_workspace=batch_workspace,
    )
    active = new_request(
        mode="simple_opt",
        definition="neg",
        workspace=batch_workspace / "definitions/neg",
        workflow_args=[],
        n_parallel=1,
        target_hardware="H800",
        eval_server=endpoint,
        batch_workspace=batch_workspace,
    )
    _persist_child(
        completed,
        state=RunState.SUCCEEDED,
        process_state=ProcessState.EXITED,
    )
    _persist_child(active, state=RunState.RUNNING, process_state=ProcessState.RUNNING)
    save_batch_request(
        BatchRequestRecord(
            batch_id="batch-1",
            workspace=batch_workspace,
            source_path=tmp_path / "batch.yaml",
            children=[
                BatchChildRecord(
                    definition=item.definition,
                    mode=item.mode,
                    workspace=item.workspace,
                    run_id=item.run_id,
                )
                for item in (completed, active)
            ],
        )
    )

    status = cli_main._status(batch_workspace)
    assert status["kind"] == "batch"
    assert status["state"] == "RUNNING"
    assert status["counts"] == {"RUNNING": 1, "SUCCEEDED": 1}

    assert cli_main.main(
        ["cancel", str(batch_workspace), "--reason", "stop Batch"]
    ) == 0
    assert cli_main._status(batch_workspace)["state"] == "CANCEL_REQUESTED"
    output = capsys.readouterr().out
    assert "tasks_cancel_requested: 1" in output
    assert "tasks_already_terminal: 1" in output


def test_batch_yaml_rejects_unknown_fields_and_non_yaml_suffix(tmp_path):
    source = _write_batch(
        tmp_path / "batch.yaml",
        """
version: 1
defaults:
  mode: simple_opt
  auth_token: secret
operators:
  - definition: square
""",
    )
    with pytest.raises(ValueError, match="unsupported option 'auth_token'"):
        load_batch_file(source)

    json_source = tmp_path / "batch.json"
    json_source.write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="must use .yaml or .yml"):
        load_batch_file(json_source)
