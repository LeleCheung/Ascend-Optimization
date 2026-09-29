"""Host-only tests for the local ``kg`` process supervisor."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from kernelgen.cli.main import (
    _explicit_run_options,
    _command_cancel,
    _status,
    build_parser,
)
from kernelgen.cli.models import (
    LeaseRecord,
    ProcessState,
    RunProcessRecord,
    utc_now,
)
from kernelgen.cli.runner import execute_request, new_request
from kernelgen.cli.state import (
    WorkerLeasePool,
    cli_home,
    normalize_worker_pool,
    process_start_identity,
    runner_log_path,
    save_process,
    set_max_workers,
)
from kernelgen.framework.run_control import RunState, WorkspaceRunControl
from kernelgen.framework.run_options import optimization_argv, resolve_run_options


def _lease(run_id, workspace, endpoint, weight=1):
    pid = os.getpid()
    identity = process_start_identity(pid)
    assert identity is not None
    return LeaseRecord(
        run_id=run_id,
        pid=pid,
        process_start=identity,
        workspace=workspace,
        worker_pool=endpoint,
        weight=weight,
    )


def test_cli_home_defaults_to_current_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("KERNELGEN_CLI_HOME", raising=False)
    monkeypatch.chdir(tmp_path)

    assert cli_home() == tmp_path / ".kernelgen"


def test_cli_home_allows_explicit_override(tmp_path, monkeypatch):
    override = tmp_path / "shared-state"
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(override))

    assert cli_home() == override


def test_worker_pool_normalizes_kgs_endpoint_and_rejects_credentials():
    assert normalize_worker_pool("http://localhost:8000/") == (
        "http://127.0.0.1:8000"
    )
    assert normalize_worker_pool("HTTPS://KGS.EXAMPLE") == (
        "https://kgs.example:443"
    )
    with pytest.raises(ValueError, match="credentials"):
        normalize_worker_pool("http://user:secret@kgs.example:8000")


def test_worker_limits_are_isolated_by_kgs_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    set_max_workers("http://ascend-kgs:8000", 1)
    set_max_workers("http://cuda-kgs:8000", 2)
    ascend = WorkerLeasePool("http://ascend-kgs:8000")
    cuda = WorkerLeasePool("http://cuda-kgs:8000")
    first = _lease("ascend", tmp_path / "ascend", "http://ascend-kgs:8000")
    second = _lease("cuda", tmp_path / "cuda", "http://cuda-kgs:8000", weight=2)

    ascend.acquire(first, cancelled=lambda: False)
    cuda.acquire(second, cancelled=lambda: False)
    try:
        assert ascend.snapshot()["available_workers"] == 0
        assert cuda.snapshot()["available_workers"] == 0
        assert ascend.snapshot()["leases"][0]["run_id"] == "ascend"
        assert cuda.snapshot()["leases"][0]["run_id"] == "cuda"
    finally:
        ascend.release(first.run_id, first.pid, first.process_start)
        cuda.release(second.run_id, second.pid, second.process_start)


def test_queued_run_can_be_cancelled_without_invoking_workflow(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    set_max_workers("http://ascend-kgs:8000", 1)
    pool = WorkerLeasePool("http://ascend-kgs:8000")
    holder = _lease("holder", tmp_path / "holder", "http://ascend-kgs:8000")
    pool.acquire(holder, cancelled=lambda: False)
    workspace = tmp_path / "queued"
    request = new_request(
        mode="simple_opt",
        definition="gcd",
        workspace=workspace,
        workflow_args=["--definition-name", "gcd", "--workspace", str(workspace)],
        n_parallel=1,
        target_hardware="Ascend910B",
        eval_server="http://ascend-kgs:8000",
    )
    result = []
    worker = threading.Thread(target=lambda: result.append(execute_request(request)))
    worker.start()
    control = WorkspaceRunControl(workspace)
    deadline = time.monotonic() + 3
    while control.progress().state != RunState.QUEUED and time.monotonic() < deadline:
        time.sleep(0.02)
    control.request_cancel("cancel queued test")
    worker.join(timeout=3)
    pool.release(holder.run_id, holder.pid, holder.process_start)

    assert worker.is_alive() is False
    assert result == [130]
    assert control.progress().state == RunState.CANCELLED
    assert "RUN_STARTED" not in {event.event_type for event in control.read_events()}


def test_cli_builds_kernelgen_request_with_chip_weight(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    set_max_workers("http://ascend-kgs:8000", 4)
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--mode",
            "kernelgen",
            "--definition",
            "gcd",
            "--target-hardware",
            "Ascend910B",
            "--eval-server",
            "http://ascend-kgs:8000",
            "--n-parallel",
            "3",
            "--workspace",
            str(tmp_path / "run"),
        ]
    )
    assert args.n_parallel == 3
    request = new_request(
        mode=args.mode,
        definition=args.definition,
        workspace=args.workspace,
        workflow_args=[],
        n_parallel=args.n_parallel,
        target_hardware=args.target_hardware,
        eval_server=args.eval_server,
    )
    assert request.worker_weight == 3
    assert request.worker_pool == "http://ascend-kgs:8000"


def test_cli_builds_simple_opt_measurement_arguments(tmp_path):
    args = build_parser().parse_args(
        [
            "run",
            "--mode",
            "simple_opt",
            "--definition",
            "gcd",
            "--workspace",
            str(tmp_path / "run"),
            "--warmup-ms",
            "25",
            "--benchmark-ms",
            "40",
            "--num-trials",
            "3",
        ]
    )

    workflow_args = optimization_argv(resolve_run_options(_explicit_run_options(args)))

    assert workflow_args[workflow_args.index("--warmup-ms") + 1] == "25"
    assert workflow_args[workflow_args.index("--benchmark-ms") + 1] == "40"
    assert workflow_args[workflow_args.index("--num-trials") + 1] == "3"


def test_legacy_simple_opt_launcher_forwards_measurement_arguments(
    tmp_path, monkeypatch, capsys
):
    from kernelgen.cli import legacy_simple_opt as run_example

    captured = {}

    class FakeWorkflow:
        def __init__(self, **kwargs):
            pass

        def run(self, value):
            captured.update(value)
            return SimpleNamespace(
                definition_name=value["definition_name"],
                op_type="gcd",
                status="PASSED",
                rounds=1,
                best_geo_mean=1.0,
                workspace=str(tmp_path / "run"),
                summary="",
            )

    monkeypatch.setattr(run_example, "SimpleOptWorkflow", FakeWorkflow)
    monkeypatch.setattr(
        run_example,
        "copy_claude_directory",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        run_example,
        "copy_mcp_configuration",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_example.legacy_main(
            [
                "--definition-name",
                "gcd",
                "--workspace",
                str(tmp_path / "run"),
                "--runtime",
                "codex",
                "--warmup-ms",
                "25",
                "--benchmark-ms",
                "40",
                "--num-trials",
                "3",
            ]
        )

    assert exc_info.value.code == 0
    assert captured["warmup_ms"] == 25
    assert captured["benchmark_ms"] == 40
    assert captured["num_trials"] == 3
    capsys.readouterr()


def test_legacy_simple_opt_launcher_binds_mcp_to_active_python(
    tmp_path, monkeypatch, capsys
):
    from kernelgen.cli import legacy_simple_opt as run_example

    class FakeWorkflow:
        def __init__(self, **kwargs):
            pass

        def run(self, value):
            return SimpleNamespace(
                definition_name=value["definition_name"],
                op_type="gcd",
                status="PASSED",
                rounds=1,
                best_geo_mean=1.0,
                workspace=str(tmp_path / "run"),
                summary="",
            )

    monkeypatch.setattr(run_example, "SimpleOptWorkflow", FakeWorkflow)
    workspace = tmp_path / "run"

    with pytest.raises(SystemExit) as exc_info:
        run_example.legacy_main(
            [
                "--definition-name",
                "gcd",
                "--workspace",
                str(workspace),
                "--runtime",
                "codex",
            ]
        )

    assert exc_info.value.code == 0
    neutral = json.loads(
        (workspace / ".kernelgen" / "mcp.json").read_text(encoding="utf-8")
    )
    claude = json.loads((workspace / ".mcp.json").read_text(encoding="utf-8"))
    assert neutral["servers"]["kernelgen"]["command"][0] == sys.executable
    assert claude["mcpServers"]["kernelgen"]["command"] == sys.executable
    capsys.readouterr()


def test_status_reports_interrupted_when_process_disappeared(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    set_max_workers("http://cuda-kgs:8000", 1)
    workspace = tmp_path / "run"
    request = new_request(
        mode="simple_opt",
        definition="demo",
        workspace=workspace,
        workflow_args=[],
        n_parallel=1,
        target_hardware="A100",
        eval_server="http://cuda-kgs:8000",
    )
    from kernelgen.cli.state import register_workspace, save_request

    workspace.mkdir()
    save_request(request)
    register_workspace(workspace)
    control = WorkspaceRunControl(workspace)
    control.update_progress(state=RunState.RUNNING, stage="CODING")

    assert _status(workspace)["state"] == "INTERRUPTED"


def test_status_prioritizes_cancel_request_for_queued_process(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELGEN_CLI_HOME", str(tmp_path / "state"))
    set_max_workers("http://cuda-kgs:8000", 1)
    workspace = tmp_path / "run"
    request = new_request(
        mode="simple_opt",
        definition="demo",
        workspace=workspace,
        workflow_args=[],
        n_parallel=1,
        target_hardware="A100",
        eval_server="http://cuda-kgs:8000",
    )
    from kernelgen.cli.state import register_workspace, save_request

    workspace.mkdir()
    save_request(request)
    register_workspace(workspace)
    identity = process_start_identity(os.getpid())
    assert identity is not None
    save_process(
        RunProcessRecord(
            run_id=request.run_id,
            pid=os.getpid(),
            process_start=identity,
            state=ProcessState.QUEUED,
            workspace=workspace,
            log_path=runner_log_path(workspace),
            worker_weight=1,
            submitted_at=utc_now(),
        )
    )
    WorkspaceRunControl(workspace).request_cancel("stop queue")

    assert _status(workspace)["state"] == "CANCEL_REQUESTED"


def test_cli_cancel_forwards_active_server_operations(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "run"
    workspace.mkdir()
    identity = process_start_identity(os.getpid())
    assert identity is not None
    save_process(
        RunProcessRecord(
            run_id="run-1",
            pid=os.getpid(),
            process_start=identity,
            state=ProcessState.RUNNING,
            workspace=workspace,
            log_path=runner_log_path(workspace),
            worker_weight=1,
            submitted_at=utc_now(),
        )
    )
    control = WorkspaceRunControl(workspace)
    control.register_server_operation(
        "evaluate",
        "http://127.0.0.1:18000",
        operation_id="operation-1",
    )
    calls = []

    def cancel_operation(operation_id, server_url, timeout):
        calls.append((operation_id, server_url, timeout))
        return {"operation_id": operation_id, "state": "CANCEL_REQUESTED"}

    monkeypatch.setattr(
        "kernelgen_client.http.cancel_operation",
        cancel_operation,
    )

    exit_code = _command_cancel(
        SimpleNamespace(workspace=str(workspace), reason="stop evaluation")
    )

    assert exit_code == 0
    assert calls == [("operation-1", "http://127.0.0.1:18000", 10)]
    assert control.cancellation_state().requested is True
    assert "server_operations_cancel_requested: 1" in capsys.readouterr().out
