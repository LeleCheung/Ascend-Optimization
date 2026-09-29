"""Detached process launcher and workflow worker used by ``kg run``."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from kernelgen.cli.models import (
    ProcessState,
    RunMode,
    RunProcessRecord,
    RunRequest,
    utc_now,
)
from kernelgen.cli.state import (
    active_process,
    load_process,
    process_start_identity,
    register_workspace,
    request_path,
    runner_log_path,
    save_process,
    save_request,
    workspace_submission_lock,
)
from kernelgen.framework.cancellation import cooperative_sigint, reconcile_server_operations
from kernelgen.data.ledger import Ledger
from kernelgen.framework.run_control import RunCancelled, RunState, WorkspaceRunControl
from kernelgen.framework.worker_pool import (
    LeaseRecord, WorkerLeasePool, get_max_workers, normalize_worker_pool,
)


def new_request(
    *,
    mode: str,
    definition: str,
    workspace: Path,
    workflow_args: list[str],
    n_parallel: int,
    target_hardware: str | None,
    eval_server: str,
    batch_workspace: Path | None = None,
) -> RunRequest:
    normalized_mode = RunMode(mode)
    weight = 1 if normalized_mode == RunMode.SIMPLE_OPT else n_parallel
    worker_pool = normalize_worker_pool(eval_server)
    maximum = get_max_workers(worker_pool)
    if weight > maximum:
        raise ValueError(
            f"run requests {weight} workers but run.max-workers is {maximum}"
        )
    return RunRequest(
        run_id=uuid.uuid4().hex,
        mode=normalized_mode,
        definition=definition,
        target_hardware=target_hardware,
        eval_server=eval_server,
        worker_pool=worker_pool,
        workspace=workspace.expanduser().resolve(),
        batch_workspace=(
            batch_workspace.expanduser().resolve() if batch_workspace else None
        ),
        worker_weight=weight,
        workflow_args=workflow_args,
    )


def _without_option(arguments: list[str], option: str, *, takes_value: bool) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(arguments):
        item = arguments[index]
        if item == option:
            index += 2 if takes_value else 1
            continue
        if takes_value and item.startswith(f"{option}="):
            index += 1
            continue
        result.append(item)
        index += 1
    return result


def resumed_workflow_args(request: RunRequest) -> list[str]:
    arguments = list(request.workflow_args)
    arguments = _without_option(arguments, "--clean", takes_value=False)
    if request.mode != RunMode.KERNELGEN:
        return arguments
    for option in ("--start-mode", "--start-epoch", "--seed-code-path", "--finalize-epoch"):
        arguments = _without_option(arguments, option, takes_value=True)
    control = WorkspaceRunControl(request.workspace, source="cli:resume")
    epoch = control.progress().current_epoch or 1
    return [*arguments, "--start-mode", "resume", "--start-epoch", str(epoch)]


def prepare_resume(request: RunRequest) -> int:
    """Clear only resumable cancellation state and return restored ledger count."""
    control = WorkspaceRunControl(request.workspace, source="cli:resume")
    if reconcile_server_operations(control):
        raise RuntimeError("resume requires all previously registered KGS operations to be terminal")
    cancellation = control.cancellation_state()
    restored = 0
    for path in request.workspace.rglob(".ledger.json"):
        ledger = Ledger(path.parent)
        stopped = ledger.stopped_round()
        if (
            stopped is not None
            and stopped.next_verdict is not None
            and stopped.next_verdict.code == "user_cancelled"
        ):
            ledger.clear_supervisor_stop(code="user_cancelled")
            restored += 1
    if cancellation.requested:
        control.clear_cancellation(expected_generation=cancellation.generation)
    control.update_progress(
        state=RunState.PENDING,
        stage="RESUMING",
        stop_reason="",
        message="Explicit resume submitted",
    )
    control.record_event(
        "RUN_RESUME_SUBMITTED",
        stage="RESUMING",
        data={"restored_ledgers": restored},
    )
    return restored


def _invoke_workflow(request: RunRequest, *, resume: bool) -> int:
    if request.schema_version == "2.0":
        from kernelgen.cli.optimization import run_operator_optimization
        if request.workflow_input is None or request.workflow_args:
            raise ValueError("v2 run requires workflow_input and no launcher argv")
        return run_operator_optimization(request.workflow_input, workspace=request.workspace,
                                    runtime=request.runtime, model=request.model,
                                    base_url=request.base_url, resume=resume)
    arguments = resumed_workflow_args(request) if resume else request.workflow_args
    try:
        if request.mode == RunMode.SIMPLE_OPT:
            from kernelgen.cli.legacy_simple_opt import legacy_main as main
            main(arguments)
        elif request.mode == RunMode.KERNELGEN:
            from kernelgen.examples.kernel_gen.run_example import legacy_main as main
            main(arguments)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def execute_request(request: RunRequest, *, resume: bool = False) -> int:
    """Own one run process from queue registration through lease release."""
    workspace = request.workspace
    workspace.mkdir(parents=True, exist_ok=True)
    control = WorkspaceRunControl(workspace, source="cli:worker")
    pid = os.getpid()
    process_start = process_start_identity(pid)
    if process_start is None:  # pragma: no cover - current process always exists
        raise RuntimeError("cannot determine worker process identity")
    record = RunProcessRecord(
        run_id=request.run_id,
        pid=pid,
        process_start=process_start,
        state=ProcessState.QUEUED,
        workspace=workspace,
        log_path=runner_log_path(workspace),
        worker_weight=request.worker_weight,
        submitted_at=request.submitted_at,
        message="Waiting for local Coder capacity",
    )
    save_process(record)
    control.update_progress(
        state=RunState.QUEUED,
        stage="WAITING_FOR_WORKER",
        mode=request.mode.value,
        message=record.message,
    )
    control.record_event(
        "RUN_QUEUED",
        stage="WAITING_FOR_WORKER",
        data={"worker_weight": request.worker_weight},
    )
    lease = LeaseRecord(
        run_id=request.run_id,
        pid=pid,
        process_start=process_start,
        workspace=workspace,
        worker_pool=request.worker_pool,
        weight=request.worker_weight,
    )
    pool = WorkerLeasePool(request.worker_pool)
    acquired = False
    exit_code = 1
    try:
        with cooperative_sigint(control):
            if request.schema_version == "1.0":
                pool.acquire(lease, cancelled=control.is_cancellation_requested)
                acquired = True
            record.state = ProcessState.RUNNING
            record.started_at = utc_now()
            record.message = "Workflow running"
            save_process(record)
            control.update_progress(
                state=RunState.RUNNING,
                stage="STARTING",
                message="Worker capacity acquired" if acquired else "Starting Catalog workflow",
            )
            control.record_event(
                "RUN_STARTED",
                stage="STARTING",
                data={"worker_weight": request.worker_weight},
            )
            exit_code = _invoke_workflow(request, resume=resume)
        current = control.progress()
        if control.is_cancellation_requested():
            control.acknowledge_cancellation(stage="CANCELLED")
            exit_code = 130
        elif current.stage.startswith("WAITING_"):
            pass
        elif current.state not in {
            RunState.SUCCEEDED,
            RunState.FAILED,
            RunState.INFRASTRUCTURE_ERROR,
        }:
            control.update_progress(
                state=RunState.SUCCEEDED if exit_code == 0 else RunState.FAILED,
                stage="COMPLETED",
                message=f"Workflow exited with code {exit_code}",
            )
    except InterruptedError:
        control.acknowledge_cancellation(stage="CANCELLED_WHILE_QUEUED")
        exit_code = 130
    except RunCancelled:
        control.acknowledge_cancellation(stage="CANCELLED")
        exit_code = 130
    except BaseException as exc:  # noqa: BLE001
        control.update_progress(
            state=RunState.INFRASTRUCTURE_ERROR,
            stage="FAILED",
            message=f"{type(exc).__name__}: {exc}",
        )
        control.record_event(
            "RUN_PROCESS_FAILED",
            message=f"{type(exc).__name__}: {exc}",
            stage="FAILED",
            level="ERROR",
        )
        print(f"kg worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 2
    finally:
        if acquired:
            pool.release(request.run_id, pid, process_start)
        control.record_event(
            "RUN_PROCESS_EXITED",
            message=f"Worker process exited with code {exit_code}",
            stage="PROCESS_EXIT",
            level="INFO" if exit_code == 0 else "WARNING",
            data={"exit_code": exit_code},
        )
        record.state = ProcessState.EXITED
        record.finished_at = utc_now()
        record.exit_code = exit_code
        record.message = f"Process exited with code {exit_code}"
        save_process(record)
    return exit_code


def submit_request(
    request: RunRequest,
    *,
    foreground: bool,
    resume: bool = False,
) -> RunProcessRecord:
    """Persist and launch a run, rejecting a second owner of the workspace."""
    request.workspace.mkdir(parents=True, exist_ok=True)
    with workspace_submission_lock(request.workspace):
        active = active_process(request.workspace)
        if active is not None:
            raise RuntimeError(
                f"workspace already has active process {active.pid}: {request.workspace}"
            )
        if not resume and request_path(request.workspace).exists():
            raise RuntimeError(
                "workspace already contains a kg run; use kg resume or choose a new workspace"
            )
        if not resume and request.schema_version == "2.0" and (
            any((request.workspace / name).exists() for name in (
                ".ledger.json", "optimize_definition_output.json",
                "kernelgen_output.json", "batch_simple_opt_definition_output.json",
            )) or any(request.workspace.glob("*R/agent*/.ledger.json"))
        ):
            raise RuntimeError(
                "legacy Python workspace: use its legacy resume entry or choose a new workspace"
            )
        if resume:
            prepare_resume(request)
        save_request(request)
        register_workspace(request.workspace)
        if foreground:
            execute_request(request, resume=resume)
            record = load_process(request.workspace)
            if record is None:  # pragma: no cover
                raise RuntimeError("foreground worker did not write process metadata")
            return record

        log_path = runner_log_path(request.workspace)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with Path(os.devnull).open("rb") as stdin_handle, log_path.open(
            "ab", buffering=0
        ) as log_handle:
            command = [
                sys.executable,
                "-u",
                "-m",
                "kernelgen.cli.worker",
                "--workspace",
                str(request.workspace),
            ]
            if resume:
                command.append("--resume")
            process = subprocess.Popen(
                command,
                stdin=stdin_handle,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            record = load_process(request.workspace)
            if record is not None and record.pid == process.pid:
                return record
            if process.poll() is not None:
                break
            time.sleep(0.05)
        raise RuntimeError(
            f"background worker failed to register; inspect {log_path}"
        )
