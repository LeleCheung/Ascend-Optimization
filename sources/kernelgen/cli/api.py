"""Argparse-free, print-free core operations shared by the ``kg`` CLI and the HTTP service.

Both :mod:`kernelgen.cli.main` (the local CLI) and :mod:`kernelgen.service`
(the HTTP API) call into these functions so there is a single public contract
for submitting, inspecting, listing, and cancelling runs. Functions here take
plain values and return plain dicts/objects; they never read ``argparse``
namespaces and never print.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from pathlib import Path

from kernelgen.cli.batch import batch_request_path, load_batch_request
from kernelgen.cli.history import load_run_history, project_progress
from kernelgen.cli.models import ProcessState, RunRequest
from kernelgen.cli.runner import new_request, submit_request
from kernelgen.cli.state import (
    WorkerLeasePool,
    active_process,
    cli_home,
    indexed_workspaces,
    load_process,
    load_request,
    process_is_alive,
    read_json,
    request_path,
)
from kernelgen.framework.cancellation import (
    forward_server_cancellation,
    request_run_cancellation,
)
from kernelgen.framework.run_control import RunEvent, RunState, WorkspaceRunControl
from kernelgen.framework.progress_schema import progress_v2


TERMINAL_STATES = {
    RunState.CANCELLED,
    RunState.SUCCEEDED,
    RunState.FAILED,
    RunState.INFRASTRUCTURE_ERROR,
}

_ACTIVITY_MESSAGE_LIMIT = 2_000
_ACTIVITY_URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.IGNORECASE)
_ACTIVITY_SECRET_PATTERN = re.compile(
    r"\b(?:[A-Z0-9_-]*(?:TOKEN|API[_-]?KEY|SECRET|PASSWORD)[A-Z0-9_-]*"
    r"|AUTHORIZATION)\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)
_ACTIVITY_BEARER_PATTERN = re.compile(
    r"\bBearer\s+[A-Za-z0-9._~+/=-]+",
    re.IGNORECASE,
)
_ACTIVITY_EXCLUDED_EVENT_TYPES = {
    "RUNTIME_LOG",
    "MODEL_INVOCATION_FAILED",
    "EVALUATION_SNAPSHOT_FAILED",
    "SERVER_OPERATION_STARTED",
    "SERVER_OPERATION_FINISHED",
    "SERVER_OPERATION_RECONCILE_FAILED",
    "SERVER_OPERATION_CANCEL_FAILED",
    "CANCEL_CLEARED",
}


def _sanitize_activity_message(message: str) -> str:
    message = _ACTIVITY_BEARER_PATTERN.sub("Bearer [redacted]", message)
    message = _ACTIVITY_SECRET_PATTERN.sub("[credential redacted]", message)
    message = _ACTIVITY_URL_PATTERN.sub("[url redacted]", message)
    if len(message) > _ACTIVITY_MESSAGE_LIMIT:
        message = message[: _ACTIVITY_MESSAGE_LIMIT - 1] + "…"
    return message


def _project_activity_event(event: RunEvent) -> dict:
    return {
        "sequence": event.sequence,
        "recorded_at": event.recorded_at.isoformat(),
        "event_type": event.event_type,
        "scope": event.scope,
        "stage": event.stage,
        "level": event.level,
        "message": _sanitize_activity_message(event.message),
    }


def _activity_event_is_visible(event: RunEvent) -> bool:
    return (
        event.visibility == "USER"
        and event.level != "DEBUG"
        and event.event_type not in _ACTIVITY_EXCLUDED_EVENT_TYPES
    )


def validate_input_paths(values: dict, definition: str) -> None:
    for name in (
        "reference_code_path",
        "reference_code_prompt_path",
        "seed_code_path",
    ):
        path = values.get(name)
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"{definition} {name} does not exist: {path}")


def build_request(
    values: dict,
    *,
    definition: str,
    workspace: Path,
    batch_workspace: Path | None = None,
):
    """Build a :class:`RunRequest` from resolved option ``values``.

    CLI and HTTP submissions use the same normalized CatalogOptimize input.
    """
    validate_input_paths(values, definition)
    from kernelgen.framework.catalog_options import catalog_input
    inp = catalog_input(values, definition)
    request = new_request(
        mode=values["mode"], definition=definition, workspace=workspace, workflow_args=[],
        n_parallel=getattr(inp.optimization, "n_parallel", 1),
        target_hardware=inp.optimization.target_hardware,
        eval_server=inp.optimization.eval_server_url, batch_workspace=batch_workspace,
    )
    return RunRequest.model_validate({**request.model_dump(), "schema_version": "2.0", "workflow_input": inp.model_dump(mode="json"),
                                      "runtime": values["runtime"], "model": values.get("model"),
                                      "base_url": values.get("base_url")})


def submit_run(
    values: dict,
    *,
    definition: str,
    workspace: Path,
    foreground: bool = False,
):
    """Build and submit a single run. Returns ``(request, record)``."""
    request = build_request(values, definition=definition, workspace=workspace)
    record = submit_request(request, foreground=foreground)
    return request, record


def export_gems_definition(values: dict, *, workspace: str | Path) -> dict:
    """Export original pytest provenance and ABI without running tests or a model."""
    import subprocess

    from kernelgen.workflows.gems_adapter_definition import GemsAdapterDefinitionWorkflow

    root = Path(workspace).expanduser().resolve()
    try:
        output = GemsAdapterDefinitionWorkflow(cwd=root).run(values)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Gems Definition export failed: {exc}") from exc
    return {"workspace": str(root), **output.model_dump(mode="json")}


def extract_catalog(values: dict, *, workspace: str | Path, runtime="claude",
                    model=None, timeout=900) -> dict:
    """Run independent extraction in a new workspace, without an optimizer lease.

    No background RunRequest or resume contract is implied by this entry.
    """
    from kernelgen.framework import copy_claude_directory
    from kernelgen.framework.cancellation import cooperative_sigint
    from kernelgen.framework.runtime import (
        create_cli_runtime, resolve_cli_runtime_options, materialize_claude_runtime_config,
    )
    from kernelgen.workflows.catalog_extract import CatalogExtractInput, CatalogExtractWorkflow

    inp = CatalogExtractInput.model_validate(values)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if inp.flaggems_repo:
        repo = Path(inp.flaggems_repo).expanduser().resolve()
        if not repo.is_dir():
            raise FileNotFoundError(f"source checkout not found: {repo}")
        inp = inp.model_copy(update={"flaggems_repo": str(repo)})
    root = Path(workspace).expanduser().resolve()
    if inp.flaggems_repo and root.is_relative_to(repo):
        raise ValueError("extraction workspace must be outside the source checkout")
    options = resolve_cli_runtime_options(runtime, model=model)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError("extraction workspace already exists; preserve it and use a new workspace") from None
    control = WorkspaceRunControl(root, source="cli:extract", root_workspace=root)
    source = Path(__file__).resolve().parents[1]

    def make_runtime(path):
        path = Path(path).resolve()
        copy_claude_directory(source / ".claude", path / ".claude", include_skills=False)
        config = materialize_claude_runtime_config(path, options["model"]) if runtime == "claude" else None
        permissions = {"allowed_tools": "Read"} if runtime == "claude" else {"sandbox_mode": "read-only"}
        return create_cli_runtime(runtime, workspace=path, **options, **permissions,
                                  claude_config_dir=config, timeout=timeout,
                                  idle_timeout=max(120, timeout // 2), verbose=False)

    with cooperative_sigint(control):
        output = CatalogExtractWorkflow(cwd=str(root), runtime_factory=make_runtime).run(inp.model_dump())
    return {"operator": output.operator, "workspace": str(root),
            "catalog_path": str(output.catalog_path), "operator_dir": str(output.operator_dir),
            "case_list_path": output.case_list_path,
            "review_path": str(output.review_path) if output.review_path else None,
            "target_validation": "NOT_RUN"}


def run_status(workspace: str | Path) -> dict:
    path = Path(workspace).expanduser().resolve()
    if not request_path(path).is_file():
        raise FileNotFoundError(f"kg run not found: {path}")
    request = load_request(path)
    process = load_process(path)
    control = WorkspaceRunControl(path, source="cli:status")
    progress = control.progress()
    alive = (
        process is not None
        and process.state != ProcessState.EXITED
        and process_is_alive(process.pid, process.process_start)
    )
    if progress.state in TERMINAL_STATES:
        state = progress.state.value
    elif (
        progress.state == RunState.PENDING
        and progress.stage == "WAITING_CODE_REVIEW"
    ):
        state = "GENERATED"
    elif progress.state == RunState.PENDING and progress.stage.startswith("WAITING_"):
        state = RunState.PENDING.value
    elif not alive:
        state = "INTERRUPTED"
    elif progress.state == RunState.CANCEL_REQUESTED:
        state = RunState.CANCEL_REQUESTED.value
    elif alive and process is not None and process.state == ProcessState.QUEUED:
        state = RunState.QUEUED.value
    elif alive:
        state = progress.state.value
    else:
        state = "INTERRUPTED"
    return {
        "schema_version": "2.0",
        "kind": "run",
        "run_id": request.run_id,
        "mode": request.mode.value,
        "definition": request.definition,
        "target_hardware": request.target_hardware,
        "eval_server": request.eval_server,
        "worker_pool": request.worker_pool,
        "batch_workspace": (
            str(request.batch_workspace) if request.batch_workspace else None
        ),
        "workspace": str(path),
        "state": state,
        "process_alive": alive,
        "process": process.model_dump(mode="json") if process else None,
        "progress": progress_v2(project_progress(
            progress.model_dump(mode="json"), load_run_history(path)
        )),
    }


def batch_status(workspace: str | Path) -> dict:
    path = Path(workspace).expanduser().resolve()
    request = load_batch_request(path)
    runs = []
    for child in request.children:
        try:
            runs.append(run_status(child.workspace))
        except (FileNotFoundError, ValueError) as exc:
            runs.append(
                {
                    "schema_version": "2.0",
                    "kind": "run",
                    "run_id": child.run_id,
                    "mode": child.mode.value,
                    "definition": child.definition,
                    "workspace": str(child.workspace),
                    "state": "SUBMISSION_FAILED",
                    "process_alive": False,
                    "process": None,
                    "progress": None,
                    "error": str(exc),
                }
            )
    counts = Counter(item["state"] for item in runs)
    states = set(counts)
    terminal = {
        RunState.SUCCEEDED.value,
        RunState.CANCELLED.value,
        RunState.FAILED.value,
        RunState.INFRASTRUCTURE_ERROR.value,
        "GENERATED",
        "INTERRUPTED",
        "SUBMISSION_FAILED",
    }
    completed = {RunState.SUCCEEDED.value, "GENERATED"}
    if states <= completed:
        state = "GENERATED" if "GENERATED" in states else RunState.SUCCEEDED.value
    elif states <= terminal:
        state = (
            RunState.CANCELLED.value
            if states == {RunState.CANCELLED.value}
            else RunState.FAILED.value
        )
    elif RunState.CANCEL_REQUESTED.value in states:
        state = RunState.CANCEL_REQUESTED.value
    elif RunState.RUNNING.value in states:
        state = RunState.RUNNING.value
    elif RunState.PENDING.value in states:
        state = RunState.PENDING.value
    else:
        state = RunState.QUEUED.value
    modes = {item["mode"] for item in runs}
    return {
        "schema_version": "2.0",
        "kind": "batch",
        "batch_id": request.batch_id,
        "mode": next(iter(modes)) if len(modes) == 1 else "mixed",
        "definition": f"{len(runs)} operators",
        "source_path": str(request.source_path),
        "workspace": str(path),
        "state": state,
        "total_tasks": len(runs),
        "counts": dict(sorted(counts.items())),
        "runs": runs,
    }


def status(workspace: str | Path) -> dict:
    path = Path(workspace).expanduser().resolve()
    return batch_status(path) if batch_request_path(path).is_file() else run_status(path)


def history(workspace: str | Path) -> dict:
    """Return the per-round history for one non-batch run workspace."""
    path = Path(workspace).expanduser().resolve()
    if batch_request_path(path).is_file():
        raise ValueError("history is not defined for a batch workspace")
    if not request_path(path).is_file():
        raise FileNotFoundError(f"kg run not found: {path}")
    return load_run_history(path)


def run_activity(
    workspace: str | Path,
    *,
    after_sequence: int | None = None,
    limit: int = 50,
) -> dict:
    """Return a bounded, user-visible projection of one run's event stream."""
    path = Path(workspace).expanduser().resolve()
    if batch_request_path(path).is_file():
        raise ValueError("activity is not defined for a batch workspace")
    if not request_path(path).is_file():
        raise FileNotFoundError(f"kg run not found: {path}")
    if after_sequence is not None and after_sequence < 0:
        raise ValueError("after_sequence must be non-negative")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")

    control = WorkspaceRunControl(path, source="cli:activity")
    cursor = after_sequence or 0
    raw_events = control.read_events(after_sequence=cursor)

    if after_sequence is None:
        recent = deque(maxlen=limit)
        for event in raw_events:
            if _activity_event_is_visible(event):
                recent.append(_project_activity_event(event))
        events = list(recent)
        next_sequence = raw_events[-1].sequence if raw_events else 0
        has_more = False
    else:
        visible_events = [
            _project_activity_event(event)
            for event in raw_events
            if _activity_event_is_visible(event)
        ]
        has_more = len(visible_events) > limit
        events = visible_events[:limit]
        if has_more:
            next_sequence = events[-1]["sequence"]
        elif raw_events:
            next_sequence = raw_events[-1].sequence
        else:
            next_sequence = after_sequence

    return {
        "schema_version": "1.0",
        "kind": "run_activity",
        "workspace": str(path),
        "events": events,
        "next_sequence": next_sequence,
        "has_more": has_more,
    }


def list_run_statuses() -> list[dict]:
    """Return one status dict per locally submitted run/batch root, newest first."""
    roots: dict[str, Path] = {}
    for workspace in reversed(indexed_workspaces()):
        try:
            request = load_request(workspace)
            root = request.batch_workspace or workspace
            roots.setdefault(str(root), root)
        except (FileNotFoundError, ValueError):
            continue
    statuses = []
    for workspace in roots.values():
        try:
            statuses.append(status(workspace))
        except (FileNotFoundError, ValueError):
            continue
    return statuses


def cancel_run(workspace: str | Path, reason: str, *, source: str = "cli:cancel") -> dict:
    """Request cooperative cancellation of a single (non-batch) run.

    Returns a dict describing what happened. Raises ``RuntimeError`` if there is
    no active process and no server operations to cancel.
    """
    path = Path(workspace).expanduser().resolve()
    process = active_process(path)
    control = WorkspaceRunControl(path, source=source)
    if process is None:
        if control.active_server_operations():
            requested, failed = forward_server_cancellation(control)
            return {
                "workspace": str(path),
                "status": "NO_ACTIVE_PROCESS",
                "server_operations_cancel_requested": requested,
                "server_operations_cancel_failed": failed,
            }
        raise RuntimeError(f"run has no active process: {path}")
    state, server_requested, server_failed = request_run_cancellation(control, reason)
    return {
        "workspace": str(path),
        "status": "CANCEL_REQUESTED",
        "generation": state.generation,
        "server_operations_cancel_requested": server_requested,
        "server_operations_cancel_failed": server_failed,
    }


def worker_pool_status(eval_server: str) -> dict:
    """Return capacity and waiting runs for one explicit eval server.

    Unlike :func:`list_worker_pools`, this always returns a snapshot, even when
    the pool has never been configured and has no active leases. This lets
    clients inspect a selected device before submitting its first run.
    """
    snapshot = WorkerLeasePool(eval_server).snapshot()
    queued_runs = []
    for item in list_run_statuses():
        runs = item.get("runs", []) if item.get("kind") == "batch" else [item]
        for run in runs:
            if (
                run.get("kind") == "run"
                and run.get("state") == RunState.QUEUED.value
                and run.get("worker_pool") == snapshot["worker_pool"]
            ):
                process = run.get("process") or {}
                queued_runs.append(
                    {
                        "run_id": run.get("run_id"),
                        "definition": run.get("definition"),
                        "workspace": run.get("workspace"),
                        "worker_weight": process.get("worker_weight") or 1,
                        "submitted_at": process.get("submitted_at"),
                    }
                )
    snapshot["queued_runs"] = queued_runs
    snapshot["queued_count"] = len(queued_runs)
    snapshot["queued_workers"] = sum(
        item["worker_weight"] for item in queued_runs
    )
    return snapshot


def list_worker_pools() -> list[dict]:
    """Return snapshots of every known worker pool.

    Enumerates pools from two sources:

    - ``config.json`` — every ``eval_server`` for which ``run.max_workers``
      has been explicitly set via ``kg config set``.
    - ``worker-leases.json`` — every pool with a live lease (i.e. a run
      currently occupying capacity, even if its max was never customized).

    Both sources are unioned so that:

    - unconfigured-but-active pools show up with ``max_workers`` at the default;
    - configured-but-idle pools show up with ``used_workers`` == 0.

    Each entry is the same shape as :meth:`WorkerLeasePool.snapshot()`.
    """
    home = cli_home()
    pools: set[str] = set()
    config = read_json(home / "config.json", {}) or {}
    maximums = config.get("run", {}).get("max_workers", {})
    if isinstance(maximums, dict):
        pools.update(str(name) for name in maximums)
    leases_doc = read_json(home / "worker-leases.json", {}) or {}
    for item in leases_doc.get("leases", []) or []:
        pool = item.get("worker_pool") if isinstance(item, dict) else None
        if isinstance(pool, str) and pool:
            pools.add(pool)
    snapshots = []
    for pool in sorted(pools):
        try:
            snapshots.append(WorkerLeasePool(pool).snapshot())
        except ValueError:
            continue
    return snapshots
