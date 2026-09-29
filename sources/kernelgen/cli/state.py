"""Persisted request, process and workspace index state for ``kg``."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from kernelgen.cli.models import RunProcessRecord, RunRequest
# Keep existing import paths as aliases, not a second implementation.
from kernelgen.framework.local_state import (
    STATE_DIRNAME, atomic_write_json, file_lock, process_is_alive,
    process_start_identity, read_json, state_home as cli_home,
)
from kernelgen.framework.worker_pool import (
    DEFAULT_MAX_WORKERS, WorkerLeasePool, get_max_workers,
    normalize_worker_pool, set_max_workers,
)

REQUEST_FILENAME = "run-request.json"
PROCESS_FILENAME = "run-process.json"
RUNNER_LOG_FILENAME = "runner.log"


def run_state_dir(workspace: str | Path) -> Path:
    return Path(workspace).expanduser().resolve() / STATE_DIRNAME


def request_path(workspace: str | Path) -> Path:
    return run_state_dir(workspace) / REQUEST_FILENAME


def process_path(workspace: str | Path) -> Path:
    return run_state_dir(workspace) / PROCESS_FILENAME


def runner_log_path(workspace: str | Path) -> Path:
    return run_state_dir(workspace) / RUNNER_LOG_FILENAME


def save_request(request: RunRequest) -> None:
    atomic_write_json(
        request_path(request.workspace),
        request.model_dump(mode="json"),
    )


def load_request(workspace: str | Path) -> RunRequest:
    path = request_path(workspace)
    raw = read_json(path)
    if raw is None:
        raise FileNotFoundError(f"run request not found: {path}")
    return RunRequest.model_validate(raw)


def save_process(record: RunProcessRecord) -> None:
    atomic_write_json(
        process_path(record.workspace),
        record.model_dump(mode="json"),
    )


def load_process(workspace: str | Path) -> RunProcessRecord | None:
    raw = read_json(process_path(workspace))
    return None if raw is None else RunProcessRecord.model_validate(raw)


@contextmanager
def workspace_submission_lock(workspace: str | Path) -> Iterator[None]:
    with file_lock(run_state_dir(workspace) / "submission.lock"):
        yield


def active_process(workspace: str | Path) -> RunProcessRecord | None:
    record = load_process(workspace)
    if record is None:
        return None
    if record.state.value == "EXITED":
        return None
    return record if process_is_alive(record.pid, record.process_start) else None


def register_workspace(workspace: str | Path) -> None:
    root = cli_home()
    index_path = root / "run-index.json"
    with file_lock(root / "state.lock"):
        raw = read_json(index_path, {"schema_version": "1.0", "workspaces": []})
        resolved = str(Path(workspace).expanduser().resolve())
        items = [item for item in raw.get("workspaces", []) if item != resolved]
        items.append(resolved)
        atomic_write_json(
            index_path,
            {"schema_version": "1.0", "workspaces": items},
        )


def indexed_workspaces() -> list[Path]:
    raw = read_json(cli_home() / "run-index.json", {}) or {}
    workspaces = [Path(item) for item in raw.get("workspaces", [])]
    default_runs = cli_home() / "runs"
    if default_runs.is_dir():
        for path in default_runs.rglob(REQUEST_FILENAME):
            if path.parent.name == STATE_DIRNAME:
                workspaces.append(path.parent.parent)
    unique: dict[str, Path] = {}
    for workspace in workspaces:
        resolved = workspace.expanduser().resolve()
        unique[str(resolved)] = resolved
    return list(unique.values())
