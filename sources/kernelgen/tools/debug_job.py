"""Workspace-bound adapter for trusted remote Debug Jobs.

Debug Jobs are bounded diagnostic experiments, not formal optimization rounds.
This module keeps the KernelGen Server HTTP protocol out of agent prompts,
snapshots the exact submitted files, binds remote job IDs to one agent
workspace, and downloads verified artifacts into that workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kernelgen.data.tool_context import (
    ToolContext,
    resolve_workspace_file,
)


DEBUG_JOB_SCHEMA_VERSION = "1.0"
DEBUG_JOB_RELATIVE_ROOT = Path(".kernelgen") / "debug-jobs"
DEBUG_TERMINAL_STATUSES = {
    "SUCCEEDED",
    "FAILED",
    "TIMEOUT",
    "CANCELLED",
}
_DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_MAX_RETURNED_LOG_CHARS = 32 * 1024
_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_ENV_NAMES = {
    "KGS_BACKEND",
    "KGS_CATALOG_ROOT",
    "KGS_DEBUG_ARTIFACTS",
    "KGS_DEBUG_JOB_ID",
    "KGS_DEBUG_WORKSPACE",
    "KGS_DEVICE",
    "KGS_ASSIGNED_DEVICE",
    "KGS_PYTHON",
    "FIB_DEBUG_ARTIFACTS",
    "FIB_DEBUG_JOB_ID",
    "FIB_DEBUG_WORKSPACE",
    "FIB_DEVICE",
    "FIB_ASSIGNED_DEVICE",
    "FIB_TRACE_ROOT",
    "ASCEND_RT_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "MLU_VISIBLE_DEVICES",
    "MTHREADS_VISIBLE_DEVICES",
    "MUSA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _safe_debug_path(value: str) -> str:
    path = Path(value)
    if (
        not value
        or "\x00" in value
        or path.is_absolute()
        or ".." in path.parts
        or value in {".", "./"}
    ):
        raise ValueError("debug file path must be a safe relative path")
    if not path.parts or path.parts[0] != "tmp":
        raise ValueError("debug files must be under tmp/")
    return value


class DebugWorkspaceFile(BaseModel):
    """One existing UTF-8 text file uploaded from the current workspace."""

    model_config = ConfigDict(extra="forbid")

    path: str
    executable: bool = False

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _safe_debug_path(value)


class WorkspaceDebugJobRequest(BaseModel):
    """A complete and auditable remote diagnostic experiment."""

    model_config = ConfigDict(extra="forbid")

    purpose: str = Field(min_length=1, max_length=500)
    command: Annotated[list[str], Field(min_length=1, max_length=64)]
    files: Annotated[list[DebugWorkspaceFile], Field(min_length=1, max_length=64)]
    env: Annotated[dict[str, str], Field(max_length=64)] = Field(
        default_factory=dict
    )
    timeout_seconds: int = Field(default=300, ge=1, le=1800)

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: list[str]) -> list[str]:
        if any(not item or "\x00" in item for item in value):
            raise ValueError(
                "debug command arguments must be non-empty and contain no NUL bytes"
            )
        return value

    @field_validator("env")
    @classmethod
    def validate_env(cls, value: dict[str, str]) -> dict[str, str]:
        for name, item in value.items():
            if not _ENV_NAME.fullmatch(name):
                raise ValueError(f"invalid debug environment variable: {name!r}")
            if (
                name in _RESERVED_ENV_NAMES
                or name.startswith("KGS_DEBUG_")
                or name.startswith("FIB_DEBUG_")
            ):
                raise ValueError(f"reserved debug environment variable: {name}")
            if "\x00" in item:
                raise ValueError(
                    f"debug environment variable contains NUL bytes: {name}"
                )
        return value

    @model_validator(mode="after")
    def validate_unique_files(self) -> "WorkspaceDebugJobRequest":
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("debug file paths must be unique")
        return self


def _append_event(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        )


def _validated_job_id(job_id: str) -> str:
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("invalid Debug Job ID")
    return job_id


def _job_dir(workspace: Path, job_id: str, *, require_owned: bool) -> Path:
    validated = _validated_job_id(job_id)
    path = workspace / DEBUG_JOB_RELATIVE_ROOT / validated
    if require_owned and not (path / "request.json").is_file():
        raise ValueError(
            f"Debug Job {validated} was not submitted by this workspace"
        )
    return path


def _read_sources(
    workspace: Path,
    request: WorkspaceDebugJobRequest,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    service_files = []
    metadata = []
    total_bytes = 0
    for item in request.files:
        source = resolve_workspace_file(workspace, item.path)
        if not source.is_file():
            raise ValueError(f"debug file not found: {item.path}")
        try:
            content = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"debug file must be UTF-8 text: {item.path}"
            ) from exc
        encoded = content.encode("utf-8")
        total_bytes += len(encoded)
        service_files.append(
            {
                "path": item.path,
                "content": content,
                "executable": item.executable,
            }
        )
        metadata.append(
            {
                "path": item.path,
                "executable": item.executable,
                "size_bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    if total_bytes > _MAX_SOURCE_BYTES:
        raise ValueError(
            f"debug source files exceed {_MAX_SOURCE_BYTES} bytes"
        )
    return service_files, metadata


def _snapshot_sources(job_dir: Path, service_files: list[dict[str, Any]]) -> None:
    root = (job_dir / "sources").resolve()
    root.mkdir(parents=True, exist_ok=False)
    for item in service_files:
        destination = (root / item["path"]).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"unsafe debug source snapshot path: {item['path']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(item["content"], encoding="utf-8")
        destination.chmod(0o700 if item["executable"] else 0o600)


def _relative_path(workspace: Path, value: str | Path) -> str:
    resolved = Path(value).resolve()
    try:
        return str(resolved.relative_to(workspace.resolve()))
    except ValueError as exc:
        raise ValueError(f"Debug Job output escaped workspace: {value}") from exc


def _resolved_server_url(context: ToolContext) -> str:
    from kernelgen.tools.kernelgen_server_adapter import resolve_server_url

    return resolve_server_url(context.eval_server_url) or _DEFAULT_SERVER_URL


def _check_service_capability(
    context: ToolContext,
    server_url: str,
) -> dict[str, Any]:
    from kernelgen.tools.kernelgen_server_adapter import hardware_family
    from kernelgen_client.http import status

    service = status(server_url)
    debug = service.get("debug", {})
    if not debug.get("enabled"):
        raise RuntimeError(
            "KernelGen Server Debug Jobs are unavailable. Start the trusted "
            "loopback service with --enable-debug-jobs."
        )
    expected_backend = hardware_family(context.target_hardware)
    actual_backend = service.get("backend")
    if expected_backend and actual_backend != expected_backend:
        raise RuntimeError(
            f"target hardware {context.target_hardware!r} maps to backend "
            f"{expected_backend!r}, but KernelGen Server runs "
            f"{actual_backend!r}"
        )
    return service


def _record_status(
    workspace: Path,
    job_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    status = str(result.get("status", ""))
    metadata = dict(result)
    stdout = str(metadata.pop("stdout", ""))
    stderr = str(metadata.pop("stderr", ""))
    if stdout:
        (job_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    if stderr:
        (job_dir / "stderr.log").write_text(stderr, encoding="utf-8")

    artifacts = []
    for raw_artifact in metadata.get("artifacts", []):
        artifact = dict(raw_artifact)
        if artifact.get("local_path"):
            artifact["local_path"] = _relative_path(
                workspace,
                artifact["local_path"],
            )
        artifacts.append(artifact)
    metadata["artifacts"] = artifacts
    metadata.update(
        {
            "schema_version": DEBUG_JOB_SCHEMA_VERSION,
            "observed_at": _utc_now(),
        }
    )
    _atomic_json(job_dir / "status.json", metadata)
    _append_event(
        job_dir / "events.jsonl",
        {
            "observed_at": metadata["observed_at"],
            "status": status,
            "exit_code": result.get("exit_code"),
            "artifact_count": len(artifacts),
            "error": result.get("error", ""),
        },
    )

    stdout_truncated = len(stdout) > _MAX_RETURNED_LOG_CHARS
    stderr_truncated = len(stderr) > _MAX_RETURNED_LOG_CHARS
    response = {
        **metadata,
        "stdout": stdout[-_MAX_RETURNED_LOG_CHARS:],
        "stderr": stderr[-_MAX_RETURNED_LOG_CHARS:],
        "client_output_truncated": stdout_truncated or stderr_truncated,
        "record_path": _relative_path(workspace, job_dir / "status.json"),
    }
    if stdout:
        response["stdout_path"] = _relative_path(
            workspace,
            job_dir / "stdout.log",
        )
    if stderr:
        response["stderr_path"] = _relative_path(
            workspace,
            job_dir / "stderr.log",
        )
    if status in {"QUEUED", "RUNNING"}:
        response["instruction"] = (
            "KernelGen is waiting for this Debug Job to reach a terminal "
            "status before returning control to the agent."
        )
    else:
        response["instruction"] = (
            "Treat this result as diagnostic evidence only. Formal correctness "
            "and performance require eval_round, and every edited candidate "
            "must still pass preflight_kernel."
        )
    return response


def submit_workspace_debug_job(
    workspace: str | Path,
    context: ToolContext,
    request: WorkspaceDebugJobRequest,
) -> dict[str, Any]:
    """Upload exact workspace files and queue one remote diagnostic job."""

    root = Path(workspace).resolve()
    service_files, file_metadata = _read_sources(root, request)
    from kernelgen_client.http import (
        download_debug_artifacts,
        submit_debug_job,
    )
    from kernelgen_client.debug.models import DebugJobRequest, DebugSourceFile

    server_url = _resolved_server_url(context)
    service = _check_service_capability(context, server_url)
    server_request = DebugJobRequest(
        command=request.command,
        files=[DebugSourceFile(**item) for item in service_files],
        env=request.env,
        timeout_seconds=request.timeout_seconds,
    )
    transport_timeout = max(
        float(context.eval_transport_timeout_seconds),
        float(request.timeout_seconds + 10),
    )
    job = submit_debug_job(
        server_request,
        server_url,
        timeout=transport_timeout,
    )
    if job.status not in DEBUG_TERMINAL_STATUSES:
        raise RuntimeError(
            "KernelGen Server returned a non-terminal Debug Job from submit; "
            "deploy the synchronous Debug Job Server API"
        )
    job_id = _validated_job_id(job.job_id)
    job_dir = _job_dir(root, job_id, require_owned=False)
    if job_dir.exists():
        raise ValueError(f"Debug Job directory already exists: {job_id}")
    job_dir.mkdir(parents=True)
    _snapshot_sources(job_dir, service_files)
    _atomic_json(
        job_dir / "request.json",
        {
            "schema_version": DEBUG_JOB_SCHEMA_VERSION,
            "job_id": job_id,
            "submitted_at": _utc_now(),
            "purpose": request.purpose,
            "command": request.command,
            "files": file_metadata,
            "env": request.env,
            "timeout_seconds": request.timeout_seconds,
            "catalog_name": context.catalog_name,
            "definition": context.definition,
            "target_hardware": context.target_hardware,
            "server_backend": service.get("backend"),
        },
    )
    result = job.model_dump(mode="json")
    if job.artifacts:
        downloaded = download_debug_artifacts(
            job,
            job_dir / "artifacts",
            server_url,
        )
        for artifact in result["artifacts"]:
            local_path = downloaded.get(artifact["id"])
            if local_path is not None:
                artifact["local_path"] = str(local_path)
    return _record_status(root, job_dir, result)


def get_workspace_debug_job(
    workspace: str | Path,
    context: ToolContext,
    job_id: str,
    *,
    wait_seconds: float = 300,
) -> dict[str, Any]:
    """Wait across bounded HTTP polls and return one owned Debug Job."""

    root = Path(workspace).resolve()
    job_dir = _job_dir(root, job_id, require_owned=True)
    from kernelgen_client.http import (
        download_debug_artifacts,
        get_debug_job,
    )

    server_url = _resolved_server_url(context)
    if not 0 <= wait_seconds <= 1800:
        raise ValueError("wait_seconds must be between 0 and 1800")
    deadline = time.monotonic() + wait_seconds
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        job = get_debug_job(
            job_id,
            server_url,
            wait_seconds=min(30.0, remaining),
        )
        if job.job_id != job_id:
            raise RuntimeError("KernelGen Server returned a mismatched Debug Job ID")
        if (
            job.status in DEBUG_TERMINAL_STATUSES
            or time.monotonic() >= deadline
        ):
            break

    result = job.model_dump(mode="json")
    if job.status in DEBUG_TERMINAL_STATUSES and job.artifacts:
        downloaded = download_debug_artifacts(
            job,
            job_dir / "artifacts",
            server_url,
        )
        for artifact in result["artifacts"]:
            local_path = downloaded.get(artifact["id"])
            if local_path is not None:
                artifact["local_path"] = str(local_path)
    return _record_status(root, job_dir, result)


def cancel_workspace_debug_job(
    workspace: str | Path,
    context: ToolContext,
    job_id: str,
) -> dict[str, Any]:
    """Cancel one queued/running job previously submitted by this workspace."""

    root = Path(workspace).resolve()
    job_dir = _job_dir(root, job_id, require_owned=True)
    from kernelgen_client.http import cancel_debug_job

    server_url = _resolved_server_url(context)
    job = cancel_debug_job(job_id, server_url)
    if job.job_id != job_id:
        raise RuntimeError("KernelGen Server returned a mismatched Debug Job ID")
    return _record_status(
        root,
        job_dir,
        job.model_dump(mode="json"),
    )
