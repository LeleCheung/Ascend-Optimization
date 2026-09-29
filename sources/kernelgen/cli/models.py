"""Persisted models used by the local ``kg`` process supervisor."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kernelgen.framework.worker_pool import LeaseRecord  # Existing import path.


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunMode(str, Enum):
    SIMPLE_OPT = "simple_opt"
    KERNELGEN = "kernelgen"


class ProcessState(str, Enum):
    SUBMITTED = "SUBMITTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    EXITED = "EXITED"


class RunRequest(BaseModel):
    """One secret-free, replayable workflow invocation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "2.0"] = "1.0"
    run_id: str = Field(min_length=1)
    mode: RunMode
    definition: str = Field(min_length=1)
    target_hardware: str | None = Field(min_length=1)
    eval_server: str = Field(min_length=1)
    worker_pool: str = Field(min_length=1)
    workspace: Path
    batch_workspace: Path | None = None
    worker_weight: int = Field(ge=1)
    workflow_args: list[str] = Field(default_factory=list)
    workflow_input: dict | None = None
    runtime: str = "claude"
    model: str | None = None
    base_url: str | None = None
    submitted_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_invocation(self):
        if self.schema_version == "2.0":
            if self.workflow_input is None or self.workflow_args:
                raise ValueError("v2 run requires workflow_input and no launcher argv")
            options = self.workflow_input.get("optimization", {})
            expected = (self.mode.value, self.definition, self.target_hardware, self.eval_server, self.worker_weight)
            observed = (options.get("mode"), self.workflow_input.get("operator"), options.get("target_hardware"),
                        options.get("eval_server_url"), options.get("n_parallel", 1))
            if expected != observed:
                raise ValueError("run metadata must match the normalized Catalog request")
        elif self.workflow_input is not None:
            raise ValueError("legacy run uses launcher argv, not workflow_input")
        return self


class BatchChildRecord(BaseModel):
    """One independently supervised operator belonging to a CLI Batch."""

    model_config = ConfigDict(extra="forbid")

    definition: str = Field(min_length=1)
    mode: RunMode
    workspace: Path
    run_id: str = Field(min_length=1)


class BatchRequestRecord(BaseModel):
    """Secret-free index for one YAML-backed CLI Batch."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    batch_id: str = Field(min_length=1)
    workspace: Path
    source_path: Path
    children: list[BatchChildRecord] = Field(min_length=1)
    submitted_at: datetime = Field(default_factory=utc_now)


class RunProcessRecord(BaseModel):
    """PID identity and lifecycle facts for one detached run process."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(min_length=1)
    pid: int = Field(gt=0)
    process_start: str = Field(min_length=1)
    state: ProcessState = ProcessState.SUBMITTED
    workspace: Path
    log_path: Path
    worker_weight: int = Field(ge=1)
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    message: str = ""


class ServerProcessRecord(BaseModel):
    """Identity for a managed local KGS process or remote KGS proxy."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    instance: str = Field(min_length=1)
    target: Literal["local", "remote"] = "local"
    pid: int = Field(gt=0)
    process_start: str = Field(min_length=1)
    log_path: Path
    server_url: str | None = None
    kgs_release: str = Field(min_length=1)
    kgs_commit: str = Field(min_length=1)
    protocol_version: str = Field(min_length=1)
    backend: str = Field(min_length=1)
    devices: list[str] = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    max_workers: int = Field(ge=1)
    remote_pid: int | None = Field(default=None, gt=0)
    remote_process_start: str | None = None
    remote_log_path: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
