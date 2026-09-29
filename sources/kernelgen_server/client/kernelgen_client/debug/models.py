"""Debug Job wire models and validation without execution imports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..device_visibility import ALL_VISIBILITY_VARIABLES
from ..protocol.version import ApiVersion, KERNELGEN_API_VERSION


DEFAULT_MAX_SOURCE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_ARTIFACT_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_ARTIFACTS = 32
_PROCESS_TERMINATION_GRACE_SECONDS = 2.0

DebugJobStatus = Literal[
    "QUEUED",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    "TIMEOUT",
    "CANCELLED",
]

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INHERITED_ENV_NAMES = {
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "ROCM_PATH",
}
_INHERITED_ENV_PREFIXES = (
    "ASCEND_",
    "ATB_",
    "CANN_",
    "CUDA_",
    "HCCL_",
    "HIP_",
    "MACA_",
    "MLU_",
    "MTHREADS_",
    "MUSA_",
    "NCCL_",
    "PPU_",
    "PYTORCH_",
    "ROCR_",
    "TORCH_",
    "TRITON_",
    "TVM_",
    "XPU_",
)
_RESERVED_ENV_NAMES = {
    "KGS_BACKEND",
    "KGS_CATALOG_ROOT",
    "KGS_DEBUG_ARTIFACTS",
    "KGS_DEBUG_JOB_ID",
    "KGS_DEBUG_WORKSPACE",
    "KGS_DEVICE",
    "KGS_ASSIGNED_DEVICE",
    "KGS_PYTHON",
    *ALL_VISIBILITY_VARIABLES,
}
_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "TIMEOUT", "CANCELLED"}
def _validated_relative_path(value: str, *, label: str) -> str:
    path = Path(value)
    if not value or "\x00" in value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a safe relative path")
    if value in {".", "./"}:
        raise ValueError(f"{label} must identify a file")
    return value


class DebugSourceFile(BaseModel):
    """One source file materialized into a debug workspace."""

    model_config = ConfigDict(extra="forbid")

    path: str
    content: str
    executable: bool = False

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validated_relative_path(value, label="source path")


class DebugJobRequest(BaseModel):
    """A complete request for one target-side diagnostic command."""

    model_config = ConfigDict(extra="forbid")

    command: List[str] = Field(min_length=1, max_length=64)
    files: List[DebugSourceFile] = Field(default_factory=list, max_length=64)
    env: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=300, ge=1, le=1800)

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: List[str]) -> List[str]:
        if any(not item or "\x00" in item for item in value):
            raise ValueError(
                "command arguments must be non-empty and contain no NUL bytes"
            )
        return value

    @field_validator("env")
    @classmethod
    def validate_env(cls, value: Dict[str, str]) -> Dict[str, str]:
        for name, item in value.items():
            if not _ENV_NAME.fullmatch(name):
                raise ValueError(f"invalid environment variable name: {name!r}")
            if name in _RESERVED_ENV_NAMES or name.startswith("KGS_DEBUG_"):
                raise ValueError(f"reserved environment variable: {name}")
            if "\x00" in item:
                raise ValueError(f"environment variable contains NUL bytes: {name}")
        return value

    @model_validator(mode="after")
    def validate_unique_files(self) -> "DebugJobRequest":
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("debug source paths must be unique")
        return self

    def source_bytes(self) -> int:
        return sum(len(item.content.encode("utf-8")) for item in self.files)


class DebugArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    path: str
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str
    download_url: str


class DebugJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: ApiVersion = KERNELGEN_API_VERSION
    job_id: str
    status: DebugJobStatus
    backend: str
    device: Optional[str] = None
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    timeout_seconds: int
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    output_truncated: bool = False
    artifacts: List[DebugArtifact] = Field(default_factory=list)
    error: str = ""

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES


@dataclass
class DebugExecutionResult:
    status: DebugJobStatus
    exit_code: Optional[int]
    stdout: str
    stderr: str
    output_truncated: bool
    artifacts: List[DebugArtifact]
    error: str = ""
