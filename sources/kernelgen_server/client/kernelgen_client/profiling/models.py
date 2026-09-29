# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral profiling models for the v6 protocol."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..protocol.version import ApiVersion, KERNELGEN_API_VERSION
from ..protocol.schema import Definition, EvaluatorBinding, Implementation, Workload


ProfileLevel = Literal["metrics", "instruction"]
ProfileStatus = Literal["completed", "failed", "unsupported"]


class ProfileOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: ProfileLevel = "metrics"
    warmup: int = Field(default=10, ge=0)
    iterations: int = Field(default=100, gt=0)
    timeout_sec: int = Field(default=600, gt=0)
    backend_options: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("level", mode="before")
    @classmethod
    def normalize_legacy_source_level(cls, value: Any) -> Any:
        # ``source`` was briefly exposed as a third level.  Keep the input
        # boundary compatible, but never propagate it into internal state,
        # OpenAPI, results, or profiler implementations.
        return "instruction" if value == "source" else value


class ProfileTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluation_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1)
    implementation: Implementation
    definition: Definition
    workload: Workload
    expected_backend: str = Field(min_length=1)
    oracle_path: Optional[str] = None

    @model_validator(mode="after")
    def validate_target(self) -> "ProfileTarget":
        if self.implementation.definition != self.definition.name:
            raise ValueError("implementation.definition must match definition.name")
        return self


class ProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: ApiVersion = KERNELGEN_API_VERSION
    evaluation_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1)
    binding: EvaluatorBinding
    implementation: Implementation
    benchmark_fingerprint: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    expected_backend: str = Field(min_length=1)
    options: ProfileOptions = Field(default_factory=ProfileOptions)

    @model_validator(mode="after")
    def validate_request(self) -> "ProfileRequest":
        if self.implementation.definition != self.binding.definition:
            raise ValueError("implementation.definition must match binding.definition")
        return self


class ProfileCommand(BaseModel):
    """Exact adapter-owned process launched under a vendor profiler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    argv: Tuple[str, ...] = Field(min_length=1)
    cwd: str = Field(min_length=1)
    env: Dict[str, str] = Field(default_factory=dict)
    source_roots: Tuple[str, ...] = Field(default_factory=tuple)
    completion_marker_path: Optional[str] = None


class ProfileArtifact(BaseModel):
    id: str
    kind: str
    format: str
    filename: str
    media_type: str = "application/octet-stream"
    size_bytes: int = Field(ge=0)
    download_url: str


class ProfileResult(BaseModel):
    api_version: ApiVersion = KERNELGEN_API_VERSION
    profile_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: ProfileStatus
    backend: str
    profiler: str
    device: str
    hardware: Dict[str, Any] = Field(default_factory=dict)
    evaluation_id: str
    workload_name: str
    options: ProfileOptions
    capabilities: List[str] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[ProfileArtifact] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


@dataclass(frozen=True)
class ArtifactSource:
    kind: str
    format: str
    path: Path
    media_type: str = "application/octet-stream"


@dataclass
class BackendProfileResult:
    status: ProfileStatus
    profiler: str
    capabilities: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[ArtifactSource] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    hardware: Dict[str, Any] = field(default_factory=dict)
