"""Agent contract for integrating one Native KernelGen result into FlagGems."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kernelgen.framework.base import BaseAgent


class NativeToFlagGemsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    source_operator: str = ""
    vendor: str = Field(min_length=1)
    flaggems_root: str
    native_kernel_code: str = Field(min_length=1)
    native_definition: dict[str, Any]
    accuracy_files: list[str] = Field(default_factory=list)
    benchmark_files: list[str] = Field(default_factory=list)
    standard_path: str = "docs/content/zh-cn/testing/kernelgen-integration.md"
    run_tests: bool = False


class MigrationCommandResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str
    exit_code: int | None = None
    outcome: Literal["passed", "failed", "skipped"]
    summary: str = ""


class NativeToFlagGemsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator: str
    status: Literal["migrated", "already_integrated", "blocked", "failed"]
    files_modified: list[str] = Field(default_factory=list)
    commands: list[MigrationCommandResult] = Field(default_factory=list)
    remaining_checks: list[str] = Field(default_factory=list)
    protocol_gaps: list[str] = Field(default_factory=list)
    summary: str = ""


class NativeToFlagGemsAgent(BaseAgent):
    """Prepare a reviewable FlagGems diff; validation remains external."""

    name = "native_to_flaggems"
    InputModel = NativeToFlagGemsInput
    OutputModel = NativeToFlagGemsOutput


__all__ = [
    "MigrationCommandResult",
    "NativeToFlagGemsAgent",
    "NativeToFlagGemsInput",
    "NativeToFlagGemsOutput",
]
