"""Workspace-bound configuration for KernelGen agent tools.

The orchestrator writes this file before the Coder agent starts. Tool handlers
derive their configuration from it instead of accepting workspace, trace, and
evaluation-service locations as per-call model arguments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kernelgen.data.implementation import ImplementationLanguage
from kernelgen.data.target_context import TargetContext


_CONTEXT_RELATIVE_PATH = Path(".kernelgen") / "tool-context.json"


class ToolContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    definition: str
    target_hardware: str
    implementation_language: ImplementationLanguage = ImplementationLanguage.TRITON
    eval_server_url: str = ""
    catalog_name: str = ""
    evaluation_snapshot_path: str = ""
    destination_passing_style: bool = True
    profile_enabled: bool = True
    warmup_ms: int = 1000
    benchmark_ms: int = 100
    num_trials: int = 1
    eval_tolerance_mode: Literal["strict", "fixed"] = "strict"
    eval_atol: float = Field(default=1e-2, ge=0)
    eval_rtol: float = Field(default=1e-2, ge=0)
    eval_timeout_seconds: int = 1500
    eval_transport_timeout_seconds: int = 1800
    target_context: TargetContext | None = None

    def write(self, workspace: str | Path) -> Path:
        path = Path(workspace) / _CONTEXT_RELATIVE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                self.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path


def workspace_from_env() -> Path:
    """Resolve the workspace assigned by the agent runtime."""
    raw = os.environ.get("KERNELGEN_WORKSPACE") or os.environ.get("CLAUDE_PROJECT_DIR")
    return Path(raw).resolve() if raw else Path.cwd().resolve()


def load_tool_context(workspace: str | Path | None = None) -> ToolContext:
    root = Path(workspace).resolve() if workspace else workspace_from_env()
    path = root / _CONTEXT_RELATIVE_PATH
    if not path.is_file():
        raise RuntimeError(
            f"KernelGen tool context is missing: {path}. "
            "Run tools through OptimizeDefinitionWorkflow or prepare the workspace first."
        )
    return ToolContext.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_workspace_file(workspace: str | Path, relative_path: str) -> Path:
    """Resolve a model-supplied path without allowing workspace escape."""
    root = Path(workspace).resolve()
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("kernel_path must be relative to the current workspace")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes the current workspace: {relative_path}") from exc
    return resolved
