"""Parallel orchestration: run_parallel + Workspace + Runnable.bind (ADR-3 #9).

Core engine of the unified orchestration framework:
- **Workspace**: allocates isolated paths (directory/worktree/docker). Only gives
  paths — does NOT know how to launch anything.
- **Runnable.bind(path, runtime_factory)**: each Runnable type knows how to
  construct itself from a workspace path (polymorphism, not if/else).
- **run_parallel**: allocate → bind → run → collect. Zero branching. Works for
  both agents (one LLM invoke) and workflows (a whole orchestrated graph).

``runtime_factory`` is a plain callable ``(path) -> Runtime`` — Python's lambda,
not a framework abstraction. It lets run_parallel stay generic (no ClaudeRuntime
import) and testable (swap in FakeRuntime).
"""

from __future__ import annotations

import os
import shutil
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Protocol, Sequence, Tuple, Type, cast

from pydantic import BaseModel

from kernelgen.framework.agent_roles import materialize_agent_roles
from kernelgen.framework.agent_skills import (
    materialize_agent_skills,
    remove_materialized_agent_skills,
)
from kernelgen.framework.mcp_config import materialize_mcp_configuration
from kernelgen.framework.run_control import (
    CancellationToken,
    RunCancelled,
    RunControl,
)
from kernelgen.framework.runnable import Runnable


def copy_mcp_configuration(
    source: os.PathLike | str,
    destination: os.PathLike | str,
    *,
    tool_timeout_seconds: int,
) -> None:
    """Materialize neutral MCP config at a legacy provider-file destination.

    ``destination`` remains the generated Claude ``.mcp.json`` path for API
    compatibility. The neutral snapshot is written beside it under
    ``.kernelgen/mcp.json``.
    """
    destination_path = Path(destination)
    if destination_path.name != ".mcp.json":
        raise ValueError("destination must be a generated .mcp.json path")
    try:
        materialize_mcp_configuration(
            Path(source),
            destination_path.parent,
            tool_timeout_seconds=tool_timeout_seconds,
        )
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc


def copy_claude_directory(
    source: os.PathLike | str,
    destination: os.PathLike | str,
    *,
    include_skills: bool = True,
) -> None:
    """Copy Claude settings and materialize neutral roles and skills.

    Nested Claude sessions must inherit the run workspace's endpoint, model,
    plugin, and permission settings. Keep the ignored file out of Git, but copy
    it into isolated workflow workspaces when present. Canonical roles and
    skills live under the sibling ``.kernelgen`` directory and are materialized
    for both Claude and Codex in the destination workspace. The source
    ``.claude`` directory may be absent when no static Claude settings exist.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.is_dir():
        shutil.copytree(
            source_path,
            destination_path,
            dirs_exist_ok=True,
            symlinks=True,
            ignore=shutil.ignore_patterns("agents", "skills"),
        )
    source_local_settings = source_path / "settings.local.json"
    local_settings = destination_path / "settings.local.json"
    if not source_local_settings.exists() and (
        local_settings.is_file() or local_settings.is_symlink()
    ):
        local_settings.unlink()
    workspace = destination_path.parent
    agent_roles_source = source_path.parent / ".kernelgen" / "agents"
    if agent_roles_source.is_dir():
        materialize_agent_roles(agent_roles_source, workspace)
    agent_skills_source = source_path.parent / ".kernelgen" / "skills"
    if include_skills and agent_skills_source.is_dir():
        materialize_agent_skills(agent_skills_source, workspace)
    else:
        remove_materialized_agent_skills(workspace)


class Workspace(ABC):
    """Allocates isolated paths. That's it — no launch logic (that's Runnable.bind).

    ``allocate(name) → path``: create an isolated space, return its path.
    ``cleanup(name)``: tear it down (default: no-op, keep products for downstream).
    ``path_of(name)``: look up a previously allocated path.
    """

    def __init__(self):
        self._paths: dict[str, str] = {}

    def allocate(self, name: str) -> str:
        path = self._create(name)
        self._paths[name] = path
        return path

    def cleanup(self, name: str) -> None:
        path = self._paths.pop(name, None)
        if path is not None:
            self._cleanup(path)

    def path_of(self, name: str) -> str:
        return self._paths[name]

    @abstractmethod
    def _create(self, name: str) -> str:
        ...

    def _cleanup(self, path: str) -> None:
        """Default: no-op (keep products). Subclasses override to tear down."""


class Directory(Workspace):
    """Local sub-directory workspace. No real isolation — used for tests and
    single-machine runs. GitWorktree / Docker are 9b."""

    def __init__(self, base: os.PathLike | str = "."):
        super().__init__()
        self.base = Path(base)

    def _create(self, name: str) -> str:
        p = self.base / name
        p.mkdir(parents=True, exist_ok=True)
        return str(p)


class WorkspaceMaterializer(Protocol):
    """Framework-neutral hook for seeding an isolated workspace."""

    def materialize(self, workspace: Path) -> None:
        ...


@dataclass(frozen=True)
class ParallelTaskFailure:
    """One failed input from :func:`run_parallel`."""

    index: int
    name: str
    error: Exception


class ParallelExecutionError(RuntimeError):
    """All failures and completed outputs from one parallel fan-out."""

    def __init__(
        self,
        *,
        failures: Sequence[ParallelTaskFailure],
        partial_results: Sequence[Tuple[BaseModel, str] | None],
    ):
        if not failures:
            raise ValueError("ParallelExecutionError requires at least one failure")
        self.failures = tuple(sorted(failures, key=lambda item: item.index))
        self.partial_results = tuple(partial_results)
        first = self.failures[0]
        super().__init__(
            f"{len(self.failures)} parallel task(s) failed; first failure "
            f"{first.name}: {type(first.error).__name__}: {first.error}"
        )


class IsolatedDirectory(Workspace):
    """Production workspace: each agent gets a clean directory with:
    - tmp/           (agent writes kernel code here)
    - .kernelgen/agents/ (provider-neutral role snapshot)
    - .kernelgen/skills/ (provider-neutral Agent Skill snapshot)
    - .claude/       (generated agents/skills + copied settings)
    - .agents/       (generated Codex-discoverable skills)
    - .codex/        (generated project-scoped custom agents)
    - .kernelgen/mcp.json (provider-neutral MCP configuration snapshot)
    - .mcp.json      (generated Claude MCP registration)
    - kb/            (copied snapshot of the base KB)

    No git worktree — just mkdir + copytree. Benchmark-safe: directory is built
    from scratch (only what we put in), so agent can't Glob/Read repo source.
    The shared MCP server imports KernelGen through PYTHONPATH supplied by Runtime.

    The base KB's ``.git`` directory is never copied. Only the run-level KB owns
    history; agent and synthesis workspaces receive plain snapshots.
    """

    def __init__(
        self,
        base: os.PathLike | str,
        claude_source: os.PathLike | str | None = None,
        mcp_source: os.PathLike | str | None = None,
        kb_source: os.PathLike | str | None = None,
        materializers: List[WorkspaceMaterializer] | None = None,
        include_claude_skills: bool = True,
        include_skills: bool | None = None,
    ):
        super().__init__()
        self.base = Path(base)
        self.claude_source = Path(claude_source) if claude_source else None
        self.mcp_source = Path(mcp_source) if mcp_source else None
        self.kb_source = Path(kb_source) if kb_source else None
        self.materializers = list(materializers or [])
        self.include_skills = (
            include_claude_skills if include_skills is None else include_skills
        )
        self.include_claude_skills = self.include_skills

    def _create(self, name: str) -> str:
        p = self.base / name
        p.mkdir(parents=True, exist_ok=True)
        (p / "tmp").mkdir(exist_ok=True)
        try:
            if self.claude_source:
                copy_claude_directory(
                    self.claude_source,
                    p / ".claude",
                    include_skills=self.include_skills,
                )
            if self.mcp_source and self.mcp_source.is_file():
                materialize_mcp_configuration(self.mcp_source, p)
            if self.kb_source and self.kb_source.is_dir():
                shutil.copytree(
                    self.kb_source, p / "kb",
                    dirs_exist_ok=True, symlinks=True,
                    ignore=shutil.ignore_patterns(".git"),
                )
            for materializer in self.materializers:
                materializer.materialize(p)
        except BaseException:
            shutil.rmtree(p, ignore_errors=True)
            raise
        return str(p)

    def _cleanup(self, path: str) -> None:
        """Remove the workspace directory. Override default no-op."""
        p = Path(path)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

def run_parallel(
    runnable_cls: Type[Runnable],
    inputs: List[Any],
    *,
    workspace: Workspace,
    runtime_factory: Callable[[str], Any],
    max_workers: int = 0,
    task_name: Callable[[int, Any], str] | str | None = None,
    launch_interval_seconds: float = 0.0,
    cancellation_token: CancellationToken | None = None,
) -> List[Tuple[BaseModel, str]]:
    """Run ``runnable_cls`` on each input in parallel, one isolated space each.

    Flow per task: ``workspace.allocate → runnable_cls.bind(path, factory) → run(inp)``.
    Zero branching on whether runnable_cls is an Agent or Workflow — bind is polymorphic.

    ``task_name`` controls workspace directory naming:
      - None (default): ``task0``, ``task1``, ... (sequential index)
      - str: use that key from the input dict (e.g. "operator" → "relu", "gelu")
      - callable(i, inp) -> str: full control (e.g. lambda i, inp: f"agent{i}")

    Returns ``[(result, ws_name), ...]`` in input order. Spaces are NOT cleaned up
    (downstream Distiller/Synthesizer may reuse them). Every submitted future is
    collected. If any task fails, ``ParallelExecutionError`` is raised after the
    remaining tasks finish, with each task's original exception and all completed
    outputs available to the caller. When ``cancellation_token`` is supplied,
    queued tasks check it before allocating a workspace and cancellation is
    propagated after already-running tasks reach their own safe points.

    ``launch_interval_seconds`` spaces executor submissions to avoid a connection
    burst when each task initializes a remote runtime. The generic default remains
    zero; callers that use a constrained remote gateway can opt in.
    """
    if not inputs:
        return []
    if launch_interval_seconds < 0:
        raise ValueError("launch_interval_seconds must be non-negative")
    workers = max_workers or min(len(inputs), 16)
    results: List[Tuple[BaseModel, str] | None] = [None] * len(inputs)
    failures: List[ParallelTaskFailure] = []
    cancellation: RunCancelled | None = None
    completed_tasks = 0
    failed_tasks = 0
    cancelled_tasks = 0
    observable_control = (
        cancellation_token
        if isinstance(cancellation_token, RunControl)
        else None
    )
    if observable_control is not None:
        observable_control.update_progress(
            total_tasks=len(inputs),
            completed_tasks=0,
            failed_tasks=0,
            cancelled_tasks=0,
        )
        observable_control.record_event(
            "PARALLEL_BATCH_STARTED",
            stage="PARALLEL_EXECUTION",
            source="framework:run_parallel",
            data={"total_tasks": len(inputs), "max_workers": workers},
        )

    def _publish_parallel_progress() -> None:
        if observable_control is None:
            return
        observable_control.update_progress(
            total_tasks=len(inputs),
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            cancelled_tasks=cancelled_tasks,
        )

    def _resolve_name(i: int, inp: Any) -> str:
        if task_name is None:
            return f"task{i}"
        if callable(task_name):
            return task_name(i, inp)
        # str: use as key into the input dict
        if isinstance(inp, dict) and task_name in inp:
            return str(inp[task_name])
        if hasattr(inp, task_name):
            return str(getattr(inp, task_name))
        return f"task{i}"

    def _one(i: int, inp: Any, name: str) -> Tuple[int, BaseModel, str]:
        if cancellation_token is not None:
            cancellation_token.checkpoint("BEFORE_PARALLEL_TASK")
        path = workspace.allocate(name)
        if observable_control is not None:
            observable_control.record_event(
                "PARALLEL_TASK_STARTED",
                stage="PARALLEL_EXECUTION",
                source="framework:run_parallel",
                data={"index": i, "name": name, "workspace": path},
            )
        instance = runnable_cls.bind(path, runtime_factory)   # polymorphic!
        out = instance.run(inp)
        return i, out, name

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for i, inp in enumerate(inputs):
            name = _resolve_name(i, inp)
            future = ex.submit(_one, i, inp, name)
            futs[future] = (i, name)
            if launch_interval_seconds > 0 and i + 1 < len(inputs):
                time.sleep(launch_interval_seconds)
        for f in as_completed(futs):
            try:
                i, out, name = f.result()
            except RunCancelled as error:
                if cancellation is None:
                    cancellation = error
                completed_tasks += 1
                cancelled_tasks += 1
                if observable_control is not None:
                    observable_control.record_event(
                        "PARALLEL_TASK_CANCELLED",
                        message=str(error),
                        stage="PARALLEL_EXECUTION",
                        source="framework:run_parallel",
                        data={"index": futs[f][0], "name": futs[f][1]},
                    )
                _publish_parallel_progress()
                continue
            except Exception as error:
                i, name = futs[f]
                completed_tasks += 1
                failed_tasks += 1
                failures.append(
                    ParallelTaskFailure(index=i, name=name, error=error)
                )
                if observable_control is not None:
                    observable_control.record_event(
                        "PARALLEL_TASK_FAILED",
                        message=f"{type(error).__name__}: {error}",
                        stage="PARALLEL_EXECUTION",
                        level="ERROR",
                        source="framework:run_parallel",
                        data={"index": i, "name": name},
                    )
                _publish_parallel_progress()
                continue
            completed_tasks += 1
            results[i] = (out, name)
            if observable_control is not None:
                observable_control.record_event(
                    "PARALLEL_TASK_COMPLETED",
                    stage="PARALLEL_EXECUTION",
                    source="framework:run_parallel",
                    data={"index": i, "name": name},
                )
            _publish_parallel_progress()

    if observable_control is not None:
        _publish_parallel_progress()
        observable_control.record_event(
            "PARALLEL_BATCH_COMPLETED",
            stage="PARALLEL_EXECUTION",
            level=(
                "WARNING"
                if failures or cancellation is not None
                else "INFO"
            ),
            source="framework:run_parallel",
            data={
                "total_tasks": len(inputs),
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "cancelled_tasks": cancelled_tasks,
            },
        )

    if cancellation is not None:
        raise cancellation
    if failures:
        raise ParallelExecutionError(
            failures=failures,
            partial_results=results,
        ) from failures[0].error
    return cast(List[Tuple[BaseModel, str]], results)
