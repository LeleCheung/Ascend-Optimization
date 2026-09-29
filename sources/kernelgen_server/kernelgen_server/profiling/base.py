# Modified for KernelGen Server in 2026.
# SPDX-License-Identifier: Apache-2.0

"""Managed lifecycle shared by all vendor profiling toolchains."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from ..runtime.operations import OperationCancelled
from .models import (
    ArtifactSource,
    BackendProfileResult,
    ProfileCommand,
    ProfileOptions,
)
from .process import (
    ProcessExecution,
    ProfileDeadlineExceeded,
    ProfileProcessRunner,
    ProfileStageError,
    RequestDeadline,
)


class ProfilerUnsupportedError(RuntimeError):
    """The requested level cannot run in the installed vendor environment."""


class ProfilerExecutionError(RuntimeError):
    """A profile ran but did not produce valid required evidence."""


@dataclass
class ProfileExecutionContext:
    command: ProfileCommand
    options: ProfileOptions
    artifact_dir: Path
    device: str
    deadline: RequestDeadline
    process_runner: ProfileProcessRunner
    capabilities: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[ArtifactSource] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    hardware: Dict[str, Any] = field(default_factory=dict)
    stage_count: int = 0

    @property
    def source_roots(self) -> tuple[Path, ...]:
        return tuple(Path(item).resolve() for item in self.command.source_roots)

    @property
    def completion_marker(self) -> Path | None:
        value = self.command.completion_marker_path
        return Path(value).resolve() if value else None

    def environment(
        self,
        overrides: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.command.env)
        if overrides:
            env.update({str(name): str(value) for name, value in overrides.items()})
        return env

    def run_profile_stage(
        self,
        argv: Sequence[str],
        *,
        label: str,
        env_overrides: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> ProcessExecution:
        self.stage_count += 1
        return self.process_runner.run(
            argv,
            label=label,
            cwd=cwd or self.command.cwd,
            env=self.environment(env_overrides),
            completion_marker=self.completion_marker,
        )

    def run_tool(
        self,
        argv: Sequence[str],
        *,
        label: str,
        env_overrides: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
    ) -> ProcessExecution:
        return self.process_runner.run(
            argv,
            label=label,
            cwd=cwd or self.command.cwd,
            env=self.environment(env_overrides),
        )

    def add_artifact(
        self,
        kind: str,
        format: str,
        path: str | Path,
        media_type: str = "application/octet-stream",
    ) -> None:
        self.artifacts.append(
            ArtifactSource(kind, format, Path(path), media_type)
        )

    def add_capability(self, *values: str) -> None:
        for value in values:
            if value and value not in self.capabilities:
                self.capabilities.append(value)

    def has_artifact(self, kind: str) -> bool:
        return any(artifact.kind == kind for artifact in self.artifacts)


class Profiler(ABC):
    """Backend adapter for one vendor profiling toolchain."""

    backend: str = ""
    name: str = ""

    @abstractmethod
    def available(self) -> bool:
        """Return whether the base profiler executable is available."""

    def available_for(self, options: ProfileOptions) -> bool:
        """Return availability after applying request-local tool paths.

        ``available()`` describes the Server's default installation for
        ``/status``.  This request-aware form lets a validated backend option
        select an executable outside ``PATH`` without constructing or running
        the evaluator command first.
        """

        del options
        return self.available()

    @abstractmethod
    def capabilities(self) -> List[str]:
        """Return capabilities this installed adapter may produce."""

    def levels(self) -> List[str]:
        return ["metrics"]

    def levels_for(self, options: ProfileOptions) -> List[str]:
        """Return levels supported by request-local tool configuration."""

        del options
        return self.levels()

    def normalize_options(
        self,
        options: ProfileOptions,
    ) -> tuple[ProfileOptions, List[str]]:
        return options, []

    def describe(self) -> Dict[str, Any]:
        return {
            "supported": self.available(),
            "profiler": self.name,
            "levels": self.levels(),
            "capabilities": self.capabilities(),
        }

    @abstractmethod
    def profile_command(
        self,
        command: ProfileCommand,
        options: ProfileOptions,
        artifact_dir: Path,
        device: str,
        *,
        deadline: RequestDeadline,
    ) -> BackendProfileResult:
        """Profile one exact evaluator-owned command."""


class ManagedProfiler(Profiler):
    """Template lifecycle with explicit vendor preparation and postprocessing."""

    def prepare(self, context: ProfileExecutionContext) -> Any:
        return None

    def resource_scope(
        self,
        context: ProfileExecutionContext,
        prepared: Any,
    ) -> AbstractContextManager[Any]:
        return nullcontext()

    @abstractmethod
    def collect(self, context: ProfileExecutionContext, prepared: Any) -> Any:
        """Execute the required vendor collection stages."""

    def postprocess(
        self,
        context: ProfileExecutionContext,
        prepared: Any,
        collected: Any,
    ) -> None:
        """Export and normalize vendor reports after collection."""

    @abstractmethod
    def validate_evidence(self, context: ProfileExecutionContext) -> None:
        """Raise when the requested level lacks mandatory evidence."""

    def cleanup(self, context: ProfileExecutionContext, prepared: Any) -> None:
        """Release vendor-local resources after every terminal path."""

    def _validate_command(
        self,
        command: ProfileCommand,
        artifact_dir: Path,
    ) -> None:
        root = artifact_dir.resolve()
        for value in command.source_roots:
            source_root = Path(value).resolve()
            if not source_root.is_dir() or not source_root.is_relative_to(root):
                raise ProfilerExecutionError(
                    f"unsafe or missing ProfileCommand source_root: {value}"
                )
        if command.completion_marker_path:
            marker = Path(command.completion_marker_path).resolve()
            if not marker.is_relative_to(root):
                raise ProfilerExecutionError(
                    "ProfileCommand completion marker escapes artifact directory"
                )

    def profile_command(
        self,
        command: ProfileCommand,
        options: ProfileOptions,
        artifact_dir: Path,
        device: str,
        *,
        deadline: RequestDeadline,
    ) -> BackendProfileResult:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        context = ProfileExecutionContext(
            command=command,
            options=options,
            artifact_dir=artifact_dir.resolve(),
            device=device,
            deadline=deadline,
            process_runner=ProfileProcessRunner(deadline),
        )
        prepared: Any = None
        status = "completed"
        error: str | None = None
        try:
            self._validate_command(command, artifact_dir)
            context.process_runner.raise_if_cancelled()
            deadline.remaining("vendor preparation")
            prepared = self.prepare(context)
            context.process_runner.raise_if_cancelled()
            deadline.remaining("vendor collection")
            with self.resource_scope(context, prepared):
                collected = self.collect(context, prepared)
            context.process_runner.raise_if_cancelled()
            deadline.remaining("vendor postprocessing")
            self.postprocess(context, prepared, collected)
            context.process_runner.raise_if_cancelled()
            deadline.remaining("vendor evidence validation")
            self.validate_evidence(context)
            context.process_runner.raise_if_cancelled()
            deadline.remaining("vendor completion")
        except OperationCancelled:
            raise
        except ProfilerUnsupportedError as exc:
            status = "unsupported"
            error = str(exc)
        except (ProfilerExecutionError, ProfileStageError) as exc:
            status = "failed"
            error = str(exc)
        except ProfileDeadlineExceeded as exc:
            status = "failed"
            error = str(exc)
            if exc.output:
                timeout_log = artifact_dir / "timeout.log"
                timeout_log.write_text(exc.output, encoding="utf-8")
                context.add_artifact(
                    "profile_log",
                    "text",
                    timeout_log,
                    "text/plain; charset=utf-8",
                )
        except Exception as exc:
            status = "failed"
            error = f"unhandled {self.name} profiler error: {type(exc).__name__}: {exc}"
        finally:
            try:
                self.cleanup(context, prepared)
            except Exception as exc:
                status = "failed"
                cleanup_error = (
                    f"{self.name} cleanup failed: {type(exc).__name__}: {exc}"
                )
                error = f"{error}; {cleanup_error}" if error else cleanup_error

        context.summary.setdefault("stage_count", context.stage_count)
        return BackendProfileResult(
            status=status,  # type: ignore[arg-type]
            profiler=self.name,
            capabilities=context.capabilities,
            summary=context.summary,
            metrics=context.metrics,
            artifacts=context.artifacts,
            warnings=context.warnings,
            error=error,
            hardware=context.hardware,
        )


class UnsupportedProfiler(Profiler):
    """Explicit result for backends without a profiling adapter."""

    def __init__(self, backend: str):
        self.backend = backend
        self.name = "unsupported"

    def available(self) -> bool:
        return False

    def capabilities(self) -> List[str]:
        return []

    def levels(self) -> List[str]:
        return []

    def profile_command(
        self,
        command: ProfileCommand,
        options: ProfileOptions,
        artifact_dir: Path,
        device: str,
        *,
        deadline: RequestDeadline,
    ) -> BackendProfileResult:
        del command, options, artifact_dir, device, deadline
        return BackendProfileResult(
            status="unsupported",
            profiler=self.name,
            error=f"Profiling not implemented for backend {self.backend!r}",
        )


__all__ = [
    "ManagedProfiler",
    "ProfileExecutionContext",
    "Profiler",
    "ProfilerExecutionError",
    "ProfilerUnsupportedError",
    "UnsupportedProfiler",
]
