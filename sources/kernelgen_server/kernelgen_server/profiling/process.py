"""Deadline-aware subprocess execution shared by vendor profilers."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..runtime.operations import current_operation
from ..runtime.process_control import terminate_process_tree


class ProfileDeadlineExceeded(TimeoutError):
    def __init__(self, label: str, output: str = "") -> None:
        super().__init__(f"profiling deadline expired during {label}")
        self.label = label
        self.output = output


class ProfileStageError(RuntimeError):
    """A common execution invariant failed for one vendor stage."""

    def __init__(self, message: str, output: str = "") -> None:
        super().__init__(message)
        self.output = output


@dataclass(frozen=True)
class RequestDeadline:
    expires_at: float

    @classmethod
    def from_timeout(cls, timeout_seconds: float) -> "RequestDeadline":
        return cls(time.monotonic() + timeout_seconds)

    def remaining(self, label: str = "profiling request") -> float:
        value = self.expires_at - time.monotonic()
        if value <= 0:
            raise ProfileDeadlineExceeded(label)
        return value


@dataclass(frozen=True)
class ProcessExecution:
    argv: tuple[str, ...]
    returncode: int
    output: str


class ProfileProcessRunner:
    """Run external stages in their own process group under one deadline."""

    def __init__(self, deadline: RequestDeadline) -> None:
        self.deadline = deadline
        self.operation = current_operation()

    def raise_if_cancelled(self) -> None:
        if self.operation is not None:
            self.operation.raise_if_cancelled()

    def run(
        self,
        argv: Sequence[str],
        *,
        label: str,
        cwd: str | Path,
        env: Mapping[str, str],
        completion_marker: str | Path | None = None,
    ) -> ProcessExecution:
        command = tuple(str(item) for item in argv)
        if not command:
            raise ValueError("profile stage command must not be empty")
        marker = Path(completion_marker).resolve() if completion_marker else None
        if marker is not None:
            marker.unlink(missing_ok=True)
        self.raise_if_cancelled()
        remaining = self.deadline.remaining(label)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        last_timeout: subprocess.TimeoutExpired | None = None
        while True:
            if self.operation is not None and self.operation.cancel_requested:
                terminate_process_tree(process)
                process.communicate()
                self.operation.raise_if_cancelled()
            remaining = self.deadline.expires_at - time.monotonic()
            if remaining <= 0:
                terminate_process_tree(process)
                output, _ = process.communicate()
                if not output and last_timeout is not None:
                    partial = last_timeout.output or ""
                    output = (
                        partial.decode("utf-8", "replace")
                        if isinstance(partial, bytes)
                        else partial
                    )
                raise ProfileDeadlineExceeded(label, output or "") from last_timeout
            try:
                output, _ = process.communicate(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired as exc:
                last_timeout = exc

        execution = ProcessExecution(
            argv=command,
            returncode=int(process.returncode),
            output=output or "",
        )
        if execution.returncode == 0 and marker is not None and not marker.is_file():
            raise ProfileStageError(
                f"{label} exited successfully without profile completion marker",
                execution.output,
            )
        return execution


__all__ = [
    "ProcessExecution",
    "ProfileDeadlineExceeded",
    "ProfileProcessRunner",
    "ProfileStageError",
    "RequestDeadline",
]
