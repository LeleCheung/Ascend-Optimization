"""Shared, bounded process-tree termination for Eval, Profile, and Debug."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TERMINATION_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class TerminationResult:
    group_signaled: bool
    forced: bool
    returncode: int | None


def _pid(process: Any) -> int:
    value = getattr(process, "pid", None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("process has no valid pid")
    return value


def _is_alive(process: Any) -> bool:
    if hasattr(process, "poll"):
        return process.poll() is None
    return bool(process.is_alive())


def _wait(process: Any, timeout: float | None = None) -> None:
    if hasattr(process, "poll"):
        try:
            process.wait(timeout=timeout)
        except (TimeoutError, subprocess.TimeoutExpired):
            pass
        return
    process.join(timeout=timeout)


def _returncode(process: Any) -> int | None:
    if hasattr(process, "poll"):
        return process.poll()
    return getattr(process, "exitcode", None)


def _group_exists(group_id: int) -> bool:
    if os.name != "posix" or group_id == os.getpgrp():
        return False
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(group_id: int, sig: int) -> bool:
    if os.name != "posix" or group_id == os.getpgrp():
        return False
    try:
        os.killpg(group_id, sig)
    except (PermissionError, ProcessLookupError):
        return False
    return True


def _session_pids(session_id: int) -> set[int]:
    """Return members of one explicitly-created POSIX job session."""

    if os.name != "posix":
        return set()
    result: set[int] = set()
    try:
        entries = Path("/proc").iterdir()
    except OSError:
        return result
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            if os.getsid(pid) == session_id:
                result.add(pid)
        except (OSError, ProcessLookupError):
            continue
    return result


def _signal_session(session_id: int, sig: int) -> bool:
    signaled = False
    for pid in sorted(_session_pids(session_id), reverse=True):
        try:
            os.kill(pid, sig)
            signaled = True
        except (PermissionError, ProcessLookupError):
            continue
    return signaled


def _signal_direct(process: Any, *, force: bool) -> None:
    if not _is_alive(process):
        return
    action = getattr(process, "kill" if force else "terminate", None)
    if callable(action):
        action()


def terminate_process_tree(
    process: Any,
    *,
    grace_seconds: float = DEFAULT_TERMINATION_GRACE_SECONDS,
) -> TerminationResult:
    """Terminate one explicitly-created job process group and reap its leader."""

    group_id = _pid(process)
    group_signaled = _signal_group(group_id, signal.SIGTERM)
    session_signaled = _signal_session(group_id, signal.SIGTERM)
    if not group_signaled and not session_signaled:
        _signal_direct(process, force=False)

    deadline = time.monotonic() + max(0.0, grace_seconds)
    while (
        _is_alive(process)
        or (group_signaled and _group_exists(group_id))
        or bool(_session_pids(group_id))
    ):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        _wait(process, min(0.05, remaining))

    group_alive = group_signaled and _group_exists(group_id)
    session_alive = bool(_session_pids(group_id))
    process_alive = _is_alive(process)
    forced = group_alive or session_alive or process_alive
    if group_alive:
        _signal_group(group_id, signal.SIGKILL)
    if session_alive:
        _signal_session(group_id, signal.SIGKILL)
    if process_alive and not (group_alive or session_alive):
        _signal_direct(process, force=True)
    _wait(process, None)
    return TerminationResult(
        group_signaled=group_signaled or session_signaled,
        forced=forced,
        returncode=_returncode(process),
    )


__all__ = [
    "DEFAULT_TERMINATION_GRACE_SECONDS",
    "TerminationResult",
    "terminate_process_tree",
]
