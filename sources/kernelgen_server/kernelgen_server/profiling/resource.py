"""Deadline-aware locks for profiler resources shared by a whole host."""

from __future__ import annotations

import fcntl
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .process import RequestDeadline


_THREAD_LOCKS: dict[Path, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(path, threading.Lock())


@contextmanager
def host_global_lock(path: Path, deadline: RequestDeadline) -> Iterator[float]:
    """Acquire one fixed cross-thread/process lock within the request deadline."""

    path = path.resolve()
    started = time.monotonic()
    thread_lock = _thread_lock(path)
    remaining = deadline.remaining(f"waiting for global profiler lock {path.name}")
    if not thread_lock.acquire(timeout=remaining):
        deadline.remaining(f"waiting for global profiler lock {path.name}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            while True:
                deadline.remaining(f"waiting for global profiler lock {path.name}")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(min(0.05, deadline.remaining()))
            try:
                yield time.monotonic() - started
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        thread_lock.release()


__all__ = ["host_global_lock"]
