"""Shared local state primitives; no dependency on CLI commands or models."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:  # pragma: no cover - production path is POSIX
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


STATE_DIRNAME = ".kernelgen"

_THREAD_LOCK = threading.RLock()


def state_home() -> Path:
    override = os.environ.get("KERNELGEN_CLI_HOME")
    return (
        Path(override).expanduser().resolve()
        if override
        else (Path.cwd() / STATE_DIRNAME).resolve()
    )


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCK:
        with path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def process_start_identity(pid: int) -> str | None:
    from kernelgen_client.process_utils import process_start_identity as identity
    return identity(pid)


def process_is_alive(pid: int, expected_start: str) -> bool:
    from kernelgen_client.process_utils import process_is_alive as alive
    return alive(pid, expected_start)
