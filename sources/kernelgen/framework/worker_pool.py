"""Process-safe weighted Coder leases and their existing local configuration."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from kernelgen.framework.local_state import (
    atomic_write_json, file_lock, process_is_alive, read_json, state_home,
)

DEFAULT_MAX_WORKERS = 1


class LeaseRecord(BaseModel):
    """One process-owned allocation from the local Coder budget."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    pid: int = Field(gt=0)
    process_start: str = Field(min_length=1)
    workspace: Path
    worker_pool: str = Field(min_length=1)
    weight: int = Field(ge=1)
    acquired_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def _config_path() -> Path:
    return state_home() / "config.json"


def normalize_worker_pool(eval_server: str) -> str:
    """Return a stable local capacity key for one KGS HTTP endpoint."""
    parsed = urlsplit(eval_server.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("eval server must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("eval server URL must not contain credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("eval server URL must not contain a path")
    host = parsed.hostname.casefold()
    if host == "localhost":
        host = "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    authority = f"[{host}]" if ":" in host else host
    return f"{parsed.scheme.casefold()}://{authority}:{port}"


def get_max_workers(eval_server: str) -> int:
    raw = read_json(_config_path(), {}) or {}
    pool = normalize_worker_pool(eval_server)
    maximums = raw.get("run", {}).get("max_workers", {})
    value = (
        maximums.get(pool, DEFAULT_MAX_WORKERS)
        if isinstance(maximums, dict)
        else DEFAULT_MAX_WORKERS
    )
    if not isinstance(value, int) or value < 1:
        raise ValueError("run.max-workers must be a positive integer")
    return value


def set_max_workers(eval_server: str, value: int) -> None:
    if value < 1:
        raise ValueError("run.max-workers must be a positive integer")
    with file_lock(state_home() / "config.lock"):
        raw = read_json(_config_path(), {}) or {}
        run_config = raw.setdefault("run", {})
        maximums = run_config.get("max_workers")
        if not isinstance(maximums, dict):
            maximums = {}
            run_config["max_workers"] = maximums
        maximums[normalize_worker_pool(eval_server)] = value
        atomic_write_json(_config_path(), raw)


class WorkerLeasePool:
    """A process-safe weighted semaphore persisted under the CLI home."""

    def __init__(self, eval_server: str) -> None:
        self.root = state_home()
        self.worker_pool = normalize_worker_pool(eval_server)
        self.path = self.root / "worker-leases.json"
        self.lock_path = self.root / "worker-leases.lock"

    def _load_live_unlocked(self) -> list[LeaseRecord]:
        raw = read_json(self.path, {}) or {}
        leases: list[LeaseRecord] = []
        for item in raw.get("leases", []):
            try:
                lease = LeaseRecord.model_validate(item)
            except ValueError:
                continue
            if process_is_alive(lease.pid, lease.process_start):
                leases.append(lease)
        return leases

    def _save_unlocked(self, leases: list[LeaseRecord]) -> None:
        atomic_write_json(
            self.path,
            {
                "schema_version": "1.0",
                "leases": [item.model_dump(mode="json") for item in leases],
            },
        )

    def acquire(self, lease: LeaseRecord, *, cancelled) -> None:
        while True:
            if cancelled():
                raise InterruptedError("run cancelled while waiting for worker capacity")
            with file_lock(self.lock_path):
                maximum = get_max_workers(self.worker_pool)
                if lease.weight > maximum:
                    raise ValueError(
                        f"run requests {lease.weight} workers but run.max-workers is {maximum}"
                    )
                leases = self._load_live_unlocked()
                used = sum(
                    item.weight
                    for item in leases
                    if item.worker_pool == self.worker_pool
                )
                if used + lease.weight <= maximum:
                    leases.append(lease)
                    self._save_unlocked(leases)
                    return
                self._save_unlocked(leases)
            time.sleep(0.25)

    def release(self, run_id: str, pid: int, process_start: str) -> None:
        with file_lock(self.lock_path):
            leases = self._load_live_unlocked()
            leases = [
                item
                for item in leases
                if not (
                    item.run_id == run_id
                    and item.pid == pid
                    and item.process_start == process_start
                )
            ]
            self._save_unlocked(leases)

    def snapshot(self) -> dict:
        with file_lock(self.lock_path):
            leases = self._load_live_unlocked()
            self._save_unlocked(leases)
            maximum = get_max_workers(self.worker_pool)
            used = sum(
                item.weight
                for item in leases
                if item.worker_pool == self.worker_pool
            )
            return {
                "worker_pool": self.worker_pool,
                "max_workers": maximum,
                "used_workers": used,
                "available_workers": max(maximum - used, 0),
                "leases": [
                    item.model_dump(mode="json")
                    for item in leases
                    if item.worker_pool == self.worker_pool
                ],
            }
