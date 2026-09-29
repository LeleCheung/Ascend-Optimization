"""In-memory control plane for cancellable Server operations."""

from __future__ import annotations

import asyncio
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Literal


OperationKind = Literal["preflight", "evaluate", "profile", "reference"]
OperationState = Literal[
    "QUEUED",
    "RUNNING",
    "CANCEL_REQUESTED",
    "CANCELLED",
    "SUCCEEDED",
    "FAILED",
]

_TERMINAL_STATES = {"CANCELLED", "SUCCEEDED", "FAILED"}
_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CURRENT_OPERATION: ContextVar[OperationControl | None] = ContextVar(
    "kernelgen_server_operation",
    default=None,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OperationCancelled(RuntimeError):
    """A registered operation observed a client cancellation request."""

    def __init__(self, operation_id: str, kind: OperationKind) -> None:
        self.operation_id = operation_id
        self.kind = kind
        super().__init__(f"{kind} operation {operation_id} was cancelled")


class OperationConflictError(ValueError):
    """An operation identifier is already registered."""


@dataclass
class OperationControl:
    operation_id: str
    kind: OperationKind
    created_at: str = field(default_factory=_utc_now)
    state: OperationState = "QUEUED"
    device: str | None = None
    finished_at: str | None = None
    error: str = ""
    _cancel_event: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )
    _finished_monotonic: float | None = field(default=None, repr=False)

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel_event

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    @property
    def terminal(self) -> bool:
        with self._lock:
            return self.state in _TERMINAL_STATES

    def request_cancel(self) -> None:
        with self._lock:
            if self.state in _TERMINAL_STATES:
                return
            self._cancel_event.set()
            self.state = "CANCEL_REQUESTED"

    def raise_if_cancelled(self) -> None:
        if self.cancel_requested:
            raise OperationCancelled(self.operation_id, self.kind)

    def mark_running(self, device: str) -> None:
        with self._lock:
            if self._cancel_event.is_set():
                raise OperationCancelled(self.operation_id, self.kind)
            self.state = "RUNNING"
            self.device = device

    def finish(self, state: Literal["CANCELLED", "SUCCEEDED", "FAILED"], error: str = "") -> None:
        with self._lock:
            self.state = state
            self.finished_at = _utc_now()
            self.error = error
            self._finished_monotonic = time.monotonic()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "operation_id": self.operation_id,
                "kind": self.kind,
                "state": self.state,
                "cancel_requested": self._cancel_event.is_set(),
                "device": self.device,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "error": self.error,
            }


class OperationRegistry:
    """Keep active operations and a bounded window of terminal results."""

    def __init__(
        self,
        *,
        retention_seconds: float = 60 * 60,
        max_records: int = 2048,
    ) -> None:
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self._retention_seconds = retention_seconds
        self._max_records = max_records
        self._operations: dict[str, OperationControl] = {}
        self._lock = threading.RLock()

    def create(
        self,
        kind: OperationKind,
        operation_id: str | None = None,
    ) -> OperationControl:
        resolved = operation_id or uuid.uuid4().hex
        if not _OPERATION_ID.fullmatch(resolved):
            raise ValueError(
                "operation id must be 1-128 ASCII letters, digits, '.', '_', or '-'"
            )
        with self._lock:
            self._prune_unlocked()
            if resolved in self._operations:
                raise OperationConflictError(
                    f"operation id {resolved!r} is already registered"
                )
            operation = OperationControl(resolved, kind)
            self._operations[resolved] = operation
            self._trim_unlocked()
            return operation

    def get(self, operation_id: str) -> OperationControl:
        with self._lock:
            self._prune_unlocked()
            try:
                return self._operations[operation_id]
            except KeyError as exc:
                raise KeyError(operation_id) from exc

    def cancel(self, operation_id: str) -> OperationControl:
        operation = self.get(operation_id)
        operation.request_cancel()
        return operation

    def _prune_unlocked(self) -> None:
        cutoff = time.monotonic() - self._retention_seconds
        expired = [
            operation_id
            for operation_id, operation in self._operations.items()
            if operation._finished_monotonic is not None
            and operation._finished_monotonic < cutoff
        ]
        for operation_id in expired:
            self._operations.pop(operation_id, None)

    def _trim_unlocked(self) -> None:
        overflow = len(self._operations) - self._max_records
        if overflow <= 0:
            return
        terminal = sorted(
            (
                operation._finished_monotonic,
                operation_id,
            )
            for operation_id, operation in self._operations.items()
            if operation._finished_monotonic is not None
        )
        for _, operation_id in terminal[:overflow]:
            self._operations.pop(operation_id, None)


@contextmanager
def bind_operation(operation: OperationControl | None) -> Iterator[None]:
    token = _CURRENT_OPERATION.set(operation)
    try:
        yield
    finally:
        _CURRENT_OPERATION.reset(token)


def current_operation() -> OperationControl | None:
    return _CURRENT_OPERATION.get()


def raise_if_operation_cancelled() -> None:
    operation = current_operation()
    if operation is not None:
        operation.raise_if_cancelled()


async def acquire_device_or_cancel(
    device_pool: Any,
    operation: OperationControl | None,
) -> str:
    """Acquire one device without leaving a cancelled queue waiter behind."""
    if operation is None:
        return await device_pool.acquire()
    operation.raise_if_cancelled()
    acquisition = asyncio.create_task(device_pool.acquire())
    while True:
        done, _ = await asyncio.wait({acquisition}, timeout=0.05)
        if operation.cancel_requested:
            acquisition.cancel()
            outcomes = await asyncio.gather(
                acquisition,
                return_exceptions=True,
            )
            if outcomes and isinstance(outcomes[0], str):
                await asyncio.shield(device_pool.release(outcomes[0]))
            operation.raise_if_cancelled()
        if done:
            return acquisition.result()


__all__ = [
    "OperationCancelled",
    "OperationConflictError",
    "OperationControl",
    "OperationKind",
    "OperationRegistry",
    "acquire_device_or_cancel",
    "bind_operation",
    "current_operation",
    "raise_if_operation_cancelled",
]
