"""Durable, provider-neutral control and observation for one KernelGen run.

The optimization ledger remains authoritative for measured round facts.  This
module owns the orthogonal control-plane facts that a caller needs while a run
is active: cooperative cancellation, a lightweight progress projection, and an
append-only stream of structured events.

The default implementation is workspace-backed so a CLI process, an MCP child,
and a later service worker can cooperate without sharing Python objects.  Other
frontends may implement the protocols below without changing Workflow code.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Optional, Protocol, runtime_checkable
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field
from kernelgen.framework.progress_schema import ProgressKind, RoundProgress, EpochProgress

try:  # pragma: no cover - exercised on the Linux production/test path
    import fcntl
except ImportError:  # pragma: no cover - fallback for non-POSIX development
    fcntl = None


RUN_CONTROL_SCHEMA_VERSION = "1.0"
RUN_EVENT_SCHEMA_VERSION = "1.0"
RUN_PROGRESS_SCHEMA_VERSION = "1.0"

RUN_CONTROL_DIRECTORY = ".kernelgen"
RUN_CONTROL_FILENAME = "run-control.json"
RUN_CONTROL_REFERENCE_FILENAME = "run-control-ref.json"
RUN_EVENTS_FILENAME = "run-events.jsonl"
RUN_EVENT_SEQUENCE_FILENAME = "run-events.sequence"
RUN_PROGRESS_FILENAME = "run-progress.json"
RUN_ROOT_MARKER_FILENAME = "run-root.json"
ACTIVE_SERVER_OPERATIONS_FILENAME = "active-server-operations.json"
ACTIVE_SERVER_OPERATIONS_SCHEMA_VERSION = "1.0"

# Kept in the response model for the stable status Schema, never persisted.
# CLI status/history obtain these measured facts from the ledger.
LEDGER_PROGRESS_FIELDS = {
    "current_round", "completed_rounds", "best_round", "best_geo_mean",
    "last_evaluation_status",
}

_PROCESS_LOCK = threading.RLock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunState(str, Enum):
    """Stable lifecycle states shared by CLI and future service adapters."""

    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"


class RunEvent(BaseModel):
    """One immutable event emitted by Workflow, Runtime, or an MCP tool."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = RUN_EVENT_SCHEMA_VERSION
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1)
    sequence: int = Field(default=0, ge=0)
    recorded_at: datetime = Field(default_factory=_utc_now)
    event_type: str = Field(min_length=1)
    source: str = Field(default="kernelgen", min_length=1)
    scope: str = ""
    stage: str = ""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    visibility: Literal["USER", "DEBUG", "INTERNAL"] = "USER"
    message: str = ""
    data: Dict[str, Any] = Field(default_factory=dict)


class ProgressRecord(RoundProgress, EpochProgress):
    """Progress fields for a run or one nested agent scope."""

    model_config = ConfigDict(extra="forbid")

    progress_kind: ProgressKind = "basic"

    state: RunState = RunState.PENDING
    stage: str = ""
    mode: str = ""
    stop_reason: str = ""
    message: str = ""
    updated_at: datetime = Field(default_factory=_utc_now)


class RunProgressSnapshot(ProgressRecord):
    """Durable progress projection; measured values are copied from the ledger."""

    schema_version: Literal["1.0"] = RUN_PROGRESS_SCHEMA_VERSION
    revision: int = Field(default=0, ge=0)
    cancel_requested: bool = False
    scopes: Dict[str, ProgressRecord] = Field(default_factory=dict)


class CancellationState(BaseModel):
    """Durable cancellation request shared across process boundaries."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = RUN_CONTROL_SCHEMA_VERSION
    requested: bool = False
    requested_at: Optional[datetime] = None
    reason: str = ""
    generation: int = Field(default=0, ge=0)


class ActiveServerOperation(BaseModel):
    """One cancellable KGS call currently owned by this run."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(min_length=1, max_length=128)
    kind: Literal["preflight", "evaluate", "profile", "reference"]
    server_url: str = Field(min_length=1)
    scope: str = ""
    started_at: datetime = Field(default_factory=_utc_now)


class ActiveServerOperations(BaseModel):
    """Atomic workspace projection used by ``kg cancel``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = ACTIVE_SERVER_OPERATIONS_SCHEMA_VERSION
    operations: Dict[str, ActiveServerOperation] = Field(default_factory=dict)


class RunCancelled(RuntimeError):
    """Raised only at a cooperative boundary after cancellation was requested."""

    def __init__(self, state: CancellationState, *, stage: str = "") -> None:
        self.state = state
        self.stage = stage
        detail = state.reason or "cancellation requested"
        if stage:
            detail = f"{detail} (safe point: {stage})"
        super().__init__(detail)


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: RunEvent) -> RunEvent:
        """Persist one event and return it with its assigned sequence."""


@runtime_checkable
class EventStream(EventSink, Protocol):
    def read_events(
        self,
        *,
        after_sequence: int = 0,
        limit: int | None = None,
    ) -> list[RunEvent]:
        """Replay persisted events after an exclusive sequence cursor."""


@runtime_checkable
class CancellationToken(Protocol):
    def cancellation_state(self) -> CancellationState:
        """Return the latest durable cancellation state."""

    def is_cancellation_requested(self) -> bool:
        """Return whether work must stop at the next safe point."""

    def observe_cancellation(self, stage: str = "") -> Optional[CancellationState]:
        """Record and return a request without raising, for tool responses."""

    def checkpoint(self, stage: str = "") -> None:
        """Raise :class:`RunCancelled` if cancellation was requested."""


@runtime_checkable
class CancellationController(CancellationToken, Protocol):
    def request_cancel(self, reason: str = "") -> CancellationState:
        """Persist an idempotent request for cooperative cancellation."""

    def clear_cancellation(
        self,
        *,
        expected_generation: int | None = None,
    ) -> CancellationState:
        """Clear a request before explicitly resuming a run."""

    def acknowledge_cancellation(
        self,
        *,
        stage: str = "",
    ) -> RunProgressSnapshot:
        """Mark cancellation terminal after work has reached a safe point."""


@runtime_checkable
class ProgressStore(Protocol):
    def progress(self) -> RunProgressSnapshot:
        """Return the latest durable progress projection."""

    def update_progress(self, **changes: Any) -> RunProgressSnapshot:
        """Atomically update progress for the current scope."""


@runtime_checkable
class RunControl(EventStream, CancellationController, ProgressStore, Protocol):
    def record_event(
        self,
        event_type: str,
        *,
        message: str = "",
        stage: str = "",
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
        visibility: Literal["USER", "DEBUG", "INTERNAL"] = "USER",
        data: Optional[Dict[str, Any]] = None,
        source: str | None = None,
        scope: str | None = None,
    ) -> RunEvent:
        """Emit a structured event without constructing the model manually."""


class WorkspaceRunControl(RunControl):
    """File-backed control plane rooted in one run workspace.

    Nested workspaces discover the nearest marked run root.  A caller allocating
    a workspace outside that directory can use :meth:`link_workspace` to bind it
    explicitly to the same cancellation and event stream.
    """

    def __init__(
        self,
        workspace: os.PathLike[str] | str,
        *,
        source: str = "kernelgen",
        scope: str | None = None,
        root_workspace: os.PathLike[str] | str | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.source = source
        discovered_root, discovered_scope = self._discover_root(
            self.workspace,
            explicit_root=(
                Path(root_workspace).expanduser().resolve()
                if root_workspace is not None
                else None
            ),
        )
        self.root_workspace = discovered_root
        self.scope = discovered_scope if scope is None else scope
        self._state_dir = self.root_workspace / RUN_CONTROL_DIRECTORY
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_root_marker()

    @property
    def control_path(self) -> Path:
        return self._state_dir / RUN_CONTROL_FILENAME

    @property
    def events_path(self) -> Path:
        return self._state_dir / RUN_EVENTS_FILENAME

    @property
    def progress_path(self) -> Path:
        return self._state_dir / RUN_PROGRESS_FILENAME

    @property
    def active_server_operations_path(self) -> Path:
        return self._state_dir / ACTIVE_SERVER_OPERATIONS_FILENAME

    def register_server_operation(
        self,
        kind: Literal["preflight", "evaluate", "profile", "reference"],
        server_url: str,
        *,
        operation_id: str | None = None,
    ) -> ActiveServerOperation:
        """Persist a KGS operation before issuing its HTTP request."""
        parsed = urlsplit(server_url)
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "server_url must not contain credentials, query, or fragment"
            )
        resolved_id = operation_id or uuid.uuid4().hex
        operation = ActiveServerOperation(
            operation_id=resolved_id,
            kind=kind,
            server_url=server_url.rstrip("/"),
            scope=self.scope,
        )
        with self._locked():
            cancellation = self._read_cancellation_unlocked()
            if cancellation.requested:
                raise RunCancelled(cancellation, stage=f"BEFORE_{kind.upper()}")
            active = self._read_active_server_operations_unlocked()
            if resolved_id in active.operations:
                raise ValueError(
                    f"server operation {resolved_id!r} is already active"
                )
            updated = active.model_copy(
                update={
                    "operations": {
                        **active.operations,
                        resolved_id: operation,
                    }
                }
            )
            self._atomic_write_json(
                self.active_server_operations_path,
                updated.model_dump(mode="json"),
            )
        self.record_event(
            "SERVER_OPERATION_STARTED",
            message=f"KGS {kind} operation started",
            stage=kind.upper(),
            visibility="DEBUG",
            data={
                "operation_id": resolved_id,
                "operation": kind,
                "server_url": operation.server_url,
            },
        )
        return operation

    def unregister_server_operation(self, operation_id: str) -> None:
        """Remove a completed KGS operation from the active projection."""
        removed: ActiveServerOperation | None = None
        with self._locked():
            active = self._read_active_server_operations_unlocked()
            removed = active.operations.get(operation_id)
            if removed is None:
                return
            operations = dict(active.operations)
            operations.pop(operation_id, None)
            updated = active.model_copy(update={"operations": operations})
            self._atomic_write_json(
                self.active_server_operations_path,
                updated.model_dump(mode="json"),
            )
        self.record_event(
            "SERVER_OPERATION_FINISHED",
            message=f"KGS {removed.kind} operation finished",
            stage=removed.kind.upper(),
            visibility="DEBUG",
            data={
                "operation_id": operation_id,
                "operation": removed.kind,
                "server_url": removed.server_url,
            },
        )

    def active_server_operations(self) -> list[ActiveServerOperation]:
        with self._locked():
            active = self._read_active_server_operations_unlocked()
        return sorted(
            active.operations.values(),
            key=lambda operation: (operation.started_at, operation.operation_id),
        )

    def emit(self, event: RunEvent) -> RunEvent:
        """Append an event with a run-global, monotonically increasing sequence."""
        with self._locked():
            sequence_path = self._state_dir / RUN_EVENT_SEQUENCE_FILENAME
            try:
                sequence = int(sequence_path.read_text(encoding="utf-8")) + 1
            except (FileNotFoundError, OSError, ValueError):
                sequence = self._last_event_sequence() + 1
            persisted = event.model_copy(
                update={
                    "sequence": sequence,
                    "source": event.source or self.source,
                    "scope": event.scope or self.scope,
                }
            )
            self._atomic_write_text(sequence_path, str(sequence))
            encoded = persisted.model_dump_json() + "\n"
            with self.events_path.open("a", encoding="utf-8") as stream:
                os.chmod(self.events_path, 0o600)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        return persisted

    def record_event(
        self,
        event_type: str,
        *,
        message: str = "",
        stage: str = "",
        level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
        visibility: Literal["USER", "DEBUG", "INTERNAL"] = "USER",
        data: Optional[Dict[str, Any]] = None,
        source: str | None = None,
        scope: str | None = None,
    ) -> RunEvent:
        """Convenience wrapper used by KernelGen internals."""
        return self.emit(
            RunEvent(
                event_type=event_type,
                source=source or self.source,
                scope=self.scope if scope is None else scope,
                stage=stage,
                level=level,
                visibility=visibility,
                message=message,
                data=data or {},
            )
        )

    def read_events(
        self,
        *,
        after_sequence: int = 0,
        limit: int | None = None,
    ) -> list[RunEvent]:
        """Read replayable events without treating a truncated tail as state."""
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        events: list[RunEvent] = []
        try:
            stream = self.events_path.open(
                "r", encoding="utf-8", errors="strict"
            )
        except FileNotFoundError:
            return events
        with stream:
            for line in stream:
                try:
                    event = RunEvent.model_validate_json(line)
                except ValueError:
                    continue
                if event.sequence <= after_sequence:
                    continue
                events.append(event)
                if limit is not None and len(events) >= limit:
                    break
        return events

    def cancellation_state(self) -> CancellationState:
        with self._locked():
            return self._read_cancellation_unlocked()

    def is_cancellation_requested(self) -> bool:
        return self.cancellation_state().requested

    def request_cancel(self, reason: str = "") -> CancellationState:
        """Persist an idempotent request; running work observes it at safe points."""
        created = False
        with self._locked():
            current = self._read_cancellation_unlocked()
            if current.requested:
                return current
            state = CancellationState(
                requested=True,
                requested_at=_utc_now(),
                reason=reason.strip(),
                generation=current.generation + 1,
            )
            self._atomic_write_json(self.control_path, state.model_dump(mode="json"))
            self._update_progress_unlocked(
                {
                    "state": RunState.CANCEL_REQUESTED,
                    "message": state.reason or "cancellation requested",
                },
                root=True,
            )
            created = True
        if created:
            self.record_event(
                "CANCEL_REQUESTED",
                message=state.reason or "Cancellation requested",
                stage="CANCELLING",
                data={"generation": state.generation},
            )
        return state

    def clear_cancellation(
        self,
        *,
        expected_generation: int | None = None,
    ) -> CancellationState:
        """Clear a prior request before an explicit resume operation."""
        changed = False
        with self._locked():
            current = self._read_cancellation_unlocked()
            if (
                expected_generation is not None
                and current.generation != expected_generation
            ):
                raise ValueError(
                    "cancellation generation changed; refusing stale clear"
                )
            if not current.requested:
                return current
            state = CancellationState(generation=current.generation + 1)
            self._atomic_write_json(self.control_path, state.model_dump(mode="json"))
            changed = True
        if changed:
            self.record_event(
                "CANCEL_CLEARED",
                message="Cancellation cleared for an explicit resume",
                visibility="INTERNAL",
                data={"generation": state.generation},
            )
        return state

    def checkpoint(self, stage: str = "") -> None:
        state = self.observe_cancellation(stage)
        if state is None:
            return

        raise RunCancelled(state, stage=stage)

    def observe_cancellation(
        self,
        stage: str = "",
    ) -> CancellationState | None:
        state = self.cancellation_state()
        if not state.requested:
            return None
        self.update_progress(
            state=RunState.CANCEL_REQUESTED,
            stage=stage or "CANCELLING",
            stop_reason=state.reason or "user_cancelled",
        )
        self.record_event(
            "CANCEL_OBSERVED",
            message=state.reason or "Cancellation observed at a safe point",
            stage=stage,
            data={"generation": state.generation},
        )
        return state

    def acknowledge_cancellation(self, *, stage: str = "") -> RunProgressSnapshot:
        """Mark cancellation terminal after the caller has stopped launching work."""
        current = self.progress()
        current_record: ProgressRecord = current
        if self.scope:
            current_record = current.scopes.get(self.scope, ProgressRecord())
        if current_record.state == RunState.CANCELLED:
            return current
        snapshot = self.update_progress(
            state=RunState.CANCELLED,
            stage=stage or "CANCELLED",
            stop_reason=self.cancellation_state().reason or "user_cancelled",
        )
        self.record_event(
            "RUN_CANCELLED",
            message=snapshot.stop_reason,
            stage=snapshot.stage,
        )
        return snapshot

    def progress(self) -> RunProgressSnapshot:
        with self._locked():
            return self._read_progress_unlocked()

    def update_progress(self, **changes: Any) -> RunProgressSnapshot:
        """Update the root record or this controller's nested scope."""
        with self._locked():
            return self._update_progress_unlocked(changes, root=not self.scope)

    def link_workspace(
        self,
        workspace: os.PathLike[str] | str,
        *,
        scope: str,
    ) -> "WorkspaceRunControl":
        """Bind an allocated workspace to this run, including non-descendants."""
        child = Path(workspace).expanduser().resolve()
        reference_path = (
            child / RUN_CONTROL_DIRECTORY / RUN_CONTROL_REFERENCE_FILENAME
        )
        self._atomic_write_json(
            reference_path,
            {
                "schema_version": RUN_CONTROL_SCHEMA_VERSION,
                "root_workspace": str(self.root_workspace),
                "scope": scope,
            },
        )
        return WorkspaceRunControl(child, source=self.source)

    def _discover_root(
        self,
        workspace: Path,
        *,
        explicit_root: Path | None,
    ) -> tuple[Path, str]:
        reference_path = (
            workspace / RUN_CONTROL_DIRECTORY / RUN_CONTROL_REFERENCE_FILENAME
        )
        try:
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            reference = None
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"invalid run-control reference: {reference_path}") from exc
        if reference is not None:
            if reference.get("schema_version") != RUN_CONTROL_SCHEMA_VERSION:
                raise ValueError(f"unsupported run-control reference: {reference_path}")
            root_raw = reference.get("root_workspace")
            if not isinstance(root_raw, str) or not Path(root_raw).is_absolute():
                raise ValueError(f"invalid run-control root: {reference_path}")
            return Path(root_raw).resolve(), str(reference.get("scope") or "")
        if explicit_root is not None:
            return explicit_root, self._relative_scope(explicit_root, workspace)
        for candidate in (workspace, *workspace.parents):
            marker = (
                candidate / RUN_CONTROL_DIRECTORY / RUN_ROOT_MARKER_FILENAME
            )
            if marker.is_file():
                return candidate, self._relative_scope(candidate, workspace)
        return workspace, ""

    @staticmethod
    def _relative_scope(root: Path, workspace: Path) -> str:
        try:
            relative = workspace.relative_to(root)
        except ValueError:
            return workspace.name
        return "" if relative == Path(".") else relative.as_posix()

    def _ensure_root_marker(self) -> None:
        marker = self._state_dir / RUN_ROOT_MARKER_FILENAME
        if marker.exists():
            return
        self._atomic_write_json(
            marker,
            {
                "schema_version": RUN_CONTROL_SCHEMA_VERSION,
                "root_workspace": str(self.root_workspace),
            },
        )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._state_dir / ".run-control.lock"
        with _PROCESS_LOCK:
            with lock_path.open("a+") as lock:
                os.chmod(lock_path, 0o600)
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _read_cancellation_unlocked(self) -> CancellationState:
        try:
            return CancellationState.model_validate_json(
                self.control_path.read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            return CancellationState()
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid run control state: {self.control_path}") from exc

    def _read_progress_unlocked(self) -> RunProgressSnapshot:
        try:
            payload = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("scopes", {}), dict):
                raise ValueError("progress must contain an object and scope mapping")
            for record in [payload, *payload.get("scopes", {}).values()]:
                if not isinstance(record, dict):
                    raise ValueError("progress scope must be an object")
                for field in LEDGER_PROGRESS_FIELDS:
                    record.pop(field, None)
            snapshot = RunProgressSnapshot.model_validate(payload)
        except FileNotFoundError:
            snapshot = RunProgressSnapshot(progress_kind="basic")
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid run progress state: {self.progress_path}") from exc
        cancellation = self._read_cancellation_unlocked()
        if snapshot.cancel_requested != cancellation.requested:
            snapshot = snapshot.model_copy(
                update={"cancel_requested": cancellation.requested}
            )
        return snapshot

    def _read_active_server_operations_unlocked(self) -> ActiveServerOperations:
        try:
            return ActiveServerOperations.model_validate_json(
                self.active_server_operations_path.read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            return ActiveServerOperations()
        except (OSError, ValueError) as exc:
            raise ValueError(
                "invalid active Server operation state: "
                f"{self.active_server_operations_path}"
            ) from exc

    def _update_progress_unlocked(
        self,
        changes: Dict[str, Any],
        *,
        root: bool,
    ) -> RunProgressSnapshot:
        snapshot = self._read_progress_unlocked()
        forbidden = {
            "schema_version",
            "revision",
            "cancel_requested",
            "scopes",
            "updated_at",
        }
        forbidden |= LEDGER_PROGRESS_FIELDS
        unknown = set(changes) - set(ProgressRecord.model_fields)
        if unknown or forbidden & set(changes):
            invalid = sorted(unknown | (forbidden & set(changes)))
            raise ValueError(f"unsupported progress field(s): {', '.join(invalid)}")
        now = _utc_now()
        if root:
            base = ProgressRecord.model_validate(
                snapshot.model_dump(
                    exclude={
                        "schema_version",
                        "revision",
                        "cancel_requested",
                        "scopes",
                    }
                )
            )
            updated = base.model_copy(update={**changes, "updated_at": now})
            values = updated.model_dump()
            values.update(
                schema_version=RUN_PROGRESS_SCHEMA_VERSION,
                revision=snapshot.revision + 1,
                cancel_requested=self._read_cancellation_unlocked().requested,
                scopes=snapshot.scopes,
            )
            snapshot = RunProgressSnapshot.model_validate(values)
        else:
            current = snapshot.scopes.get(self.scope, ProgressRecord(progress_kind="basic"))
            updated = ProgressRecord.model_validate(
                {
                    **current.model_dump(),
                    **changes,
                    "updated_at": now,
                }
            )
            values = snapshot.model_dump()
            values.update(
                revision=snapshot.revision + 1,
                updated_at=now,
                cancel_requested=self._read_cancellation_unlocked().requested,
                scopes={**snapshot.scopes, self.scope: updated},
            )
            snapshot = RunProgressSnapshot.model_validate(values)
        self._atomic_write_json(
            self.progress_path,
            snapshot.model_dump(mode="json", exclude={
                **{field: True for field in LEDGER_PROGRESS_FIELDS},
                "cancel_requested": True,
                "scopes": {"__all__": LEDGER_PROGRESS_FIELDS},
            }),
        )
        return snapshot

    def _last_event_sequence(self) -> int:
        try:
            with self.events_path.open("rb") as stream:
                lines = stream.readlines()
        except FileNotFoundError:
            return 0
        for line in reversed(lines):
            try:
                return RunEvent.model_validate_json(line).sequence
            except ValueError:
                continue
        return 0

    @staticmethod
    def _atomic_write_json(path: Path, value: Any) -> None:
        WorkspaceRunControl._atomic_write_text(
            path,
            json.dumps(value, indent=2, ensure_ascii=False),
        )

    @staticmethod
    def _atomic_write_text(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def cooperative_cancel_result(
    token: CancellationToken,
    *,
    stage: str,
) -> dict[str, Any] | None:
    """Return the common MCP response when cancellation reaches a safe point."""
    state = token.observe_cancellation(stage)
    if state is None:
        return None
    return {
        "status": "RUN_CANCELLED",
        "cancel_requested": True,
        "reason_code": "user_cancelled",
        "reason": state.reason or "cancellation requested by the run owner",
        "stage": stage,
        "instruction": (
            "STOP NOW. Do not start another Agent or Server call. "
            "Return the final report using only results already persisted."
        ),
    }


__all__ = [
    "ActiveServerOperation",
    "ActiveServerOperations",
    "CancellationController",
    "CancellationState",
    "CancellationToken",
    "EventSink",
    "EventStream",
    "ProgressRecord",
    "ProgressStore",
    "RunCancelled",
    "RunControl",
    "RunEvent",
    "RunProgressSnapshot",
    "RunState",
    "WorkspaceRunControl",
    "cooperative_cancel_result",
]
