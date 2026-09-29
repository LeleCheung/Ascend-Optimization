"""Append-only workspace query and candidate records."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Type, TypeVar
from uuid import uuid4

import fcntl
from pydantic import BaseModel

from kernelgen.knowledge.models import RuntimeCandidate
from kernelgen.knowledge.contracts.runtime import (
    QueryRecord,
    RetrievalRecord,
)


RecordT = TypeVar("RecordT", bound=BaseModel)


def _append(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = record.model_dump_json() + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read(path: Path, model: Type[RecordT]) -> list[RecordT]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate(json.loads(line)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid record {path}:{line_number}: {exc}") from exc
    return records


class FileQueryAudit:
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: QueryRecord) -> None:
        _append(self.path, record)

    def read(self, query_id: str) -> QueryRecord:
        for record in reversed(_read(self.path, QueryRecord)):
            if record.query_id == query_id:
                return record
        raise KeyError(f"query id not found: {query_id}")


class FileRetrievalAudit:
    def __init__(
        self,
        path: Path,
        *,
        legacy_query_path: Path | None = None,
    ):
        self.path = Path(path)
        self.legacy_query_path = (
            Path(legacy_query_path)
            if legacy_query_path is not None
            else None
        )

    def append(self, record: RetrievalRecord) -> None:
        _append(self.path, record)

    def read(self, query_id: str) -> RetrievalRecord:
        return self.read_query(query_id, operation="query_knowledge")

    def read_query(
        self,
        query_id: str,
        *,
        operation: str,
    ) -> RetrievalRecord:
        for record in reversed(_read(self.path, RetrievalRecord)):
            if (
                record.query_id == query_id
                and record.operation == operation
                and record.status == "success"
            ):
                return record
        legacy = self._read_legacy_query(query_id, operation)
        if legacy is not None:
            return legacy
        raise KeyError(
            f"{operation} retrieval query id not found: {query_id}"
        )

    def read_all(self) -> list[RetrievalRecord]:
        return _read(self.path, RetrievalRecord)

    def _read_legacy_query(
        self,
        query_id: str,
        operation: str,
    ) -> RetrievalRecord | None:
        if (
            operation != "query_knowledge"
            or self.legacy_query_path is None
        ):
            return None
        for record in reversed(
            _read(self.legacy_query_path, QueryRecord)
        ):
            if record.query_id != query_id:
                continue
            return RetrievalRecord(
                event_id=record.query_id,
                operation="query_knowledge",
                origin=record.origin,
                status="success",
                query_id=record.query_id,
                snapshot=record.snapshot,
                request={
                    "phase": record.phase,
                    "task": record.task,
                    "question": record.question,
                },
                returned_refs=record.returned_refs,
                result_levels=record.result_levels,
                created_at=record.created_at,
            )
        return None


class FileCandidateOutbox:
    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, candidate: RuntimeCandidate) -> None:
        existing = {item.candidate_id for item in self.read_all()}
        if candidate.candidate_id not in existing:
            _append(self.path, candidate)

    def read_all(self) -> list[RuntimeCandidate]:
        return _read(self.path, RuntimeCandidate)


def retrieval_record(
    *,
    operation: str,
    origin: str,
    status: str,
    event_id: str = "",
    query_id: str = "",
    snapshot: str = "",
    request: dict[str, object] | None = None,
    returned_refs: list[str] | None = None,
    returned_sources: list[str] | None = None,
    result_levels: dict[str, str] | None = None,
    error: str = "",
) -> RetrievalRecord:
    """Build one timestamped retrieval event."""

    return RetrievalRecord(
        event_id=event_id or f"retrieval:{uuid4().hex}",
        operation=operation,
        origin=origin,
        status=status,
        query_id=query_id,
        snapshot=snapshot,
        request=request or {},
        returned_refs=returned_refs or [],
        returned_sources=returned_sources or [],
        result_levels=result_levels or {},
        error=error,
        created_at=datetime.now(timezone.utc),
    )
