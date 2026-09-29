"""Durable evidence capture for preflight and evaluation requests."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..protocol.schema import BoundEvaluateRequest, ReferenceRequest


AUDIT_SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_value(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return value


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class RequestAuditHandle:
    request_id: str
    operation: str
    request_dir: Path
    started_monotonic: float


class RequestAudit:
    """Persist exact requests and correlated execution events.

    The recorder is deliberately fail-closed. If an enabled audit cannot save
    the request before scheduling, the request must not reach an accelerator.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        manifest: dict[str, Any],
    ) -> None:
        root_path = Path(root).expanduser().resolve()
        root_path.mkdir(parents=True, exist_ok=True, mode=0o700)
        session_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + f"_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        )
        self.session_root = root_path / f"server_{session_id}"
        self.requests_root = self.session_root / "requests"
        self.requests_root.mkdir(parents=True, mode=0o700)
        self.events_path = self.session_root / "events.jsonl"
        self._lock = threading.Lock()
        self._write_json(
            self.session_root / "manifest.json",
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "session_id": session_id,
                "started_at": _utc_now(),
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                **manifest,
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "schema_version": AUDIT_SCHEMA_VERSION,
            "session_root": str(self.session_root),
        }

    def begin(
        self,
        operation: str,
        request: BoundEvaluateRequest | ReferenceRequest,
    ) -> RequestAuditHandle:
        payload = request.wire_payload()
        request_id = uuid.uuid4().hex
        request_dir = self.requests_root / request_id
        request_dir.mkdir(mode=0o700)
        handle = RequestAuditHandle(
            request_id=request_id,
            operation=operation,
            request_dir=request_dir,
            started_monotonic=time.monotonic(),
        )
        implementation = payload.get("implementation")
        definition_name = payload["binding"]["definition"]
        metadata = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "request_id": request_id,
            "operation": operation,
            "received_at": _utc_now(),
            "definition": definition_name,
            "request_sha256": _sha256(payload),
        }
        if implementation is not None:
            metadata.update(implementation=implementation["name"], candidate_sha256=_sha256(implementation))
        self._write_json(
            request_dir / "request.json",
            {
                "metadata": metadata,
                "payload": payload,
            },
        )
        self.event(handle, "request_persisted", **metadata)
        return handle

    def event(
        self,
        handle: RequestAuditHandle,
        event: str,
        **details: Any,
    ) -> None:
        record = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "recorded_at": _utc_now(),
            "elapsed_ms": round(
                (time.monotonic() - handle.started_monotonic) * 1000,
                3,
            ),
            "request_id": handle.request_id,
            "operation": handle.operation,
            "event": event,
            **details,
        }
        encoded = _canonical_json(record) + b"\n"
        with self._lock:
            with self.events_path.open("ab") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())

    def response(
        self,
        handle: RequestAuditHandle,
        response: Any,
        *,
        device: str,
    ) -> None:
        payload = _json_value(response)
        self._write_json(
            handle.request_dir / "response.json",
            {
                "recorded_at": _utc_now(),
                "device": device,
                "payload": payload,
            },
        )
        status = payload.get("status") if isinstance(payload, dict) else None
        self.event(
            handle,
            "response_persisted",
            device=device,
            status=status,
        )

    def failure(
        self,
        handle: RequestAuditHandle,
        error: BaseException,
        *,
        device: str | None,
    ) -> None:
        payload = {
            "recorded_at": _utc_now(),
            "device": device,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        self._write_json(handle.request_dir / "error.json", payload)
        self.event(handle, "request_failed", **payload)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        encoded = _canonical_json(value) + b"\n"
        try:
            with temporary.open("xb") as output:
                os.chmod(temporary, 0o600)
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
