"""Lightweight protocol client with no accelerator-runtime imports."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import requests
from requests.adapters import HTTPAdapter

from .debug.models import DebugJob, DebugJobRequest
from .operator_bundles import (
    OPERATOR_BUNDLE_MEDIA_TYPE,
    OperatorBundleInfo,
    normalize_operator_bundle_sha256,
    pack_operator_bundle,
)
from .profiling.models import ProfileRequest, ProfileResult
from .protocol.schema import (
    AdapterManifest,
    BoundEvaluateRequest,
    EvaluateResponse,
    InspectRequest,
    OperatorContract,
    ReferenceRequest,
    ReferenceResult,
)


class ServerError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class OperationCancelledError(ServerError):
    """The Server stopped a registered operation after client cancellation."""

    def __init__(self, operation_id: str, operation: str) -> None:
        self.operation_id = operation_id
        self.operation = operation
        super().__init__(f"{operation} operation {operation_id} was cancelled", status_code=409)


EVALUATION_MAX_ATTEMPTS = 1
EVALUATION_TRANSPORT_GRACE_SECONDS = 90
_HTTP_POOL_SIZE = 16


def _build_session() -> requests.Session:
    """Build one process-local HTTP pool without implicit request retries."""

    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=_HTTP_POOL_SIZE,
        pool_maxsize=_HTTP_POOL_SIZE,
        max_retries=0,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_SESSION = _build_session()


def _raise_response_error(response: Any) -> None:
    if response.status_code < 400:
        return
    if response.status_code == 409:
        try:
            detail = response.json().get("detail", {})
        except (AttributeError, ValueError):
            detail = {}
        if (
            isinstance(detail, dict)
            and detail.get("code") == "OPERATION_CANCELLED"
        ):
            raise OperationCancelledError(
                str(detail.get("operation_id") or "unknown"),
                str(detail.get("operation") or "server"),
            )
    raise ServerError(
        f"server returned HTTP {response.status_code}: {response.text}",
        status_code=response.status_code,
    )


def _evaluation_transport_timeout(
    request: BoundEvaluateRequest | ReferenceRequest,
) -> float:
    """Cover one isolated attempt, its strong recovery probe, and HTTP overhead."""
    return (
        EVALUATION_MAX_ATTEMPTS * request.settings.timeout_seconds
        + EVALUATION_TRANSPORT_GRACE_SECONDS
    )


def _request_json(
    server_url: str,
    path: str,
    *,
    method: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float,
) -> Any:
    url = server_url.rstrip("/") + path
    if query:
        url += "?" + urlencode(query)
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    try:
        with _SESSION.request(
            method,
            url,
            data=data,
            headers={
                **({"Content-Type": "application/json"} if data is not None else {}),
                **(headers or {}),
            }
            or None,
            timeout=timeout,
            allow_redirects=False,
        ) as response:
            _raise_response_error(response)
            try:
                return response.json()
            except ValueError as exc:
                raise ServerError(
                    "evaluation server returned invalid JSON"
                ) from exc
    except requests.RequestException as exc:
        raise ServerError(f"cannot reach evaluation server: {exc}") from exc


def _post(
    server_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: float,
    *,
    operation_id: str | None = None,
) -> Any:
    return _request_json(
        server_url,
        path,
        method="POST",
        payload=payload,
        headers=(
            {"X-KernelGen-Operation-Id": operation_id}
            if operation_id is not None
            else None
        ),
        timeout=timeout,
    )


def status(server_url: str, timeout: float = 45) -> dict[str, Any]:
    value = _request_json(
        server_url,
        "/status",
        method="GET",
        timeout=timeout,
    )
    if (
        not isinstance(value, dict)
        or value.get("status") not in {"ok", "degraded"}
    ):
        raise ServerError(f"invalid status response: {value!r}")
    return value


def operator_bundle_exists(
    bundle_sha256: str,
    server_url: str,
    timeout: float = 30,
) -> bool:
    """Return whether one content-addressed operator bundle is installed."""

    digest = normalize_operator_bundle_sha256(bundle_sha256)
    url = server_url.rstrip("/") + f"/operator-bundles/{digest}"
    try:
        with _SESSION.request(
            "HEAD",
            url,
            timeout=timeout,
            allow_redirects=False,
        ) as response:
            if response.status_code == 404:
                return False
            _raise_response_error(response)
            return True
    except requests.RequestException as exc:
        raise ServerError(f"cannot reach evaluation server: {exc}") from exc


def get_operator_bundle(
    bundle_sha256: str,
    server_url: str,
    timeout: float = 30,
) -> OperatorBundleInfo | None:
    """Return installed bundle metadata, or ``None`` when it is absent."""

    digest = normalize_operator_bundle_sha256(bundle_sha256)
    url = server_url.rstrip("/") + f"/operator-bundles/{digest}"
    try:
        with _SESSION.request(
            "GET",
            url,
            timeout=timeout,
            allow_redirects=False,
        ) as response:
            if response.status_code == 404:
                return None
            _raise_response_error(response)
            try:
                return OperatorBundleInfo.model_validate(response.json())
            except ValueError as exc:
                raise ServerError("evaluation server returned invalid JSON") from exc
    except requests.RequestException as exc:
        raise ServerError(f"cannot reach evaluation server: {exc}") from exc


def upload_operator_bundle(
    operator_dir: str | os.PathLike[str],
    server_url: str,
    timeout: float = 600,
) -> OperatorBundleInfo:
    """Pack and upload one v6.2 native operator unless KGS already has it."""

    with tempfile.TemporaryFile(mode="w+b") as archive:
        digest, size = pack_operator_bundle(operator_dir, archive)
        existing = get_operator_bundle(digest, server_url, timeout=min(timeout, 30))
        if existing is not None:
            return existing
        url = server_url.rstrip("/") + f"/operator-bundles/{digest}"
        try:
            with _SESSION.request(
                "PUT",
                url,
                data=archive,
                headers={
                    "Content-Type": OPERATOR_BUNDLE_MEDIA_TYPE,
                    "Content-Length": str(size),
                },
                timeout=timeout,
                allow_redirects=False,
            ) as response:
                _raise_response_error(response)
                try:
                    info = OperatorBundleInfo.model_validate(response.json())
                except ValueError as exc:
                    raise ServerError("evaluation server returned invalid JSON") from exc
        except requests.RequestException as exc:
            raise ServerError(f"cannot upload operator bundle: {exc}") from exc
    if info.sha256 != digest:
        raise ServerError("evaluation server returned a different operator bundle digest")
    return info


def inspect(
    request: InspectRequest,
    server_url: str,
    timeout: float = 600,
) -> AdapterManifest:
    value = _post(server_url, "/inspect", request.wire_payload(), timeout)
    return AdapterManifest.model_validate(value)


def get_operator_contract(
    request: InspectRequest,
    server_url: str,
    timeout: float = 60,
    *, include_test_sources: bool = False,
) -> OperatorContract:
    """Read the target's contract without running reference code or pytest."""
    payload = request.wire_payload()
    if include_test_sources:
        payload["include_test_sources"] = True
    value = _post(server_url, "/operator-contract", payload, timeout)
    contract = OperatorContract.model_validate(value)
    if contract.binding != request.binding:
        raise ServerError("evaluation server returned a different operator binding")
    if include_test_sources and contract.test_sources is None:
        raise ServerError("evaluation server did not return requested test sources")
    return contract


def preflight(
    request: BoundEvaluateRequest,
    server_url: str,
    timeout: float | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    transport_timeout = timeout or _evaluation_transport_timeout(request)
    return _post(
        server_url,
        "/preflight",
        request.wire_payload(),
        transport_timeout,
        operation_id=operation_id,
    )


def reference(
    request: ReferenceRequest,
    server_url: str,
    timeout: float | None = None,
    operation_id: str | None = None,
) -> ReferenceResult:
    value = _post(server_url, "/reference", request.wire_payload(),
                  timeout or _evaluation_transport_timeout(request), operation_id=operation_id)
    result = ReferenceResult.model_validate(value)
    if result.status == "PASSED" and result.benchmark_fingerprint != request.benchmark_fingerprint:
        raise ServerError("reference result has a different benchmark fingerprint")
    return result


def evaluate(
    request: BoundEvaluateRequest,
    server_url: str,
    timeout: float | None = None,
    operation_id: str | None = None,
) -> EvaluateResponse:
    transport_timeout = timeout or _evaluation_transport_timeout(request)
    value = _post(server_url, "/evaluate", request.wire_payload(), transport_timeout, operation_id=operation_id)
    return EvaluateResponse.model_validate(value)


def profile(
    request: ProfileRequest,
    server_url: str,
    timeout: float | None = None,
    operation_id: str | None = None,
) -> ProfileResult:
    payload = request.model_dump(mode="json", exclude_unset=True, by_alias=True)
    transport_timeout = timeout or request.options.timeout_sec + 10
    value = _post(server_url, "/profile", payload, transport_timeout, operation_id=operation_id)
    return ProfileResult.model_validate(value)


def get_operation(
    operation_id: str,
    server_url: str,
    timeout: float = 30,
) -> dict[str, Any]:
    encoded_operation_id = quote(operation_id, safe="")
    value = _request_json(
        server_url,
        f"/operations/{encoded_operation_id}",
        method="GET",
        timeout=timeout,
    )
    if not isinstance(value, dict):
        raise ServerError(f"invalid operation response: {value!r}")
    return value


def cancel_operation(
    operation_id: str,
    server_url: str,
    timeout: float = 30,
) -> dict[str, Any]:
    encoded_operation_id = quote(operation_id, safe="")
    value = _request_json(
        server_url,
        f"/operations/{encoded_operation_id}",
        method="DELETE",
        timeout=timeout,
    )
    if not isinstance(value, dict):
        raise ServerError(f"invalid operation response: {value!r}")
    return value


def submit_debug_job(
    request: DebugJobRequest,
    server_url: str,
    timeout: float | None = None,
) -> DebugJob:
    """Submit one trusted debug command and return its terminal result."""
    value = _request_json(
        server_url,
        "/debug/jobs",
        method="POST",
        payload=request.model_dump(mode="json"),
        timeout=timeout or request.timeout_seconds + 10,
    )
    return DebugJob.model_validate(value)


def get_debug_job(
    job_id: str,
    server_url: str,
    *,
    wait_seconds: float = 0,
    timeout: float | None = None,
) -> DebugJob:
    """Return current state after an optional server-side bounded wait."""
    if not 0 <= wait_seconds <= 30:
        raise ValueError("wait_seconds must be between 0 and 30")
    encoded_job_id = quote(job_id, safe="")
    value = _request_json(
        server_url,
        f"/debug/jobs/{encoded_job_id}",
        method="GET",
        query={"wait_seconds": wait_seconds},
        timeout=timeout or max(30, wait_seconds + 5),
    )
    return DebugJob.model_validate(value)


def cancel_debug_job(
    job_id: str,
    server_url: str,
    timeout: float = 30,
) -> DebugJob:
    """Cancel a queued or running debug job."""
    encoded_job_id = quote(job_id, safe="")
    value = _request_json(
        server_url,
        f"/debug/jobs/{encoded_job_id}",
        method="DELETE",
        timeout=timeout,
    )
    return DebugJob.model_validate(value)


def download_debug_artifacts(
    job: DebugJob,
    output_dir: str | os.PathLike[str],
    server_url: str,
    timeout: float = 600,
) -> dict[str, Path]:
    """Download every artifact and verify its declared size and SHA-256."""
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    encoded_job_id = quote(job.job_id, safe="")
    downloaded: dict[str, Path] = {}
    for artifact in job.artifacts:
        relative = Path(artifact.path)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative in {Path("."), Path("")}
        ):
            raise ValueError(f"unsafe debug artifact path: {relative}")
        destination = (root / relative).resolve()
        if not destination.is_relative_to(root):
            raise ValueError(f"debug artifact escapes output directory: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == artifact.size_bytes:
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            if digest == artifact.sha256:
                downloaded[artifact.id] = destination
                continue

        encoded_artifact_id = quote(artifact.id, safe="")
        url = (
            server_url.rstrip("/")
            + f"/debug/jobs/{encoded_job_id}/artifacts/{encoded_artifact_id}"
        )
        temporary = destination.with_name(f".{destination.name}.part")
        try:
            with _SESSION.get(
                url,
                timeout=timeout,
                stream=True,
                allow_redirects=False,
            ) as response:
                if response.status_code >= 400:
                    raise ServerError(
                        f"server returned HTTP {response.status_code}: {response.text}"
                    )
                digest = hashlib.sha256()
                size = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            if size != artifact.size_bytes:
                raise ValueError(
                    f"debug artifact {artifact.id} size mismatch: "
                    f"expected {artifact.size_bytes}, received {size}"
                )
            if digest.hexdigest() != artifact.sha256:
                raise ValueError(
                    f"debug artifact {artifact.id} SHA-256 mismatch"
                )
            temporary.replace(destination)
        except requests.RequestException as exc:
            raise ServerError(f"cannot download debug artifact: {exc}") from exc
        finally:
            if temporary.exists():
                temporary.unlink()
        downloaded[artifact.id] = destination
    return downloaded
