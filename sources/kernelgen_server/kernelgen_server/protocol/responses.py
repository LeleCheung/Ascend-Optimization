"""Structured protocol responses for isolated execution failures."""

from __future__ import annotations

from .schema import BoundEvaluateRequest, EvaluateResponse, PreflightResult, ReferenceRequest, ReferenceResult
from .version import KERNELGEN_API_VERSION


def timeout_response(
    operation: str,
    request: BoundEvaluateRequest | ReferenceRequest,
    *,
    backend: str,
    device_string: str,
    message: str,
) -> EvaluateResponse | PreflightResult | ReferenceResult:
    """Build an HTTP-200 protocol result for one isolated execution timeout."""
    if operation == "reference":
        return ReferenceResult(status="TIMEOUT", log=message)
    if operation == "evaluate":
        return EvaluateResponse(
            status="TIMEOUT",
            device=device_string,
            server_backend=backend,
            num_workloads=0,
            num_passed=0,
            log=message,
        )
    return PreflightResult(status="TIMEOUT", stage="isolated_execution", log=message)


def suspected_device_error_response(
    operation: str,
    request: BoundEvaluateRequest | ReferenceRequest,
    *,
    backend: str,
    device_string: str,
    message: str,
) -> EvaluateResponse | PreflightResult | ReferenceResult:
    """Report a failed strong device probe after quarantining its slot."""
    if operation == "reference":
        return ReferenceResult(status="SUSPECTED_DEVICE_ERROR", log=message)
    if operation == "evaluate":
        return EvaluateResponse(
            status="SUSPECTED_DEVICE_ERROR",
            device=device_string,
            server_backend=backend,
            num_workloads=0,
            num_passed=0,
            log=message,
        )
    return PreflightResult(
        status="SUSPECTED_DEVICE_ERROR",
        stage="device_probe",
        log=message,
    )
