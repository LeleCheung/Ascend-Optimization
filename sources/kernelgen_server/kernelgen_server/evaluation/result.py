"""Authoritative evaluation-result aggregation.

This module is intentionally runtime-light: it depends only on wire models, so
the result contract can be tested without importing torch or a vendor runtime.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Literal

from ..protocol.schema import (
    CorrectnessResult,
    EvaluateRequest,
    EvaluateResponse,
    EvaluationStatus,
    TimingResult,
    WorkloadResult,
    WorkloadStatus,
)
from ..protocol.version import ApiVersion, KERNELGEN_API_VERSION


def aggregate_evaluation_response(
    request: EvaluateRequest,
    correctness: Dict[str, CorrectnessResult],
    timing: Dict[str, TimingResult],
    *,
    device: str,
    backend: str,
    is_hack: bool = False,
    hack_reason: str = "",
    reference_source: Literal["primary", "torch_fallback"] = "primary",
) -> EvaluateResponse:
    per_workload: list[WorkloadResult] = []

    for workload in request.correctness_workloads:
        item = correctness[workload.name]
        result = WorkloadResult(
            uuid=workload.name,
            phase="correctness",
            status=item.status,
            abs_err=item.max_absolute_error,
            rel_err=item.max_relative_error,
            matched_ratio=item.matched_ratio,
            metrics=item.metrics,
            log=item.message,
        )
        per_workload.append(result)

    for workload in request.timing_workloads:
        item = timing[workload.name]
        result = WorkloadResult(
            uuid=workload.name,
            phase="timing",
            status=item.status,
            speedup=item.speedup,
            latency_ms=item.latency_ms,
            reference_latency_ms=item.reference_latency_ms,
            log=item.message,
        )
        per_workload.append(result)

    return aggregate_workload_results(
        per_workload,
        device=device,
        backend=backend,
        is_hack=is_hack,
        hack_reason=hack_reason,
        api_version=request.api_version,
        reference_source=reference_source,
    )


def aggregate_workload_results(
    results: Iterable[WorkloadResult],
    *,
    device: str,
    backend: str,
    is_hack: bool = False,
    hack_reason: str = "",
    api_version: ApiVersion = KERNELGEN_API_VERSION,
    reference_source: Literal["primary", "torch_fallback"] = "primary",
) -> EvaluateResponse:
    """Aggregate results emitted by native or framework evaluator adapters."""

    per_workload = list(results)
    passing_timing: list[tuple[float, float, str]] = []
    abs_errs: list[float] = []
    rel_errs: list[float] = []
    first_failure_uuid: str | None = None
    first_failure_log = ""
    for item in per_workload:
        if item.abs_err is not None:
            abs_errs.append(item.abs_err)
        if item.rel_err is not None:
            rel_errs.append(item.rel_err)
        if (
            item.phase == "timing"
            and item.status == WorkloadStatus.PASSED
            and item.speedup is not None
            and item.latency_ms is not None
        ):
            passing_timing.append((item.speedup, item.latency_ms, item.uuid))
        if item.status not in {WorkloadStatus.PASSED, WorkloadStatus.SKIP} and first_failure_uuid is None:
            first_failure_uuid = item.uuid
            first_failure_log = item.log

    statuses = [item.status for item in per_workload]
    if not statuses:
        return EvaluateResponse(
            api_version=api_version,
            reference_source=reference_source,
            status=EvaluationStatus.RUNTIME_ERROR,
            device=device,
            server_backend=backend,
            is_hack=is_hack,
            hack_reason=hack_reason,
            num_workloads=0,
            num_passed=0,
            log="runner returned no workload results",
        )
    num_passed = sum(status == WorkloadStatus.PASSED for status in statuses)
    applicable = [status for status in statuses if status != WorkloadStatus.SKIP]
    correctness_statuses = [item.status for item in per_workload if item.phase == "correctness"]
    all_correctness_skipped = bool(correctness_statuses) and all(
        status == WorkloadStatus.SKIP for status in correctness_statuses)
    timing_statuses = [item.status for item in per_workload if item.phase == "timing"]
    all_timing_skipped = bool(timing_statuses) and all(
        status == WorkloadStatus.SKIP for status in timing_statuses)
    if not applicable or (num_passed == len(applicable) and (all_correctness_skipped or all_timing_skipped)):
        aggregate_status = EvaluationStatus.ALL_SKIP
    elif num_passed == len(applicable):
        aggregate_status = EvaluationStatus.PASSED
    elif num_passed:
        aggregate_status = EvaluationStatus.PARTIAL_PASS
    else:
        aggregate_status = EvaluationStatus(applicable[0].value)

    if aggregate_status == EvaluationStatus.PASSED and passing_timing:
        speedups = [item[0] for item in passing_timing]
        geo_mean = math.exp(
            sum(math.log(speedup) for speedup in speedups) / len(speedups)
        )
        min_speedup, latency_ms, worst_uuid = min(
            passing_timing, key=lambda item: item[0]
        )
    else:
        geo_mean = None
        min_speedup = None
        latency_ms = None
        worst_uuid = first_failure_uuid

    return EvaluateResponse(
        api_version=api_version,
        reference_source=reference_source,
        status=aggregate_status,
        device=device,
        server_backend=backend,
        is_hack=is_hack,
        hack_reason=hack_reason,
        geo_mean=geo_mean,
        min_speedup=min_speedup,
        worst_workload_uuid=worst_uuid,
        latency_ms=latency_ms,
        abs_err=max(abs_errs) if abs_errs else None,
        rel_err=max(rel_errs) if rel_errs else None,
        num_workloads=len(statuses),
        num_passed=num_passed,
        log=first_failure_log,
        per_workload=per_workload,
    )


__all__ = ["aggregate_evaluation_response", "aggregate_workload_results"]
