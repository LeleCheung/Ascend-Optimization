import math

import pytest

from kernelgen_server.evaluation.result import aggregate_evaluation_response
from kernelgen_server.schema import (
    CorrectnessResult,
    Definition,
    EvaluateRequest,
    EvaluationStatus,
    Implementation,
    SourceFile,
    TimingResult,
    Workload,
    WorkloadStatus,
)


def _workload(name: str) -> Workload:
    return Workload(
        name=name,
        inputs={"x": {"type": "scalar", "value": 1}},
    )


def _request() -> EvaluateRequest:
    return EvaluateRequest(
        definition=Definition(
            name="identity",
            parameters=[
                {
                    "name": "x",
                    "kind": "positional_or_keyword",
                    "required": True,
                }
            ],
            outputs=["output"],
            reference="def run(x): return x",
        ),
        implementation=Implementation(
            name="candidate",
            definition="identity",
            language="python",
            entrypoint="main.py::run",
            sources=[SourceFile(path="main.py", content="def run(x): return x")],
        ),
        correctness_workloads=[_workload("corr-0"), _workload("corr-1")],
        timing_workloads=[_workload("time-0"), _workload("time-1")],
    )


def test_all_passed_result_has_authoritative_headline_metrics():
    response = aggregate_evaluation_response(
        _request(),
        {
            "corr-0": CorrectnessResult(
                status=WorkloadStatus.PASSED,
                max_absolute_error=0.1,
                max_relative_error=0.2,
            ),
            "corr-1": CorrectnessResult(
                status=WorkloadStatus.PASSED,
                max_absolute_error=0.3,
                max_relative_error=0.4,
            ),
        },
        {
            "time-0": TimingResult(
                status=WorkloadStatus.PASSED,
                latency_ms=2.0,
                reference_latency_ms=4.0,
                speedup=2.0,
            ),
            "time-1": TimingResult(
                status=WorkloadStatus.PASSED,
                latency_ms=2.0,
                reference_latency_ms=8.0,
                speedup=4.0,
            ),
        },
        device="cuda:0",
        backend="cuda",
    )

    assert response.api_version == "v6.2"
    assert response.reference_source == "primary"
    assert response.status == EvaluationStatus.PASSED
    assert response.is_hack is False
    assert response.hack_reason == ""
    assert response.geo_mean == pytest.approx(math.sqrt(8.0))
    assert response.min_speedup == 2.0
    assert response.worst_workload_uuid == "time-0"
    assert response.latency_ms == 2.0
    assert response.abs_err == 0.3
    assert response.rel_err == 0.4
    assert response.num_workloads == 4
    assert response.num_passed == 4
    assert [item.uuid for item in response.per_workload] == [
        "corr-0",
        "corr-1",
        "time-0",
        "time-1",
    ]


def test_partial_pass_does_not_publish_optimistic_headline():
    request = _request()
    response = aggregate_evaluation_response(
        request,
        {
            "corr-0": CorrectnessResult(status=WorkloadStatus.PASSED),
            "corr-1": CorrectnessResult(
                status=WorkloadStatus.INCORRECT_NUMERICAL,
                message="mismatch",
            ),
        },
        {
            "time-0": TimingResult(
                status=WorkloadStatus.PASSED,
                latency_ms=1.0,
                reference_latency_ms=2.0,
                speedup=2.0,
            ),
            "time-1": TimingResult(
                status=WorkloadStatus.RUNTIME_ERROR,
                message="launch failed",
            ),
        },
        device="npu:0",
        backend="npu",
    )

    assert response.status == EvaluationStatus.PARTIAL_PASS
    assert response.geo_mean is None
    assert response.min_speedup is None
    assert response.latency_ms is None
    assert response.worst_workload_uuid == "corr-1"
    assert response.log == "mismatch"
    assert response.num_passed == 2


def test_hack_detection_is_preserved_in_aggregate_result():
    request = _request()
    response = aggregate_evaluation_response(
        request,
        {
            "corr-0": CorrectnessResult(status=WorkloadStatus.PASSED),
            "corr-1": CorrectnessResult(status=WorkloadStatus.PASSED),
        },
        {
            "time-0": TimingResult(
                status=WorkloadStatus.PASSED,
                latency_ms=1.0,
                reference_latency_ms=2.0,
                speedup=2.0,
            ),
            "time-1": TimingResult(
                status=WorkloadStatus.PASSED,
                latency_ms=1.0,
                reference_latency_ms=2.0,
                speedup=2.0,
            ),
        },
        device="cuda:0",
        backend="cuda",
        is_hack=True,
        hack_reason="forbidden Torch API: main.py:1 torch.einsum",
    )

    assert response.status == EvaluationStatus.PASSED
    assert response.is_hack is True
    assert response.hack_reason == (
        "forbidden Torch API: main.py:1 torch.einsum"
    )
