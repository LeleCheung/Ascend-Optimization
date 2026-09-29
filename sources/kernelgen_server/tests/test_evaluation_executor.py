from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from kernelgen_server.evaluation.audit import RequestAudit
from kernelgen_server.evaluation.executor import EvaluationExecutor
from kernelgen_server.runtime.isolated import IsolatedWorkerCrashed
from kernelgen_server.runtime.device_pool import DevicePool, DeviceSlot
from kernelgen_server.runtime.operations import (
    OperationCancelled,
    OperationRegistry,
    current_operation,
)
from kernelgen_server.schema import (
    BoundEvaluateRequest,
    EvaluateResponse,
    EvaluationStatus,
    EvaluatorBinding,
    Implementation,
    SourceFile,
)


def _request() -> BoundEvaluateRequest:
    return BoundEvaluateRequest(
        binding=EvaluatorBinding(
            catalog_name="simple-v6-test", definition="identity"
        ),
        implementation=Implementation(
            name="candidate",
            definition="identity",
            language="python",
            entrypoint="main.py::run",
            sources=[
                SourceFile(
                    path="main.py",
                    content="def run(x): return x",
                )
            ],
        ),
    )


def _passed(device: str, backend: str) -> EvaluateResponse:
    return EvaluateResponse(
        status=EvaluationStatus.PASSED,
        device=device,
        server_backend=backend,
        geo_mean=1.0,
        min_speedup=1.0,
        latency_ms=1.0,
        num_workloads=1,
        num_passed=1,
    )


def _execute(
    runner,
    *,
    device_count: int = 2,
    probe=lambda device: None,
    request_audit=None,
):
    with ThreadPoolExecutor(max_workers=device_count) as executor:
        pool = DevicePool(
            [
                DeviceSlot(f"cuda:{index}", "available")
                for index in range(device_count)
            ],
            worker_count=device_count,
            executor=executor,
            probe=probe,
        )
        evaluator = EvaluationExecutor(
            backend="cuda",
            timing="triton",
            device_pool=pool,
            executor=executor,
            runner=runner,
            request_audit=request_audit,
        )
        result = asyncio.run(evaluator.run("evaluate", _request()))
        return result, pool.snapshot(probe_timeout_seconds=30)


def test_timeout_with_passing_probe_returns_timeout_without_retry():
    devices = []

    def runner(operation, request, backend, device, timing):
        devices.append(device)
        raise TimeoutError("too slow")

    result, status = _execute(runner)

    assert result.status == EvaluationStatus.TIMEOUT
    assert devices == ["cuda:0"]
    assert status["incidents"] == 1
    assert status["recovered"] == 1
    assert status["broken"] == 0


def test_timeout_with_failed_probe_returns_device_error_without_retry():
    devices = []

    def runner(operation, request, backend, device, timing):
        devices.append(device)
        raise TimeoutError("too slow")

    def failed_probe(device):
        raise RuntimeError("queue unhealthy")

    result, status = _execute(runner, probe=failed_probe)

    assert result.status == EvaluationStatus.SUSPECTED_DEVICE_ERROR
    assert devices == ["cuda:0"]
    assert "request was not retried" in result.log
    assert status["incidents"] == 1
    assert status["recovered"] == 0
    assert status["broken"] == 1


def test_worker_crash_with_passing_probe_restores_slot_without_retry():
    devices = []

    def runner(operation, request, backend, device, timing):
        devices.append(device)
        raise IsolatedWorkerCrashed("exitcode=17")

    with pytest.raises(IsolatedWorkerCrashed, match="exitcode=17"):
        _execute(runner)

    assert devices == ["cuda:0"]


def test_worker_crash_with_failed_probe_returns_device_error_without_retry():
    devices = []

    def runner(operation, request, backend, device, timing):
        devices.append(device)
        raise IsolatedWorkerCrashed("exitcode=17")

    def failed_probe(device):
        raise RuntimeError("queue unhealthy")

    result, status = _execute(runner, probe=failed_probe)

    assert result.status == EvaluationStatus.SUSPECTED_DEVICE_ERROR
    assert devices == ["cuda:0"]
    assert status["incidents"] == 1
    assert status["recovered"] == 0
    assert status["broken"] == 1


def test_runtime_error_is_not_retried():
    devices = []

    def runner(operation, request, backend, device, timing):
        devices.append(device)
        raise RuntimeError("operator failure")

    with pytest.raises(RuntimeError, match="operator failure"):
        _execute(runner)

    assert devices == ["cuda:0"]


def test_running_operation_cancellation_probes_and_releases_slot():
    started = threading.Event()

    def runner(operation, request, backend, device, timing):
        control = current_operation()
        assert control is not None
        started.set()
        assert control.cancel_event.wait(timeout=5)
        control.raise_if_cancelled()

    async def scenario():
        with ThreadPoolExecutor(max_workers=1) as executor:
            pool = DevicePool(
                [DeviceSlot("cuda:0", "available")],
                worker_count=1,
                executor=executor,
                probe=lambda device: None,
            )
            evaluator = EvaluationExecutor(
                backend="cuda",
                timing="triton",
                device_pool=pool,
                executor=executor,
                runner=runner,
            )
            control = OperationRegistry().create("evaluate", "running")
            task = asyncio.create_task(
                evaluator.run("evaluate", _request(), control)
            )
            while not started.is_set():
                await asyncio.sleep(0.01)
            control.request_cancel()
            with pytest.raises(OperationCancelled):
                await task
            return pool.snapshot(probe_timeout_seconds=30)

    status = asyncio.run(scenario())

    assert status["active"] == 0
    assert status["available"] == 1
    assert status["broken"] == 0
    assert status["incidents"] == 1
    assert status["recovered"] == 1


def test_queued_operation_cancellation_does_not_run_or_probe():
    invocations = 0

    def runner(operation, request, backend, device, timing):
        nonlocal invocations
        invocations += 1

    async def scenario():
        with ThreadPoolExecutor(max_workers=1) as executor:
            probes = []
            pool = DevicePool(
                [DeviceSlot("cuda:0", "available")],
                worker_count=1,
                executor=executor,
                probe=probes.append,
            )
            held_device = await pool.acquire()
            evaluator = EvaluationExecutor(
                backend="cuda",
                timing="triton",
                device_pool=pool,
                executor=executor,
                runner=runner,
            )
            control = OperationRegistry().create("evaluate", "queued")
            task = asyncio.create_task(
                evaluator.run("evaluate", _request(), control)
            )
            while pool.snapshot(probe_timeout_seconds=30)["waiting"] != 1:
                await asyncio.sleep(0.01)
            control.request_cancel()
            with pytest.raises(OperationCancelled):
                await task
            await pool.release(held_device)
            return probes, pool.snapshot(probe_timeout_seconds=30)

    probes, status = asyncio.run(scenario())

    assert invocations == 0
    assert probes == []
    assert status["incidents"] == 0
    assert status["active"] == 0
    assert status["available"] == 1


def _audit_events(audit: RequestAudit) -> list[dict]:
    return [
        json.loads(line)
        for line in audit.events_path.read_text(encoding="utf-8").splitlines()
    ]


def test_request_audit_persists_exact_request_and_response(tmp_path):
    audit = RequestAudit(
        tmp_path,
        manifest={"backend": "cuda", "devices": ["cuda:0"]},
    )

    result, _ = _execute(
        lambda operation, request, backend, device, timing: _passed(
            device,
            backend,
        ),
        device_count=1,
        request_audit=audit,
    )

    request_dirs = list(audit.requests_root.iterdir())
    assert len(request_dirs) == 1
    request_record = json.loads(
        (request_dirs[0] / "request.json").read_text(encoding="utf-8")
    )
    response_record = json.loads(
        (request_dirs[0] / "response.json").read_text(encoding="utf-8")
    )
    assert request_record["metadata"]["operation"] == "evaluate"
    assert request_record["payload"] == _request().wire_payload()
    assert len(request_record["metadata"]["candidate_sha256"]) == 64
    assert len(request_record["metadata"]["request_sha256"]) == 64
    assert response_record["payload"] == result.model_dump(mode="json")
    assert [event["event"] for event in _audit_events(audit)] == [
        "request_persisted",
        "waiting_for_device",
        "device_acquired",
        "response_persisted",
        "device_released",
    ]


def test_request_audit_correlates_timeout_failed_probe_and_quarantine(tmp_path):
    audit = RequestAudit(
        tmp_path,
        manifest={"backend": "cuda", "devices": ["cuda:0"]},
    )

    def runner(operation, request, backend, device, timing):
        raise TimeoutError("too slow")

    def failed_probe(device):
        raise RuntimeError("queue unhealthy")

    result, status = _execute(
        runner,
        device_count=1,
        probe=failed_probe,
        request_audit=audit,
    )

    assert result.status == EvaluationStatus.SUSPECTED_DEVICE_ERROR
    assert status["broken"] == 1
    events = _audit_events(audit)
    assert [event["event"] for event in events] == [
        "request_persisted",
        "waiting_for_device",
        "device_acquired",
        "isolated_execution_timed_out",
        "recovery_probe_started",
        "recovery_probe_failed",
        "response_persisted",
        "device_released",
    ]
    request_ids = {event["request_id"] for event in events}
    assert len(request_ids) == 1
    assert events[-1]["scheduler"]["broken"] == 1


def test_request_audit_persists_unhandled_worker_error(tmp_path):
    audit = RequestAudit(
        tmp_path,
        manifest={"backend": "cuda", "devices": ["cuda:0"]},
    )

    def runner(operation, request, backend, device, timing):
        raise RuntimeError("compiler failed")

    with pytest.raises(RuntimeError, match="compiler failed"):
        _execute(
            runner,
            device_count=1,
            request_audit=audit,
        )

    request_dir = next(audit.requests_root.iterdir())
    error = json.loads(
        (request_dir / "error.json").read_text(encoding="utf-8")
    )
    assert error["device"] == "cuda:0"
    assert error["error_type"] == "RuntimeError"
    assert error["error"] == "compiler failed"
    assert [event["event"] for event in _audit_events(audit)][-2:] == [
        "request_failed",
        "device_released",
    ]


def test_request_does_not_execute_when_enabled_audit_cannot_persist(
    tmp_path,
    monkeypatch,
):
    audit = RequestAudit(
        tmp_path,
        manifest={"backend": "cuda", "devices": ["cuda:0"]},
    )
    executed = []

    def fail_before_scheduling(operation, request):
        raise OSError("audit disk unavailable")

    def runner(operation, request, backend, device, timing):
        executed.append(device)
        return _passed(device, backend)

    monkeypatch.setattr(audit, "begin", fail_before_scheduling)

    with pytest.raises(OSError, match="audit disk unavailable"):
        _execute(
            runner,
            device_count=1,
            request_audit=audit,
        )

    assert executed == []
