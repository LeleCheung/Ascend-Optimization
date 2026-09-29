"""Spawn-isolated execution for untrusted candidate failures.

This isolates imports, accelerator contexts, crashes, and leaks. It is not a
security sandbox: submitted code still runs with the server account's OS
permissions.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import shutil
import signal
import tempfile
import time
import traceback
from multiprocessing.connection import Connection
from typing import Any, Literal

from ..protocol.schema import BoundEvaluateRequest, EvaluateResponse, PreflightResult, ReferenceRequest, ReferenceResult
from .process_control import (
    DEFAULT_TERMINATION_GRACE_SECONDS,
    terminate_process_tree,
)
from .operations import OperationCancelled, current_operation


Operation = Literal["evaluate", "preflight", "reference"]
EvaluationRequest = BoundEvaluateRequest | ReferenceRequest

_HEALTH_PROBE_MATRIX_SIZE = 1024
_HEALTH_PROBE_WARMUP_MS = 10
_HEALTH_PROBE_BENCHMARK_MS = 20
_PROCESS_TERMINATION_GRACE_SECONDS = DEFAULT_TERMINATION_GRACE_SECONDS


class IsolatedWorkerCrashed(RuntimeError):
    """The isolated process exited without returning a protocol response."""


class DeviceHealthCheckError(RuntimeError):
    """A device failed the isolated execution probe."""


class _TerminationRequested(BaseException):
    """Unwind an isolated worker after its process group receives SIGTERM."""


def _request_graceful_termination(_signum: int, _frame: Any) -> None:
    raise _TerminationRequested()


def _enter_isolated_process_group() -> None:
    """Make this worker the leader of a job-scoped POSIX process group."""

    if os.name != "posix":
        return
    if os.getpgrp() == os.getpid():
        return
    os.setsid()


def _install_graceful_termination_handler() -> None:
    if os.name == "posix":
        signal.signal(signal.SIGTERM, _request_graceful_termination)


def _stop_process(
    process: multiprocessing.Process,
    *,
    grace_seconds: float = _PROCESS_TERMINATION_GRACE_SECONDS,
) -> None:
    """Stop an isolated worker and every subprocess in its job process group."""

    terminate_process_tree(process, grace_seconds=grace_seconds)


def _worker(
    connection: Connection,
    operation: Operation,
    request_data: dict[str, Any],
    backend: str,
    device_string: str,
    timing: str,
    temp_root: str,
    operator_bundle_root: str | None = None,
) -> None:
    try:
        _enter_isolated_process_group()
        _install_graceful_termination_handler()
        os.environ["TMPDIR"] = temp_root
        tempfile.tempdir = temp_root
        from .device import configure_device, get_device
        from ..evaluation.adapters import create_adapter

        configure_kwargs = {"timing_strategy": timing}
        if backend == "npu":
            configure_kwargs["perf_mode"] = timing
        configure_device(backend, **configure_kwargs)
        device = get_device(backend)
        request = (ReferenceRequest if operation == "reference" else BoundEvaluateRequest).model_validate(request_data)
        adapter = create_adapter(
            request.binding,
            device=device,
            device_string=device_string,
            backend=backend,
            operator_bundle_root=operator_bundle_root,
        )
        result = getattr(adapter, operation)(request)
        if isinstance(result, EvaluateResponse):
            payload: Any = {"kind": "evaluation", "value": result.model_dump(mode="json")}
        elif isinstance(result, (PreflightResult, ReferenceResult)):
            payload = {"kind": "mapping", "value": result.model_dump(mode="json")}
        else:
            payload = {"kind": "mapping", "value": result}
        connection.send({"ok": True, "payload": payload})
    except BaseException:
        try:
            connection.send({"ok": False, "error": traceback.format_exc()})
        except Exception:
            pass
    finally:
        connection.close()


def _device_health_worker(
    connection: Connection,
    backend: str,
    device_string: str,
    timing: str,
) -> None:
    try:
        _enter_isolated_process_group()
        _install_graceful_termination_handler()
        from .device import configure_device, get_device

        configure_kwargs = {"timing_strategy": timing}
        if backend == "npu":
            configure_kwargs["perf_mode"] = timing
        configure_device(backend, **configure_kwargs)
        device = get_device(backend)
        device.set_device(device_string)

        import torch

        matrix_size = _HEALTH_PROBE_MATRIX_SIZE
        value = torch.linspace(
            -2.0,
            2.0,
            matrix_size * matrix_size,
            dtype=torch.float32,
            device=device_string,
        ).reshape(matrix_size, matrix_size)
        identity = torch.eye(
            matrix_size,
            dtype=torch.float32,
            device=device_string,
        )
        product = value @ identity
        probabilities = torch.softmax(product, dim=-1)
        device.synchronize(device_string)

        row_sum = float(probabilities[0].sum().item())
        if not math.isclose(row_sum, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise RuntimeError(
                "device execution returned an invalid softmax row sum: "
                f"{row_sum!r}"
            )

        # Non-Ascend production timing uses do_bench. Exercise that exact event
        # and repeated-launch path so a partially poisoned runtime cannot pass
        # merely because allocation and one tiny elementwise kernel still work.
        if backend != "npu" and timing == "triton":
            softmax_ms = device.time(
                torch.softmax,
                [product, -1],
                _HEALTH_PROBE_WARMUP_MS,
                _HEALTH_PROBE_BENCHMARK_MS,
                device_string,
            )
            add_ms = device.time(
                lambda tensor: tensor + 1.0,
                [product],
                _HEALTH_PROBE_WARMUP_MS,
                _HEALTH_PROBE_BENCHMARK_MS,
                device_string,
            )
            if not all(
                math.isfinite(latency_ms) and latency_ms > 0
                for latency_ms in (softmax_ms, add_ms)
            ):
                raise RuntimeError(
                    "device timing returned invalid latency values: "
                    f"softmax={softmax_ms!r}, add={add_ms!r}"
                )
        else:
            repeated = probabilities
            for _ in range(3):
                repeated = torch.softmax(product, dim=-1) + 1.0
            device.synchronize(device_string)
            del repeated

        del probabilities, product, identity, value
        device.empty_cache()
        connection.send({"ok": True})
    except BaseException:
        try:
            connection.send({"ok": False, "error": traceback.format_exc()})
        except Exception:
            pass
    finally:
        connection.close()


def probe_device(
    backend: str,
    device_string: str,
    timing: str,
    timeout_seconds: float = 30,
) -> None:
    """Run a representative compute and timing chain in a disposable process."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_device_health_worker,
        args=(child, backend, device_string, timing),
        daemon=False,
    )
    process.start()
    child.close()
    try:
        if not parent.poll(timeout_seconds):
            _stop_process(process)
            raise DeviceHealthCheckError(
                f"device probe timed out after {timeout_seconds}s"
            )
        try:
            message = parent.recv()
        except EOFError as exc:
            process.join(timeout=5)
            if process.is_alive():
                _stop_process(process)
            raise DeviceHealthCheckError(
                "device probe exited without a response "
                f"(exitcode={process.exitcode})"
            ) from exc
    finally:
        parent.close()
        process.join(timeout=5)

    forced_cleanup = process.is_alive()
    if forced_cleanup:
        _stop_process(process)
    if not message.get("ok"):
        raise DeviceHealthCheckError(
            message.get("error", "device probe failed")
        )
    if not forced_cleanup and process.exitcode != 0:
        raise DeviceHealthCheckError(
            f"device probe exited abnormally (exitcode={process.exitcode})"
        )


def run_isolated(
    operation: Operation,
    request: EvaluationRequest,
    backend: str,
    device_string: str,
    timing: str,
    *,
    operator_bundle_root: str | None = None,
) -> EvaluateResponse | dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    temp_root = tempfile.mkdtemp(prefix="kernelgen-isolated-")
    process = context.Process(
        target=_worker,
        args=(
            child,
            operation,
            request.wire_payload(),
            backend,
            device_string,
            timing,
            temp_root,
            operator_bundle_root,
        ),
        daemon=False,
    )
    try:
        process.start()
    except BaseException:
        parent.close()
        child.close()
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    child.close()
    timeout = request.settings.timeout_seconds
    deadline = time.monotonic() + timeout
    operation_control = current_operation()
    forced_cleanup = False
    try:
        while True:
            if (
                operation_control is not None
                and operation_control.cancel_requested
            ):
                _stop_process(process)
                raise OperationCancelled(
                    operation_control.operation_id,
                    operation_control.kind,
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _stop_process(process)
                raise TimeoutError(
                    f"{operation} timed out after {timeout}s on {device_string}"
                )
            if parent.poll(min(0.1, remaining)):
                break
        try:
            message = parent.recv()
        except EOFError as exc:
            process.join(timeout=5)
            _stop_process(process)
            raise IsolatedWorkerCrashed(
                f"{operation} worker exited without a response on "
                f"{device_string} (exitcode={process.exitcode})"
            ) from exc
    finally:
        parent.close()
        process.join(timeout=5)
        forced_cleanup = process.is_alive()
        if forced_cleanup:
            _stop_process(process)
        shutil.rmtree(temp_root, ignore_errors=True)

    if not forced_cleanup and process.exitcode != 0:
        raise IsolatedWorkerCrashed(
            f"{operation} worker exited abnormally on {device_string} "
            f"(exitcode={process.exitcode})"
        )
    if not message.get("ok"):
        raise RuntimeError(message.get("error", "isolated worker failed"))
    payload = message["payload"]
    if payload["kind"] == "evaluation":
        return EvaluateResponse.model_validate(payload["value"])
    return payload["value"]
