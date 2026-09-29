"""Isolated evaluation execution with post-failure device verification."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Executor
from typing import Any, Callable

from ..runtime.isolated import IsolatedWorkerCrashed, Operation
from ..runtime.device_pool import DevicePool
from ..runtime.operations import (
    OperationCancelled,
    OperationControl,
    acquire_device_or_cancel,
    bind_operation,
)
from ..protocol.responses import (
    suspected_device_error_response,
    timeout_response,
)
from ..protocol.schema import BoundEvaluateRequest, ReferenceRequest
from .audit import RequestAudit, RequestAuditHandle


logger = logging.getLogger(__name__)

IsolatedRunner = Callable[
    [Operation, BoundEvaluateRequest | ReferenceRequest, str, str, str],
    Any,
]


class EvaluationExecutor:
    """Run one preflight/evaluation and verify its slot after infrastructure failure."""

    def __init__(
        self,
        *,
        backend: str,
        timing: str,
        device_pool: DevicePool,
        executor: Executor,
        runner: IsolatedRunner,
        request_audit: RequestAudit | None = None,
        probe_timeout_seconds: float = 30,
    ) -> None:
        self._backend = backend
        self._timing = timing
        self._device_pool = device_pool
        self._executor = executor
        self._runner = runner
        self._request_audit = request_audit
        self._probe_timeout_seconds = probe_timeout_seconds

    def _audit_event(
        self,
        handle: RequestAuditHandle | None,
        event: str,
        **details: Any,
    ) -> None:
        if self._request_audit is not None and handle is not None:
            self._request_audit.event(handle, event, **details)

    def _scheduler_snapshot(self) -> dict[str, Any]:
        return self._device_pool.snapshot(
            probe_timeout_seconds=self._probe_timeout_seconds
        )

    async def run(
        self,
        operation: Operation,
        request: BoundEvaluateRequest | ReferenceRequest,
        control: OperationControl | None = None,
    ) -> Any:
        audit_handle = (
            self._request_audit.begin(operation, request)
            if self._request_audit is not None
            else None
        )
        device: str | None = None
        broken_reason = ""
        isolated_started = False
        try:
            self._audit_event(
                audit_handle,
                "waiting_for_device",
                scheduler=self._scheduler_snapshot(),
            )
            device = await acquire_device_or_cancel(
                self._device_pool,
                control,
            )
            if control is not None:
                control.mark_running(device)
            self._audit_event(
                audit_handle,
                "device_acquired",
                device=device,
                scheduler=self._scheduler_snapshot(),
            )
            loop = asyncio.get_running_loop()
            try:
                def invoke_runner() -> Any:
                    with bind_operation(control):
                        return self._runner(
                            operation,
                            request,
                            self._backend,
                            device,
                            self._timing,
                        )

                isolated_started = True
                result = await loop.run_in_executor(
                    self._executor,
                    invoke_runner,
                )
                if self._request_audit is not None and audit_handle is not None:
                    self._request_audit.response(
                        audit_handle,
                        result,
                        device=device,
                    )
                return result
            except OperationCancelled as exc:
                self._audit_event(
                    audit_handle,
                    "operation_cancelled",
                    device=device,
                    operation_id=exc.operation_id,
                )
                if isolated_started:
                    cause = f"{operation} cancelled by client"
                    self._audit_event(
                        audit_handle,
                        "recovery_probe_started",
                        device=device,
                        cause=cause,
                        scheduler=self._scheduler_snapshot(),
                    )
                    broken_reason = await self._device_pool.probe_after_failure(
                        device,
                        cause,
                    )
                    self._audit_event(
                        audit_handle,
                        (
                            "recovery_probe_failed"
                            if broken_reason
                            else "recovery_probe_passed"
                        ),
                        device=device,
                        reason=broken_reason,
                        scheduler=self._scheduler_snapshot(),
                    )
                raise
            except TimeoutError as exc:
                cause = f"{operation} timeout: {exc}"
                self._audit_event(
                    audit_handle,
                    "isolated_execution_timed_out",
                    device=device,
                    error=str(exc),
                )
                self._audit_event(
                    audit_handle,
                    "recovery_probe_started",
                    device=device,
                    cause=cause,
                    scheduler=self._scheduler_snapshot(),
                )
                broken_reason = await self._device_pool.probe_after_failure(
                    device,
                    cause,
                )
                if not broken_reason:
                    message = (
                        f"{exc}; post-timeout strong device probe passed; "
                        "slot restored; request was not retried"
                    )
                    logger.warning(
                        "%s timed out on %s; strong probe passed; slot restored",
                        operation,
                        device,
                    )
                    self._audit_event(
                        audit_handle,
                        "recovery_probe_passed",
                        device=device,
                        scheduler=self._scheduler_snapshot(),
                    )
                    result = timeout_response(
                        operation,
                        request,
                        backend=self._backend,
                        device_string=device,
                        message=message,
                    )
                    if (
                        self._request_audit is not None
                        and audit_handle is not None
                    ):
                        self._request_audit.response(
                            audit_handle,
                            result,
                            device=device,
                        )
                    return result

                logger.error(
                    "%s timed out on %s; strong probe failed; slot quarantined",
                    operation,
                    device,
                )
                self._audit_event(
                    audit_handle,
                    "recovery_probe_failed",
                    device=device,
                    reason=broken_reason,
                    scheduler=self._scheduler_snapshot(),
                )
                result = suspected_device_error_response(
                    operation,
                    request,
                    backend=self._backend,
                    device_string=device,
                    message=(
                        f"{broken_reason}; slot quarantined; request was not "
                        "retried on another device"
                    ),
                )
                if self._request_audit is not None and audit_handle is not None:
                    self._request_audit.response(
                        audit_handle,
                        result,
                        device=device,
                    )
                return result
            except IsolatedWorkerCrashed as exc:
                cause = f"{operation} isolated worker crashed: {exc}"
                self._audit_event(
                    audit_handle,
                    "isolated_worker_crashed",
                    device=device,
                    error=str(exc),
                )
                self._audit_event(
                    audit_handle,
                    "recovery_probe_started",
                    device=device,
                    cause=cause,
                    scheduler=self._scheduler_snapshot(),
                )
                broken_reason = await self._device_pool.probe_after_failure(
                    device,
                    cause,
                )
                if not broken_reason:
                    logger.warning(
                        "%s worker crashed on %s; strong probe passed; "
                        "slot restored",
                        operation,
                        device,
                    )
                    self._audit_event(
                        audit_handle,
                        "recovery_probe_passed",
                        device=device,
                        scheduler=self._scheduler_snapshot(),
                    )
                    raise

                logger.error(
                    "%s worker crashed on %s; strong probe failed; "
                    "slot quarantined",
                    operation,
                    device,
                )
                self._audit_event(
                    audit_handle,
                    "recovery_probe_failed",
                    device=device,
                    reason=broken_reason,
                    scheduler=self._scheduler_snapshot(),
                )
                result = suspected_device_error_response(
                    operation,
                    request,
                    backend=self._backend,
                    device_string=device,
                    message=(
                        f"{broken_reason}; slot quarantined; request was not "
                        "retried on another device"
                    ),
                )
                if self._request_audit is not None and audit_handle is not None:
                    self._request_audit.response(
                        audit_handle,
                        result,
                        device=device,
                    )
                return result
        except BaseException as exc:
            if self._request_audit is not None and audit_handle is not None:
                self._request_audit.failure(
                    audit_handle,
                    exc,
                    device=device,
                )
            raise
        finally:
            if device is not None:
                await asyncio.shield(
                    self._device_pool.release(
                        device,
                        broken_reason=broken_reason,
                    )
                )
                self._audit_event(
                    audit_handle,
                    "device_released",
                    device=device,
                    broken=bool(broken_reason),
                    scheduler=self._scheduler_snapshot(),
                )
