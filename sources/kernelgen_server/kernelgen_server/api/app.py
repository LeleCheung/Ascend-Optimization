"""FastAPI application assembly."""

from __future__ import annotations

from ..evaluation.candidate_admission import admission_capability

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from ..catalog import builtin_catalog_path
from ..operator_bundles import (
    DEFAULT_OPERATOR_BUNDLE_MAX_BYTES,
    OperatorBundleStore,
)
from ..debug.jobs import (
    DebugJobRequest,
    DebugJobStore,
    LocalDebugJobRunner,
)
from ..debug.service import DebugService
from ..evaluation.executor import EvaluationExecutor
from ..evaluation.adapters import create_adapter
from ..evaluation.adapters.registry import resolve_binding_catalog
from ..evaluation.audit import RequestAudit
from ..profiling import (
    ProfileArtifactNotFound,
    ProfileRequest,
    ProfileResult,
    ProfileService,
    get_profiler,
)
from ..runtime.device_pool import (
    DevicePool,
    NoHealthyDeviceError,
    probe_startup_slots,
)
from ..runtime.device import _make_device, configure_device, runtime_device_type
from ..runtime.environment import environment_info
from ..runtime.isolated import probe_device, run_isolated
from ..runtime.operations import (
    OperationCancelled,
    OperationConflictError,
    OperationControl,
    OperationKind,
    OperationRegistry,
    acquire_device_or_cancel,
    bind_operation,
)
from ..protocol.schema import (
    AdapterManifest,
    BoundEvaluateRequest,
    EvaluateResponse,
    InspectRequest,
    OperatorContract,
    OperatorContractRequest,
    PreflightResult,
    ReferenceRequest,
    ReferenceResult,
)
from ..protocol.version import KERNELGEN_API_VERSION, KERNELGEN_SERVER_VERSION
from .operator_bundles import create_operator_bundle_router


def _resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    for candidate in (
        "npu",
        "musa",
        "mlu",
        "txda",
        "hygon",
        "metax",
        "iluvatar",
        "kunlunxin",
        "thead",
        "enflame",
        "cuda",
    ):
        try:
            if _make_device(candidate).count_devices_safe() > 0:
                return candidate
        except Exception:
            continue
    raise RuntimeError("no supported accelerator is available")


def create_app(
    backend: str = "auto",
    max_workers: int | None = None,
    timing: str = "auto",
    profile_artifact_root: str | Path = "/tmp/kernelgen_server_profiles",
    *,
    enable_debug_jobs: bool = True,
    debug_artifact_root: str | Path = "/tmp/kernelgen_server_debug_jobs",
    debug_retention_seconds: int = 24 * 60 * 60,
    startup_probe_timeout: float = 30,
    request_audit_root: str | Path | None = None,
    operator_bundle_root: str | Path = "/tmp/kernelgen_server_operator_bundles",
    operator_bundle_max_bytes: int = DEFAULT_OPERATOR_BUNDLE_MAX_BYTES,
) -> Any:
    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import FileResponse
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the 'server' extra to run the HTTP service") from exc

    backend = _resolve_backend(backend)
    if timing == "auto":
        timing = "profiler" if backend == "npu" else "triton"
    configure_kwargs = {"timing_strategy": timing}
    if backend == "npu":
        configure_kwargs["perf_mode"] = timing
    configure_device(backend, **configure_kwargs)
    device_count = _make_device(backend).count_devices_safe()
    if device_count <= 0:
        raise RuntimeError(f"backend {backend!r} has no visible devices")
    device_type = runtime_device_type(backend)
    devices = [f"{device_type}:{index}" for index in range(device_count)]
    if startup_probe_timeout <= 0:
        raise ValueError("startup_probe_timeout must be positive")

    def check_device(device: str) -> None:
        probe_device(
            backend,
            device,
            timing,
            timeout_seconds=startup_probe_timeout,
        )

    startup_slots = probe_startup_slots(devices, check_device)
    healthy_devices = [
        slot.device for slot in startup_slots if slot.state == "available"
    ]
    if not healthy_devices:
        reasons = "; ".join(
            f"{slot.device}: {slot.reason}" for slot in startup_slots
        )
        raise RuntimeError(f"backend {backend!r} has no healthy devices: {reasons}")

    worker_count = max_workers or len(healthy_devices)
    if worker_count <= 0:
        raise ValueError("max_workers must be positive")
    executor = ThreadPoolExecutor(max_workers=worker_count)
    device_pool = DevicePool(
        startup_slots,
        worker_count=worker_count,
        executor=executor,
        probe=check_device,
    )
    request_audit = None
    if request_audit_root:
        request_audit = RequestAudit(
            request_audit_root,
            manifest={
                "api_version": KERNELGEN_API_VERSION,
                "server_version": KERNELGEN_SERVER_VERSION,
                "backend": backend,
                "timing": timing,
                "devices": devices,
                "workers": worker_count,
                "startup_slots": [slot.as_dict() for slot in startup_slots],
            },
        )
    def run_bound_operation(operation, request, backend, device_string, timing):
        kwargs = ({"operator_bundle_root": str(Path(operator_bundle_root).resolve())}
                  if request.binding.bundle_id is not None else {})
        return run_isolated(operation, request, backend, device_string, timing, **kwargs)

    evaluation_executor = EvaluationExecutor(
        backend=backend,
        timing=timing,
        device_pool=device_pool,
        executor=executor,
        runner=run_bound_operation,
        request_audit=request_audit,
        probe_timeout_seconds=startup_probe_timeout,
    )
    status_environment = environment_info(backend, devices)
    profile_service = ProfileService(
        profile_artifact_root,
        backend=backend,
        hardware=status_environment,
        operator_bundle_root=operator_bundle_root,
    )
    operation_registry = OperationRegistry()
    operator_bundle_store = OperatorBundleStore(
        operator_bundle_root,
        max_bytes=operator_bundle_max_bytes,
    )
    debug_root = Path(debug_artifact_root).resolve()
    debug_service = None
    if enable_debug_jobs:
        debug_root.mkdir(parents=True, exist_ok=True)
        debug_store = DebugJobStore(
            debug_root,
            backend=backend,
            retention_seconds=debug_retention_seconds,
        )
        debug_runner = LocalDebugJobRunner(
            debug_root,
            backend=backend,
            catalog_root=builtin_catalog_path("simple-v6-test"),
        )
        debug_service = DebugService(
            debug_store,
            debug_runner,
            device_pool=device_pool,
            executor=executor,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            if debug_service is not None:
                await debug_service.close()
            executor.shutdown(wait=True)

    app = FastAPI(
        title="KernelGen Server",
        version=KERNELGEN_SERVER_VERSION,
        lifespan=lifespan,
    )
    app.include_router(create_operator_bundle_router(operator_bundle_store))

    def require_bundle(binding):
        if binding.bundle_id is None:
            return
        info = operator_bundle_store.get(binding.bundle_id)
        if info is None:
            raise HTTPException(status_code=404, detail="operator bundle not found")
        if info.definition != binding.definition:
            raise HTTPException(status_code=422, detail="definition does not match operator bundle")

    def require_debug_service() -> DebugService:
        if debug_service is None:
            raise HTTPException(status_code=404, detail="debug jobs are disabled")
        return debug_service

    def begin_operation(
        kind: OperationKind,
        requested_id: str | None,
    ) -> OperationControl:
        try:
            return operation_registry.create(kind, requested_id)
        except OperationConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "OPERATION_ID_CONFLICT",
                    "operation_id": requested_id,
                    "message": str(exc),
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def cancelled_http_detail(
        operation: OperationControl,
    ) -> dict[str, Any]:
        return {
            "code": "OPERATION_CANCELLED",
            "operation_id": operation.operation_id,
            "operation": operation.kind,
            "state": "CANCELLED",
        }

    @app.get("/status")
    async def status() -> dict[str, Any]:
        from kernelgen_server.runtime.source_policy import policy_snapshot
        scheduler = device_pool.snapshot(
            probe_timeout_seconds=startup_probe_timeout
        )
        return {
            "status": (
                "degraded"
                if scheduler["broken"] or scheduler["checking"]
                else "ok"
            ),
            "api_version": KERNELGEN_API_VERSION,
            "server_version": KERNELGEN_SERVER_VERSION,
            "backend": backend,
            "devices": devices,
            "workers": worker_count,
            "evaluation_adapters": ["native", "flaggems"],
            "scheduler": scheduler,
            "timing": timing,
            "profile": get_profiler(backend).describe(),
            "capabilities": {
                "native_source_policy": {"enabled": True, "workload_conditions": True,
                                         **policy_snapshot(backend)},
                "operator_contract": {"enabled": True, "test_sources": True},
                "benchmark_reference": {"enabled": True, "evaluators": ["flaggems"], "level": "core"},
                "candidate_admission": admission_capability(),
                "operation_cancel": {
                    "enabled": True,
                    "operations": ["preflight", "evaluate", "profile", "reference"],
                },
                "operator_bundle_upload": operator_bundle_store.describe(),
            },
            "debug": {
                "enabled": enable_debug_jobs,
                "runner": "local-trusted" if enable_debug_jobs else None,
                "security_boundary": False,
            },
            "request_audit": (
                request_audit.describe()
                if request_audit is not None
                else {"enabled": False}
            ),
            "trusted_clients_only": True,
            **status_environment,
        }

    @app.post("/operator-contract", response_model=OperatorContract, response_model_exclude_unset=True)
    async def export_operator_contract(request: OperatorContractRequest) -> OperatorContract:
        require_bundle(request.binding)

        def read_contract():
            catalog = resolve_binding_catalog(request.binding, operator_bundle_root=operator_bundle_root)
            operator = catalog.load(request.binding.definition)
            evidence = {}
            if request.include_test_sources:
                from ..evaluation.test_review_sources import export_test_sources
                evidence["test_sources"] = export_test_sources(catalog, operator)
            return OperatorContract(
                binding=request.binding, kind=operator.evaluator,
                catalog_api_version=catalog.api_version, definition=operator.definition,
                correctness_workloads=operator.correctness_workloads,
                timing_workloads=operator.timing_workloads,
                **evidence,
            )

        try:
            return await asyncio.get_running_loop().run_in_executor(executor, read_contract)
        except (FileNotFoundError, KeyError) as exc:
            raise HTTPException(status_code=404, detail="operator contract not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/inspect", response_model=AdapterManifest)
    async def inspect_binding(request: InspectRequest) -> AdapterManifest:
        require_bundle(request.binding)
        try:
            loop = asyncio.get_running_loop()
            adapter = create_adapter(
                request.binding,
                backend=backend,
                operator_bundle_root=operator_bundle_root,
            )
            return await loop.run_in_executor(executor, adapter.inspect)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    async def run_evaluation_operation(
        kind: OperationKind,
        request: BoundEvaluateRequest | ReferenceRequest,
        operation_id: str | None,
    ) -> PreflightResult | EvaluateResponse | ReferenceResult:
        require_bundle(request.binding)
        operation = begin_operation(kind, operation_id)
        try:
            result = await evaluation_executor.run(kind, request, operation)
        except OperationCancelled as exc:
            operation.finish("CANCELLED", str(exc))
            raise HTTPException(
                status_code=409,
                detail=cancelled_http_detail(operation),
            ) from exc
        except NoHealthyDeviceError as exc:
            operation.finish("FAILED", str(exc))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            operation.finish("FAILED", str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        operation.finish("SUCCEEDED")
        return result

    @app.post("/preflight", response_model=PreflightResult)
    async def preflight(
        request: BoundEvaluateRequest,
        operation_id: str | None = Header(
            default=None,
            alias="X-KernelGen-Operation-Id",
        ),
    ) -> PreflightResult:
        return await run_evaluation_operation("preflight", request, operation_id)

    @app.post("/reference", response_model=ReferenceResult)
    async def reference(
        request: ReferenceRequest,
        operation_id: str | None = Header(default=None, alias="X-KernelGen-Operation-Id"),
    ) -> ReferenceResult:
        return await run_evaluation_operation("reference", request, operation_id)

    @app.post("/evaluate", response_model=EvaluateResponse)
    async def evaluate(
        request: BoundEvaluateRequest,
        operation_id: str | None = Header(
            default=None,
            alias="X-KernelGen-Operation-Id",
        ),
    ) -> EvaluateResponse:
        return await run_evaluation_operation("evaluate", request, operation_id)

    @app.post("/profile", response_model=ProfileResult)
    async def profile(
        request: ProfileRequest,
        operation_id: str | None = Header(
            default=None,
            alias="X-KernelGen-Operation-Id",
        ),
    ) -> ProfileResult:
        require_bundle(request.binding)
        if request.expected_backend != backend:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"profile target expects {request.expected_backend!r}, "
                    f"server uses {backend!r}"
                ),
            )
        operation = begin_operation("profile", operation_id)
        device_string: str | None = None
        broken_reason = ""
        cancelled_error: OperationCancelled | None = None
        try:
            device_string = await acquire_device_or_cancel(
                device_pool,
                operation,
            )
            operation.mark_running(device_string)
            loop = asyncio.get_running_loop()

            def run_profile() -> ProfileResult:
                with bind_operation(operation):
                    return profile_service.run(request, device_string)

            result = await loop.run_in_executor(executor, run_profile)
        except OperationCancelled as exc:
            if device_string is not None:
                cause = "profile cancelled by client"
                broken_reason = await device_pool.probe_after_failure(
                    device_string,
                    cause,
                )
            cancelled_error = exc
        except NoHealthyDeviceError as exc:
            operation.finish("FAILED", str(exc))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            operation.finish("FAILED", str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if device_string is not None:
                await asyncio.shield(
                    device_pool.release(
                        device_string,
                        broken_reason=broken_reason,
                    )
                )
        if cancelled_error is not None:
            operation.finish("CANCELLED", str(cancelled_error))
            raise HTTPException(
                status_code=409,
                detail=cancelled_http_detail(operation),
            ) from cancelled_error
        operation.finish("SUCCEEDED")
        return result

    @app.get("/operations/{operation_id}")
    async def get_operation(operation_id: str) -> dict[str, Any]:
        try:
            operation = operation_registry.get(operation_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="operation not found",
            ) from exc
        return operation.snapshot()

    @app.delete("/operations/{operation_id}")
    async def cancel_operation(operation_id: str) -> dict[str, Any]:
        try:
            operation = operation_registry.cancel(operation_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="operation not found",
            ) from exc
        return operation.snapshot()

    @app.get("/profile_artifacts/{profile_id}/{artifact_id}")
    async def profile_artifact(profile_id: str, artifact_id: str):
        try:
            artifact_path, media_type, filename = (
                profile_service.resolve_artifact(profile_id, artifact_id)
            )
        except ProfileArtifactNotFound:
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(
            artifact_path,
            media_type=media_type,
            filename=filename,
        )

    @app.post("/debug/jobs")
    async def create_debug_job(request: DebugJobRequest) -> dict[str, Any]:
        service = require_debug_service()
        try:
            job = await service.create(request)
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        return job.model_dump(mode="json")

    @app.get("/debug/jobs/{job_id}")
    async def get_debug_job(
        job_id: str,
        wait_seconds: float = 0,
    ) -> dict[str, Any]:
        if not 0 <= wait_seconds <= 30:
            raise HTTPException(
                status_code=422,
                detail="wait_seconds must be between 0 and 30",
            )
        service = require_debug_service()
        try:
            job = await service.get(job_id, wait_seconds=wait_seconds)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="debug job not found") from exc
        return job.model_dump(mode="json")

    @app.delete("/debug/jobs/{job_id}")
    async def cancel_debug_job(job_id: str) -> dict[str, Any]:
        service = require_debug_service()
        try:
            job = service.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="debug job not found") from exc
        return job.model_dump(mode="json")

    @app.get("/debug/jobs/{job_id}/artifacts/{artifact_id}")
    async def debug_artifact(job_id: str, artifact_id: str):
        service = require_debug_service()
        try:
            path, media_type, filename = service.resolve_artifact(
                job_id,
                artifact_id,
            )
        except (KeyError, StopIteration) as exc:
            raise HTTPException(status_code=404, detail="artifact not found") from exc
        return FileResponse(
            path,
            media_type=media_type,
            filename=filename,
        )

    return app
