"""Request construction and client access for the standalone KernelGen Server.

The server already returns KernelGen's flat authoritative result, so this layer
does not aggregate or reinterpret measurements.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from kernelgen_client import KERNELGEN_API_VERSION

from kernelgen.data.target_context import (
    TargetContext,
    TargetContextError,
    build_target_context,
    normalize_device,
)

_DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
SERVER_API_VERSION = KERNELGEN_API_VERSION
_HARDWARE_FAMILIES = (
    ("ascend", "npu"),
    ("npu", "npu"),
    ("910", "npu"),
    ("cuda", "cuda"),
    ("a100", "cuda"),
    ("a800", "cuda"),
    ("h100", "cuda"),
    ("h200", "cuda"),
    ("h800", "cuda"),
    ("v100", "cuda"),
    ("l40", "cuda"),
    ("l20", "cuda"),
    ("rtx", "cuda"),
    ("4090", "cuda"),
    ("3090", "cuda"),
    ("nvidia", "cuda"),
    ("musa", "musa"),
    ("mthreads", "musa"),
    ("s5000", "musa"),
    ("mlu", "mlu"),
    ("cambricon", "mlu"),
    ("590", "mlu"),
    ("metax", "metax"),
    ("c500", "metax"),
    ("c550", "metax"),
    ("iluvatar", "iluvatar"),
    ("bi-v", "iluvatar"),
    ("hygon", "hygon"),
    ("dcu", "hygon"),
    ("bw1000", "hygon"),
    ("kunlun", "kunlunxin"),
    ("p800", "kunlunxin"),
    ("ppu", "thead"),
    ("thead", "thead"),
    ("zw810", "thead"),
)


@dataclass(frozen=True)
class EvaluationBundle:
    """Objects shared by preflight, evaluation, snapshotting, and profiling."""

    kernel_code: str
    solution: Any
    definition: Any
    workloads: list[Any]
    correctness_workloads: list[Any] = field(default_factory=list)
    timing_workloads: list[Any] = field(default_factory=list)
    is_phased: bool = True
    profile_workload_uuids: list[str] | None = None
    adapter_kind: str = "native"
    binding: Any | None = None
    benchmark_fingerprint: str = ""
    case_list: list[dict[str, Any]] = field(default_factory=list)


def hardware_family(hardware: str | None) -> str | None:
    normalized = (hardware or "").lower()
    return next(
        (family for marker, family in _HARDWARE_FAMILIES if marker in normalized),
        None,
    )


def resolve_server_url(server_url: str = "") -> str:
    return (
        server_url
        or os.environ.get("KERNELGEN_SERVER_URL", "").strip()
    )


def resolve_catalog_path(
    *,
    catalog_name: str = "",
) -> Path:
    from kernelgen.data.catalog import resolve_builtin_catalog_path

    return resolve_builtin_catalog_path(catalog_name)


def _load_evaluation_contract(context: Any) -> tuple[Any, list[Any], list[Any]]:
    snapshot_path = str(
        getattr(context, "evaluation_snapshot_path", "") or ""
    ).strip()
    if snapshot_path:
        from kernelgen.data.evaluation_snapshot import (
            load_catalog_evaluation_snapshot,
        )

        snapshot = load_catalog_evaluation_snapshot(snapshot_path)
        expected = (
            getattr(context, "catalog_name", ""),
            context.definition,
        )
        actual = (snapshot.catalog_name, snapshot.definition_name)
        if actual != expected:
            raise ValueError(
                "tool context and evaluation snapshot identities differ: "
                f"context={expected!r}, snapshot={actual!r}"
            )
        return snapshot.native_operator()

    from kernelgen_client import Catalog

    catalog = Catalog(
        resolve_catalog_path(
            catalog_name=getattr(context, "catalog_name", ""),
        )
    )
    operator = catalog.load(context.definition)
    return (
        operator.definition,
        list(operator.correctness_workloads),
        list(operator.timing_workloads),
    )


def get_service_status(server_url: str = "", *, timeout: float = 45) -> dict[str, Any]:
    from kernelgen_client.http import status

    resolved = resolve_server_url(server_url) or _DEFAULT_SERVER_URL
    return status(resolved, timeout=timeout)


def supports_operation_cancel(
    status: dict[str, Any],
    operation: str,
) -> bool:
    capabilities = status.get("capabilities")
    if not isinstance(capabilities, dict):
        return False
    capability = capabilities.get("operation_cancel")
    if not isinstance(capability, dict) or not capability.get("enabled"):
        return False
    operations = capability.get("operations")
    return isinstance(operations, list) and operation in operations


@contextmanager
def tracked_server_operation(
    run_control: Any | None,
    status: dict[str, Any],
    server_url: str,
    operation: str,
) -> Iterator[str | None]:
    """Expose an active cancellable KGS request to ``kg cancel``."""
    if run_control is not None:
        run_control.checkpoint(f"BEFORE_{operation.upper()}")
    if run_control is None or not supports_operation_cancel(status, operation):
        yield None
        return
    active = run_control.register_server_operation(operation, server_url)
    try:
        yield active.operation_id
    except BaseException as exc:
        from kernelgen_client.http import OperationCancelledError, ServerError
        from kernelgen.framework.cancellation import reconcile_server_operations

        if isinstance(exc, OperationCancelledError) or (
            isinstance(exc, ServerError) and exc.status_code in {400, 401, 403, 404, 422}
        ):
            run_control.unregister_server_operation(active.operation_id)
        else:
            reconcile_server_operations(run_control, operation_ids={active.operation_id})
        raise
    else:
        run_control.unregister_server_operation(active.operation_id)


def require_candidate_admission(status: dict[str, Any]) -> dict[str, Any]:
    """Require the shared pre-import gate, independent of KGS release numbering."""
    capabilities = status.get("capabilities")
    policy = capabilities.get("candidate_admission") if isinstance(capabilities, dict) else None
    digest = policy.get("policy_sha256") if isinstance(policy, dict) else None
    if (
        not isinstance(policy, dict)
        or type(policy.get("version")) is not int or policy["version"] != 1
        or not isinstance(digest, str) or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest)
        or not isinstance(policy.get("stages"), list)
        or not all(isinstance(stage, str) for stage in policy["stages"])
        or "preflight" not in policy["stages"]
    ):
        raise RuntimeError("Server lacks candidate_admission v1; use a Server with the unified admission gate before submitting candidates")
    return {"version": policy["version"], "policy_sha256": digest}


def service_signature(status: dict[str, Any]) -> dict[str, Any]:
    """Return fields that affect whether a preflight receipt remains usable."""
    target = status.get("target") if isinstance(status.get("target"), dict) else {}
    capabilities = status.get("capabilities") if isinstance(status.get("capabilities"), dict) else {}
    return {
        "api_version": status.get("api_version"),
        "backend": status.get("backend"),
        "timing": status.get("timing"),
        "target_device": normalize_device(target.get("device")),
        "candidate_admission": capabilities.get("candidate_admission"),
    }


def require_target_context(context: Any, status: dict[str, Any]) -> TargetContext:
    """Validate Server target identity and return its normalized TargetContext."""

    target = build_target_context(
        target_hardware=context.target_hardware,
        implementation_language=context.implementation_language.value,
        service_status=status,
    )
    expected = hardware_family(context.target_hardware)
    actual = status.get("backend")
    if expected and expected != actual:
        raise TargetContextError(
            "TARGET_HARDWARE_MISMATCH",
            f"input target_hardware {context.target_hardware!r} maps to backend "
            f"{expected!r}, but KernelGen Server runs {actual!r}",
        )
    return target


def _settings_from_context(context: Any) -> Any:
    from kernelgen_client import EvaluationSettings

    return EvaluationSettings(
        warmup_ms=context.warmup_ms,
        benchmark_ms=context.benchmark_ms,
        num_trials=context.num_trials,
        tolerance_mode=getattr(context, "eval_tolerance_mode", "strict"),
        rtol=getattr(context, "eval_rtol", 1e-2),
        atol=getattr(context, "eval_atol", 1e-2),
        timeout_seconds=context.eval_timeout_seconds,
    )


def _transport_timeout_from_context(context: Any) -> float:
    """Return the explicit HTTP budget for queueing plus one Server request."""
    configured = getattr(context, "eval_transport_timeout_seconds", 0)
    if configured:
        return float(configured)
    return float(context.eval_timeout_seconds + 300)


def _build_implementation(
    kernel_code: str,
    definition_name: str,
    *,
    language: str = "triton",
    entry_point: str = "main.py::run",
) -> Any:
    from kernelgen_client import Implementation, SourceFile

    source_path = entry_point.split("::", 1)[0]
    return Implementation(
        name=f"kernelgen-candidate-{definition_name}",
        definition=definition_name,
        language=language,
        entrypoint=entry_point,
        sources=[SourceFile(path=source_path, content=kernel_code)],
    )


def prepare_evaluation_bundle(
    kernel_path: str | Path,
    context: Any,
    *,
    entry_point: str = "main.py::run",
) -> EvaluationBundle:
    from kernelgen_client import Catalog, EvaluatorBinding, Workload
    from kernelgen_client.http import inspect
    from kernelgen_client.protocol.schema import InspectRequest

    from kernelgen.data.evaluation_snapshot import load_catalog_evaluation_snapshot
    snapshot_path = getattr(context, "evaluation_snapshot_path", "")
    snapshot = load_catalog_evaluation_snapshot(snapshot_path) if snapshot_path else None
    bundle_id = snapshot.bundle_id if snapshot is not None else None
    server_snapshot = snapshot is not None and bool(snapshot.benchmark_fingerprint)
    catalog = None if server_snapshot else Catalog(
        resolve_catalog_path(
            catalog_name=getattr(context, "catalog_name", ""),
        )
    )
    operator = catalog.load(context.definition) if catalog is not None else None
    definition, correctness, timing = _load_evaluation_contract(context)
    adapter_kind = catalog.evaluator if catalog is not None else snapshot.evaluator_kind
    if snapshot_path and not server_snapshot:
        if catalog.evaluator != "native":
            raise ValueError("evaluation snapshots are supported only by native catalogs")
        live_contract = (
            operator.definition.model_dump(mode="json", exclude_none=True),
            [
                item.model_dump(mode="json", exclude_none=True)
                for item in operator.correctness_workloads
            ],
            [
                item.model_dump(mode="json", exclude_none=True)
                for item in operator.timing_workloads
            ],
        )
        frozen_contract = (
            definition.model_dump(mode="json", exclude_none=True),
            [item.model_dump(mode="json", exclude_none=True) for item in correctness],
            [item.model_dump(mode="json", exclude_none=True) for item in timing],
        )
        if live_contract != frozen_contract:
            raise RuntimeError(
                "the native catalog changed after this run's evaluation snapshot "
                "was frozen; start a fresh run against the current catalog"
            )
    if adapter_kind == "native" and not correctness and not timing:
        raise ValueError(f"no workloads for definition {context.definition!r}")
    duplicate_names = {item.name for item in correctness} & {
        item.name for item in timing
    }
    if duplicate_names:
        raise ValueError(
            "correctness and timing workload names must be globally unique for "
            f"KernelGen ledger identity: {sorted(duplicate_names)}"
        )

    path = Path(kernel_path)
    kernel_code = path.read_text(encoding="utf-8")
    implementation = _build_implementation(
        kernel_code,
        definition.name,
        language=context.implementation_language.value,
        entry_point=entry_point,
    )
    binding = EvaluatorBinding(
        definition=definition.name,
        **({"bundle_id": bundle_id} if bundle_id else {"catalog_name": getattr(context, "catalog_name", "")}),
    )
    server_url = resolve_server_url(context.eval_server_url) or _DEFAULT_SERVER_URL
    if server_snapshot and not bundle_id:
        from kernelgen_client.http import get_operator_contract
        from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot
        status = get_service_status(server_url)
        if status.get("capabilities", {}).get("operator_contract", {}).get("enabled") is not True:
            raise RuntimeError("KGS operator contract export is required for this frozen input")
        contract = get_operator_contract(InspectRequest(binding=binding), server_url)
        observed = CatalogEvaluationSnapshot.from_server_contract(contract, snapshot.benchmark_fingerprint)
        if observed != snapshot:
            raise RuntimeError("the server Catalog changed after the evaluation snapshot was frozen")
    if bundle_id:
        capability = get_service_status(server_url).get("capabilities", {}).get("operator_bundle_upload", {})
        if not capability.get("enabled") or capability.get("evaluation_binding") is not True:
            raise RuntimeError("KGS does not support operator bundle execution")
        if adapter_kind == "flaggems" and "flaggems" not in capability.get("evaluators", []):
            raise RuntimeError("KGS does not support uploaded Gems Definitions")
    manifest = inspect(InspectRequest(binding=binding), server_url)
    if server_snapshot and manifest.kind != adapter_kind:
        raise RuntimeError("KGS inspect and frozen snapshot disagree on evaluator kind")
    benchmark_fingerprint = manifest.benchmark_fingerprint
    if server_snapshot and benchmark_fingerprint != snapshot.benchmark_fingerprint:
        raise RuntimeError("operator fingerprint differs from the verified snapshot")
    case_list: list[dict[str, Any]] = [
        case.model_dump(mode="json") for case in manifest.case_list.cases
    ]
    if adapter_kind == "flaggems":
        timing = [
            Workload(name=case["case_id"], inputs={"adapter_case": case})
            for case in case_list
        ]
    workloads = correctness + timing
    return EvaluationBundle(
        kernel_code=kernel_code,
        solution=implementation,
        definition=definition,
        workloads=workloads,
        correctness_workloads=correctness,
        timing_workloads=timing,
        profile_workload_uuids=[item.name for item in timing],
        adapter_kind=adapter_kind,
        binding=binding,
        benchmark_fingerprint=benchmark_fingerprint,
        case_list=case_list,
    )


def build_evaluate_request(bundle: EvaluationBundle, context: Any) -> Any:
    from kernelgen_client import BoundEvaluateRequest, EvaluatorBinding

    binding = bundle.binding or EvaluatorBinding(
        catalog_name=getattr(context, "catalog_name", ""),
        definition=bundle.definition.name,
    )
    return BoundEvaluateRequest(
        binding=binding,
        implementation=bundle.solution,
        settings=_settings_from_context(context),
    )


def preflight_bundle(
    bundle: EvaluationBundle,
    context: Any,
    *,
    run_control: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from kernelgen_client.http import preflight

    server_url = resolve_server_url(context.eval_server_url) or _DEFAULT_SERVER_URL
    status = get_service_status(server_url)
    require_target_context(context, status)
    require_candidate_admission(status)
    request = build_evaluate_request(bundle, context)
    transport_timeout = _transport_timeout_from_context(context)
    with tracked_server_operation(
        run_control,
        status,
        server_url,
        "preflight",
    ) as operation_id:
        result = preflight(request, server_url, timeout=transport_timeout, operation_id=operation_id)
    return result, status


def evaluate_bundle(
    bundle: EvaluationBundle,
    context: Any,
    *,
    run_control: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from kernelgen_client.http import evaluate

    server_url = resolve_server_url(context.eval_server_url) or _DEFAULT_SERVER_URL
    status = get_service_status(server_url)
    require_target_context(context, status)
    request = build_evaluate_request(bundle, context)
    transport_timeout = _transport_timeout_from_context(context)
    with tracked_server_operation(
        run_control,
        status,
        server_url,
        "evaluate",
    ) as operation_id:
        response = evaluate(request, server_url, timeout=transport_timeout, operation_id=operation_id)
    if response.server_backend != status.get("backend"):
        raise RuntimeError(
            "KernelGen Server response backend does not match GET /status: "
            f"{response.server_backend!r} != {status.get('backend')!r}"
        )
    return response.model_dump(mode="json"), status


def make_error_result(
    message: str,
    *,
    status: str = "ERROR",
    requested_hardware: str | None = None,
    server_backend: str | None = None,
) -> dict[str, Any]:
    return {
        "api_version": SERVER_API_VERSION,
        "status": status,
        "is_hack": False,
        "hack_reason": "",
        "geo_mean": None,
        "min_speedup": None,
        "worst_workload_uuid": None,
        "latency_ms": None,
        "abs_err": None,
        "rel_err": None,
        "num_workloads": 0,
        "num_passed": 0,
        "requested_hardware": requested_hardware,
        "server_backend": server_backend,
        "log": message,
        "error": message,
        "per_workload": [],
    }


def evaluate_kernel_file(
    kernel_path: str | Path,
    definition_name: str,
    *,
    catalog_name: str = "",
    target_hardware: str = "",
    server_url: str = "",
    entry_point: str = "main.py::run",
    destination_passing_style: bool = False,
    language: str | None = None,
    warmup_ms: int = 1000,
    benchmark_ms: int = 100,
    num_trials: int = 1,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Stateless file evaluator used by the ``eval_only`` MCP tool."""
    requested_hardware = (
        target_hardware
        or os.environ.get("KERNELGEN_TARGET_HARDWARE", "").strip()
    )
    resolved_server_url = resolve_server_url(server_url)
    if not resolved_server_url:
        return make_error_result(
            "server_url not given and KERNELGEN_SERVER_URL is unset",
            requested_hardware=requested_hardware or None,
        )
    if destination_passing_style:
        return make_error_result(
            "KernelGen Server schema v1 supports value-returning run() only",
            requested_hardware=requested_hardware or None,
        )
    kernel = Path(kernel_path)
    if not kernel.is_file():
        return make_error_result(
            f"kernel file not found: {kernel}",
            requested_hardware=requested_hardware or None,
        )

    try:
        resolve_catalog_path(catalog_name=catalog_name)
        service = get_service_status(resolved_server_url)
        backend = service.get("backend")
        # A small namespace keeps eval_only independent of workflow models.
        context = SimpleNamespace(
            definition=definition_name,
            target_hardware=requested_hardware,
            implementation_language=SimpleNamespace(value=language or "triton"),
            eval_server_url=resolved_server_url,
            warmup_ms=warmup_ms,
            benchmark_ms=benchmark_ms,
            num_trials=num_trials,
            eval_timeout_seconds=timeout_seconds,
            eval_transport_timeout_seconds=max(
                timeout_seconds + 10,
                2 * timeout_seconds + 90,
            ),
            catalog_name=catalog_name,
        )
        target_context = require_target_context(context, service)
        context.target_hardware = target_context.device
        bundle = prepare_evaluation_bundle(
            kernel,
            context,
            entry_point=entry_point,
        )
        result, _ = evaluate_bundle(bundle, context)
    except TargetContextError as exc:
        return make_error_result(
            str(exc),
            status=exc.code,
            requested_hardware=requested_hardware or None,
            server_backend=locals().get("backend"),
        )
    except Exception as exc:  # noqa: BLE001
        return make_error_result(
            f"KernelGen Server evaluation failed: {exc}",
            requested_hardware=requested_hardware or None,
        )

    result["requested_hardware"] = requested_hardware or None
    result["server_backend"] = backend
    return result
