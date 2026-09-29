"""Target compile/smoke preflight and content-bound evaluation receipts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kernelgen.data.tool_context import (
    ToolContext,
    load_tool_context,
    resolve_workspace_file,
)
from kernelgen.tools.profile_round import (
    EvaluationBundle,
    prepare_evaluation_bundle,
)
from kernelgen.tools.kernelgen_server_adapter import SERVER_API_VERSION


PREFLIGHT_SCHEMA_VERSION = "3.0"
PREFLIGHT_RELATIVE_DIR = Path(".kernelgen") / "preflight"
_RECEIPT_NAME = "receipt.json"
_ATTEMPTS_NAME = "attempts.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _append_attempt(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _bundle_identity(bundle: EvaluationBundle, context: ToolContext) -> dict[str, Any]:
    solution = bundle.solution.model_dump(mode="json")
    definition = bundle.definition.model_dump(mode="json")
    workloads = [item.model_dump(mode="json") for item in bundle.workloads]
    workload_uuids = [item["name"] for item in workloads]
    profile_workload_uuids = (
        list(bundle.profile_workload_uuids)
        if bundle.profile_workload_uuids is not None
        else list(workload_uuids)
    )
    return {
        "kernel_sha256": _sha256_bytes(bundle.kernel_code.encode("utf-8")),
        "solution_sha256": _canonical_sha256(solution),
        "definition_name": bundle.definition.name,
        "definition_sha256": _canonical_sha256(definition),
        "workload_uuids": workload_uuids,
        "workload_sha256": [_canonical_sha256(item) for item in workloads],
        "workload_mode": "phased" if bundle.is_phased else "legacy",
        "profile_workload_uuids": profile_workload_uuids,
        "target_hardware": context.target_hardware,
        "implementation_language": context.implementation_language.value,
        "destination_passing_style": context.destination_passing_style,
        "catalog_name": _catalog_identity(context),
        "adapter_kind": bundle.adapter_kind,
        "benchmark_fingerprint": bundle.benchmark_fingerprint,
    }


def _catalog_identity(context: ToolContext) -> str:
    return context.catalog_name


def _receipt_path(workspace: Path) -> Path:
    return workspace / PREFLIGHT_RELATIVE_DIR / _RECEIPT_NAME


def _remove_current_receipt(workspace: Path) -> None:
    try:
        _receipt_path(workspace).unlink()
    except FileNotFoundError:
        pass


def run_preflight(
    workspace: str | Path,
    bundle: EvaluationBundle,
    context: ToolContext,
    *,
    run_control: Any | None = None,
) -> dict[str, Any]:
    """Ask KernelGen Server to compile/smoke every candidate specialization."""
    root = Path(workspace).resolve()
    preflight_root = root / PREFLIGHT_RELATIVE_DIR
    attempt_id = uuid.uuid4().hex
    _remove_current_receipt(root)

    from kernelgen.tools.kernelgen_server_adapter import (
        preflight_bundle,
        require_candidate_admission,
        require_target_context,
        service_signature,
    )

    if run_control is None:
        service_result, service_status = preflight_bundle(bundle, context)
    else:
        service_result, service_status = preflight_bundle(
            bundle,
            context,
            run_control=run_control,
        )
    target_context = require_target_context(context, service_status)
    context = context.model_copy(
        update={
            "target_hardware": target_context.device,
            "target_context": target_context,
        }
    )
    context.write(root)
    identity = _bundle_identity(bundle, context)
    target_result = {
        **service_result,
        "backend": service_status.get("backend"),
        "device": target_context.device,
    }
    base = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "api_version": service_status.get("api_version", ""),
        "service_signature": service_signature(service_status),
        "attempt_id": attempt_id,
        "created_at": _utc_now(),
        **identity,
    }
    policy_error = ""
    try:
        require_candidate_admission(service_status)
    except RuntimeError as exc:
        policy_error = str(exc)
    rejected = service_result.get("is_hack") is True
    passed = service_result.get("status") == "PASSED" and not rejected and not policy_error
    timed_out = service_result.get("status") == "TIMEOUT"
    suspected_device_error = (
        service_result.get("status") == "SUSPECTED_DEVICE_ERROR"
    )
    result = {
        **base,
        "status": (
            "PASSED"
            if passed
            else "SUSPECTED_DEVICE_ERROR"
            if suspected_device_error
            else "TIMEOUT"
            if timed_out
            else "FAILED"
        ),
        "stage": "candidate_admission" if rejected or policy_error else service_result.get(
            "stage",
            "complete" if passed else "service_preflight",
        ),
        "passed": passed,
        "target": target_result,
    }
    if not passed:
        _remove_current_receipt(root)
        if policy_error:
            instruction = policy_error + ". "
        elif rejected:
            instruction = (
                "Candidate admission rejected: " + str(service_result.get("hack_reason", ""))
                + ". Fix only the candidate, then retry preflight_kernel. "
            )
        elif suspected_device_error:
            instruction = (
                "Server reported a suspected device error. Do not edit the candidate. "
                "Inspect get_server_status.scheduler: only broken>0 confirms a failed device probe. "
                "If probes recovered, retry outside the optimization loop with lower concurrency "
                "or a larger configured timeout. "
            )
        elif timed_out:
            instruction = (
                "Target preflight exceeded the Server execution budget. "
                "Simplify the candidate or retry preflight_kernel when the "
                "timeout was caused by transient target load. "
            )
        else:
            instruction = "Fix the target compile/smoke failure, then call preflight_kernel again. "
        result["instruction"] = instruction + "Do not call eval_round yet."
        _append_attempt(preflight_root / _ATTEMPTS_NAME, result)
        return result

    if (
        base["api_version"] != SERVER_API_VERSION
        or not target_result.get("backend")
        or not target_result.get("device")
        or not base["service_signature"].get("timing")
    ):
        result.update({
            "status": "FAILED",
            "stage": "protocol",
            "passed": False,
            "instruction": (
                "The eval service returned an incomplete versioned preflight result. "
                "Deploy/restart a compatible eval service, then retry."
            ),
        })
        _remove_current_receipt(root)
        _append_attempt(preflight_root / _ATTEMPTS_NAME, result)
        return result

    candidates = preflight_root / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    candidate = candidates / f"{attempt_id}.py"
    candidate.write_text(bundle.kernel_code, encoding="utf-8")
    receipt = {
        **base,
        "status": "PASSED",
        "stage": "complete",
        "passed": True,
        "target": target_result,
        "candidate_path": str(candidate.relative_to(root)),
    }
    _atomic_json(_receipt_path(root), receipt)
    _append_attempt(preflight_root / _ATTEMPTS_NAME, receipt)
    return {
        **receipt,
        "receipt_path": str(_receipt_path(root).relative_to(root)),
        "instruction": (
            "Preflight passed for this exact kernel. Submit eval_round before editing it."
        ),
    }


def preflight_candidate(
    workspace: str | Path,
    kernel_path: str = "tmp/main.py",
) -> dict[str, Any]:
    """Run the complete workspace-bound preflight use case."""
    from kernelgen.data.ledger import Ledger
    from kernelgen.framework.run_control import (
        RunState,
        WorkspaceRunControl,
        cooperative_cancel_result,
    )
    from kernelgen.tools.lifecycle import lifecycle_gate
    from kernelgen_client.http import OperationCancelledError

    root = Path(workspace).resolve()
    run_control = WorkspaceRunControl(root, source="mcp:preflight_kernel")
    cancelled = cooperative_cancel_result(
        run_control,
        stage="BEFORE_PREFLIGHT",
    )
    if cancelled is not None:
        return {**cancelled, "passed": False, "exit_code": 0}
    ledger = Ledger(root)
    gate = lifecycle_gate(ledger, action="preflight")
    if gate is not None:
        return gate

    run_control.update_progress(
        state=RunState.RUNNING,
        stage="PREFLIGHT",
        message="Checking candidate admission, compilation and smoke execution",
    )
    run_control.record_event(
        "PREFLIGHT_STARTED",
        stage="PREFLIGHT",
    )
    try:
        context = load_tool_context(root)
        kernel = resolve_workspace_file(root, kernel_path)
        if not kernel.is_file():
            raise ValueError(f"kernel file not found: {kernel_path}")
        bundle = prepare_evaluation_bundle(kernel, context)
        result = run_preflight(
            root,
            bundle,
            context,
            run_control=run_control,
        )
        response = {
            **result,
            "exit_code": 0 if result.get("status") == "PASSED" else 1,
        }
    except OperationCancelledError as exc:
        _remove_current_receipt(root)
        cancelled = cooperative_cancel_result(
            run_control,
            stage="DURING_PREFLIGHT",
        ) or {
            "status": "RUN_CANCELLED",
            "cancel_requested": True,
            "reason_code": "server_operation_cancelled",
            "reason": str(exc),
            "stage": "DURING_PREFLIGHT",
            "instruction": (
                "STOP NOW. Return the final report using only results "
                "already persisted."
            ),
        }
        return {**cancelled, "passed": False, "exit_code": 0}
    except Exception as exc:  # noqa: BLE001
        from kernelgen.data.target_context import TargetContextError

        _remove_current_receipt(root)
        status = exc.code if isinstance(exc, TargetContextError) else "ERROR"
        response = {
            "status": status,
            "stage": "configuration",
            "passed": False,
            "error": str(exc),
            "instruction": "Fix the configuration error, then call preflight_kernel again.",
            "exit_code": 2,
        }
    run_control.update_progress(
        state=RunState.RUNNING,
        stage="CODING",
        message=f"Preflight {response.get('status', 'UNKNOWN')}",
    )
    run_control.record_event(
        "PREFLIGHT_COMPLETED",
        message=f"Preflight {response.get('status', 'UNKNOWN')}",
        stage="PREFLIGHT",
        level=(
            "INFO"
            if response.get("status") == "PASSED"
            else "WARNING"
        ),
        data={
            "status": response.get("status"),
            "passed": bool(response.get("passed")),
            "attempt_id": response.get("attempt_id"),
        },
    )
    return response


def validate_preflight_receipt(
    workspace: str | Path,
    bundle: EvaluationBundle,
    context: ToolContext,
) -> tuple[dict[str, Any] | None, str]:
    """Validate current code/context/service against the most recent receipt."""
    root = Path(workspace).resolve()
    path = _receipt_path(root)
    if not path.is_file():
        return None, "no successful preflight receipt exists"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"preflight receipt is unreadable: {exc}"
    if receipt.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        return None, "preflight receipt schema version is incompatible"
    if (
        receipt.get("api_version") != SERVER_API_VERSION
        or not receipt.get("service_signature")
    ):
        return None, "preflight receipt lacks service protocol identity"
    if receipt.get("status") != "PASSED" or not receipt.get("passed"):
        return None, "latest preflight did not pass"
    if receipt.get("target", {}).get("is_hack") is True:
        return None, "preflight candidate admission was rejected"

    expected = _bundle_identity(bundle, context)
    for field, value in expected.items():
        if receipt.get(field) != value:
            return None, f"preflight receipt is stale: {field} changed"

    candidate_raw = receipt.get("candidate_path", "")
    candidate = (root / candidate_raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None, "preflight candidate path escapes the workspace"
    if not candidate.is_file():
        return None, "preflight candidate snapshot is missing"
    if _sha256_bytes(candidate.read_bytes()) != expected["kernel_sha256"]:
        return None, "preflight candidate snapshot hash does not match"

    from kernelgen.tools.kernelgen_server_adapter import (
        get_service_status,
        require_candidate_admission,
        require_target_context,
        service_signature,
    )

    service = get_service_status(context.eval_server_url)
    require_target_context(context, service)
    try:
        require_candidate_admission(service)
    except RuntimeError as exc:
        return None, str(exc)
    if service_signature(service) != receipt.get("service_signature"):
        return None, "KernelGen Server backend, protocol, timing strategy, or admission policy changed"
    return receipt, ""


def consume_preflight_receipt(workspace: str | Path, receipt: dict[str, Any]) -> Path:
    """Consume the current receipt and return its immutable candidate path."""
    root = Path(workspace).resolve()
    path = _receipt_path(root)
    consumed = root / PREFLIGHT_RELATIVE_DIR / "last-consumed-receipt.json"
    _atomic_json(consumed, {**receipt, "consumed_at": _utc_now()})
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    candidate = (root / receipt["candidate_path"]).resolve()
    candidate.relative_to(root)
    return candidate
