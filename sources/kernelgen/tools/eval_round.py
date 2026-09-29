"""Frozen-plan client evaluation and atomic ledger recording for MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernelgen.tools.kernelgen_server_adapter import SERVER_API_VERSION

if TYPE_CHECKING:
    from kernelgen.data.experiment_plan import ExperimentPlan
    from kernelgen.data.tool_context import ToolContext
    from kernelgen.tools.profile_round import EvaluationBundle

# Statuses that mean "no measurement happened" -> do not record a round.
_UNMEASURED = {
    "ERROR",
    "HARDWARE_MISMATCH",
    "TARGET_BACKEND_MISSING",
    "TARGET_DEVICE_MISSING",
    "TARGET_HARDWARE_MISMATCH",
    "TRANSPORT_ERROR",
    "RUN_CANCELLED",
}

def _error_result(
    message: str,
    *,
    status: str = "ERROR",
    requested_hardware: str | None = None,
    server_backend: str | None = None,
) -> dict:
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
        "per_workload": [],
    }


def record_and_augment(
    ledger_dir,
    result: dict,
    kernel_code: str,
    experiment_plan: dict,
    *,
    profile_enabled: bool = False,
    definition_name: str = "",
    target_hardware: str = "",
    implementation_language: str = "triton",
    candidate_path: str = "tmp/main.py",
) -> dict:
    """Record an eval result to the worktree ledger and return it augmented with
    round_num + is_new_best. Host-testable (no torch/server). Skips recording for
    unmeasured statuses."""
    if result.get("status") in _UNMEASURED:
        return result
    from kernelgen.data.ledger import Ledger
    led = Ledger(ledger_dir)
    rec = led.record_eval(
        result,
        kernel_code,
        experiment_plan,
        profile_enabled=profile_enabled,
        definition_name=definition_name,
        target_hardware=target_hardware,
        implementation_language=implementation_language,
        candidate_path=candidate_path,
    )
    return {
        **result,
        "round_num": rec.round_num,
        "is_new_best": rec.is_new_best,
        "profile_required": rec.profile_required,
        "profile_status": rec.profile_status,
        "conclusion_recorded": rec.conclusion_recorded,
    }


def evaluate_kernel_and_record(
    ledger_dir: str | Path,
    kernel: str | Path,
    bundle: EvaluationBundle,
    context: ToolContext,
    *,
    experiment_plan: dict,
    server_backend: str,
    candidate_path: str = "tmp/main.py",
    run_control: Any | None = None,
) -> tuple[dict, int]:
    """Evaluate through KernelGen Server and atomically record the result."""

    ledger_path = Path(ledger_dir)
    kernel_path = Path(kernel)
    if not kernel_path.is_file():
        return _error_result(f"kernel file not found: {kernel_path}"), 2

    try:
        from kernelgen.data.ledger import Ledger

        normalized_plan = Ledger(ledger_path).validate_experiment_plan(
            experiment_plan
        ).model_dump(mode="python")
    except Exception as exc:  # noqa: BLE001
        return _error_result(f"invalid experiment plan: {exc}"), 2

    from kernelgen.tools.kernelgen_server_adapter import hardware_family

    requested_hardware = context.target_hardware or None
    expected_backend = hardware_family(context.target_hardware)
    if expected_backend and expected_backend != server_backend:
        return _error_result(
            (
                f"target hardware {context.target_hardware!r} requires backend "
                f"{expected_backend!r}, but the eval service runs {server_backend!r}"
            ),
            status="TARGET_HARDWARE_MISMATCH",
            requested_hardware=requested_hardware,
            server_backend=server_backend,
        ), 2

    try:
        from kernelgen.data.target_context import TargetContextError
        from kernelgen.tools.kernelgen_server_adapter import evaluate_bundle
        from kernelgen_client.http import OperationCancelledError

        kernel_code = kernel_path.read_text(encoding="utf-8")
        if run_control is None:
            result, service = evaluate_bundle(bundle, context)
        else:
            result, service = evaluate_bundle(
                bundle,
                context,
                run_control=run_control,
            )
        actual_backend = str(service.get("backend", ""))
        if actual_backend != server_backend:
            return _error_result(
                "KernelGen Server backend changed after preflight: "
                f"{server_backend!r} -> {actual_backend!r}",
                status="TARGET_HARDWARE_MISMATCH",
                requested_hardware=requested_hardware,
                server_backend=actual_backend,
            ), 2
    except TargetContextError as exc:
        return _error_result(
            str(exc),
            status=exc.code,
            requested_hardware=requested_hardware,
            server_backend=server_backend,
        ), 2
    except OperationCancelledError as exc:
        if run_control is not None:
            from kernelgen.framework.run_control import cooperative_cancel_result

            cancelled = cooperative_cancel_result(
                run_control,
                stage="DURING_EVALUATION",
            )
            if cancelled is not None:
                return cancelled, 0
        return {
            "status": "RUN_CANCELLED",
            "cancel_requested": True,
            "reason_code": "server_operation_cancelled",
            "reason": str(exc),
            "stage": "DURING_EVALUATION",
            "instruction": (
                "STOP NOW. Return the final report using only results "
                "already persisted."
            ),
        }, 0
    except Exception as exc:  # noqa: BLE001
        return _error_result(
            f"KernelGen Server client failed: {exc}",
            requested_hardware=requested_hardware,
            server_backend=server_backend,
        ), 2

    result["requested_hardware"] = requested_hardware
    result["server_backend"] = server_backend
    # Freeze the exposed runtime identity alongside the measured result.
    result["evaluation_service"] = service
    try:
        result = record_and_augment(
            ledger_path,
            result,
            kernel_code,
            normalized_plan,
            profile_enabled=context.profile_enabled,
            definition_name=context.definition,
            target_hardware=context.target_hardware,
            implementation_language=context.implementation_language.value,
            candidate_path=candidate_path,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "error": f"authoritative evaluation completed but ledger record failed: {exc}",
            "evaluation_result": result,
        }, 2

    status = result.get("status")
    if status == "PASSED":
        return result, 0
    if status in _UNMEASURED:
        return result, 2
    return result, 1


def evaluate_round(
    workspace: str | Path,
    kernel_path: str,
    experiment_plan: ExperimentPlan | dict,
    *,
    confirm_no_knowledge_applied: bool = False,
) -> dict:
    """Run the complete workspace-bound evaluation-round use case."""
    from kernelgen.data.ledger import Ledger
    from kernelgen.data.tool_context import (
        load_tool_context,
        resolve_workspace_file,
    )
    from kernelgen.framework.run_control import (
        RunState,
        WorkspaceRunControl,
        cooperative_cancel_result,
    )
    from kernelgen.tools.preflight import (
        consume_preflight_receipt,
        validate_preflight_receipt,
    )
    from kernelgen.tools.lifecycle import lifecycle_gate
    from kernelgen.tools.profile_round import (
        prepare_evaluation_bundle,
        write_evaluation_snapshot,
    )

    root = Path(workspace).resolve()
    run_control = WorkspaceRunControl(root, source="mcp:eval_round")
    cancelled = cooperative_cancel_result(
        run_control,
        stage="BEFORE_EVALUATION",
    )
    if cancelled is not None:
        return {**cancelled, "exit_code": 0}
    ledger = Ledger(root)
    gate = lifecycle_gate(ledger, action="evaluate")
    if gate is not None:
        return gate

    try:
        normalized_plan = ledger.validate_experiment_plan(experiment_plan)
        if confirm_no_knowledge_applied and normalized_plan.knowledge_uses:
            raise ValueError(
                "confirm_no_knowledge_applied cannot be true when "
                "knowledge_uses is non-empty"
            )
    except ValueError as exc:
        return {
            "status": "ERROR",
            "error": f"invalid experiment plan: {exc}",
            "exit_code": 2,
        }

    if not normalized_plan.knowledge_uses and not confirm_no_knowledge_applied:
        after = None
        boundary_known = True
        if ledger.history.rounds:
            after = ledger.history.rounds[-1].evaluation.evaluated_at
            boundary_known = after is not None
        if boundary_known:
            from kernelgen.workflows.knowledge_bridge import recent_detail_read_refs

            detail_read_refs = recent_detail_read_refs(root, after=after)
            if detail_read_refs:
                return {
                    "status": "KNOWLEDGE_USE_REVIEW_REQUIRED",
                    "detail_read_count": len(detail_read_refs),
                    "detail_read_refs": detail_read_refs[:8],
                    "detail_reads_truncated": len(detail_read_refs) > 8,
                    "instruction": (
                        "Review the detail-read knowledge that may have changed "
                        "the submitted solution. Add only knowledge embodied in "
                        "the solution to knowledge_uses, or call eval_round again "
                        "with confirm_no_knowledge_applied=true."
                    ),
                    "exit_code": 2,
                }

    try:
        context = load_tool_context(root)
        kernel = resolve_workspace_file(root, kernel_path)
        if not kernel.is_file():
            raise ValueError(f"kernel file not found: {kernel_path}")
        bundle = prepare_evaluation_bundle(kernel, context)
        receipt, receipt_error = validate_preflight_receipt(root, bundle, context)
    except Exception as exc:  # noqa: BLE001
        from kernelgen.data.target_context import TargetContextError

        return {
            "status": (
                exc.code
                if isinstance(exc, TargetContextError)
                else "ERROR"
            ),
            "stage": "configuration",
            "error": str(exc),
            "exit_code": 2,
        }

    if receipt is None:
        return {
            "status": "PREFLIGHT_REQUIRED",
            "required_tool": "preflight_kernel",
            "error": receipt_error,
            "instruction": (
                "Call preflight_kernel for the current kernel and fix any failures "
                "before submitting eval_round again."
            ),
            "exit_code": 2,
        }

    evaluated_kernel = consume_preflight_receipt(root, receipt)
    expected_round = len(ledger.history.rounds) + 1
    run_control.update_progress(
        state=RunState.RUNNING,
        stage="EVALUATING",
        max_round=(ledger.stop_config.max_round if ledger.stop_config else None),
        message=f"Evaluating round {expected_round}",
    )
    run_control.record_event(
        "EVALUATION_STARTED",
        message=f"Evaluating round {expected_round}",
        stage="EVALUATING",
        data={"round_num": expected_round},
    )
    result, exit_code = evaluate_kernel_and_record(
        root,
        evaluated_kernel,
        bundle,
        context,
        experiment_plan=normalized_plan.model_dump(mode="python"),
        server_backend=str((receipt.get("target") or {}).get("backend", "")),
        candidate_path=kernel_path,
        run_control=run_control,
    )

    if result.get("status") == "RUN_CANCELLED":
        run_control.record_event(
            "EVALUATION_CANCELLED",
            message=result.get("reason", "Evaluation cancelled"),
            stage="EVALUATING",
            level="WARNING",
            data={"round_num": expected_round},
        )
        return {**result, "exit_code": exit_code}

    if result.get("round_num"):
        try:
            snapshot = write_evaluation_snapshot(
                root,
                result["round_num"],
                bundle,
                result,
                context,
            )
            profile_state = Ledger(root).attach_evaluation_snapshot(
                result["round_num"],
                evaluation_fingerprint=snapshot["evaluation_fingerprint"],
                snapshot_path=snapshot["snapshot_path"],
            )
        except Exception as exc:  # noqa: BLE001
            run_control.record_event(
                "EVALUATION_SNAPSHOT_FAILED",
                message=str(exc),
                stage="EVALUATING",
                level="ERROR",
                visibility="DEBUG",
                data={"round_num": result["round_num"]},
            )
            return {
                "status": "ERROR",
                "error": (
                    "evaluation was recorded but snapshot attachment failed: "
                    f"{exc}"
                ),
                "evaluation_result": result,
                "round_num": result["round_num"],
                "exit_code": 2,
            }
        result.update(snapshot)
        result.update(profile_state)

    if result.get("profile_required"):
        result["profile_task"] = {
            "recommended_agent": "kernel-profile-analyzer",
            "round_num": result["round_num"],
            "instruction": (
                "Profile this new-best round when the measurements need diagnosis."
            ),
        }
    if result.get("round_num"):
        result["conclusion_task"] = {
            "required_tool": "finalize_round",
            "round_num": result["round_num"],
            "instruction": (
                "Call finalize_round exactly once with the expected-versus-observed "
                "conclusion. Its returned CONTINUE/STOP verdict controls whether "
                "another eval_round is allowed; pending profile analysis is advisory."
            ),
        }
    observed_ledger = Ledger(root)
    observed_round = result.get("round_num")
    next_stage = (
        "PROFILING"
        if result.get("profile_required")
        else "FINALIZING_ROUND"
        if observed_round
        else "CODING"
    )
    run_control.update_progress(
        state=RunState.RUNNING,
        stage=next_stage,
        message=f"Evaluation {result.get('status', 'UNKNOWN')}",
    )
    run_control.record_event(
        "EVALUATION_COMPLETED",
        message=f"Evaluation {result.get('status', 'UNKNOWN')}",
        stage="EVALUATING",
        level=("INFO" if result.get("status") == "PASSED" else "WARNING"),
        data={
            "round_num": observed_round,
            "status": result.get("status"),
            "geo_mean": result.get("geo_mean"),
            "is_new_best": bool(result.get("is_new_best")),
            "best_round": observed_ledger.history.best_round,
            "best_geo_mean": observed_ledger.history.best_geo_mean or None,
            "profile_required": bool(result.get("profile_required")),
        },
    )
    return {**result, "exit_code": exit_code}
