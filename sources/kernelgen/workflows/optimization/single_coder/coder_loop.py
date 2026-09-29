"""Python-owned supervision for the Coder's long optimization loop."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from kernelgen.agents.coder import CoderAgent, CoderInput, CoderReport
from kernelgen.data.ledger import Ledger
from kernelgen.framework.base import AgentContractError
from kernelgen.framework.run_control import RunControl, RunState, WorkspaceRunControl

if TYPE_CHECKING:
    from kernelgen.workflows.optimization.single_coder.workflow import (
        SingleCoderOptimizationInput,
    )


def run_coder_loop(
    inp: "SingleCoderOptimizationInput",
    workspace: Path,
    runtime: Any,
    *,
    run_control: RunControl | None = None,
) -> CoderReport:
    """Run or resume Coder until its durable verdict reaches terminal STOP."""
    control = run_control or WorkspaceRunControl(
        workspace,
        source="workflow:coder_loop",
    )
    control.checkpoint("BEFORE_CODER")
    control.update_progress(
        state=RunState.RUNNING,
        stage="CODING",
        progress_kind="rounds",
        max_round=inp.max_round,
        message="Coder optimization loop is running",
    )
    coder_input = inp.model_dump(include=set(CoderInput.model_fields))
    coder_input["ledger_dir"] = str(workspace)

    report: CoderReport | None = None
    coder = CoderAgent()
    continuation_round: int | None = None
    pending_recovery_round: int | None = None
    premature_return_pending = False

    initial_ledger = Ledger(workspace)
    initial_ledger.validate_identity(
        definition_name=inp.definition.name,
        target_hardware=inp.target_hardware,
        implementation_language=inp.implementation_language.value,
    )
    if initial_ledger.coder_completed:
        return _reuse_terminal_report(initial_ledger)

    restored_session_id = _restore_runtime_session(runtime)
    if restored_session_id:
        pending_conclusion = initial_ledger.pending_conclusion_round()
        if pending_conclusion is not None:
            pending_recovery_round = pending_conclusion.round_num
            restored_state = f"pending conclusion for R{pending_recovery_round}"
        elif initial_ledger.history.rounds:
            latest_round = _validate_coder_boundary(initial_ledger, inp)
            continuation_round = latest_round.round_num
            restored_state = f"CONTINUE after R{continuation_round}"
        else:
            premature_return_pending = True
            restored_state = "zero measured rounds"
        print(
            "[OptimizeDefinition] Restored provider session "
            f"{restored_session_id} ({restored_state})",
            flush=True,
        )
    elif initial_ledger.history.rounds:
        print(
            "[OptimizeDefinition] Existing nonterminal ledger has no recoverable "
            "provider session; starting a ledger-seeded Coder session",
            flush=True,
        )

    for invocation_index in range(inp.max_coder_sessions):
        control.checkpoint("BEFORE_CODER_INVOCATION")
        invocation_kind = "fresh"
        contract_error: AgentContractError | None = None
        if pending_recovery_round is not None:
            invocation_kind = "pending_conclusion"
            if not _can_resume(runtime):
                raise RuntimeError(
                    "Coder returned before finalize_round completed for round "
                    f"{pending_recovery_round}, but its provider session cannot "
                    "be resumed"
                )
            prompt = _pending_conclusion_prompt(pending_recovery_round)
        elif continuation_round is not None:
            invocation_kind = "continue"
            if not _can_resume(runtime):
                raise RuntimeError(
                    "Coder returned after finalize_round authorized another round "
                    f"{continuation_round}, but its provider session cannot be "
                    "resumed"
                )
            prompt = _continuation_prompt(continuation_round)
        elif premature_return_pending:
            invocation_kind = "premature_return"
            if not _can_resume(runtime):
                raise RuntimeError(
                    "Coder returned before recording any measured round, but "
                    "its provider session cannot be resumed"
                )
            prompt = _premature_return_prompt()

        try:
            control.record_event(
                "CODER_INVOCATION_STARTED",
                stage="CODING",
                data={
                    "invocation": invocation_index + 1,
                    "max_invocations": inp.max_coder_sessions,
                    "kind": invocation_kind,
                },
            )
            if invocation_kind == "fresh":
                _add_fresh_context(coder_input, Ledger(workspace))
                report = coder.run(coder_input, runtime)
            else:
                report = coder.continue_session(
                    prompt,
                    runtime,
                    knowledge_enabled=inp.knowledge_enabled,
                )
        except AgentContractError as exc:
            # CoderReport is narrative only. Inspect the authoritative ledger
            # before deciding whether malformed model output is fatal.
            contract_error = exc
            report = None
        control.checkpoint("AFTER_CODER_INVOCATION")
        control.record_event(
            "CODER_INVOCATION_COMPLETED",
            stage="CODING",
            data={
                "invocation": invocation_index + 1,
                "kind": invocation_kind,
            },
        )

        completed_ledger = Ledger(workspace)
        completed_ledger.validate_identity(
            definition_name=inp.definition.name,
            target_hardware=inp.target_hardware,
            implementation_language=inp.implementation_language.value,
        )
        if (
            pending_recovery_round is not None
            and (
                not completed_ledger.history.rounds
                or completed_ledger.history.rounds[-1].round_num
                < pending_recovery_round
            )
        ):
            raise RuntimeError(
                "Coder ledger regressed unexpectedly during same-session "
                f"recovery from round {pending_recovery_round}"
            ) from contract_error
        pending_conclusion = completed_ledger.pending_conclusion_round()
        if pending_conclusion is not None:
            if (
                pending_recovery_round is not None
                and pending_conclusion.round_num < pending_recovery_round
            ):
                raise RuntimeError(
                    "Coder pending conclusion regressed unexpectedly during "
                    f"same-session recovery from round {pending_recovery_round} "
                    f"to round "
                    f"{pending_conclusion.round_num}"
                ) from contract_error
            if not _can_resume(runtime):
                raise RuntimeError(
                    "Coder returned before finalize_round completed for round "
                    f"{pending_conclusion.round_num}, but its provider session "
                    "cannot be resumed"
                ) from contract_error
            if invocation_index + 1 >= inp.max_coder_sessions:
                raise RuntimeError(
                    "Coder returned before finalize_round completed for round "
                    f"{pending_conclusion.round_num}; no Coder invocation remains "
                    "for directed recovery"
                ) from contract_error
            pending_recovery_round = pending_conclusion.round_num
            premature_return_pending = False
            continuation_round = None
            print(
                "[OptimizeDefinition] Coder returned with a pending conclusion "
                f"for round {pending_recovery_round}; resuming the same provider "
                f"session to finalize it ({invocation_index + 2}/"
                f"{inp.max_coder_sessions})",
                flush=True,
            )
            continue

        latest_round = _validate_coder_boundary(completed_ledger, inp)
        if latest_round is None:
            if not _can_resume(runtime):
                # A schema-valid zero-round report can be a legitimate terminal
                # result for providers that do not persist conversations (for
                # example, a startup/target validation failure). Invalid output
                # still fails loudly because there is no session to recover.
                if contract_error is not None:
                    raise contract_error
                break
            if invocation_index + 1 >= inp.max_coder_sessions:
                raise RuntimeError(
                    "Coder returned before recording any measured round; no "
                    "Coder invocation remains for same-session recovery"
                ) from contract_error
            premature_return_pending = True
            pending_recovery_round = None
            continuation_round = None
            print(
                "[OptimizeDefinition] Coder returned before recording any "
                "measured round; resuming the same provider session to "
                f"continue unfinished work ({invocation_index + 2}/"
                f"{inp.max_coder_sessions})",
                flush=True,
            )
            continue
        if completed_ledger.coder_completed:
            if contract_error is not None:
                report = _recover_terminal_report(
                    completed_ledger,
                    contract_error,
                )
            break

        premature_return_pending = False
        pending_recovery_round = None
        continuation_round = latest_round.round_num
        if invocation_index + 1 >= inp.max_coder_sessions:
            if _stop_at_session_limit(completed_ledger, inp):
                if contract_error is not None:
                    report = _recover_terminal_report(
                        Ledger(workspace),
                        contract_error,
                    )
                break
            raise RuntimeError(
                "Coder returned after finalize_round authorized another round "
                f"{latest_round.round_num}; a terminal STOP verdict is required "
                f"after {inp.max_coder_sessions} Coder invocation(s)"
            )
        print(
            "[OptimizeDefinition] Coder returned with CONTINUE after round "
            f"{latest_round.round_num}; resuming the same provider session "
            f"({invocation_index + 2}/{inp.max_coder_sessions})",
            flush=True,
        )

    if report is None:
        raise RuntimeError("Coder did not run")
    return report


def _stop_with_valid_best(
    ledger: Ledger,
    inp: "SingleCoderOptimizationInput",
    *,
    code: str,
    reason: str,
) -> bool:
    """Persist a supervisor stop only after the minimum viable run completed."""
    if len(ledger.history.rounds) < inp.min_rounds:
        return False
    if ledger.history.best_round <= 0:
        return False
    best = ledger.get_round(ledger.history.best_round)
    if (
        best.evaluation.status != "PASSED"
        or best.evaluation.is_hack
        or not best.solution.code
    ):
        return False
    latest = ledger.history.rounds[-1]
    full_reason = f"{reason}; preserving validated best R{best.round_num}"
    ledger.record_supervisor_stop(
        latest.round_num,
        code=code,
        reason=full_reason,
    )
    print(
        f"[OptimizeDefinition] STOP ({code}): {full_reason}",
        flush=True,
    )
    return True


def _stop_at_session_limit(
    ledger: Ledger,
    inp: "SingleCoderOptimizationInput",
) -> bool:
    """End a sufficiently measured run even when it never produced a best."""
    reason = (
        "maximum Coder invocation limit reached without a terminal stop verdict"
    )
    if _stop_with_valid_best(
        ledger,
        inp,
        code="coder_session_limit_reached",
        reason=reason,
    ):
        return True
    if len(ledger.history.rounds) < inp.min_rounds:
        return False

    latest = ledger.history.rounds[-1]
    full_reason = (
        f"{reason}; no validated PASSED candidate after "
        f"{len(ledger.history.rounds)} measured round(s)"
    )
    ledger.record_supervisor_stop(
        latest.round_num,
        code="coder_session_limit_reached",
        reason=full_reason,
    )
    print(
        f"[OptimizeDefinition] STOP (coder_session_limit_reached): {full_reason}",
        flush=True,
    )
    return True


def _can_resume(runtime: Any) -> bool:
    return bool(getattr(runtime, "last_session_id", None)) and callable(
        getattr(runtime, "resume", None)
    )


def _restore_runtime_session(runtime: Any) -> str | None:
    """Ask the bound Runtime to recover only its own workspace conversation."""
    session_id = getattr(runtime, "last_session_id", None)
    if session_id:
        return str(session_id)
    restore = getattr(runtime, "restore_session", None)
    if not callable(restore):
        return None
    restored = restore()
    return str(restored) if restored else None


def _reuse_terminal_report(ledger: Ledger) -> CoderReport:
    """Skip Coder when the authoritative ledger already contains terminal STOP."""
    stopped = ledger.stopped_round()
    if not ledger.coder_completed:
        raise RuntimeError("terminal Coder report requested without a STOP verdict")

    if ledger.history.best_round > 0:
        best = ledger.get_round(ledger.history.best_round)
        status = best.evaluation.status
        best_summary = f"validated best R{best.round_num}"
    else:
        status = stopped.evaluation.status
        best_summary = "no validated best candidate"

    print(
        "[OptimizeDefinition] Reusing terminal authoritative Coder ledger "
        f"through R{stopped.round_num} ({best_summary})",
        flush=True,
    )
    return CoderReport(
        status=status,
        summary=(
            "Reused the terminal authoritative Coder ledger through round "
            f"{stopped.round_num}; {best_summary}."
        ),
    )


def _continuation_prompt(round_num: int) -> str:
    return (
        f"The authoritative finalize_round verdict for round {round_num} is "
        "CONTINUE. Your previous response ended the Coder loop too early. "
        "Continue optimizing in this same conversation and workspace from the "
        "persisted candidate transition. Do not repeat the baseline and do not "
        "emit the final CoderReport until finalize_round returns continue=false."
    )


def _premature_return_prompt() -> str:
    return (
        "The authoritative ledger still has zero measured rounds, so this "
        "Coder workflow is not terminal. Your previous response ended before "
        "the first eval_round, even if it looked like a final report. Continue "
        "the original optimization in this same conversation and workspace "
        "from the next unfinished step. Do not restart or repeat setup, status "
        "checks, or knowledge retrieval that already completed. Use the "
        "available tools to establish or repair the candidate, then run the "
        "required preflight, eval_round, and finalize_round loop. Do not emit "
        "the final CoderReport until finalize_round returns continue=false."
    )


def _pending_conclusion_prompt(round_num: int) -> str:
    return (
        "The authoritative ledger shows that eval_round completed for round "
        f"{round_num}, but finalize_round did not. Before doing anything else, "
        f"you MUST call finalize_round for exactly round {round_num} with your "
        "evidence-based RoundConclusion. Do not use Python or direct file edits "
        "to invent or patch a conclusion. Then obey the tool's authoritative "
        "CONTINUE/STOP verdict: continue optimization on CONTINUE, and emit the "
        "final CoderReport only after STOP."
    )


def _add_fresh_context(coder_input: dict[str, Any], ledger: Ledger) -> None:
    """Inject durable state for an explicitly reused workspace."""
    if not ledger.history.rounds:
        return
    coder_input["history_summary"] = (
        "RESUMED WORKSPACE CONTRACT:\n"
        "- This is a continuation in an existing workspace, not a fresh "
        "optimization run.\n"
        "- Do not submit plan.kind='baseline'; the next measured round must use "
        "a non-baseline plan kind.\n"
        "- Continue from the persisted candidate transition and history until "
        "finalize_round returns STOP.\n\n"
        + ledger.history.format_for_prompt()
    )
    coder_input["seed_code"] = ledger.best.get("code", "")


def _validate_coder_boundary(
    ledger: Ledger,
    inp: "SingleCoderOptimizationInput",
):
    """Validate the authoritative state left behind by a Coder invocation."""
    pending_conclusion = ledger.pending_conclusion_round()
    if pending_conclusion is not None:
        raise RuntimeError(
            "Coder returned before finalize_round completed for round "
            f"{pending_conclusion.round_num}"
        )
    if not ledger.history.rounds:
        return None

    latest_round = ledger.history.rounds[-1]
    if latest_round.next_verdict is None:
        raise RuntimeError(
            "Coder returned before finalize_round finalized round "
            f"{latest_round.round_num}"
        )
    if latest_round.next_verdict.code == "user_cancelled":
        raise RuntimeError("cancelled Coder ledger must be resumed before continuing")
    return latest_round


def _recover_terminal_report(
    ledger: Ledger,
    error: AgentContractError,
) -> CoderReport:
    """Recover narrative output only from a terminal validated PASSED best."""
    if not ledger.coder_completed or ledger.history.best_round <= 0:
        raise error
    best = ledger.get_round(ledger.history.best_round)
    if (
        best.evaluation.status != "PASSED"
        or best.evaluation.is_hack
        or best.evaluation.geo_mean is None
        or not best.solution.code
    ):
        raise error
    print(
        "[OptimizeDefinition] CoderReport validation failed after terminal "
        f"STOP; recovering narrative result from authoritative ledger best "
        f"R{best.round_num}",
        flush=True,
    )
    return CoderReport(
        status=best.evaluation.status,
        summary=(
            "Recovered the final narrative report from the terminal "
            "authoritative ledger after CoderReport validation failed; "
            f"preserved validated best round {best.round_num}."
        ),
    )
