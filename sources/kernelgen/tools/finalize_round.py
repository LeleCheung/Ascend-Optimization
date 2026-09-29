"""Atomically finalize one measured ledger round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_EXIT_OK = 0
_EXIT_ERROR = 2


def _emit(obj: dict, code: int) -> int:
    print(json.dumps(obj, indent=2, ensure_ascii=False))
    return code


def _advisory_recommendations(ledger) -> list[str]:
    """Return non-blocking follow-up work for the finalized ledger."""
    pending = ledger.pending_profile_round()
    if pending is None:
        return []
    return [f"profile best round {pending.round_num} if diagnosis is useful"]


def _verdict_result(
    verdict,
    snapshot: dict,
    *,
    recommendations: list[str] | None = None,
) -> dict:
    """Format one persisted CONTINUE/STOP verdict for the Coder."""
    if verdict.should_continue:
        instruction = (
            f"CONTINUE optimizing. {verdict.reason}. "
            f"Current best: {snapshot['best_geo_mean']:.6f}x "
            f"(round {snapshot['best_round']}). "
            f"Rounds without improvement: "
            f"{snapshot['rounds_without_improvement']}."
        )
    else:
        instruction = (
            "STOP NOW. You MUST stop the optimization loop immediately and output "
            "your final CoderReport JSON. "
            f"Reason: {verdict.reason}. "
            f"Best achieved: {snapshot['best_geo_mean']:.6f}x "
            f"(round {snapshot['best_round']}). "
            "Do NOT attempt another round. Output the ```json CoderReport now."
        )
    return {
        "status": "CONTINUE" if verdict.should_continue else "RUN_STOPPED",
        "continue": verdict.should_continue,
        "instruction": instruction,
        "reason_code": verdict.code,
        "reason": verdict.reason,
        "snapshot": snapshot,
        "recommendations": list(recommendations or []),
    }


def finalize_round(ledger_dir: str | Path, conclusion: dict) -> dict:
    """Atomically finalize a round or replay its persisted finalization."""
    ledger_path = Path(ledger_dir)
    if not ledger_path.is_dir():
        return {"recorded": False, "error": f"ledger dir not found: {ledger_path}"}
    if not isinstance(conclusion, dict):
        return {"recorded": False, "error": "conclusion must be an object"}

    round_num = conclusion.get("round_num")
    if not isinstance(round_num, int):
        return {"recorded": False, "error": "conclusion must include integer 'round_num'"}
    try:
        from kernelgen.data.round_conclusion import RoundConclusion

        normalized = RoundConclusion.model_validate(conclusion)
    except ValueError as exc:
        return {
            "recorded": False,
            "round_num": round_num,
            "error": f"invalid round conclusion: {exc}",
        }

    from kernelgen.data.ledger import Ledger
    from kernelgen.framework.run_control import RunState, WorkspaceRunControl

    run_control = WorkspaceRunControl(
        ledger_path,
        source="mcp:finalize_round",
    )
    run_control.record_event(
        "ROUND_FINALIZATION_STARTED",
        stage="FINALIZING_ROUND",
        data={"round_num": round_num},
    )
    idempotent_replay = False
    cancellation_requested = False
    try:
        ledger = Ledger(ledger_path)
        known_rounds = {record.round_num for record in ledger.history.rounds}
        if round_num not in known_rounds:
            return {
                "recorded": False,
                "round_num": round_num,
                "error": f"no eval round {round_num} in ledger; call eval first",
            }
        pending_conclusion = ledger.pending_conclusion_round()
        if pending_conclusion is None:
            record = ledger.get_round(round_num)
            if record is not ledger.history.rounds[-1]:
                return {
                    "recorded": False,
                    "round_num": round_num,
                    "error": "ROUND_FINALIZATION_ALREADY_ADVANCED",
                    "instruction": (
                        "This finalized round is no longer the latest round; "
                        "use the current ledger lifecycle state."
                    ),
                }
            if record.conclusion is None or record.next_verdict is None:
                return {
                    "recorded": False,
                    "round_num": round_num,
                    "error": "ROUND_FINALIZATION_INCOMPLETE",
                }
            if (
                record.conclusion.model_dump(mode="json")
                != normalized.model_dump(mode="json")
            ):
                return {
                    "recorded": False,
                    "round_num": round_num,
                    "error": "ROUND_CONCLUSION_CONFLICT",
                    "instruction": (
                        "Round conclusion is already recorded with different "
                        "content; do not overwrite authoritative ledger history."
                    ),
                }
            verdict = record.next_verdict
            snapshot = ledger.snapshot()
            idempotent_replay = True
        elif pending_conclusion.round_num != round_num:
            return {
                "recorded": False,
                "round_num": pending_conclusion.round_num,
                "error": "ROUND_CONCLUSION_REQUIRED",
                "instruction": f"Record the conclusion for round {pending_conclusion.round_num} first.",
            }
        else:
            verdict = ledger.finalize_round(round_num, normalized)
            snapshot = ledger.snapshot()

        # A Coder invocation can span multiple measured rounds without yielding
        # to the Python supervisor.  finalize_round is therefore the earliest
        # durable safe point at which an external cancellation can prevent the
        # model from starting another candidate.  Never interrupt the Eval that
        # produced this round; persist its conclusion first, then replace only a
        # CONTINUE verdict with a cooperative STOP.
        cancellation = run_control.cancellation_state()
        cancellation_requested = cancellation.requested
        if cancellation_requested and verdict.should_continue:
            verdict = ledger.record_supervisor_stop(
                round_num,
                code="user_cancelled",
                reason=(
                    cancellation.reason
                    or "cancellation requested by the run owner"
                ),
            )
            snapshot = ledger.snapshot()
        if cancellation_requested:
            run_control.record_event(
                "CANCEL_SAFE_POINT_REACHED",
                message=(
                    cancellation.reason
                    or "Cancellation reached the round-finalization safe point"
                ),
                stage="CANCELLING",
                data={
                    "round_num": round_num,
                    "reason_code": verdict.code,
                },
            )
        run_control.update_progress(
            state=(
                RunState.CANCEL_REQUESTED
                if cancellation_requested
                else RunState.RUNNING
            ),
            stage=(
                "CANCELLING"
                if cancellation_requested
                else "CODING"
                if verdict.should_continue
                else "STOPPING"
            ),
            stop_reason="" if verdict.should_continue else verdict.reason,
        )
        run_control.record_event(
            "ROUND_FINALIZED",
            message=verdict.reason,
            stage="FINALIZING_ROUND",
            data={
                "round_num": round_num,
                "continue": verdict.should_continue,
                "reason_code": verdict.code,
                "best_round": snapshot["best_round"],
                "best_geo_mean": snapshot["best_geo_mean"] or None,
                "evaluation_status": ledger.get_round(round_num).evaluation.status,
                "cancel_requested": cancellation_requested,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "recorded": False,
            "round_num": round_num,
            "error": f"ledger write failed: {exc}",
        }
    try:
        candidate_transition = ledger.transition_candidate(round_num)
    except Exception as exc:  # noqa: BLE001
        candidate_transition = {
            "candidate_action": "ERROR",
            "candidate_path": ledger.get_round(round_num).solution.candidate_path,
            "best_round": ledger.history.best_round,
            "candidate_error": str(exc),
        }
    verdict_result = _verdict_result(
        verdict,
        snapshot,
        recommendations=_advisory_recommendations(ledger),
    )
    if candidate_transition.get("candidate_action") == "KEEP":
        verdict_result["instruction"] = (
            "KEEP applied: "
            f"{candidate_transition['candidate_path']} was atomically overwritten "
            f"with the evaluated round {round_num} solution. Any edits made after "
            f"eval_round for round {round_num} were discarded. "
            f"{verdict_result['instruction']}"
        )
    return {
        "recorded": True,
        "round_num": round_num,
        "conclusion_recorded": True,
        "round_finalized": True,
        "idempotent_replay": idempotent_replay,
        "cancel_requested": cancellation_requested,
        **candidate_transition,
        **verdict_result,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="kernelgen.tools.finalize_round",
        description="Atomically attach a conclusion and finalize a measured round.",
    )
    parser.add_argument("conclusion_file", help="Path to the round-conclusion JSON file.")
    parser.add_argument("--ledger-dir", default="", help="Worktree dir holding .ledger.json (default: cwd).")
    args = parser.parse_args(argv)
    ledger_dir = Path(args.ledger_dir) if args.ledger_dir else Path.cwd()
    conclusion_path = Path(args.conclusion_file)
    if not conclusion_path.is_file():
        return _emit(
            {"recorded": False, "error": f"conclusion file not found: {conclusion_path}"},
            _EXIT_ERROR,
        )
    try:
        conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        return _emit(
            {"recorded": False, "error": f"invalid JSON in {conclusion_path}: {exc}"},
            _EXIT_ERROR,
        )
    result = finalize_round(ledger_dir, conclusion)
    return _emit(result, _EXIT_OK if result.get("recorded") else _EXIT_ERROR)


if __name__ == "__main__":
    raise SystemExit(main())
