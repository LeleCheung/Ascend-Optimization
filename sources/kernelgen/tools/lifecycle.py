"""Shared agent-facing formatting for ledger lifecycle gates."""

from __future__ import annotations

from typing import Literal

from kernelgen.data.ledger import Ledger


def lifecycle_gate(
    ledger: Ledger,
    *,
    action: Literal["preflight", "evaluate"],
) -> dict | None:
    """Return a blocking tool result, or ``None`` when the run may advance."""
    blocker = ledger.lifecycle_blocker()
    if blocker is None:
        return None

    record = blocker.record
    action_text = (
        "preflighting another candidate"
        if action == "preflight"
        else "evaluating another candidate"
    )
    if blocker.kind == "conclusion":
        return {
            "status": "ROUND_CONCLUSION_REQUIRED",
            "round_num": record.round_num,
            "required_tool": "finalize_round",
            "instruction": (
                f"Call finalize_round exactly once for round {record.round_num} "
                f"before {action_text}."
            ),
            "exit_code": 2,
        }
    if blocker.kind == "stopped":
        verdict = record.next_verdict
        return {
            "status": "RUN_STOPPED",
            "continue": False,
            "round_num": record.round_num,
            "reason_code": verdict.code,
            "reason": verdict.reason,
            "round_count": len(ledger.history.rounds),
            "max_round": ledger.stop_config.max_round,
            "instruction": "Do not edit or evaluate again. Return the final CoderReport.",
            "exit_code": 2,
        }
    raise AssertionError(f"unknown lifecycle blocker: {blocker.kind}")
