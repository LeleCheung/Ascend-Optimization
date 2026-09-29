"""Small deterministic ranking helpers for measured knowledge usage."""

from __future__ import annotations

from kernelgen.knowledge.models import KnowledgeUsageSummary


def usage_score(summary: KnowledgeUsageSummary) -> float:
    """Reward independent success more than combined use; retain regressions."""

    return (
        min(summary.single_success_count, 3) * 4.0
        + min(summary.combined_success_count, 3) * 1.0
        + min(summary.correctness_recovery_count, 3) * 2.0
        - min(summary.regression_count, 3) * 4.0
    )


def usage_reasons(
    summary: KnowledgeUsageSummary,
    *,
    scope_label: str,
) -> list[str]:
    reasons = [f"scope match: {scope_label}"]
    if summary.single_success_count:
        reasons.append(
            "same-scope single successes: "
            f"{summary.single_success_count}"
        )
    if summary.combined_success_count:
        reasons.append(
            "same-scope combined successes: "
            f"{summary.combined_success_count} (not independent attribution)"
        )
    if summary.correctness_recovery_count:
        reasons.append(
            "same-scope correctness recoveries: "
            f"{summary.correctness_recovery_count}"
        )
    if summary.regression_count:
        reasons.append(
            f"same-scope regressions: {summary.regression_count}"
        )
    if summary.agent_confirmed_count:
        reasons.append(
            "Agent-confirmed applications: "
            f"{summary.agent_confirmed_count}"
        )
    if summary.agent_partially_confirmed_count:
        reasons.append(
            "Agent-partially-confirmed applications: "
            f"{summary.agent_partially_confirmed_count}"
        )
    if summary.agent_not_confirmed_count:
        reasons.append(
            "Agent-not-confirmed applications: "
            f"{summary.agent_not_confirmed_count}"
        )
    if summary.agent_inconclusive_count:
        reasons.append(
            "Agent-inconclusive applications: "
            f"{summary.agent_inconclusive_count}"
        )
    if not summary.evaluated_count:
        reasons.append("no same-scope evaluated use yet")
    return reasons
