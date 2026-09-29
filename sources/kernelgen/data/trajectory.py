"""Measured source plus diffs for distillation; bounded facts for epoch synthesis.

Keep host dispatch, helpers and kernel bodies together: all affect the measured
candidate. Compress repeated source with diffs, never by dropping executable code.
"""

from __future__ import annotations

import difflib
import math
from typing import Any, Dict, List, Sequence


def build_trajectory(rounds: List[dict], rewrite_threshold: float = 0.2) -> str:
    """Build code evolution plus plan-to-outcome evidence from schema-v2 rounds."""
    if not rounds:
        return "(no rounds)"

    parts = []
    by_round = {item.get("round_num", index + 1): item for index, item in enumerate(rounds)}
    # The full initial implementation anchors later diffs, including host code.
    first_code = (rounds[0].get("solution") or {}).get("code", "")
    first_plan = rounds[0].get("plan") or {}
    first_evaluation = rounds[0].get("evaluation") or {}
    first_conclusion = rounds[0].get("conclusion") or {}
    parts.append(
        "\n".join(
            [
                f"Initial implementation (R1, {first_evaluation.get('status', '?')}, geo={_format_geo(first_evaluation.get('geo_mean'))}):",
                f"  plan: {first_plan.get('strategy', '')}",
                f"  hypothesis: {first_plan.get('hypothesis', '')}",
                f"  expectation: {_format_expected(first_plan)}",
                f"  expectation_status: {first_conclusion.get('expectation_status', '')}",
                f"  perf_gap: {first_conclusion.get('perf_gap_analysis', '')}",
                f"```python\n{first_code}\n```",
            ]
        )
    )

    for i in range(1, len(rounds)):
        prev_code = (rounds[i - 1].get("solution") or {}).get("code", "")
        curr_code = (rounds[i].get("solution") or {}).get("code", "")
        plan = rounds[i].get("plan") or {}
        evaluation = rounds[i].get("evaluation") or {}
        conclusion = rounds[i].get("conclusion") or {}
        status = evaluation.get("status", "?")
        geo = evaluation.get("geo_mean")
        round_num = rounds[i].get("round_num", i + 1)

        strategy = (plan.get("strategy", "") or "")[:150]
        root_cause = (conclusion.get("root_cause", "") or "")[:150]
        perf_gap = (conclusion.get("perf_gap_analysis", "") or "")[:180]
        debug_lesson = (conclusion.get("debug_lesson", "") or "")[:100]
        opt_level = conclusion.get("optimization_level", "")
        strategy_evolution = (conclusion.get("strategy_evolution", "") or "")[:120]

        header = f"R{round_num-1}→R{round_num} ({status}, geo={_format_geo(geo)}, level={opt_level}):"
        narrative_lines = [
            f"  plan: {strategy}",
            f"  hypothesis: {plan.get('hypothesis', '')}",
            f"  expectation: {_format_expected(plan)}",
            f"  expectation_status: {conclusion.get('expectation_status', '')}",
        ]
        comparison = evaluation.get("comparison") or {}
        performance_baseline_round = comparison.get(
            "performance_baseline_round_num",
            comparison.get("baseline_round_num"),
        )
        if performance_baseline_round is not None:
            narrative_lines.append(
                f"  measured_vs_best_R{performance_baseline_round}: geo_delta={_format_pct(comparison.get('geo_mean_delta_pct'))}"
            )
        narrative_lines.extend(_paired_timing_evidence(rounds[i], by_round))
        if root_cause:
            narrative_lines.append(f"  root_cause: {root_cause}")
        if perf_gap:
            narrative_lines.append(f"  perf_gap: {perf_gap}")
        if debug_lesson:
            narrative_lines.append(f"  debug_lesson: {debug_lesson}")
        if strategy_evolution:
            narrative_lines.append(f"  evolution: {strategy_evolution}")
        narrative = "\n".join(narrative_lines)

        # Code diff/REWRITE
        if not curr_code or not prev_code:
            parts.append(f"{header}\n{narrative}\n  (code unavailable)")
            continue

        diff_lines = list(difflib.unified_diff(
            prev_code.splitlines(), curr_code.splitlines(), lineterm="", n=2))
        change_count = sum(1 for l in diff_lines if l.startswith('+') or l.startswith('-'))
        total_lines = max(len(curr_code.splitlines()), 1)

        if change_count / total_lines <= rewrite_threshold:
            diff_text = "\n".join(diff_lines)
            parts.append(f"{header}\n{narrative}\n```diff\n{diff_text}\n```")
        else:
            parts.append(
                f"{header}\n{narrative}\n[REWRITE — {change_count}/{total_lines} lines changed "
                f"({change_count/total_lines:.0%}); showing full implementation]\n```python\n{curr_code}\n```"
            )

    return "\n\n".join(parts)


def build_synthesis_trajectory(
    rounds: List[dict],
    *,
    best_round: int = 0,
    max_chars: int = 8000,
) -> str:
    """Build a bounded, code-free trajectory for cross-agent synthesis.

    Distillation needs source evolution, while epoch synthesis only needs enough
    evidence to compare experiments and choose useful next directions.  Keep a
    compact row for every round when it fits, then reserve detailed evidence for
    the authoritative best and final rounds.  Source code is deliberately left
    out; the workflow supplies the epoch's best kernel exactly once.
    """
    if not rounds:
        return "(no rounds)"
    if max_chars < 4096:
        raise ValueError("max_chars must be at least 4096")

    ordered = sorted(rounds, key=lambda item: item.get("round_num", 0))
    by_round = {item.get("round_num", index + 1): item for index, item in enumerate(ordered)}
    if best_round not in by_round:
        best_round = _infer_best_round(ordered)

    first_round = ordered[0].get("round_num", 1)
    final_round = ordered[-1].get("round_num", len(ordered))
    best_geo = ((by_round.get(best_round) or {}).get("evaluation") or {}).get("geo_mean")
    final_geo = ((by_round.get(final_round) or {}).get("evaluation") or {}).get("geo_mean")
    overview = (
        f"total_rounds={len(ordered)} | "
        f"first=R{first_round} | "
        f"best=R{best_round or '?'} geo={_format_geo(best_geo)} | "
        f"final=R{final_round} geo={_format_geo(final_geo)}"
    )

    # Best/final evidence is placed outside the variable-size round table so it
    # cannot disappear when a long trajectory reaches the character budget.
    important_numbers = [number for number in (best_round, final_round) if number]
    important_rounds = []
    for number in important_numbers:
        if number in by_round and number not in important_rounds:
            important_rounds.append(number)
    evidence = "\n\n".join(
        _format_synthesis_evidence(by_round[number], by_round=by_round, label=(
            "BEST+FINAL" if best_round == final_round == number
            else "BEST" if number == best_round
            else "FINAL"
        ))
        for number in important_rounds
    )
    evidence_section = f"Key evidence:\n{evidence}" if evidence else ""

    table_header = "Round table:\nround | status | geo | min | level | architecture | planned change"
    rows = [_format_synthesis_row(item) for item in ordered]
    required_rounds = {first_round, best_round, final_round}
    failed_rounds = {
        item.get("round_num")
        for item in ordered
        if (item.get("evaluation") or {}).get("status") != "PASSED"
    }
    row_by_round = {
        item.get("round_num"): row
        for item, row in zip(ordered, rows)
    }
    priority = []
    for number in (
        [first_round, best_round, final_round]
        + [
            item.get("round_num")
            for item in ordered
            if item.get("round_num") in failed_rounds
        ]
        + [item.get("round_num") for item in ordered]
    ):
        if number in row_by_round and number not in priority:
            priority.append(number)

    fixed = "\n\n".join(
        part for part in (overview, evidence_section, table_header) if part
    )
    row_budget = max_chars - len(fixed) - 2
    selected = []
    used = 0
    for number in priority:
        row = row_by_round[number]
        cost = len(row) + (1 if selected else 0)
        if number in required_rounds or used + cost <= row_budget:
            selected.append(number)
            used += cost
    selected.sort()
    table_rows = "\n".join(row_by_round[number] for number in selected)
    omitted = len(rows) - len(selected)
    if omitted:
        marker = f"\n[{omitted} non-key round rows omitted by synthesis budget]"
        if len(table_rows) + len(marker) <= row_budget:
            table_rows += marker

    result = "\n\n".join(
        part
        for part in (
            overview,
            evidence_section,
            f"{table_header}\n{table_rows}" if table_rows else table_header,
        )
        if part
    )
    # Per-field bounds and row selection should keep the result within budget.
    # This final guard only handles unusually large structured key_params.
    if len(result) > max_chars:
        marker = "\n[summary truncated to synthesis budget]"
        result = result[: max_chars - len(marker)].rstrip() + marker
    return result


def _infer_best_round(rounds: Sequence[Dict[str, Any]]) -> int:
    candidates = []
    for index, item in enumerate(rounds):
        evaluation = item.get("evaluation") or {}
        geo = evaluation.get("geo_mean")
        if evaluation.get("status") == "PASSED" and geo is not None:
            candidates.append((float(geo), item.get("round_num", index + 1)))
    return max(candidates)[1] if candidates else 0


def _format_synthesis_row(item: Dict[str, Any]) -> str:
    plan = item.get("plan") or {}
    evaluation = item.get("evaluation") or {}
    conclusion = item.get("conclusion") or {}
    row = (
        f"R{item.get('round_num', '?')} | "
        f"{_clip(evaluation.get('status', '?'), 16)} | "
        f"{_format_geo(evaluation.get('geo_mean'))} | "
        f"{_format_geo(evaluation.get('min_speedup'))} | "
        f"{_clip(conclusion.get('optimization_level', ''), 18)} | "
        f"{_clip(plan.get('strategy', ''), 80)}"
    )
    if evaluation.get("status") != "PASSED" and conclusion.get("root_cause"):
        row += f" | cause={_clip(conclusion['root_cause'], 100)}"
    return row


def _format_synthesis_evidence(item: Dict[str, Any], *, by_round: dict, label: str) -> str:
    plan = item.get("plan") or {}
    expected = plan.get("expected_effect") or {}
    evaluation = item.get("evaluation") or {}
    comparison = evaluation.get("comparison") or {}
    profile = item.get("profile") or {}
    conclusion = item.get("conclusion") or {}
    params = ", ".join(
        f"{key}={_clip(value, 48)}"
        for key, value in sorted((plan.get("key_params") or {}).items())
    )
    lines = [
        f"{label} R{item.get('round_num', '?')}: "
        f"status={evaluation.get('status', '?')}, "
        f"geo={_format_geo(evaluation.get('geo_mean'))}, "
        f"min={_format_geo(evaluation.get('min_speedup'))}",
        f"  strategy: {_clip(plan.get('strategy', ''), 180)}",
        f"  hypothesis: {_clip(plan.get('hypothesis', ''), 220)}",
        f"  code_changes: {_clip(plan.get('code_changes', ''), 220)}",
        "  expected: "
        f"{_clip(expected.get('metric', ''), 48)} "
        f"{_clip(expected.get('direction', ''), 32)} via "
        f"{_clip(expected.get('mechanism', ''), 180)}",
        f"  expectation_status: {_clip(conclusion.get('expectation_status', ''), 32)}",
    ]
    if params:
        lines.append(f"  key_params: {_clip(params, 300)}")
    performance_baseline_round = comparison.get(
        "performance_baseline_round_num",
        comparison.get("baseline_round_num"),
    )
    if performance_baseline_round is not None:
        lines.append(
            f"  measured_vs_best_R{performance_baseline_round}: "
            f"geo_delta={_format_pct(comparison.get('geo_mean_delta_pct'))}"
        )
    lines.extend(_paired_timing_evidence(item, by_round))
    if conclusion.get("root_cause"):
        lines.append(f"  root_cause: {_clip(conclusion['root_cause'], 220)}")
    if conclusion.get("perf_gap_analysis"):
        lines.append(f"  perf_gap: {_clip(conclusion['perf_gap_analysis'], 240)}")
    if conclusion.get("next_suggestion"):
        lines.append(f"  next: {_clip(conclusion['next_suggestion'], 220)}")
    if profile.get("status") and profile.get("status") != "not_required":
        lines.append(
            f"  profile: {_clip(profile.get('status', ''), 32)} | "
            f"{_clip(profile.get('summary', ''), 220)}"
        )
    return "\n".join(lines)


def _paired_timing_evidence(item: dict, by_round: dict) -> list[str]:
    """Derive paired latencies from ledger rows, never from model explanations.

    A changing reference can move headline speedup without improving a candidate.
    Keep the performance incumbent and experiment parent distinct; neither is a
    randomized control. Missing/partial timing must not become a favorable subset.
    """
    comparison = (item.get("evaluation") or {}).get("comparison") or {}
    best = comparison.get("performance_baseline_round_num", comparison.get("baseline_round_num"))
    parent = item.get("experiment_parent_round_num")
    controls = []
    if best is not None:
        controls.append((best, "best+parent" if best == parent else "best"))
    if parent is not None and parent != best:
        controls.append((parent, "parent"))

    lines = []
    for number, role in controls:
        prefix = f"  paired_timing_vs_{role}_R{number}: "
        baseline = by_round.get(number)
        if baseline is None:
            lines.append(prefix + "unavailable (baseline absent from trajectory)")
            continue
        evaluations = [record.get("evaluation") or {} for record in (baseline, item)]
        if any(evaluation.get("status") != "PASSED" for evaluation in evaluations):
            lines.append(prefix + "unavailable (round not PASSED)")
            continue
        rows = [[row for row in evaluation.get("workloads", []) if row.get("phase") == "timing"]
                for evaluation in evaluations]
        mappings = [{row.get("uuid"): row for row in side} for side in rows]
        before, after = mappings
        if (not before or before.keys() != after.keys()
                or any(not isinstance(key, str) or not key for key in before)
                or any(len(mapping) != len(side) for mapping, side in zip(mappings, rows))
                or any(before[key].get("axes", {}) != after[key].get("axes", {}) for key in before)):
            lines.append(prefix + "unavailable (timing identities missing or inconsistent)")
            continue
        fields = ("latency_ms", "reference_latency_ms")
        if any(row.get("status") != "PASSED" or any(
            isinstance(row.get(field), bool) or not isinstance(row.get(field), (int, float))
            or not math.isfinite(row[field]) or row[field] <= 0 for field in fields
        ) for side in rows for row in side):
            lines.append(prefix + "unavailable (invalid or failed timing)")
            continue
        try:
            deltas = [100 * math.expm1(math.fsum(
                math.log(after[key][field]) - math.log(before[key][field]) for key in before
            ) / len(before)) for field in fields]
        except OverflowError:
            deltas = [math.inf]
        if not all(math.isfinite(delta) for delta in deltas):
            lines.append(prefix + "unavailable (non-finite comparison)")
            continue
        lines.append(
            prefix + f"cases={len(before)}, candidate_latency_delta={_format_pct(deltas[0])}, "
            f"reference_latency_delta={_format_pct(deltas[1])} "
            "(geometric mean after/before minus one; lower candidate is faster; "
            "observational, not causal or a noise estimate)"
        )
    return lines


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _format_geo(value) -> str:
    return "N/A" if value is None else f"{value:.6f}x"


def _format_pct(value) -> str:
    return "N/A" if value is None else f"{value:+.6f}%"


def _format_expected(plan: dict) -> str:
    expected = plan.get("expected_effect") or {}
    return " ".join(
        part
        for part in (
            str(expected.get("metric", "")),
            str(expected.get("direction", "")),
            str(expected.get("mechanism", "")),
        )
        if part
    )
