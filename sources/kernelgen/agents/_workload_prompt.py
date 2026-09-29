"""Bound prompt context without changing authoritative evaluation coverage."""

from __future__ import annotations

from typing import Any

MAX_PROMPT_WORKLOADS = 10


def _evenly_spaced_sample(workloads: list[dict[str, Any]], limit: int):
    """Select deterministic representatives while retaining range endpoints."""
    if limit <= 0:
        return []
    if len(workloads) <= limit:
        return list(workloads)
    if limit == 1:
        return [workloads[0]]
    last = len(workloads) - 1
    return [workloads[round(index * last / (limit - 1))] for index in range(limit)]


def sample_workloads_for_prompt(
    workloads: list[dict[str, Any]], limit: int = MAX_PROMPT_WORKLOADS,
) -> list[dict[str, Any]]:
    """Share the prompt budget across phases, retaining ordered range endpoints."""
    if len(workloads) <= limit:
        return list(workloads)
    if limit <= 0:
        return []

    by_phase: dict[str, list[dict[str, Any]]] = {}
    for workload in workloads:
        phase = str(workload.get("phase", "legacy"))
        by_phase.setdefault(phase, []).append(workload)

    quotas = {phase: 0 for phase in by_phase}
    remaining = limit
    while remaining:
        allocated = False
        for phase in by_phase:
            if quotas[phase] >= len(by_phase[phase]):
                continue
            quotas[phase] += 1
            remaining -= 1
            allocated = True
            if not remaining:
                break
        if not allocated:
            break

    sampled = []
    for phase, items in by_phase.items():
        sampled.extend(_evenly_spaced_sample(items, quotas[phase]))
    return sampled
