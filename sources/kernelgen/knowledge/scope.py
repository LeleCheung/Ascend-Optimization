"""Deterministic applicability matching with explicit uncertainty."""

from __future__ import annotations

from typing import Any

from kernelgen.knowledge.models import (
    QueryContext,
    Scope,
    WorkloadPredicate,
)
from kernelgen.knowledge.contracts.runtime import ScopeMatch
from kernelgen.knowledge.context import normalize_device


def match_scope(scope: Scope, context: QueryContext) -> ScopeMatch:
    matched: list[str] = []
    gaps: list[str] = []
    conflicts: list[str] = []

    _match_target(scope, context, matched, gaps, conflicts)
    _match_operator(scope, context, matched, gaps, conflicts)
    _match_workloads(scope, context, matched, gaps, conflicts)
    _match_numerics(scope, context, matched, gaps, conflicts)

    if conflicts:
        level = "incompatible"
    elif gaps:
        level = "analogy"
    else:
        level = "direct"
    return ScopeMatch(
        level=level,
        matched_on=sorted(set(matched)),
        gaps=sorted(set(gaps)),
        conflicts=sorted(set(conflicts)),
    )


def _match_target(
    scope: Scope,
    context: QueryContext,
    matched: list[str],
    gaps: list[str],
    conflicts: list[str],
) -> None:
    expected = scope.target
    actual = context.target_context
    if expected.level == "portable":
        matched.append("target:portable")
    elif expected.level == "exact":
        _compare(
            "target:backend",
            expected.backend,
            actual.backend,
            matched,
            gaps,
            conflicts,
        )
        _device_membership(
            "target:device",
            expected.devices,
            actual.device,
            matched,
            gaps,
            conflicts,
        )
        if expected.architecture:
            _compare(
                "target:architecture",
                expected.architecture,
                actual.architecture,
                matched,
                gaps,
                conflicts,
            )
    elif expected.level == "device":
        if expected.backend:
            _compare(
                "target:backend",
                expected.backend,
                actual.backend,
                matched,
                gaps,
                conflicts,
            )
        _device_membership(
            "target:device",
            expected.devices,
            actual.device,
            matched,
            gaps,
            conflicts,
        )
        if expected.architecture:
            _compare(
                "target:architecture",
                expected.architecture,
                actual.architecture,
                matched,
                gaps,
                conflicts,
            )
    elif expected.level == "architecture":
        if expected.backend:
            _compare(
                "target:backend",
                expected.backend,
                actual.backend,
                matched,
                gaps,
                conflicts,
            )
        _compare(
            "target:architecture",
            expected.architecture,
            actual.architecture,
            matched,
            gaps,
            conflicts,
        )
    elif expected.level == "backend":
        _compare(
            "target:backend",
            expected.backend,
            actual.backend,
            matched,
            gaps,
            conflicts,
        )

    if expected.capabilities:
        actual_capabilities = set(actual.capabilities)
        missing = sorted(set(expected.capabilities) - actual_capabilities)
        if missing and actual.capabilities:
            conflicts.extend(f"target:capability:{item}" for item in missing)
        elif missing:
            gaps.extend(f"target:capability:{item}" for item in missing)
        else:
            matched.extend(
                f"target:capability:{item}" for item in expected.capabilities
            )

    for field_name in (
        "language",
        "compiler",
        "runtime",
        "library",
    ):
        required = getattr(expected.software, field_name)
        observed = getattr(actual.software, field_name)
        if required:
            _compare(
                f"software:{field_name}",
                required,
                observed,
                matched,
                gaps,
                conflicts,
            )

    for field_name in (
        "language_version",
        "compiler_version",
        "runtime_version",
        "library_version",
        "driver_version",
    ):
        required = getattr(expected.software, field_name)
        observed = getattr(actual.software, field_name)
        if not required:
            continue
        if any(token in required for token in "<>=!~,"):
            if not observed:
                gaps.append(f"software:{field_name}")
            elif required == observed:
                matched.append(f"software:{field_name}:{observed}")
            else:
                gaps.append(f"software:{field_name}:constraint")
        else:
            _compare(
                f"software:{field_name}",
                required,
                observed,
                matched,
                gaps,
                conflicts,
            )


def _match_operator(
    scope: Scope,
    context: QueryContext,
    matched: list[str],
    gaps: list[str],
    conflicts: list[str],
) -> None:
    expected = scope.operator
    actual = context.operator_signature
    _membership(
        "operator:definition_id",
        expected.definition_ids,
        actual.definition_id,
        matched,
        gaps,
        conflicts,
    )
    _membership(
        "operator:op_type",
        expected.op_types,
        actual.op_type,
        matched,
        gaps,
        conflicts,
    )
    _required_subset(
        "operator:motif",
        expected.motifs,
        actual.motifs,
        matched,
        gaps,
        conflicts,
    )
    for label, required, observed in (
        ("operator:dataflow", expected.dataflow, actual.dataflow),
        ("operator:dtype", expected.dtypes, actual.dtypes),
        ("operator:layout", expected.layouts, actual.layouts),
    ):
        if not required:
            continue
        overlap = sorted(set(required) & set(observed))
        if overlap:
            matched.extend(f"{label}:{item}" for item in overlap)
        elif observed:
            conflicts.append(label)
        else:
            gaps.append(label)


def _required_subset(
    label: str,
    required: list[str],
    observed: list[str],
    matched: list[str],
    gaps: list[str],
    conflicts: list[str],
) -> None:
    """Require every Concept motif while preserving unknown-context analogy."""

    if not required:
        return
    if not observed:
        gaps.append(label)
        return
    missing = sorted(set(required) - set(observed))
    if missing:
        conflicts.extend(f"{label}:missing:{item}" for item in missing)
        return
    matched.extend(f"{label}:{item}" for item in sorted(set(required)))


def _match_workloads(
    scope: Scope,
    context: QueryContext,
    matched: list[str],
    gaps: list[str],
    conflicts: list[str],
) -> None:
    summary = context.operator_signature.workload_features
    for predicate in scope.workloads.all:
        result = _evaluate(predicate, summary)
        _record_predicate(result, predicate, matched, gaps, conflicts)
    if scope.workloads.any:
        results = [_evaluate(item, summary) for item in scope.workloads.any]
        if any(result is True for result in results):
            matched.append("workload:any")
        elif any(result is None for result in results):
            gaps.append("workload:any")
        else:
            conflicts.append("workload:any")
    for predicate in scope.workloads.excludes:
        result = _evaluate(predicate, summary)
        if result is True:
            conflicts.append(f"workload:excluded:{predicate.field}")
        elif result is None:
            gaps.append(f"workload:exclude:{predicate.field}")


def _match_numerics(
    scope: Scope,
    context: QueryContext,
    matched: list[str],
    gaps: list[str],
    conflicts: list[str],
) -> None:
    expected = scope.numerics
    actual = context.operator_signature.numerics
    for field_name in ("exact", "allow_tf32", "max_abs_error", "max_rel_error"):
        required = getattr(expected, field_name)
        if required is None:
            continue
        # exact=false means the Concept does not require exact semantics. It is
        # permissive, not a requirement that the query contract declare an
        # approximation mode.
        if field_name == "exact" and required is False:
            continue
        observed = getattr(actual, field_name)
        if observed is None:
            gaps.append(f"numerics:{field_name}")
        elif observed == required:
            matched.append(f"numerics:{field_name}:{observed}")
        else:
            conflicts.append(f"numerics:{field_name}")
    if expected.accumulation_dtypes:
        observed = set(actual.accumulation_dtypes)
        required = set(expected.accumulation_dtypes)
        if not observed:
            gaps.append("numerics:accumulation_dtype")
        elif required <= observed:
            matched.extend(
                f"numerics:accumulation_dtype:{item}" for item in sorted(required)
            )
        else:
            conflicts.append("numerics:accumulation_dtype")


def _evaluate(predicate: WorkloadPredicate, values: dict[str, Any]) -> bool | None:
    if predicate.field not in values:
        return None
    actual = values[predicate.field]
    expected = predicate.value
    try:
        if predicate.op == "eq":
            return actual == expected
        if predicate.op == "ne":
            return actual != expected
        if predicate.op == "lt":
            return actual < expected
        if predicate.op == "lte":
            return actual <= expected
        if predicate.op == "gt":
            return actual > expected
        if predicate.op == "gte":
            return actual >= expected
        if predicate.op == "in":
            return actual in expected
        if predicate.op == "not_in":
            return actual not in expected
    except TypeError:
        return False
    return False


def _record_predicate(
    result: bool | None,
    predicate: WorkloadPredicate,
    matched: list[str],
    gaps: list[str],
    conflicts: list[str],
) -> None:
    label = f"workload:{predicate.field}:{predicate.op}"
    if result is True:
        matched.append(label)
    elif result is None:
        gaps.append(f"workload:{predicate.field}")
    else:
        conflicts.append(label)


def _membership(
    label: str,
    accepted: list[str],
    observed: str,
    matched: list[str],
    gaps: list[str],
    conflicts: list[str],
) -> None:
    if not accepted:
        return
    if not observed:
        gaps.append(label)
    elif observed in accepted:
        matched.append(f"{label}:{observed}")
    else:
        conflicts.append(f"{label}:{observed}")


def _device_membership(
    label: str,
    accepted: list[str],
    observed: str,
    matched: list[str],
    gaps: list[str],
    conflicts: list[str],
) -> None:
    if not accepted:
        return
    normalized_observed = normalize_device(observed)
    normalized_accepted = {
        normalize_device(item).casefold()
        for item in accepted
    }
    if not normalized_observed:
        gaps.append(label)
    elif normalized_observed.casefold() in normalized_accepted:
        matched.append(f"{label}:{normalized_observed}")
    else:
        conflicts.append(f"{label}:{normalized_observed}")


def _compare(
    label: str,
    expected: str,
    observed: str,
    matched: list[str],
    gaps: list[str],
    conflicts: list[str],
) -> None:
    if not observed or observed == "unknown":
        gaps.append(label)
    elif expected == observed:
        matched.append(f"{label}:{observed}")
    else:
        conflicts.append(f"{label}:{observed}")
