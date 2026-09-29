"""Side-effect-free migrations into the merged KernelGen ledger schema."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Literal


CURRENT_LEDGER_VERSION = "3.0"
SUPPORTED_LEDGER_VERSIONS = frozenset(
    {"2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", CURRENT_LEDGER_VERSION}
)
LedgerOrigin = Literal["dev-xy", "jiabei"]


class UnsupportedLedgerVersion(ValueError):
    def __init__(self, version: object):
        super().__init__(
            f"unsupported ledger schema {version!r}; "
            f"supported versions are {sorted(SUPPORTED_LEDGER_VERSIONS)!r}. "
            "Start a fresh agent workspace or run an explicit migration."
        )
        self.version = version


class AmbiguousLedgerOrigin(ValueError):
    """A same-version branch ledger cannot be classified without guessing."""


def _rounds(raw: Dict[str, Any]) -> list[dict]:
    return [item for item in (raw.get("rounds") or []) if isinstance(item, dict)]


def detect_ledger_origin(raw: Dict[str, Any]) -> LedgerOrigin:
    """Identify the producer lineage before interpreting branch-specific v2."""

    version = raw.get("schema_version")
    if version in {"2.1", "2.2", "2.3", "2.4", "2.5", "2.6"}:
        return "jiabei"
    if version != "2.0":
        raise UnsupportedLedgerVersion(version)

    rounds = _rounds(raw)
    current_markers = False
    jiabei_markers = False
    for record in rounds:
        solution = record.get("solution") or {}
        evaluation = record.get("evaluation") or {}
        workloads = evaluation.get("workloads") or []
        plan = record.get("plan") or {}
        current_markers = current_markers or (
            "next_verdict" in record
            or "candidate_path" in solution
            or "api_version" in evaluation
            or "is_hack" in evaluation
            or "timing_skipped" in evaluation
            or any(
                isinstance(item, dict)
                and ("phase" in item or "skip_reason" in item)
                for item in workloads
            )
        )
        jiabei_markers = jiabei_markers or bool(plan.get("knowledge_uses"))

    if current_markers and not jiabei_markers:
        return "dev-xy"
    if jiabei_markers and not current_markers:
        return "jiabei"
    if current_markers and jiabei_markers:
        # Current schema may already contain forward-compatible knowledge uses.
        return "dev-xy"
    raise AmbiguousLedgerOrigin(
        "ledger schema '2.0' has no branch-specific marker; pass "
        "origin='dev-xy' or origin='jiabei' explicitly"
    )


def _normalize_knowledge_use(use: dict) -> None:
    if "query_event_id" not in use:
        use["query_event_id"] = use.pop("query_id", "")
    else:
        use.pop("query_id", None)
    if "application_note" not in use:
        use["application_note"] = use.pop("adaptation", "")
    else:
        use.pop("adaptation", None)
    use.setdefault("affected_parts", [])
    ref = str(use.get("concept_ref") or "")
    concept_id, separator, revision = ref.rpartition("@")
    if separator and revision.isdigit():
        use["concept_ref"] = concept_id


def _normalize_plan(plan: dict) -> None:
    plan.pop("target_workloads", None)
    plan.pop("risks", None)
    plan.setdefault("knowledge_uses", [])
    for use in plan["knowledge_uses"]:
        if isinstance(use, dict):
            _normalize_knowledge_use(use)

    expected = plan.get("expected_effect")
    if isinstance(expected, dict):
        expected.pop("estimated_change_pct_min", None)
        expected.pop("estimated_change_pct_max", None)

    source = plan.get("source")
    if not isinstance(source, dict):
        return
    if "origin" not in source and "kind" in source:
        source["origin"] = source.pop("kind")
    else:
        source.pop("kind", None)
    if "parent_round_num" not in source and "round_num" in source:
        source["parent_round_num"] = source.pop("round_num")
    else:
        source.pop("round_num", None)
    source.pop("analysis_path", None)
    source.pop("detail", None)


def _normalize_conclusion(conclusion: dict) -> None:
    for field in (
        "architecture_tag",
        "suggestion_followed",
        "key_numbers",
        "composable",
        "composition_group",
        "diagnostic_results",
    ):
        conclusion.pop(field, None)


def _normalize_round(record: dict) -> None:
    record.setdefault("experiment_parent_round_num", None)
    record.setdefault("next_verdict", None)

    plan = record.get("plan")
    if isinstance(plan, dict):
        _normalize_plan(plan)

    solution = record.get("solution")
    if isinstance(solution, dict):
        solution.setdefault("snapshot_path", "")
        solution.setdefault("candidate_path", "tmp/main.py")

    evaluation = record.get("evaluation")
    if isinstance(evaluation, dict):
        if "api_version" not in evaluation and "schema_version" in evaluation:
            evaluation["api_version"] = evaluation.pop("schema_version")
        else:
            evaluation.pop("schema_version", None)
        evaluation.setdefault("evaluated_at", None)
        evaluation.setdefault("api_version", "")
        evaluation.setdefault("is_hack", False)
        evaluation.setdefault("hack_reason", "")
        evaluation.setdefault("timing_skipped", False)
        evaluation.setdefault("fingerprint", "")
        comparison = evaluation.get("comparison")
        if isinstance(comparison, dict):
            if (
                "performance_baseline_round_num" not in comparison
                and "baseline_round_num" in comparison
            ):
                comparison["performance_baseline_round_num"] = comparison.pop(
                    "baseline_round_num"
                )
            else:
                comparison.pop("baseline_round_num", None)
        for workload in evaluation.get("workloads") or []:
            if isinstance(workload, dict):
                workload.setdefault("phase", "")
                workload.setdefault("skip_reason", "")

    conclusion = record.get("conclusion")
    if isinstance(conclusion, dict):
        _normalize_conclusion(conclusion)


def migrate_ledger(
    raw: Dict[str, Any],
    *,
    origin: LedgerOrigin | None = None,
) -> Dict[str, Any]:
    """Return a normalized v3 copy without overwriting the persisted source."""

    version = raw.get("schema_version")
    if version == CURRENT_LEDGER_VERSION:
        migrated = deepcopy(raw)
        migrated.setdefault("implementation_language", "triton")
        for record in _rounds(migrated):
            _normalize_round(record)
        return migrated
    if version not in SUPPORTED_LEDGER_VERSIONS:
        raise UnsupportedLedgerVersion(version)
    resolved_origin = origin or detect_ledger_origin(raw)
    if version != "2.0" and resolved_origin != "jiabei":
        raise ValueError(
            f"ledger schema {version!r} is only defined for origin='jiabei'"
        )

    migrated = deepcopy(raw)
    migrated["schema_version"] = CURRENT_LEDGER_VERSION
    migrated.setdefault("implementation_language", "triton")
    for record in _rounds(migrated):
        _normalize_round(record)
    return migrated
