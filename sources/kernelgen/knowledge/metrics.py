"""Deterministic KB usefulness metrics from run archives and Catalog facts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from kernelgen.data.optimization_history import OptimizationHistory
from kernelgen.knowledge.catalog import (
    FilesystemCatalog,
    concept_ref,
)
from kernelgen.knowledge.models import (
    KnowledgeUsageScope,
    OperatorSignature,
    TargetContext,
    classify_knowledge_effect,
)


def build_knowledge_metrics(
    catalog_root: Path,
    run_archive_root: Path,
) -> dict:
    snapshots = _latest_snapshots(run_archive_root)
    query_events = set()
    queries_with_detail = set()
    retrieved_refs = []
    considered = 0
    applied = 0
    evaluated = 0
    recovery_opportunities = 0
    recoveries = 0
    performance = {
        "single": {"comparable_rounds": 0, "improved_rounds": 0},
        "combined": {"comparable_rounds": 0, "improved_rounds": 0},
    }
    reference_outcomes = defaultdict(
        lambda: {
            "retrieved": 0,
            "considered": 0,
            "applied": 0,
            "evaluated": 0,
            "correctness_recovered": 0,
            "correctness_regressed": 0,
            "performance_improved": 0,
            "performance_regressed": 0,
            "no_material_change": 0,
            "unclassified": 0,
            "agent_confirmed": 0,
            "agent_partially_confirmed": 0,
            "agent_not_confirmed": 0,
            "agent_inconclusive": 0,
            "agent_unassessed": 0,
        }
    )
    progress = []

    for snapshot in snapshots:
        run_id = snapshot["run_id"]
        workspace_id = snapshot["workspace_id"]
        root = snapshot["root"]
        scope = _evaluation_scope(root)
        scope_key = json.dumps(
            scope,
            sort_keys=True,
            separators=(",", ":"),
        )
        records = _retrieval_records(root)
        for record in records:
            if record.get("status") != "success":
                continue
            operation = record.get("operation")
            query_id = str(
                record.get("query_id") or record.get("event_id") or ""
            )
            query_key = (run_id, workspace_id, query_id)
            if operation in {"query_knowledge", "query_sources"} and query_id:
                query_events.add(query_key)
            if operation not in {"get_knowledge", "get_source"}:
                continue
            references = (
                record.get("returned_refs")
                if operation == "get_knowledge"
                else record.get("returned_sources")
            ) or []
            if references and query_id:
                queries_with_detail.add(query_key)
            for reference in references:
                ref = str(reference)
                retrieved_refs.append(ref)
                reference_outcomes[(ref, scope_key)]["retrieved"] += 1

        history = OptimizationHistory.load(root / "ledger.json")
        rounds_by_num = {
            item.round_num: item for item in history.rounds
        }
        passed_rounds = [
            item.round_num
            for item in history.rounds
            if item.evaluation.status == "PASSED"
        ]
        progress.append(
            {
                "run_id": run_id,
                "workspace_id": workspace_id,
                "rounds": len(history.rounds),
                "first_passed_round": (
                    min(passed_rounds) if passed_rounds else None
                ),
                "best_round": history.best_round or None,
            }
        )
        for round_record in history.rounds:
            uses = round_record.plan.knowledge_uses
            applied_uses = [
                item
                for item in uses
                if item.disposition in {"adopted", "adapted"}
            ]
            usage_mode = (
                "none"
                if not applied_uses
                else "single"
                if len(applied_uses) == 1
                else "combined"
            )
            parent = rounds_by_num.get(
                round_record.experiment_parent_round_num
            )
            assessments = {
                _knowledge_ref(item): item.assessment
                for item in (
                    round_record.conclusion.knowledge_assessments
                    if round_record.conclusion is not None
                    else []
                )
            }
            effect = classify_knowledge_effect(
                usage_mode=usage_mode,
                parent_status=(
                    parent.evaluation.status if parent is not None else None
                ),
                current_status=round_record.evaluation.status,
                performance_baseline_round_num=(
                    round_record.evaluation.comparison
                    .performance_baseline_round_num
                ),
                geo_mean_delta_pct=(
                    round_record.evaluation.comparison.geo_mean_delta_pct
                ),
            )
            considered += len(uses)
            applied += len(applied_uses)
            evaluated += len(applied_uses)
            for use in uses:
                ref = _knowledge_ref(use)
                counts = reference_outcomes[(ref, scope_key)]
                counts["considered"] += 1
                if use.disposition in {"adopted", "adapted"}:
                    counts["applied"] += 1
                    counts["evaluated"] += 1
                    counts[effect] += 1
                    assessment = assessments.get(ref, "unassessed")
                    counts[f"agent_{assessment}"] += 1
            if not applied_uses or parent is None:
                continue
            if parent.evaluation.status != "PASSED":
                recovery_opportunities += 1
                if effect == "correctness_recovered":
                    recoveries += 1
            comparison = round_record.evaluation.comparison
            if (
                parent.evaluation.status == "PASSED"
                and round_record.evaluation.status == "PASSED"
                and comparison.performance_baseline_round_num is not None
                and comparison.geo_mean_delta_pct is not None
            ):
                bucket = performance[usage_mode]
                bucket["comparable_rounds"] += 1
                if effect == "performance_improved":
                    bucket["improved_rounds"] += 1

    for bucket in performance.values():
        bucket["improvement_rate"] = _rate(
            bucket["improved_rounds"],
            bucket["comparable_rounds"],
        )

    catalog = FilesystemCatalog(catalog_root)
    concepts = catalog.iter_concepts()
    concept_refs = {concept_ref(item) for item in concepts}
    retrieved_set = set(retrieved_refs)
    applied_set = {
        ref
        for (ref, _), counts in reference_outcomes.items()
        if counts["applied"]
    }
    duplicate_groups = defaultdict(list)
    for concept in concepts:
        key = (
            concept.kind,
            concept.claim_key,
            json.dumps(
                concept.scope.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        duplicate_groups[key].append(concept_ref(concept))
    exact_duplicates = [
        sorted(refs)
        for refs in duplicate_groups.values()
        if len(refs) > 1
    ]
    queries_without_results = sum(
        1
        for snapshot in snapshots
        for record in _retrieval_records(snapshot["root"])
        if record.get("status") == "success"
        and record.get("operation") in {"query_knowledge", "query_sources"}
        and not (
            record.get("returned_refs")
            or record.get("returned_sources")
        )
    )
    return {
        "workspaces": len(snapshots),
        "retrieval": {
            "query_events": len(query_events),
            "queries_with_detail_read": len(
                query_events & queries_with_detail
            ),
            "query_to_detail_read_rate": _rate(
                len(query_events & queries_with_detail),
                len(query_events),
            ),
        },
        "funnel": {
            "retrieved": len(retrieved_refs),
            "considered": considered,
            "applied": applied,
            "evaluated": evaluated,
        },
        "correctness_recovery": {
            "opportunities": recovery_opportunities,
            "recoveries": recoveries,
            "rate": _rate(recoveries, recovery_opportunities),
        },
        "performance_improvement": performance,
        "per_reference": [
            {
                "reference": ref,
                "scope": json.loads(scope_key),
                **reference_outcomes[(ref, scope_key)],
            }
            for ref, scope_key in sorted(reference_outcomes)
        ],
        "optimization_progress": sorted(
            progress,
            key=lambda item: (item["run_id"], item["workspace_id"]),
        ),
        "catalog_quality": {
            "exact_duplicate_groups": exact_duplicates,
            "contested": sorted(
                concept_ref(item)
                for item in concepts
                if item.evidence_state == "contested"
            ),
            "falsified": sorted(
                concept_ref(item)
                for item in concepts
                if item.evidence_state == "falsified"
            ),
            "never_retrieved": sorted(concept_refs - retrieved_set),
            "never_applied": sorted(concept_refs - applied_set),
            "queries_without_results": queries_without_results,
        },
        "campaign_comparison": {
            "available": False,
            "reason": "run archive does not record a campaign variant",
        },
    }


def _latest_snapshots(root: Path) -> list[dict]:
    latest = {}
    for path in sorted(
        Path(root).glob("*/*/snapshots/*/manifest.json")
    ):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        run_id = str(manifest.get("run_id") or "")
        workspace_id = str(manifest.get("workspace_id") or "")
        if not run_id or not workspace_id:
            continue
        key = (run_id, workspace_id)
        rank = str(manifest.get("archived_at") or "")
        if key not in latest or rank > latest[key]["rank"]:
            latest[key] = {
                "rank": rank,
                "run_id": run_id,
                "workspace_id": workspace_id,
                "root": path.parent,
            }
    return [latest[key] for key in sorted(latest)]


def _retrieval_records(snapshot_root: Path) -> list[dict]:
    path = snapshot_root / "knowledge" / "retrieval-log.jsonl"
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _evaluation_scope(snapshot_root: Path) -> dict:
    try:
        signature = OperatorSignature.model_validate_json(
            (snapshot_root / "context" / "operator-signature.json").read_text(
                encoding="utf-8"
            )
        )
        target = TargetContext.model_validate_json(
            (snapshot_root / "context" / "target-context.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, TypeError, ValueError):
        return {}
    return KnowledgeUsageScope.from_context(
        signature,
        target,
    ).model_dump(mode="json")


def _knowledge_ref(use) -> str:
    if use.concept_ref:
        return use.concept_ref
    source = use.source_ref
    return f"{source.resource}@{source.revision}::{source.locator}"


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
