"""Independent semantic review between fact materialization and Concept merge."""

from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal, Sequence

from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.contracts.runtime import RetrievalRecord
from kernelgen.knowledge.layout import safe_name
from kernelgen.knowledge.models import (
    CandidateConcept,
    CandidateReviewDecision,
    CandidateReviewRecord,
    CandidateReviewUnit,
    Concept,
    ConceptEvidence,
    KnowledgeReviewInput,
    KnowledgeReviewOutput,
    ObservationRecord,
    ReviewConceptContext,
)
from kernelgen.knowledge.publishing.merge import (
    candidate_group_key,
    canonical_candidate_scope,
)


MAX_REVIEW_UNITS_PER_CALL = 4
MAX_REVIEW_PACKET_CHARS = 60_000
MAX_PARALLEL_REVIEW_CALLS = 2


@dataclass(frozen=True)
class ReviewerExecutionResult:
    """Reviewer output plus the immutable audit records produced by its run."""

    output: KnowledgeReviewOutput
    retrievals: tuple[RetrievalRecord, ...] = ()


ReviewCallable = Callable[
    [KnowledgeReviewInput],
    KnowledgeReviewOutput | ReviewerExecutionResult,
]


@dataclass(frozen=True)
class ReviewBatchResult:
    candidates: list[tuple[CandidateConcept, list[ConceptEvidence]]]
    records: list[CandidateReviewRecord]
    warnings: list[str]


@dataclass(frozen=True)
class _ReviewGroup:
    unit: CandidateReviewUnit
    candidates: list[tuple[CandidateConcept, list[ConceptEvidence]]]


@dataclass(frozen=True)
class _ReviewExecution:
    output: KnowledgeReviewOutput | None
    retrievals: tuple[RetrievalRecord, ...]
    error: str = ""


def review_candidate_batch(
    catalog: FilesystemCatalog,
    candidates: Sequence[tuple[CandidateConcept, list[ConceptEvidence]]],
    observations: dict[str, ObservationRecord],
    *,
    run_id: str,
    batch_id: str,
    reviewer: ReviewCallable | None,
    reviewer_mode: Literal["shadow", "enforce"] = "enforce",
) -> ReviewBatchResult:
    """Review exact-deduplicated groups in bounded, isolated calls.

    A failed call defers only its own units.  Source and Concept citations count
    only when the same call's retrieval audit proves an exact successful detail
    read; query hits alone are never sufficient.
    """

    if not candidates:
        return ReviewBatchResult(candidates=[], records=[], warnings=[])
    if reviewer_mode not in {"shadow", "enforce"}:
        raise ValueError(
            "review_candidate_batch requires reviewer_mode=shadow or enforce"
        )
    groups = _build_review_groups(
        catalog,
        candidates,
        observations,
        run_id=run_id,
        batch_id=batch_id,
    )
    review_batches = _partition_review_groups(groups)
    executions = _execute_review_batches(
        review_batches,
        run_id=run_id,
        batch_id=batch_id,
        reviewer=reviewer,
    )
    approved: list[tuple[CandidateConcept, list[ConceptEvidence]]] = []
    records: list[CandidateReviewRecord] = []
    warnings: list[str] = []
    batch_count = len(review_batches)
    for batch_num, (review_groups, execution) in enumerate(
        zip(review_batches, executions),
        start=1,
    ):
        decisions_by_id: dict[
            str, list[CandidateReviewDecision]
        ] = defaultdict(list)
        if execution.error:
            warnings.append(
                f"Reviewer unavailable for batch {batch_num}/{batch_count}: "
                + execution.error
            )
        if execution.output is not None:
            known_ids = {group.unit.review_id for group in review_groups}
            for decision in execution.output.decisions:
                decisions_by_id[decision.review_id].append(decision)
                if decision.review_id not in known_ids:
                    warnings.append(
                        f"Reviewer batch {batch_num}/{batch_count} returned "
                        f"unknown review_id: {decision.review_id}"
                    )
        detail_reads = _successful_detail_reads(execution.retrievals)
        for group in review_groups:
            decisions = decisions_by_id.get(group.unit.review_id, [])
            if execution.error:
                decision = _fallback_decision(
                    group.unit.review_id,
                    "Reviewer unavailable; candidate deferred: "
                    + execution.error,
                )
                reviewer_status = "fallback"
            elif len(decisions) != 1:
                decision_error = (
                    "Reviewer must return exactly one decision for this "
                    "review_id"
                )
                warnings.append(
                    f"{group.unit.review_id}: {decision_error}"
                )
                decision = _fallback_decision(
                    group.unit.review_id,
                    decision_error,
                )
                reviewer_status = "fallback"
            else:
                decision = decisions[0]
                reviewer_status = "completed"

            transformed, validation_error = _apply_decision(
                catalog,
                group,
                decision,
            )
            if not validation_error:
                required_reads = _required_detail_reads(group, decision)
                missing_reads = sorted(required_reads - detail_reads.keys())
                if missing_reads:
                    validation_error = (
                        "Reviewer did not perform successful exact detail reads "
                        "for: "
                        + ", ".join(missing_reads)
                    )
            unit_read_refs, unit_event_ids = _unit_detail_read_audit(
                group,
                detail_reads,
            )
            if validation_error:
                warnings.append(
                    f"{group.unit.review_id}: {validation_error}"
                )
                decision = _fallback_decision(
                    group.unit.review_id,
                    "Reviewer decision failed program validation: "
                    + validation_error,
                )
                reviewer_status = "fallback"
                transformed = []
            if decision.decision == "approve":
                approved.extend(transformed)
            records.append(
                CandidateReviewRecord(
                    batch_id=batch_id,
                    candidate_ids=group.unit.candidate_ids,
                    reviewer_mode=reviewer_mode,
                    reviewer_status=reviewer_status,
                    review_batch_num=batch_num,
                    review_batch_count=batch_count,
                    detail_read_refs=unit_read_refs,
                    detail_read_event_ids=unit_event_ids,
                    observation_refs=[item.id for item in group.unit.observations],
                    reviewed_at=datetime.now(timezone.utc),
                    **decision.model_dump(mode="python"),
                )
            )
    return ReviewBatchResult(
        candidates=approved,
        records=sorted(records, key=lambda item: item.review_id),
        warnings=sorted(set(warnings)),
    )


def defer_candidate_batch(
    catalog: FilesystemCatalog,
    candidates: Sequence[tuple[CandidateConcept, list[ConceptEvidence]]],
    observations: dict[str, ObservationRecord],
    *,
    run_id: str,
    batch_id: str,
) -> ReviewBatchResult:
    """Record intentional Reviewer-off deferrals without invoking a model."""

    if not candidates:
        return ReviewBatchResult(candidates=[], records=[], warnings=[])
    groups = _build_review_groups(
        catalog,
        candidates,
        observations,
        run_id=run_id,
        batch_id=batch_id,
    )
    review_batches = _partition_review_groups(groups)
    records: list[CandidateReviewRecord] = []
    batch_count = len(review_batches)
    for batch_num, review_groups in enumerate(review_batches, start=1):
        for group in review_groups:
            decision = _fallback_decision(
                group.unit.review_id,
                "Reviewer disabled by configuration; Candidate retained for "
                "a later reviewed epoch.",
            )
            records.append(
                CandidateReviewRecord(
                    batch_id=batch_id,
                    candidate_ids=group.unit.candidate_ids,
                    reviewer_mode="off",
                    reviewer_status="disabled",
                    review_batch_num=batch_num,
                    review_batch_count=batch_count,
                    observation_refs=[item.id for item in group.unit.observations],
                    reviewed_at=datetime.now(timezone.utc),
                    **decision.model_dump(mode="python"),
                )
            )
    return ReviewBatchResult(
        candidates=[],
        records=sorted(records, key=lambda item: item.review_id),
        warnings=[],
    )


def _partition_review_groups(
    groups: Sequence[_ReviewGroup],
    *,
    max_units: int = MAX_REVIEW_UNITS_PER_CALL,
    max_chars: int = MAX_REVIEW_PACKET_CHARS,
) -> list[list[_ReviewGroup]]:
    if max_units < 1 or max_chars < 1:
        raise ValueError("Reviewer batch limits must be positive")
    batches: list[list[_ReviewGroup]] = []
    current: list[_ReviewGroup] = []
    current_chars = 0
    for group in groups:
        unit_chars = len(group.unit.model_dump_json()) + 256
        if current and (
            len(current) >= max_units
            or current_chars + unit_chars > max_chars
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(group)
        current_chars += unit_chars
    if current:
        batches.append(current)
    return batches


def _execute_review_batches(
    batches: Sequence[Sequence[_ReviewGroup]],
    *,
    run_id: str,
    batch_id: str,
    reviewer: ReviewCallable | None,
) -> list[_ReviewExecution]:
    inputs = [
        KnowledgeReviewInput(
            batch_id=batch_id,
            run_id=run_id,
            units=[group.unit for group in batch],
        )
        for batch in batches
    ]
    if reviewer is None:
        return [
            _ReviewExecution(
                output=None,
                retrievals=(),
                error="Reviewer is not configured",
            )
            for _ in inputs
        ]

    def execute(review_input: KnowledgeReviewInput) -> _ReviewExecution:
        try:
            raw = reviewer(review_input)
            if isinstance(raw, ReviewerExecutionResult):
                output = KnowledgeReviewOutput.model_validate(raw.output)
                retrievals = tuple(
                    RetrievalRecord.model_validate(item)
                    for item in raw.retrievals
                )
            else:
                output = KnowledgeReviewOutput.model_validate(raw)
                retrievals = ()
            return _ReviewExecution(output=output, retrievals=retrievals)
        except Exception as exc:  # fail closed at the semantic boundary
            return _ReviewExecution(
                output=None,
                retrievals=(),
                error=f"{type(exc).__name__}: {str(exc)[:1000]}",
            )

    if len(inputs) == 1:
        return [execute(inputs[0])]
    results: list[_ReviewExecution | None] = [None] * len(inputs)
    with ThreadPoolExecutor(
        max_workers=min(MAX_PARALLEL_REVIEW_CALLS, len(inputs))
    ) as executor:
        futures = {
            executor.submit(execute, review_input): index
            for index, review_input in enumerate(inputs)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [item for item in results if item is not None]


def _successful_detail_reads(
    retrievals: Sequence[RetrievalRecord],
) -> dict[str, set[str]]:
    reads: dict[str, set[str]] = defaultdict(set)
    for record in retrievals:
        if record.status != "success":
            continue
        if record.operation == "get_knowledge":
            returned = record.returned_refs
        elif record.operation == "get_source":
            returned = record.returned_sources
        else:
            continue
        for reference in returned:
            reads[str(reference)].add(record.event_id)
    return reads


def _required_detail_reads(
    group: _ReviewGroup,
    decision: CandidateReviewDecision,
) -> set[str]:
    source_refs = {
        _source_ref(source)
        for candidate, _ in group.candidates
        for source in candidate.source_refs
    }
    concept_refs = {item.id for item in group.unit.nearby_concepts}
    cited_refs = {*decision.evidence_refs, *decision.conflict_refs}
    required = cited_refs & (source_refs | concept_refs)
    if decision.target_concept_id is not None:
        required.add(decision.target_concept_id)
    return required


def _unit_detail_read_audit(
    group: _ReviewGroup,
    detail_reads: dict[str, set[str]],
) -> tuple[list[str], list[str]]:
    packet_refs = {
        _source_ref(source)
        for candidate, _ in group.candidates
        for source in candidate.source_refs
    } | {item.id for item in group.unit.nearby_concepts}
    read_refs = sorted(packet_refs & detail_reads.keys())
    event_ids = sorted(
        {
            event_id
            for reference in read_refs
            for event_id in detail_reads[reference]
        }
    )
    return read_refs, event_ids


def _build_review_groups(
    catalog: FilesystemCatalog,
    candidates: Sequence[tuple[CandidateConcept, list[ConceptEvidence]]],
    observations: dict[str, ObservationRecord],
    *,
    run_id: str,
    batch_id: str,
) -> list[_ReviewGroup]:
    grouped = defaultdict(list)
    for candidate, evidence in candidates:
        normalized = candidate.model_copy(
            update={"scope": canonical_candidate_scope(candidate.scope)}
        )
        grouped[candidate_group_key(normalized)].append((normalized, evidence))

    current = list(catalog.iter_concepts())
    groups = []
    for index, (_, group) in enumerate(
        sorted(grouped.items(), key=lambda item: str(item[0])),
        start=1,
    ):
        group = sorted(group, key=lambda item: item[0].candidate_id)
        candidate_ids = [item.candidate_id for item, _ in group]
        observation_ids = sorted(
            {
                evidence.observation_ref
                for _, links in group
                for evidence in links
            }
        )
        unit_observations = [
            observations[item]
            for item in observation_ids
            if item in observations
        ]
        representative = group[0][0]
        nearby = _nearby_concepts(representative, current)
        review_id = (
            f"review:{safe_name(batch_id)}:unit-{index:04d}"
        )
        checks = [
            "candidate_schema_valid",
            "runtime_scope_bound",
            "source_provenance_valid",
            "observation_lineage_valid",
            f"candidate_count={len(group)}",
            f"observation_count={len(unit_observations)}",
        ]
        groups.append(
            _ReviewGroup(
                unit=CandidateReviewUnit(
                    review_id=review_id,
                    candidate_ids=candidate_ids,
                    candidates=[item for item, _ in group],
                    observations=unit_observations,
                    nearby_concepts=nearby,
                    automatic_checks=checks,
                ),
                candidates=group,
            )
        )
    return groups


def _nearby_concepts(
    candidate: CandidateConcept,
    concepts: Sequence[Concept],
    *,
    limit: int = 5,
) -> list[ReviewConceptContext]:
    candidate_tokens = _tokens(
        " ".join(
            [
                candidate.claim_key,
                candidate.title,
                candidate.summary,
                *candidate.domains,
            ]
        )
    )
    scored = []
    for concept in concepts:
        if concept.kind != candidate.proposed_kind:
            continue
        score = 0
        if concept.id == candidate.target_concept_id:
            score += 10000
        if concept.claim_key == candidate.claim_key:
            score += 5000
        if canonical_candidate_scope(concept.scope) == candidate.scope:
            score += 500
        concept_tokens = _tokens(
            " ".join(
                [
                    concept.claim_key,
                    concept.title,
                    concept.summary,
                    *concept.domains,
                ]
            )
        )
        score += 10 * len(candidate_tokens & concept_tokens)
        if score:
            scored.append((score, concept.id, concept))
    contexts = []
    for _, _, concept in sorted(scored, reverse=True)[:limit]:
        contexts.append(
            ReviewConceptContext(
                id=concept.id,
                kind=concept.kind,
                claim_key=concept.claim_key,
                title=concept.title,
                summary=concept.summary,
                scope=canonical_candidate_scope(concept.scope),
                status=concept.status,
                evidence_state=concept.evidence_state,
                evidence_refs=[item.observation_ref for item in concept.evidence],
                source_refs=[_source_ref(item) for item in concept.sources],
                # Keep the packet bounded.  The Reviewer must use get_knowledge
                # when the full current Concept matters to its decision.
                body_excerpt="",
            )
        )
    return contexts


def _apply_decision(
    catalog: FilesystemCatalog,
    group: _ReviewGroup,
    decision: CandidateReviewDecision,
) -> tuple[list[tuple[CandidateConcept, list[ConceptEvidence]]], str]:
    if decision.review_id != group.unit.review_id:
        return [], "review_id does not match the review unit"
    allowed_evidence = {
        item.id for item in group.unit.observations
    } | {
        _source_ref(source)
        for candidate, _ in group.candidates
        for source in candidate.source_refs
    }
    invalid_evidence = sorted(set(decision.evidence_refs) - allowed_evidence)
    if invalid_evidence:
        return [], "unknown evidence_refs: " + ", ".join(invalid_evidence)
    allowed_conflicts = allowed_evidence | {
        item.id for item in group.unit.nearby_concepts
    }
    invalid_conflicts = sorted(set(decision.conflict_refs) - allowed_conflicts)
    if invalid_conflicts:
        return [], "unknown conflict_refs: " + ", ".join(invalid_conflicts)
    if decision.decision != "approve":
        return [], ""

    target = None
    if decision.target_concept_id is not None:
        nearby_ids = {item.id for item in group.unit.nearby_concepts}
        if decision.target_concept_id not in nearby_ids:
            return [], "target_concept_id is not in nearby_concepts"
        try:
            target = catalog.get_concept(decision.target_concept_id)
        except KeyError:
            return [], "target_concept_id does not exist"
        if target.status == "deprecated":
            return [], "target_concept_id is deprecated"
        if any(
            candidate.proposed_kind != target.kind
            for candidate, _ in group.candidates
        ):
            return [], "target Concept kind differs from candidate kind"

    if decision.merge_action == "create_new":
        if any(
            candidate.publish_action != "create"
            for candidate, _ in group.candidates
        ):
            return [], "create_new cannot replace an explicit update"
        if _matching_existing(catalog, group.candidates[0][0]) is not None:
            return [], "same claim_key and canonical Scope already exist"
        return list(group.candidates), ""

    if decision.merge_action == "update_existing":
        if target is None:
            return [], "update_existing requires an existing target"
        return [
            (
                candidate.model_copy(
                    update={
                        "publish_action": "update",
                        "target_concept_id": target.id,
                    }
                ),
                evidence,
            )
            for candidate, evidence in group.candidates
        ], ""

    if decision.merge_action == "attach_evidence":
        if target is None:
            return [], "attach_evidence requires an existing target"
        if any(
            canonical_candidate_scope(candidate.scope)
            != canonical_candidate_scope(target.scope)
            for candidate, _ in group.candidates
        ):
            return [], "attach_evidence requires the same canonical Scope"
        return [
            (
                candidate.model_copy(
                    update={
                        "publish_action": "create",
                        "target_concept_id": None,
                        "proposed_id": target.id,
                        "proposed_kind": target.kind,
                        "claim_key": target.claim_key,
                        "title": target.title,
                        "summary": target.summary,
                        "domains": target.domains,
                        "scope": canonical_candidate_scope(target.scope),
                        "retrieval": target.retrieval,
                        "body": target.body,
                        "relations": target.relations,
                    }
                ),
                evidence,
            )
            for candidate, evidence in group.candidates
        ], ""
    return [], "unsupported merge_action"


def _matching_existing(
    catalog: FilesystemCatalog,
    candidate: CandidateConcept,
) -> Concept | None:
    return next(
        (
            item
            for item in catalog.iter_concepts()
            if item.kind == candidate.proposed_kind
            and item.claim_key == candidate.claim_key
            and canonical_candidate_scope(item.scope)
            == canonical_candidate_scope(candidate.scope)
        ),
        None,
    )


def _fallback_decision(
    review_id: str,
    reason: str,
) -> CandidateReviewDecision:
    return CandidateReviewDecision(
        review_id=review_id,
        decision="defer",
        merge_action="none",
        evidence_refs=[],
        conflict_refs=[],
        scope_assessment="unverifiable",
        reusable=False,
        confidence="low",
        reason=reason,
    )


def _source_ref(source) -> str:
    return f"{source.resource}@{source.revision}::{source.locator}"


def _tokens(value: str) -> set[str]:
    return {
        item
        for item in re.split(r"[^a-z0-9]+", value.casefold())
        if len(item) > 1
    }


__all__ = [
    "ReviewBatchResult",
    "ReviewCallable",
    "ReviewerExecutionResult",
    "defer_candidate_batch",
    "review_candidate_batch",
]
