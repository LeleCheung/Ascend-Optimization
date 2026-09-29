"""Deterministic knowledge query and authorized detail retrieval."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4

from kernelgen.knowledge.catalog import concept_ref
from kernelgen.knowledge.records import retrieval_record
from kernelgen.knowledge.models import (
    KnowledgeBundle,
    QueryContext,
    BundleCoverage,
    Concept,
    KnowledgeResult,
    KnowledgeUsageScope,
    KnowledgeUsageSummary,
)
from kernelgen.knowledge.contracts.runtime import (
    KnowledgeDocument,
    SearchRoute,
)
from kernelgen.knowledge.scope import match_scope
from kernelgen.knowledge.context import normalize_device
from kernelgen.knowledge.ranking import (
    usage_reasons,
    usage_score,
)
from kernelgen.knowledge.vocabulary import Vocabulary


_EVIDENCE_WEIGHT = {
    "corroborated": 4.0,
    "observed": 3.0,
    "source_supported": 2.0,
    "contested": 1.0,
    "falsified": -100.0,
}
_TARGET_WEIGHT = {
    "exact": 5.0,
    "device": 4.0,
    "architecture": 3.0,
    "backend": 2.0,
    "portable": 1.0,
}
_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "how",
    "into",
    "should",
    "that",
    "the",
    "this",
    "use",
    "what",
    "with",
}
_TASK_COVERAGE = {
    "constraint_check": (
        frozenset({"reference"}),
        "no direct reference for constraint_check",
    ),
    "architecture_selection": (
        frozenset({"method", "reference"}),
        "no direct method or reference for architecture_selection",
    ),
    "implementation": (
        frozenset({"method"}),
        "no direct method for implementation",
    ),
    "diagnosis": (
        frozenset({"diagnostic"}),
        "no direct diagnostic for diagnosis",
    ),
    "next_experiment": (
        frozenset({"method", "experience"}),
        "no direct method or experience for next_experiment",
    ),
    "portability_analysis": (
        frozenset({"reference", "experience"}),
        "no direct reference or experience for portability_analysis",
    ),
}


class QueryKnowledge:
    def __init__(
        self,
        catalog,
        index,
        events,
        vocabulary: Vocabulary | None = None,
        *,
        allow_index_rebuild: bool = True,
    ):
        self.catalog = catalog
        self.index = index
        self.events = events
        self.allow_index_rebuild = allow_index_rebuild
        if vocabulary is None:
            catalog_layout = getattr(catalog, "layout", None)
            catalog_root = getattr(catalog_layout, "root", None)
            vocabulary = (
                Vocabulary.from_catalog(catalog_root)
                if catalog_root is not None
                else Vocabulary()
            )
        self.vocabulary = vocabulary

    def execute(
        self,
        context: QueryContext,
        *,
        origin: str = "application",
    ) -> KnowledgeBundle:
        query_id = f"query:{uuid4().hex}"
        request = context.model_dump(mode="json")
        try:
            bundle = self._execute(context, query_id=query_id)
        except Exception as exc:
            self.events.append(
                retrieval_record(
                    event_id=query_id,
                    operation="query_knowledge",
                    origin=origin,
                    status="error",
                    query_id=query_id,
                    request=request,
                    error=str(exc),
                )
            )
            raise
        levels = {
            item.concept_ref: level
            for level, items in (
                ("direct", bundle.direct),
                ("analogy", bundle.analogies),
                ("conflict", bundle.conflicts),
            )
            for item in items
        }
        self.events.append(
            retrieval_record(
                event_id=query_id,
                operation="query_knowledge",
                origin=origin,
                status="success",
                query_id=query_id,
                snapshot=bundle.snapshot,
                request=request,
                returned_refs=list(levels),
                result_levels=levels,
            )
        )
        return bundle

    def _execute(
        self,
        context: QueryContext,
        *,
        query_id: str,
    ) -> KnowledgeBundle:
        snapshot = self.catalog.snapshot()
        concepts = list(self.catalog.iter_concepts())
        observations = list(self.catalog.iter_observations())
        usage_snapshot = self.catalog.usage_snapshot()
        if (
            self.index.snapshot() != snapshot
            or self.index.usage_snapshot() != usage_snapshot
        ):
            if not self.allow_index_rebuild:
                raise RuntimeError(
                    "knowledge index is missing or stale in read_only_v1; "
                    "prebuild the Catalog .derived index or provide an "
                    "external derived_root"
                )
            self.index.rebuild(
                concepts,
                snapshot,
                observations=observations,
                retrievals=self.catalog.iter_retrieval_events(),
                usage_snapshot=usage_snapshot,
            )
        routes = _routes(context)
        hits = self.index.search(
            routes,
            max(context.max_results * 12, 128),
            lexical_queries=_lexical_queries(context, self.vocabulary),
        )
        concepts_by_ref = {item.id: item for item in concepts}

        direct = []
        analogies = []
        conflicts = []
        for hit in hits:
            ref = hit.concept_id
            concept = concepts_by_ref.get(ref)
            if concept is None or concept.status != "stable":
                continue
            if (
                concept.stale_after is not None
                and concept.stale_after < datetime.now(timezone.utc).date()
            ):
                continue
            if not _eligible(concept, context):
                continue
            scope = match_scope(concept.scope, context)
            if scope.level == "incompatible":
                continue
            usage_summary = self.index.usage(
                ref,
                _usage_scope(context),
            )
            result = _result(
                concept,
                [*hit.matched_on, *scope.matched_on],
                scope.gaps,
                usage_summary,
                usage_reasons(
                    usage_summary,
                    scope_label=scope.level,
                ),
            )
            score = (
                hit.score
                + _EVIDENCE_WEIGHT[concept.evidence_state]
                + (10.0 if scope.level == "direct" else 0.0)
                + _TARGET_WEIGHT[concept.scope.target.level]
                + (0.5 if concept.verified else 0.0)
                + usage_score(usage_summary)
            )
            item = (score, concept.claim_key, result)
            if concept.evidence_state in {"contested", "falsified"}:
                conflicts.append((score, concept.claim_key, result))
            elif scope.level == "direct":
                direct.append(item)
            else:
                result = result.model_copy(
                    update={"scope_gaps": scope.gaps}
                )
                analogies.append((score, concept.claim_key, result))

        direct = _deduplicate(direct, context.max_results)
        analogies = _deduplicate(analogies, context.max_results)
        conflicts = _deduplicate(conflicts, context.max_results)
        returned = [*direct, *analogies, *conflicts]
        coverage = BundleCoverage(
            matched_routes=sorted(
                {
                    matched
                    for item in returned
                    for matched in item.matched_on
                }
            ),
            gaps=_coverage_gaps(context, direct, analogies, conflicts),
        )
        bundle = KnowledgeBundle(
            query_id=query_id,
            snapshot=snapshot,
            direct=direct,
            analogies=analogies,
            conflicts=conflicts,
            coverage=coverage,
        )
        return bundle


class GetKnowledge:
    def __init__(
        self,
        catalog,
        events,
    ):
        self.catalog = catalog
        self.events = events

    def execute(
        self,
        query_id: str,
        concept_refs: Iterable[str],
        detail_level: str = "full",
        *,
        origin: str = "application",
    ) -> list[KnowledgeDocument]:
        requested = list(dict.fromkeys(concept_refs))
        request = {
            "concept_refs": requested,
            "detail_level": detail_level,
        }
        try:
            documents = self._execute(
                query_id,
                requested,
                detail_level,
            )
        except Exception as exc:
            self.events.append(
                retrieval_record(
                    operation="get_knowledge",
                    origin=origin,
                    status="error",
                    query_id=query_id,
                    request=request,
                    error=str(exc),
                )
            )
            raise
        self.events.append(
            retrieval_record(
                operation="get_knowledge",
                origin=origin,
                status="success",
                query_id=query_id,
                request=request,
                returned_refs=[
                    document.concept_ref for document in documents
                ],
            )
        )
        return documents

    def _execute(
        self,
        query_id: str,
        requested: list[str],
        detail_level: str,
    ) -> list[KnowledgeDocument]:
        if detail_level not in {"summary", "full", "sources"}:
            raise ValueError("detail_level must be summary, full, or sources")
        documents = []
        for ref in requested:
            concept = self.catalog.get_concept(ref)
            documents.append(
                KnowledgeDocument(
                    query_id=query_id,
                    concept_ref=ref,
                    title=concept.title,
                    summary=concept.summary,
                    body=concept.body if detail_level == "full" else "",
                    sources=(
                        [item.model_dump(mode="json") for item in concept.sources]
                        if detail_level in {"full", "sources"}
                        else []
                    ),
                    evidence=(
                        [
                            item.model_dump(mode="json")
                            for item in concept.evidence
                        ]
                        if detail_level == "full"
                        else []
                    ),
                )
            )
        return documents


def _routes(context: QueryContext) -> list[SearchRoute]:
    signature = context.operator_signature
    symptoms = [
        *[item.label for item in context.findings],
        *[
            item.split(":", 1)[0].strip()
            for item in context.errors
            if item.split(":", 1)[0].strip()
        ],
    ]
    routes = [
        SearchRoute(route="phase", term=context.phase, weight=2.0),
        SearchRoute(route="task", term=context.task, weight=2.0),
        SearchRoute(
            route="definition_id",
            term=signature.definition_id,
            weight=12.0,
        ),
    ]
    if signature.op_type:
        routes.append(
            SearchRoute(route="op_type", term=signature.op_type, weight=8.0)
        )
    target = context.target_context
    for route, values, weight in (
        ("backend", [target.backend], 6.0),
        ("architecture", [target.architecture], 5.0),
        ("device", [normalize_device(target.device)], 5.0),
        ("capability", target.capabilities, 4.0),
    ):
        routes.extend(
            SearchRoute(route=route, term=value, weight=weight)
            for value in values
            if value
        )
    for route, values, weight in (
        ("motif", signature.motifs, 6.0),
        ("dataflow", signature.dataflow, 5.0),
        ("dtype", signature.dtypes, 3.0),
        ("layout", signature.layouts, 2.0),
        ("symptom", symptoms, 5.0),
    ):
        routes.extend(
            SearchRoute(route=route, term=value, weight=weight)
            for value in values
            if value
        )
    routes.extend(
        SearchRoute(route="keyword", term=token, weight=1.0)
        for token in _tokens(context.question)
    )
    return routes


def _lexical_queries(
    context: QueryContext,
    vocabulary: Vocabulary,
) -> list[str]:
    signature = context.operator_signature
    raw_queries = [
        context.question,
        *context.errors,
        *[
            " ".join((item.category, item.label))
            for item in context.findings
        ],
        " ".join(
            [
                signature.definition_id,
                signature.definition_name,
                signature.op_type,
                *signature.motifs,
                *signature.dataflow,
                *signature.dtypes,
                *signature.layouts,
            ]
        ),
    ]
    expanded: list[str] = []
    for query in raw_queries:
        if query.strip():
            expanded.extend(vocabulary.expand(query))
    return list(dict.fromkeys(expanded))


def _result(
    concept: Concept,
    matched_on: list[str],
    scope_gaps: list[str],
    usage_summary: KnowledgeUsageSummary,
    recommendation_reasons: list[str],
) -> KnowledgeResult:
    return KnowledgeResult(
        concept_ref=concept_ref(concept),
        kind=concept.kind,
        summary=concept.summary,
        matched_on=matched_on,
        scope_gaps=scope_gaps,
        evidence_state=concept.evidence_state,
        usage_summary=usage_summary,
        recommendation_reasons=recommendation_reasons,
    )


def _usage_scope(context: QueryContext) -> KnowledgeUsageScope:
    return KnowledgeUsageScope.from_context(
        context.operator_signature,
        context.target_context,
    )


def _deduplicate(
    scored: list[tuple[float, str, KnowledgeResult]],
    limit: int,
) -> list[KnowledgeResult]:
    output = []
    claims = set()
    for _, claim_key, result in sorted(
        scored,
        key=lambda item: (-item[0], item[2].concept_ref),
    ):
        if claim_key in claims:
            continue
        claims.add(claim_key)
        output.append(result)
        if len(output) >= limit:
            break
    return output


def _coverage_gaps(
    context: QueryContext,
    direct: list[KnowledgeResult],
    analogies: list[KnowledgeResult],
    conflicts: list[KnowledgeResult],
) -> list[str]:
    gaps = []
    if not direct:
        gaps.append("no direct applicable knowledge")
    if context.phase == "post_profile" and not any(
        item.kind == "diagnostic" for item in [*direct, *analogies, *conflicts]
    ):
        gaps.append("no diagnostic matched profile findings")
    required_kinds, message = _TASK_COVERAGE[context.task]
    if not any(item.kind in required_kinds for item in direct):
        gaps.append(message)
    return gaps


def _tokens(text: str) -> list[str]:
    normalized = "".join(
        character.lower() if character.isalnum() or character in {"_", "-"} else " "
        for character in text
    )
    return sorted(
        {
            item
            for item in normalized.split()
            if len(item) >= 3 and item not in _STOPWORDS
        }
    )


def _eligible(concept: Concept, context: QueryContext) -> bool:
    phases = concept.retrieval.phases
    tasks = concept.retrieval.tasks
    return (
        (not phases or context.phase in phases)
        and (not tasks or context.task in tasks)
    )
