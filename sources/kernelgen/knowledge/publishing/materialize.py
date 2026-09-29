"""Bind workspace candidates to authoritative runtime observations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.context import normalize_operator_signature, workspace_provenance
from kernelgen.knowledge.contracts.runtime import WorkspaceKnowledgeState
from kernelgen.knowledge.layout import KnowledgeLayout
from kernelgen.knowledge.models import (
    CandidateConcept,
    ConceptEvidence,
    ObservationRecord,
    OperatorScope,
    OperatorSignature,
    RuntimeCandidate,
    RuntimeScopeHints,
    Scope,
    TargetContext,
    TargetScope,
    WorkloadPredicate,
    WorkloadScope,
)
from kernelgen.knowledge.records import FileCandidateOutbox
from kernelgen.knowledge.publishing.validation import (
    normalize_candidate_vocabulary,
    validate_runtime_candidate,
    validate_static_candidate,
)
from kernelgen.knowledge.taxonomy import (
    OperatorTaxonomy,
    normalize_taxonomy_term,
)
from kernelgen.knowledge.vocabulary import (
    Vocabulary,
    canonical_failure_symptom,
)


class CandidateMaterializer:
    def __init__(self, catalog_root: Path, fact_reader, vocabulary: Vocabulary):
        self.catalog_root = Path(catalog_root)
        self.fact_reader = fact_reader
        self.vocabulary = vocabulary
        self.operator_taxonomy = OperatorTaxonomy.from_catalog(self.catalog_root)

    def validate_static(
        self,
        candidates: Sequence[CandidateConcept],
    ) -> list[tuple[None, CandidateConcept]]:
        normalized = []
        for candidate in candidates:
            if not candidate.created_by.startswith("source-ingest:"):
                candidate = candidate.model_copy(
                    update={"created_by": f"source-ingest:{candidate.created_by}"}
                )
            normalized.append((None, candidate))
        return sorted(
            normalized,
            key=lambda item: (item[1].created_by, item[1].candidate_id),
        )

    def collect(
        self,
        workspaces: Sequence[Path],
        run_id: str,
    ) -> list[tuple[Path, RuntimeCandidate]]:
        collected = []
        for workspace in sorted(Path(item) for item in workspaces):
            layout = KnowledgeLayout(workspace)
            if not layout.state.is_file():
                continue
            for candidate in FileCandidateOutbox(layout.candidates).read_all():
                collected.append((workspace, candidate))
        return sorted(
            collected,
            key=lambda item: (str(item[0]), item[1].candidate_id),
        )

    def collect_usage_observations(
        self,
        workspaces: Sequence[Path],
        *,
        run_id: str,
    ) -> dict[str, ObservationRecord]:
        if self.fact_reader is None:
            return {}
        observations = {}
        for workspace in sorted(Path(item) for item in workspaces):
            layout = KnowledgeLayout(workspace)
            if not layout.state.is_file():
                continue
            state, signature, target = _workspace_context(workspace)
            for item in self.fact_reader.materialize_usage_observations(
                workspace,
                state,
                signature,
                target,
                run_id=run_id,
            ):
                prior = observations.get(item.id)
                if prior is not None and not _same_observation(prior, item):
                    raise ValueError(f"observation id collision: {item.id}")
                observations[item.id] = prior or item
        return observations

    def materialize_observations(
        self,
        candidates: Sequence[
            tuple[Path | None, CandidateConcept | RuntimeCandidate]
        ],
        *,
        run_id: str,
        published_observations: dict[str, ObservationRecord],
    ) -> tuple[
        dict[str, ObservationRecord],
        list[tuple[CandidateConcept, list[ConceptEvidence]]],
        list[str],
        list[str],
    ]:
        observations: dict[str, ObservationRecord] = {}
        valid = []
        rejected = []
        audit_warnings = []
        source_packages = list(
            FilesystemCatalog(self.catalog_root).iter_source_packages()
        )
        for workspace, candidate in candidates:
            candidate_evidence: list[ConceptEvidence] = []
            try:
                candidate = _canonicalize_candidate_symptoms(candidate)
                candidate, candidate_warnings = normalize_candidate_vocabulary(
                    candidate,
                    self.vocabulary,
                )
                audit_warnings.extend(candidate_warnings)
                if workspace is None:
                    if candidate.observation_intents:
                        raise ValueError(
                            "static Candidate cannot materialize Observations"
                        )
                    if candidate.proposed_kind == "experience":
                        raise ValueError(
                            "static Candidate cannot publish Experience"
                        )
                    if not isinstance(candidate, CandidateConcept):
                        raise ValueError(
                            "static publication requires a CandidateConcept"
                        )
                    if self.operator_taxonomy is not None:
                        self.operator_taxonomy.validate_operator_scope(
                            candidate.scope.operator,
                            label=candidate.candidate_id,
                        )
                    validate_static_candidate(candidate, source_packages)
                    valid.append((candidate, candidate_evidence))
                    continue
                state, signature, target = _workspace_context(workspace)
                if not isinstance(candidate, RuntimeCandidate):
                    raise ValueError(
                        "workspace outbox requires a RuntimeCandidate"
                    )
                scope_hints, unknown_scope_motifs = (
                    _canonicalize_runtime_scope_hints(
                        candidate.scope_hints,
                        self.operator_taxonomy,
                    )
                )
                audit_warnings.extend(
                    f"{candidate.candidate_id}: unknown operator motif {value!r}"
                    for value in unknown_scope_motifs
                )
                candidate = CandidateConcept(
                    **candidate.model_dump(
                        mode="python",
                        exclude={"scope_hints"},
                    ),
                    scope=_bind_runtime_scope(
                        scope_hints,
                        signature,
                        target,
                    ),
                )
                if self.operator_taxonomy is not None:
                    self.operator_taxonomy.validate_operator_scope(
                        candidate.scope.operator,
                        label=candidate.candidate_id,
                    )
                candidate_observations = []
                for intent in candidate.observation_intents:
                    if self.fact_reader is None:
                        raise ValueError("run fact reader is not configured")
                    item = self.fact_reader.materialize_observation(
                        workspace,
                        state,
                        signature,
                        target,
                        round_num=intent.round_num,
                        run_id=run_id,
                    )
                    prior = (
                        observations.get(item.id)
                        or published_observations.get(item.id)
                    )
                    if prior is not None and not _same_observation(prior, item):
                        raise ValueError(
                            f"observation id collision: {item.id}"
                        )
                    selected = prior or item
                    observations[item.id] = selected
                    candidate_observations.append(selected)
                    candidate_evidence.append(
                        ConceptEvidence(
                            observation_ref=item.id,
                            stance=intent.stance,
                            rationale=intent.rationale,
                            confidence=intent.confidence,
                        )
                    )
                candidate = _bind_runtime_retrieval(
                    candidate,
                    candidate_observations,
                )
                candidate, candidate_warnings = normalize_candidate_vocabulary(
                    candidate,
                    self.vocabulary,
                )
                audit_warnings.extend(candidate_warnings)
                _validate_workspace_provenance(candidate, workspace)
                validate_runtime_candidate(candidate)
                if candidate.proposed_kind == "experience" and not candidate_evidence:
                    raise ValueError("Experience requires measured observations")
                valid.append((candidate, candidate_evidence))
            except (OSError, KeyError, TypeError, ValueError) as exc:
                rejected.append(f"{candidate.candidate_id}: {exc}")
        return (
            observations,
            valid,
            sorted(rejected),
            sorted(set(audit_warnings)),
        )


def _workspace_context(
    workspace: Path,
) -> tuple[WorkspaceKnowledgeState, OperatorSignature, TargetContext]:
    layout = KnowledgeLayout(workspace)
    state = WorkspaceKnowledgeState.model_validate_json(
        layout.state.read_text(encoding="utf-8")
    )
    signature = OperatorSignature.model_validate_json(
        layout.operator_signature.read_text(encoding="utf-8")
    )
    target = TargetContext.model_validate_json(
        layout.target_context.read_text(encoding="utf-8")
    )
    # Old workspaces stored a content hash as the operator identity. Normalize
    # it on read so replay publishes the same human scope as a new run.
    signature = normalize_operator_signature(signature)
    return state, signature, target



def _bind_runtime_scope(
    hints: RuntimeScopeHints,
    signature: OperatorSignature,
    target: TargetContext,
) -> Scope:
    """Bind identity fields; retain Agent vocabulary as normalized search tags."""

    _validate_publish_target(target)
    missing_capabilities = sorted(
        set(signature.required_capabilities) - set(target.capabilities)
    )
    if missing_capabilities:
        raise ValueError(
            "Definition requires target capabilities not reported by the service: "
            + ", ".join(missing_capabilities)
    )
    bound_operator = OperatorScope(
        definition_ids=[signature.definition_id],
        op_types=_normalized_tags([signature.op_type]),
        motifs=_normalized_tags([*signature.motifs, *hints.motifs]),
        dataflow=_normalized_tags(signature.dataflow),
        dtypes=_normalized_tags(signature.dtypes),
        layouts=_normalized_tags(signature.layouts),
    )
    bound_target = TargetScope(
        level="exact",
        backend=target.backend,
        architecture=target.architecture,
        devices=[target.device],
        capabilities=signature.required_capabilities,
        software=target.software.model_dump(mode="python"),
    )
    return Scope(
        target=bound_target,
        operator=bound_operator,
        workloads=WorkloadScope(
            all=[
                WorkloadPredicate(field=key, op="eq", value=value)
                for key, value in sorted(signature.workload_features.items())
            ]
        ),
        numerics=signature.numerics,
    )


def _canonicalize_runtime_scope_hints(
    hints: RuntimeScopeHints,
    taxonomy: OperatorTaxonomy | None,
) -> tuple[RuntimeScopeHints, list[str]]:
    """Map runtime Scope hints to active taxonomy motifs and drop unknown tags."""

    if taxonomy is None:
        return hints, []
    canonical = []
    unknown = []
    for value in hints.motifs:
        aliases = taxonomy.alias_families.get(normalize_taxonomy_term(value))
        motif = aliases[0] if aliases and aliases[0] in taxonomy.motifs else ""
        if motif:
            canonical.append(motif)
        else:
            unknown.append(value)
    return (
        hints.model_copy(update={"motifs": _normalized_tags(canonical)}),
        sorted(set(unknown)),
    )


def _validate_publish_target(target: TargetContext) -> None:
    if target.source != "eval_service":
        raise ValueError("stable runtime knowledge requires Eval service target metadata")
    if target.metadata is not None and not target.metadata.complete:
        detail = (
            ": " + ", ".join(target.metadata.missing)
            if target.metadata.missing
            else ""
        )
        raise ValueError(
            "stable runtime knowledge requires complete Server status metadata"
            + detail
        )
    required = {
        "backend": target.backend,
        "architecture": target.architecture,
        "device": target.device,
        "software.language": target.software.language,
        "software.compiler": target.software.compiler,
        "software.compiler_version": target.software.compiler_version,
        "software.runtime": target.software.runtime,
        "software.runtime_version": target.software.runtime_version,
        "software.driver_version": target.software.driver_version,
    }
    missing = sorted(name for name, value in required.items() if not str(value).strip())
    if missing:
        raise ValueError(
            "stable runtime knowledge requires target metadata: "
            + ", ".join(missing)
        )


def _canonicalize_candidate_symptoms(
    candidate: CandidateConcept | RuntimeCandidate,
) -> CandidateConcept | RuntimeCandidate:
    """Normalize evaluator statuses stored by pre-migration candidate outboxes."""

    symptoms = sorted(
        {
            canonical_failure_symptom(item)
            for item in candidate.retrieval.symptoms
        }
    )
    if symptoms == candidate.retrieval.symptoms:
        return candidate
    return candidate.model_copy(
        update={
            "retrieval": candidate.retrieval.model_copy(
                update={"symptoms": symptoms}
            )
        }
    )


def _bind_runtime_retrieval(
    candidate: CandidateConcept,
    observations: Sequence[ObservationRecord],
) -> CandidateConcept:
    failed_statuses = {
        canonical_failure_symptom(item.outcome.status)
        for item in observations
        if item.outcome.status != "PASSED"
    }
    if not failed_statuses:
        return candidate
    retrieval = candidate.retrieval.model_copy(
        update={
            "symptoms": sorted(
                {
                    str(item).strip()
                    for item in [
                        *candidate.retrieval.symptoms,
                        *failed_statuses,
                    ]
                    if str(item).strip()
                }
            )
        }
    )
    return candidate.model_copy(update={"retrieval": retrieval})


def _validate_workspace_provenance(
    candidate: CandidateConcept,
    workspace: Path,
) -> None:
    sources, concept_ids = workspace_provenance(workspace)
    allowed_sources = {
        (item.resource, item.revision, item.locator)
        for item in sources
    }
    requested_sources = {
        (item.resource, item.revision, item.locator)
        for item in candidate.source_refs
    }
    unavailable_sources = sorted(requested_sources - allowed_sources)
    if unavailable_sources:
        raise ValueError(
            "Sources were not retrieved in this workspace: "
            + ", ".join(
                f"{resource}@{revision}::{locator}"
                for resource, revision, locator in unavailable_sources
            )
        )
    unavailable_relations = sorted(
        {item.target for item in candidate.relations} - set(concept_ids)
    )
    if unavailable_relations:
        raise ValueError(
            "relation targets were not retrieved in this workspace: "
            + ", ".join(unavailable_relations)
        )



def _normalized_tags(values: Iterable[str]) -> list[str]:
    return sorted(
        {
            str(value).strip().lower()
            for value in values
            if str(value).strip()
        }
    )



def _same_observation(
    left: ObservationRecord,
    right: ObservationRecord,
) -> bool:
    stable_left = left.model_dump(mode="json", exclude={"recorded_at"})
    stable_right = right.model_dump(mode="json", exclude={"recorded_at"})
    return stable_left == stable_right
