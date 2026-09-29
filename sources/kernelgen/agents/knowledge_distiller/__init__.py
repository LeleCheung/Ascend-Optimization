"""Distill one agent run into reports and structured V1 knowledge proposals.

Reads the agent's optimization trajectory (initial kernel + per-round code diff +
strategy/root_cause) from the prompt, loads listed profile analyses through
read-only tools, and produces two per-run Markdown fragments plus optional
CandidateDraft records.

Does NOT write KB itself; Python renders the reports and the V1 publisher owns
canonical persistence.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

from kernelgen.framework.base import BaseAgent
from kernelgen.framework.contract import extract_json
from kernelgen.data.implementation import (
    ImplementationLanguage,
    render_implementation_profile,
)
from kernelgen.knowledge.models import (
    CandidateDraft,
    KnowledgeApplicationEffect,
    KnowledgeUsageMode,
    KnowledgeUseDisposition,
    KnowledgeUseRole,
    QueryPhase,
    QueryTask,
    SourceReference,
    classify_knowledge_effect,
)
from kernelgen.knowledge.vocabulary import canonical_failure_symptom
from kernelgen.knowledge.validation import validate_runtime_candidate
from kernelgen.agents.knowledge_distiller.report_format import (
    validate_detailed_body,
    validate_experience_body,
    validate_round_references,
)


_QUERY_PHASES = frozenset(get_args(QueryPhase))
_QUERY_TASKS = frozenset(get_args(QueryTask))
_MAX_CANDIDATE_REPAIR_ATTEMPTS = 2


class SourceApplicationResult(BaseModel):
    """One Source declaration joined to its authoritative round outcome."""

    source_ref: SourceReference
    round_num: int = Field(gt=0)
    role: KnowledgeUseRole
    disposition: KnowledgeUseDisposition
    usage_mode: KnowledgeUsageMode
    effect: KnowledgeApplicationEffect
    application_note: str = ""
    affected_parts: List[str] = Field(default_factory=list)


def build_source_application_results(
    rounds: List[Dict[str, Any]],
) -> List[SourceApplicationResult]:
    """Derive promotion inputs from Ledger facts; reads alone are absent."""

    statuses = {
        int(item["round_num"]): str(
            (item.get("evaluation") or {}).get("status") or ""
        )
        for item in rounds
        if item.get("round_num") is not None
    }
    results = []
    for record in rounds:
        round_num = record.get("round_num")
        if not isinstance(round_num, int) or round_num <= 0:
            continue
        plan = record.get("plan") or {}
        uses = plan.get("knowledge_uses") or []
        applied_count = sum(
            isinstance(item, dict)
            and item.get("disposition") in {"adopted", "adapted"}
            for item in uses
        )
        usage_mode = (
            "none"
            if applied_count == 0
            else "single"
            if applied_count == 1
            else "combined"
        )
        evaluation = record.get("evaluation") or {}
        comparison = evaluation.get("comparison") or {}
        parent_round = record.get("experiment_parent_round_num")
        effect = classify_knowledge_effect(
            usage_mode=usage_mode,
            parent_status=statuses.get(parent_round),
            current_status=str(evaluation.get("status") or ""),
            performance_baseline_round_num=comparison.get(
                "performance_baseline_round_num",
                comparison.get("baseline_round_num"),
            ),
            geo_mean_delta_pct=comparison.get("geo_mean_delta_pct"),
        )
        for use in uses:
            if not isinstance(use, dict) or not use.get("source_ref"):
                continue
            try:
                results.append(
                    SourceApplicationResult(
                        source_ref=use["source_ref"],
                        round_num=round_num,
                        role=use.get("role"),
                        disposition=use.get("disposition"),
                        usage_mode=usage_mode,
                        effect=effect,
                        application_note=str(
                            use.get("application_note") or ""
                        ),
                        affected_parts=list(
                            use.get("affected_parts") or []
                        ),
                    )
                )
            except (TypeError, ValueError):
                continue
    return sorted(
        results,
        key=lambda item: (
            item.round_num,
            item.source_ref.resource,
            item.source_ref.revision,
            item.source_ref.locator,
        ),
    )


def _normalize_retrieval_axes(payload: Dict[str, Any]) -> None:
    """Repair recognized phase/task values emitted on the wrong axis."""
    candidates = payload.get("candidate_concepts")
    if not isinstance(candidates, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        retrieval = candidate.get("retrieval")
        if not isinstance(retrieval, dict):
            continue
        for source_key, target_key, allowed in (
            ("phases", "tasks", _QUERY_TASKS),
            ("tasks", "phases", _QUERY_PHASES),
        ):
            source = retrieval.get(source_key)
            target = retrieval.get(target_key)
            if not isinstance(source, list) or not isinstance(target, list):
                continue
            misplaced = [
                value for value in source
                if isinstance(value, str) and value in allowed
            ]
            if misplaced:
                retrieval[source_key] = [
                    value for value in source if value not in misplaced
                ]
                retrieval[target_key] = list(dict.fromkeys([*target, *misplaced]))


class DistillInput(BaseModel):
    """What the distiller needs: the optimization history + context."""
    definition_name: str
    op_type: str
    target_hardware: str
    implementation_language: ImplementationLanguage = ImplementationLanguage.TRITON
    definition: Dict[str, Any] = Field(default_factory=dict)
    # The optimization rounds (from ledger.history.rounds, serialized)
    rounds: List[Dict[str, Any]] = Field(default_factory=list)
    best_geo_mean: float = 0.0
    best_round: int = 0
    # Validated backend-neutral analyses remain on disk and are read on demand.
    profile_analysis_files: List[Dict[str, Any]] = Field(default_factory=list)
    # Legacy rollout context. V1 workspaces do not materialize this path.
    existing_experience: str = ""
    existing_detailed: str = ""
    available_source_refs: List[SourceReference] = Field(default_factory=list)
    known_concept_ids: List[str] = Field(default_factory=list)
    canonical_symptoms: List[str] = Field(default_factory=list)
    canonical_techniques: List[str] = Field(default_factory=list)
    source_application_results: List[SourceApplicationResult] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def derive_source_application_results(self) -> "DistillInput":
        if self.source_application_results:
            return self
        return self.model_copy(
            update={
                "source_application_results": (
                    build_source_application_results(self.rounds)
                )
            }
        )


class DistillOutput(BaseModel):
    """Per-run Markdown reports plus structured V1 knowledge proposals."""
    candidate_experience: str = Field(
        default="",
        description=(
            "Markdown fragment for the next agent; exact H2 sections are "
            "Lessons, Remaining Bottlenecks, Next Experiments"
        ),
    )
    candidate_detailed: str = Field(
        default="",
        description=(
            "Markdown analysis fragment; exact H2 sections are Strategy Evolution, "
            "Cross-Round Analysis, Failure Analysis, Open Questions"
        ),
    )
    candidate_concepts: List[CandidateDraft] = Field(
        default_factory=list,
        description="structured V1 knowledge proposals; authority fields are added by Python",
    )
    skip_reason: str = Field(default="", description="if empty = has content; if set = why skipping distillation")

    @field_validator("candidate_experience")
    @classmethod
    def validate_experience_markdown(cls, value: str) -> str:
        return validate_experience_body(value)

    @field_validator("candidate_detailed")
    @classmethod
    def validate_detailed_markdown(cls, value: str) -> str:
        return validate_detailed_body(value)

    @model_validator(mode="after")
    def validate_report_presence(self) -> "DistillOutput":
        if self.skip_reason:
            if (
                self.candidate_experience
                or self.candidate_detailed
                or self.candidate_concepts
            ):
                raise ValueError("skip_reason requires empty reports and candidates")
            return self
        if not self.candidate_experience or not self.candidate_detailed:
            raise ValueError(
                "completed distillation requires both per-run Markdown reports"
            )
        return self


class KnowledgeDistillerAgent(BaseAgent):
    name = "knowledge_distiller"
    InputModel = DistillInput
    OutputModel = DistillOutput

    def __init__(self):
        self._valid_rounds: set[int] = set()
        self._round_statuses: dict[int, str] = {}
        self._available_source_refs: set[tuple[str, str, str]] = set()
        self._known_concept_ids: set[str] = set()
        self._canonical_symptoms: tuple[str, ...] = ()
        self._canonical_techniques: tuple[str, ...] = ()

    def preprocess(self, inp: DistillInput, runtime) -> str:
        from kernelgen.framework.contract import render_contract
        from kernelgen.data.trajectory import build_trajectory

        role = self._role_for(runtime)

        if not inp.rounds:
            return ('No optimization rounds to distill. Output: '
                    '{"skip_reason": "no rounds", "candidate_experience": "", '
                    '"candidate_detailed": "", "candidate_concepts": []}')

        self._valid_rounds = {
            int(item["round_num"])
            for item in inp.rounds
            if item.get("round_num") is not None
        }
        self._round_statuses = {
            int(item["round_num"]): str(
                (item.get("evaluation") or {}).get("status") or ""
            )
            for item in inp.rounds
            if item.get("round_num") is not None
        }
        self._available_source_refs = {
            (item.resource, item.revision, item.locator)
            for item in inp.available_source_refs
        }
        self._known_concept_ids = set(inp.known_concept_ids)
        self._canonical_symptoms = tuple(inp.canonical_symptoms)
        self._canonical_techniques = tuple(inp.canonical_techniques)

        # Build trajectory (initial kernel + per-round diff/REWRITE)
        trajectory = build_trajectory(inp.rounds)

        # Context summary
        context = (
            f"Definition: {inp.definition_name} (op_type={inp.op_type}, hardware={inp.target_hardware})\n"
            f"Best: geo_mean={inp.best_geo_mean:.3f}x at round {inp.best_round}\n"
            f"Total rounds: {len(inp.rounds)}\n"
            "Canonical definition (do not infer conflicting shapes or constants):\n"
            f"{json.dumps(inp.definition, indent=2, ensure_ascii=False, default=str)[:12000]}"
        )

        existing_blocks = []
        if inp.existing_experience:
            existing_blocks.append(
                "<existing_kb>\n"
                "The following experience is ALREADY in KB — avoid duplicating it, "
                f"focus on what's NEW from this run:\n{inp.existing_experience[:2000]}\n"
                "</existing_kb>"
            )
        if inp.existing_detailed:
            existing_blocks.append(
                "<existing_detailed>\n"
                "Existing detailed evidence (use only to avoid repetition):\n"
                f"{inp.existing_detailed[:2000]}\n"
                "</existing_detailed>"
            )

        profile_block = ""
        if inp.profile_analysis_files:
            profile_block = (
                "<profile_analysis_files>\n"
                "Before drafting any report or Concept candidate, use Read to inspect "
                "every file listed below. Use the analyses together to compare profiler "
                "findings across rounds. Treat their profiler/backend fields as "
                "authoritative; do not replace msprof with NCU or vice versa. Do not "
                "read raw profiler artifacts referenced inside these files. If one file "
                "cannot be read, continue with the remaining evidence and identify the "
                "missing round explicitly.\n"
                f"{json.dumps(inp.profile_analysis_files, indent=2, ensure_ascii=False, default=str)}\n"
                "</profile_analysis_files>"
            )

        provenance_block = (
            "<available_provenance>\n"
            "Only the following exact records were actually retrieved in this "
            "workspace. source_refs and relation targets must be selected from "
            "these lists; an empty list means do not emit that metadata. Use "
            "get_knowledge or get_source only to inspect an exact listed record "
            "when its full content is needed; do not perform open-ended retrieval.\n"
            f"Sources: {json.dumps([item.model_dump(mode='json') for item in inp.available_source_refs], ensure_ascii=False)}\n"
            f"Concept IDs: {json.dumps(inp.known_concept_ids, ensure_ascii=False)}\n"
            "</available_provenance>"
        )
        source_application_block = (
            "<source_application_results>\n"
            "Python joined Source declarations to measured round outcomes. "
            "Only entries here were present in ExperimentPlan; retrieval or "
            "read counts alone are not application evidence.\n"
            f"{json.dumps([item.model_dump(mode='json') for item in inp.source_application_results], indent=2, ensure_ascii=False)}\n"
            "</source_application_results>"
        )
        contract = render_contract(self.OutputModel)
        prompt = "\n\n".join(filter(None, [
            role,
            f"<context>\n{context}\n</context>",
            render_implementation_profile(inp.implementation_language),
            f"<trajectory>\n{trajectory}\n</trajectory>",
            profile_block,
            provenance_block,
            source_application_block,
            (
                "<analysis_requirement>\n"
                "Treat every round as plan → solution → evaluation → optional profile "
                "→ conclusion. Compare the frozen expected effect with Python-owned "
                "aggregate and per-workload measurements, preserve latency as well as "
                "speedup, and identify validated, partially validated, falsified, or "
                "unevaluable hypotheses. If the exact best round has a completed "
                "profile analysis, do not call it unprofiled. If only an earlier round "
                "was profiled, state that narrower scope explicitly. When analysis "
                "status is failed, unsupported, or inconclusive, report that exact "
                "state instead of inventing profiler findings.\n"
                "</analysis_requirement>"
            ),
            (
                "<report_requirement>\n"
                "The two report fields are per-run Markdown fragments, not canonical KB entries. "
                "Do not add an H1 or run metadata; Python adds authoritative metadata from ledger.\n\n"
                "candidate_experience must contain exactly these H2 sections in this order:\n"
                "## Lessons\n"
                "- Each lesson starts with [validated], [conditional], or [refuted].\n"
                "- Include Evidence: R<n>, Scope, Action, and Reopen if when applicable.\n"
                "## Remaining Bottlenecks\n"
                "## Next Experiments\n"
                "- Each experiment states the change and expected validation signal.\n\n"
                "candidate_detailed must contain exactly these H2 sections in this order:\n"
                "## Strategy Evolution\n"
                "## Cross-Round Analysis\n"
                "- Analyze only meaningful comparisons; cite ledger rounds as R<n>.\n"
                "- For each finding state Status, Rounds, Analysis, and Uncertainty.\n"
                "## Failure Analysis\n"
                "## Open Questions\n\n"
                "Do not rewrite the ledger round by round. Do not invent run identity, "
                "best result, correctness status, latency, speedup, or profile values. "
                "Whenever reporting a measured fact, cite an existing round as R<n>.\n"
                "</report_requirement>"
            ),
            (
                "<candidate_requirement>\n"
                "When the trajectory supports reusable knowledge, emit one atomic "
                "claim per candidate_concept. Reference, Method, and Experience bodies "
                "must contain exactly these H2 sections in order: ## Claim, "
                "## Evidence, ## Applicability, ## Action, ## Limits. A Method must "
                "also contain ### Expected Metric Change inside Action and "
                "### Mechanism Requirements inside Limits. A Diagnostic body must "
                "instead contain exactly: ## Symptom, ## Likely Causes, "
                "## Candidate Techniques, ## Diagnosis Checklist, ## Caveats. "
                "Evidence or diagnostic text must cite exactly every observation "
                "intent as R<n>; ranges such as R2-R4 cover every included round. "
                "retrieval.phases and retrieval.tasks must be non-empty. Include "
                "only canonical symptom/technique values when the lists below cover "
                "the claim; otherwise use one concise new value so Publisher can "
                "report it for vocabulary review.\n"
                f"Canonical symptoms: {json.dumps(list(self._canonical_symptoms), ensure_ascii=False)}\n"
                f"Canonical techniques: {json.dumps(list(self._canonical_techniques), ensure_ascii=False)}\n"
                "Put concrete API/compiler names in retrieval.keywords, diagnostic "
                "procedures in the Diagnostic body, and only reusable optimization "
                "methods in retrieval.techniques. "
                "Python adds canonical failure symptoms for failed evidence; add only "
                "validated profile finding labels yourself. Use "
                "observation_intents with exact measured round numbers and "
                "conservative confidence. Each observation intent uses claim_stance "
                "relative only to the candidate Concept claim: supports means the "
                "observation agrees with the claim, refutes means it contradicts the "
                "claim, and illustrates means it is a relevant example without "
                "directional proof. A round that disproves its own optimization "
                "hypothesis can still support a negative Concept claim; describe the "
                "hypothesis outcome in rationale, never by reversing claim_stance. "
                "Default publish_action is create. If an exact Concept ID in "
                "available_provenance expresses the same claim and scope, use "
                "publish_action=update with that exact target_concept_id and a concise "
                "change_reason; do not create a duplicate claim_key for wording changes. "
                "Use a supersedes relation only when this candidate replaces an older "
                "claim, not for ordinary evidence updates. "
                "Do not copy measured values into scope. "
                "Python binds target, definition, constant workloads, explicit "
                "Definition capabilities/dataflow/layouts, software, and numerics. "
                "Do not emit scope; scope_hints may contain only canonical motifs "
                "already present in the trajectory operator signature and supported by "
                "this trajectory; use an empty list instead of inventing a motif. "
                "External API, compiler, or hardware facts "
                "require an exact available source_ref; otherwise state only the "
                "measured inference. Relations require an exact available Concept ID. "
                "Experience requires measured observations. Source-only facts belong "
                "to Reference, Method, or Diagnostic. A stable external fact may "
                "be proposed as Reference from Source alone. A Source-guided measured "
                "method or diagnostic must include both the exact source_ref and an "
                "observation_intent for the supporting round. A single-use "
                "correctness_recovered or performance_improved result is eligible "
                "measured support. Combined success is contextual evidence, not "
                "independent attribution. Regression, rejection, no_material_change, "
                "and unclassified results must not be promoted as positive proof. "
                "Keep distinct actionable claims from the same Source in separate "
                "candidate_concepts.\n"
                "</candidate_requirement>"
            ),
            "\n\n".join(existing_blocks),
            f"--- OUTPUT CONTRACT ---\n{contract}",
        ]))
        return prompt

    def postprocess(self, raw: str, runtime) -> DistillOutput:
        """Keep valid reports/candidates while repairing invalid candidates alone."""
        payload = extract_json(raw)
        if not isinstance(payload, dict):
            raise ValueError("distiller output must be a JSON object")
        _normalize_retrieval_axes(payload)
        raw_candidates = payload.get("candidate_concepts", [])
        if raw_candidates is None:
            raw_candidates = []
        if not isinstance(raw_candidates, list):
            raise ValueError("candidate_concepts must be a JSON array")

        report_payload = dict(payload)
        report_payload["candidate_concepts"] = []
        output = self.OutputModel.model_validate(report_payload)
        if output.skip_reason:
            if raw_candidates:
                raise ValueError("skip_reason requires empty reports and candidates")
            return output

        validate_round_references(
            experience=output.candidate_experience,
            detailed=output.candidate_detailed,
            valid_rounds=self._valid_rounds,
        )
        candidates = []
        for index, candidate_payload in enumerate(raw_candidates, start=1):
            try:
                candidate = self._validate_candidate(candidate_payload)
            except (TypeError, ValueError) as exc:
                candidate = self._repair_candidate(
                    candidate_payload,
                    exc,
                    runtime,
                    index=index,
                )
            if candidate is not None:
                candidates.append(candidate)
        output = output.model_copy(update={"candidate_concepts": candidates})
        return output

    def _validate_candidate(self, payload: Any) -> CandidateDraft:
        normalized = {"candidate_concepts": [payload]}
        _normalize_retrieval_axes(normalized)
        candidate = CandidateDraft.model_validate(
            normalized["candidate_concepts"][0]
        )
        intent_rounds = {
            item.round_num for item in candidate.observation_intents
        }
        unknown = sorted(intent_rounds - self._valid_rounds)
        if unknown:
            raise ValueError(
                "candidate cites rounds absent from ledger: "
                + ", ".join(f"R{item}" for item in unknown)
            )
        failed_statuses = sorted(
            {
                self._round_statuses.get(item, "")
                for item in intent_rounds
                if self._round_statuses.get(item, "") not in {"", "PASSED"}
            }
        )
        if failed_statuses:
            canonical_failures = {
                canonical_failure_symptom(status)
                for status in failed_statuses
            }
            candidate = candidate.model_copy(
                update={
                    "retrieval": candidate.retrieval.model_copy(
                        update={
                            "symptoms": sorted(
                                {
                                    *candidate.retrieval.symptoms,
                                    *canonical_failures,
                                }
                            )
                        }
                    )
                }
            )
        unavailable_sources = sorted(
            {
                (item.resource, item.revision, item.locator)
                for item in candidate.source_refs
            }
            - self._available_source_refs
        )
        if unavailable_sources:
            raise ValueError(
                "candidate cites Sources not retrieved in this workspace: "
                + ", ".join(
                    f"{resource}@{revision}::{locator}"
                    for resource, revision, locator in unavailable_sources
                )
            )
        unavailable_relations = sorted(
            {item.target for item in candidate.relations}
            - self._known_concept_ids
        )
        if unavailable_relations:
            raise ValueError(
                "candidate relates to Concepts not retrieved in this workspace: "
                + ", ".join(unavailable_relations)
            )
        if (
            candidate.publish_action == "update"
            and candidate.target_concept_id not in self._known_concept_ids
        ):
            raise ValueError(
                "candidate update target was not retrieved in this workspace: "
                + str(candidate.target_concept_id)
            )
        validate_runtime_candidate(candidate)
        return candidate

    def _repair_candidate(
        self,
        payload: Any,
        error: Exception,
        runtime,
        *,
        index: int,
    ) -> CandidateDraft | None:
        """Ask the model to repair one candidate without regenerating the bundle."""
        from kernelgen.framework.contract import render_contract

        current = payload
        last_error = error
        for attempt in range(1, _MAX_CANDIDATE_REPAIR_ATTEMPTS + 1):
            prompt = "\n\n".join(
                [
                    (
                        "Repair exactly one KernelGen CandidateDraft. Return only "
                        "the corrected CandidateDraft JSON object, without Markdown "
                        "fences, commentary, or a top-level wrapper."
                    ),
                    (
                        "Preserve the original claim. Fix only the validation "
                        "problem. Every candidate needs observation_intents or "
                        "source_refs. Every observation intent uses claim_stance "
                        "relative only to the candidate Concept claim: supports "
                        "agrees with the claim, refutes contradicts the claim, and "
                        "illustrates is a relevant example without directional proof. "
                        "A round that disproves its own optimization hypothesis can "
                        "still support a negative Concept claim; describe that "
                        "hypothesis outcome in rationale and do not reverse "
                        "claim_stance. Preserve publish_action, target_concept_id, and "
                        "change_reason; an update target must be an exact allowed "
                        "Concept ID. "
                        "Every R<n> cited in the body must match the "
                        "observation_intents exactly. Do not invent Source refs or "
                        "relation targets. Remove a relation when no allowed target "
                        "expresses it."
                    ),
                    f"Validation error:\n{str(last_error)[:4000]}",
                    (
                        "Valid ledger rounds and statuses:\n"
                        f"{json.dumps(self._round_statuses, ensure_ascii=False)}"
                    ),
                    (
                        "Allowed Source refs:\n"
                        f"{json.dumps(sorted(self._available_source_refs), ensure_ascii=False)}"
                    ),
                    (
                        "Allowed relation targets:\n"
                        f"{json.dumps(sorted(self._known_concept_ids), ensure_ascii=False)}"
                    ),
                    (
                        "Candidate to repair:\n"
                        f"{json.dumps(current, indent=2, ensure_ascii=False, default=str)}"
                    ),
                    (
                        "CandidateDraft contract:\n"
                        f"{render_contract(CandidateDraft)}"
                    ),
                ]
            )
            try:
                # A focused repair must not reuse the native Distiller role, whose
                # contract asks for the complete report bundle again.
                raw = runtime.invoke(prompt, model=self.model)
                current = extract_json(raw)
                candidate = self._validate_candidate(current)
            except Exception as exc:
                last_error = exc
                continue
            print(
                f"[Distiller] Repaired candidate {index} "
                f"after {attempt} attempt(s)"
            )
            return candidate
        print(
            f"[Distiller] Skipped invalid candidate {index} after "
            f"{_MAX_CANDIDATE_REPAIR_ATTEMPTS} repair attempts: {last_error}"
        )
        return None
