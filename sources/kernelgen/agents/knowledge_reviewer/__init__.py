"""Independent semantic Reviewer for runtime KB publication candidates."""

from __future__ import annotations

from kernelgen.framework.base import BaseAgent
from kernelgen.framework.contract import render_contract
from kernelgen.knowledge.models import KnowledgeReviewInput, KnowledgeReviewOutput


class KnowledgeReviewerAgent(BaseAgent):
    """Judge reusable knowledge without changing authoritative run facts."""

    name = "knowledge_reviewer"
    InputModel = KnowledgeReviewInput
    OutputModel = KnowledgeReviewOutput

    def preprocess(self, inp: KnowledgeReviewInput, runtime) -> str:
        role = self._role_for(runtime)
        requirements = (
            "Review every review unit in this packet exactly once; do not review "
            "or wait for units from another packet. Observation v2 fields are "
            "immutable records, not infallible claims, and must not be rewritten. "
            "Measurements establish outcomes for the evaluated implementation "
            "and workloads, not the claimed bottleneck or compiler root cause. "
            "experiment_plan and agent_conclusion are the author's intent and "
            "interpretation; code_changes records planned edits, not a verified "
            "code diff. Python automatic_checks do not establish semantic "
            "correctness beyond their stated checks. Verify implementation and "
            "causal claims against available source-bound evidence; defer when "
            "required evidence is unavailable, reject when it contradicts the "
            "claim, and never repair the original records to fit a proposal. "
            "Use the kernelgen MCP tools to read full Catalog or Source details "
            "only when they matter. Before citing any Source ref, call get_source "
            "for that exact revision and locator. Before using a nearby Concept as "
            "target_concept_id or conflict_refs, call get_knowledge for that exact "
            "Concept ID. A query result or packet summary is not a detail read. "
            "The publisher verifies the successful detail-read audit and will "
            "defer any decision whose external refs were not actually read. "
            "Approve only when the candidate claim is actually supported, its "
            "canonical Scope is justified, it is reusable within that Scope, and "
            "the merge action is safe. A performance win from combined knowledge "
            "does not establish single-item causality. Use defer for insufficient "
            "evidence, unresolved conflicts, ambiguous attribution, or facts that "
            "need another reproduction. Use reject for a clearly false, "
            "contradicted, or non-knowledge claim. Cite only exact Observation IDs, "
            "Source refs, and nearby Concept IDs present in the packet. For an "
            "existing same-Scope Concept, prefer attach_evidence; use "
            "update_existing only when the candidate body is a justified semantic "
            "revision. Return no IDs that are absent from the packet."
        )
        return "\n\n".join(
            part
            for part in [
                role,
                requirements,
                "--- REVIEW PACKET ---",
                inp.model_dump_json(indent=2),
                "--- OUTPUT CONTRACT ---",
                render_contract(self.OutputModel),
            ]
            if part
        )


__all__ = ["KnowledgeReviewerAgent"]
