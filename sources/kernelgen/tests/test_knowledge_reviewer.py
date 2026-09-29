"""Semantic Reviewer isolation, validation, and fail-closed behavior."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernelgen.agents.knowledge_reviewer import KnowledgeReviewerAgent
from kernelgen.framework import FakeRuntime
from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.config import KnowledgeReviewerMode
from kernelgen.knowledge.contracts.runtime import PublishResult, RetrievalRecord
from kernelgen.knowledge.layout import CatalogLayout, KnowledgeLayout
from kernelgen.knowledge.models import (
    CandidateReviewDecision,
    CandidateReviewUnit,
    ConceptEvidence,
    KnowledgeReviewInput,
    KnowledgeReviewOutput,
    ObservationLocator,
    ObservationOutcome,
    ObservationRecord,
)
from kernelgen.knowledge.publishing.publisher import BatchPublisher
from kernelgen.knowledge.publishing.reviewer import (
    ReviewerExecutionResult,
    review_candidate_batch,
)
from kernelgen.knowledge.records import FileRetrievalAudit
from kernelgen.tests._knowledge_runtime_support import (
    _lifecycle_candidate,
    _seed,
)
from kernelgen.workflows.optimization.kernelgen.finalization import _build_epoch_reviewer


def _source_key(source) -> str:
    return f"{source.resource}@{source.revision}::{source.locator}"


def _detail_read(
    suffix: str,
    operation: str,
    reference: str,
    *,
    status: str = "success",
) -> RetrievalRecord:
    return RetrievalRecord(
        event_id=f"retrieval:{suffix}",
        operation=operation,
        origin="mcp",
        status=status,
        returned_refs=(
            [reference] if operation == "get_knowledge" else []
        ),
        returned_sources=([reference] if operation == "get_source" else []),
        created_at=datetime.now(timezone.utc),
    )


def test_reviewer_can_attach_evidence_without_rewriting_existing_concept(
    tmp_path,
):
    catalog_root = tmp_path / "catalog"
    existing = _seed(catalog_root)
    candidate = _lifecycle_candidate(
        existing,
        candidate_id="runtime:attach-1",
        claim_key=existing.claim_key,
        title="A more promotional candidate title",
        body="# Claim\n\nA body that must not replace the reviewed Concept.",
    )

    def reviewer(review_input):
        unit = review_input.units[0]
        source = unit.candidates[0].source_refs[0]
        output = KnowledgeReviewOutput(
            decisions=[
                CandidateReviewDecision(
                    review_id=unit.review_id,
                    decision="approve",
                    merge_action="attach_evidence",
                    target_concept_id=existing.id,
                    evidence_refs=[_source_key(source)],
                    conflict_refs=[],
                    scope_assessment="compatible",
                    reusable=True,
                    confidence="high",
                    reason="The same canonical claim and Scope already exist.",
                )
            ]
        )
        return ReviewerExecutionResult(
            output=output,
            retrievals=(
                _detail_read(
                    "attach-source",
                    "get_source",
                    _source_key(source),
                ),
                _detail_read(
                    "attach-target",
                    "get_knowledge",
                    existing.id,
                ),
            ),
        )

    result = review_candidate_batch(
        FilesystemCatalog(catalog_root),
        [(candidate, [])],
        {},
        run_id="run-1",
        batch_id="run-1-epoch-1",
        reviewer=reviewer,
    )

    assert result.records[0].decision == "approve"
    assert result.records[0].merge_action == "attach_evidence"
    assert result.records[0].detail_read_refs == sorted(
        [existing.id, _source_key(candidate.source_refs[0])]
    )
    assert result.records[0].detail_read_event_ids == [
        "retrieval:attach-source",
        "retrieval:attach-target",
    ]
    transformed = result.candidates[0][0]
    assert transformed.publish_action == "create"
    assert transformed.target_concept_id is None
    assert transformed.title == existing.title
    assert transformed.body == existing.body


def test_reviewer_failure_defers_every_candidate(tmp_path):
    catalog_root = tmp_path / "catalog"
    existing = _seed(catalog_root)
    candidate = _lifecycle_candidate(
        existing,
        candidate_id="runtime:defer-1",
        claim_key="method.reduction.new_variant",
        title="New reduction variant",
        body="# Claim\n\nA claim awaiting independent review.",
    )

    def unavailable(_):
        raise RuntimeError("review provider unavailable")

    result = review_candidate_batch(
        FilesystemCatalog(catalog_root),
        [(candidate, [])],
        {},
        run_id="run-1",
        batch_id="run-1-epoch-1",
        reviewer=unavailable,
    )

    assert result.candidates == []
    assert result.records[0].decision == "defer"
    assert result.records[0].reviewer_status == "fallback"
    assert "Reviewer unavailable" in result.warnings[0]


def test_runtime_agent_uses_strict_packet_refs(tmp_path):
    catalog_root = tmp_path / "catalog"
    existing = _seed(catalog_root)
    candidate = _lifecycle_candidate(
        existing,
        candidate_id="runtime:agent-1",
        claim_key="method.reduction.reviewed_variant",
        title="Reviewed reduction variant",
        body="# Claim\n\nA source-supported reviewed variant.",
    )
    runtime = FakeRuntime([])

    def reviewer(review_input):
        unit = review_input.units[0]
        source = unit.candidates[0].source_refs[0]
        runtime._replies.append(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "decisions": [
                        {
                            "review_id": unit.review_id,
                            "decision": "approve",
                            "merge_action": "create_new",
                            "target_concept_id": None,
                            "evidence_refs": [_source_key(source)],
                            "conflict_refs": [],
                            "scope_assessment": "compatible",
                            "reusable": True,
                            "confidence": "medium",
                            "reason": "The supplied Source supports this scoped rule.",
                        }
                    ],
                }
            )
        )
        output = KnowledgeReviewerAgent().run(review_input, runtime)
        return ReviewerExecutionResult(
            output=output,
            retrievals=(
                _detail_read(
                    "runtime-source",
                    "get_source",
                    _source_key(source),
                ),
            ),
        )

    result = review_candidate_batch(
        FilesystemCatalog(catalog_root),
        [(candidate, [])],
        {},
        run_id="run-1",
        batch_id="run-1-epoch-1",
        reviewer=reviewer,
    )

    assert result.records[0].decision == "approve"
    assert len(result.candidates) == 1
    prompt = runtime.calls[0]["prompt"]
    assert "must not be rewritten" in prompt
    assert "Combined knowledge" in prompt
    assert candidate.candidate_id in prompt


@pytest.mark.parametrize("role_mode", ["inline", "native", "provider_role"])
def test_reviewer_distinguishes_recorded_claims_from_measurements(tmp_path, role_mode):
    existing = _seed(tmp_path / "catalog")
    candidate = _lifecycle_candidate(
        existing,
        candidate_id="runtime:reported-tile",
        claim_key="method.reduction.reported_tile",
        title="A reported tile choice needs source verification",
        body="# Claim\n\nThe author reports a 128 by 256 tile.",
    )
    observation = ObservationRecord(
        id="kg:observation:run-1:agent0:round-1",
        run_id="run-1",
        workspace_id="agent0",
        definition_id="sum_rows",
        round_num=1,
        target_context_ref="archive://run-1/agent0/context/target.json",
        operator_signature_ref="archive://run-1/agent0/context/operator.json",
        ledger_locator=ObservationLocator(path="archive://run-1/agent0/ledger.json", round_num=1),
        solution_sha256="a" * 64,
        outcome=ObservationOutcome(
            status="PASSED",
            geo_mean_speedup=1.2,
            workload_results_ref="archive://run-1/agent0/ledger.json#/rounds/0/evaluation/workloads",
        ),
        experiment_plan={"key_params": {"BLOCK_N": 256}},
        code_changes="Changed BLOCK_N to 256",
        agent_conclusion={"root_cause": "The kernel is memory-bound"},
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    packet = KnowledgeReviewInput(
        batch_id="batch-1",
        run_id="run-1",
        units=[CandidateReviewUnit(
            review_id="review:reported-tile",
            candidate_ids=[candidate.candidate_id],
            candidates=[candidate],
            observations=[observation],
            automatic_checks=["provenance references are valid"],
        )],
    )
    before = packet.model_dump_json()
    runtime = SimpleNamespace(
        supports_native_agents=role_mode == "native",
        supports_agent_roles=role_mode == "provider_role",
    )
    prompt = KnowledgeReviewerAgent().preprocess(packet, runtime)
    assert "immutable records, not infallible claims" in prompt
    assert "code_changes records planned edits, not a verified code diff" in prompt
    assert "automatic_checks do not establish semantic correctness" in prompt
    assert "not the claimed bottleneck or compiler root cause" in prompt
    assert "must not be rewritten" in prompt
    assert observation.code_changes in prompt
    assert observation.agent_conclusion["root_cause"] in prompt
    assert packet.model_dump_json() == before


def test_reviewer_role_requires_effective_code_and_valid_diagnostic_controls():
    role = KnowledgeReviewerAgent()._native_definition_path().read_text(encoding="utf-8")
    assert "code_changes is the plan's description, not a verified source diff" in role
    assert "host dispatch and effective parameters" in role
    assert "invalid control" in role
    assert "unmeasured case" in role
    assert "compiler root cause" in role
    assert "Treat the supplied automatic checks" not in role


def test_invented_evidence_ref_is_programmatically_deferred(tmp_path):
    catalog_root = tmp_path / "catalog"
    existing = _seed(catalog_root)
    candidate = _lifecycle_candidate(
        existing,
        candidate_id="runtime:invented-ref",
        claim_key="method.reduction.invalid_ref",
        title="Invalid evidence reference",
        body="# Claim\n\nA claim with an invented review citation.",
    )

    def reviewer(review_input):
        return KnowledgeReviewOutput(
            decisions=[
                CandidateReviewDecision(
                    review_id=review_input.units[0].review_id,
                    decision="approve",
                    merge_action="create_new",
                    evidence_refs=["kg:observation:invented"],
                    conflict_refs=[],
                    scope_assessment="compatible",
                    reusable=True,
                    confidence="high",
                    reason="This decision cites an ID absent from the packet.",
                )
            ]
        )

    result = review_candidate_batch(
        FilesystemCatalog(catalog_root),
        [(candidate, [])],
        {},
        run_id="run-1",
        batch_id="run-1-epoch-1",
        reviewer=reviewer,
    )

    assert result.candidates == []
    assert result.records[0].decision == "defer"
    assert result.records[0].reviewer_status == "fallback"
    assert "unknown evidence_refs" in result.warnings[0]


def test_observation_v2_embeds_reviewer_experiment_facts(tmp_path):
    catalog_root = tmp_path / "catalog"
    observation = ObservationRecord(
        id="kg:observation:run-1:agent0:round-1",
        run_id="run-1",
        workspace_id="agent0",
        definition_id="sum_rows",
        round_num=1,
        target_context_ref="archive://run-1/agent0/context/target.json",
        operator_signature_ref=(
            "archive://run-1/agent0/context/operator.json"
        ),
        ledger_locator=ObservationLocator(
            path="archive://run-1/agent0/ledger.json",
            round_num=1,
        ),
        solution_sha256="a" * 64,
        outcome=ObservationOutcome(
            status="PASSED",
            geo_mean_speedup=1.2,
            workload_results_ref="archive://run-1/agent0/ledger.json#/rounds/0",
        ),
        experiment_plan={"strategy": "two-stage reduction"},
        code_changes="split the reduction",
        agent_conclusion={"expectation_status": "met"},
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    path = CatalogLayout(catalog_root).observations / "run-1.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(observation.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )

    BatchPublisher(catalog_root, fact_reader=None)._write_observations(
        catalog_root,
        "run-1",
        [observation],
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == "2.0"
    assert persisted["experiment_plan"] == observation.experiment_plan
    assert persisted["code_changes"] == observation.code_changes
    assert persisted["agent_conclusion"] == observation.agent_conclusion

    existing = _seed(catalog_root)
    candidate = _lifecycle_candidate(
        existing,
        candidate_id="runtime:review-facts",
        claim_key="method.reduction.review_facts",
        title="Reviewer facts are embedded in Observation v2",
        body="# Claim\n\nExperiment facts are frozen with measurements.",
    )
    captured = []

    def reviewer(review_input):
        captured.append(review_input)
        return KnowledgeReviewOutput(
            decisions=[
                CandidateReviewDecision(
                    review_id=review_input.units[0].review_id,
                    decision="defer",
                    merge_action="none",
                    evidence_refs=[],
                    conflict_refs=[],
                    scope_assessment="unverifiable",
                    reusable=False,
                    confidence="low",
                    reason="This fixture verifies the Observation v2 packet.",
                )
            ]
        )

    result = review_candidate_batch(
        FilesystemCatalog(catalog_root),
        [
            (
                candidate,
                [
                    ConceptEvidence(
                        observation_ref=observation.id,
                        stance="supports",
                        rationale="Measured fixture evidence.",
                    )
                ],
            )
        ],
        {observation.id: observation},
        run_id="run-1",
        batch_id="run-1-epoch-1",
        reviewer=reviewer,
    )

    assert captured[0].units[0].observations == [observation]
    assert result.records[0].observation_refs == [observation.id]


def test_missing_group_decision_is_deferred(tmp_path):
    catalog_root = tmp_path / "catalog"
    existing = _seed(catalog_root)
    first = _lifecycle_candidate(
        existing,
        candidate_id="runtime:group-1",
        claim_key="method.reduction.group_one",
        title="Group one",
        body="# Claim\n\nFirst independent claim.",
    )
    second = _lifecycle_candidate(
        existing,
        candidate_id="runtime:group-2",
        claim_key="method.reduction.group_two",
        title="Group two",
        body="# Claim\n\nSecond independent claim.",
    )

    def reviewer(review_input):
        unit = review_input.units[0]
        source = unit.candidates[0].source_refs[0]
        output = KnowledgeReviewOutput(
            decisions=[
                CandidateReviewDecision(
                    review_id=unit.review_id,
                    decision="approve",
                    merge_action="create_new",
                    evidence_refs=[_source_key(source)],
                    conflict_refs=[],
                    scope_assessment="compatible",
                    reusable=True,
                    confidence="medium",
                    reason="Only the first group received a valid decision.",
                )
            ]
        )
        return ReviewerExecutionResult(
            output=output,
            retrievals=(
                _detail_read(
                    "first-group-source",
                    "get_source",
                    _source_key(source),
                ),
            ),
        )

    result = review_candidate_batch(
        FilesystemCatalog(catalog_root),
        [(first, []), (second, [])],
        {},
        run_id="run-1",
        batch_id="run-1-epoch-1",
        reviewer=reviewer,
    )

    assert [item.decision for item in result.records] == [
        "approve",
        "defer",
    ]
    assert [item[0].candidate_id for item in result.candidates] == [
        "runtime:group-1"
    ]
    assert "exactly one decision" in result.warnings[0]


def test_source_citation_requires_an_exact_successful_detail_read(tmp_path):
    catalog_root = tmp_path / "catalog"
    existing = _seed(catalog_root)
    candidate = _lifecycle_candidate(
        existing,
        candidate_id="runtime:unread-source",
        claim_key="method.reduction.unread_source",
        title="Unread source rule",
        body="# Claim\n\nA claim whose Source was only returned by search.",
    )

    def reviewer(review_input):
        unit = review_input.units[0]
        source_ref = _source_key(unit.candidates[0].source_refs[0])
        output = KnowledgeReviewOutput(
            decisions=[
                CandidateReviewDecision(
                    review_id=unit.review_id,
                    decision="approve",
                    merge_action="create_new",
                    evidence_refs=[source_ref],
                    conflict_refs=[],
                    scope_assessment="compatible",
                    reusable=True,
                    confidence="medium",
                    reason="Search returned the Source, but no detail was read.",
                )
            ]
        )
        query_only = RetrievalRecord(
            event_id="source-query:unread-source",
            operation="query_sources",
            origin="mcp",
            status="success",
            returned_sources=[source_ref],
            created_at=datetime.now(timezone.utc),
        )
        wrong_locator = _detail_read(
            "wrong-source-locator",
            "get_source",
            source_ref + "-different",
        )
        return ReviewerExecutionResult(
            output=output,
            retrievals=(query_only, wrong_locator),
        )

    result = review_candidate_batch(
        FilesystemCatalog(catalog_root),
        [(candidate, [])],
        {},
        run_id="run-1",
        batch_id="run-1-epoch-1",
        reviewer=reviewer,
    )

    assert result.candidates == []
    assert result.records[0].decision == "defer"
    assert result.records[0].detail_read_refs == []
    assert "successful exact detail reads" in result.warnings[0]
    assert _source_key(candidate.source_refs[0]) in result.warnings[0]


def test_target_concept_requires_get_knowledge_even_when_source_was_read(
    tmp_path,
):
    catalog_root = tmp_path / "catalog"
    existing = _seed(catalog_root)
    candidate = _lifecycle_candidate(
        existing,
        candidate_id="runtime:unread-target",
        claim_key=existing.claim_key,
        title="Candidate for an unread target",
        body="# Claim\n\nThe current Concept body was not inspected.",
    )

    def reviewer(review_input):
        unit = review_input.units[0]
        source_ref = _source_key(unit.candidates[0].source_refs[0])
        output = KnowledgeReviewOutput(
            decisions=[
                CandidateReviewDecision(
                    review_id=unit.review_id,
                    decision="approve",
                    merge_action="attach_evidence",
                    target_concept_id=existing.id,
                    evidence_refs=[source_ref],
                    conflict_refs=[],
                    scope_assessment="compatible",
                    reusable=True,
                    confidence="medium",
                    reason="The target was selected from its packet summary.",
                )
            ]
        )
        return ReviewerExecutionResult(
            output=output,
            retrievals=(
                _detail_read(
                    "unread-target-source",
                    "get_source",
                    source_ref,
                ),
            ),
        )

    result = review_candidate_batch(
        FilesystemCatalog(catalog_root),
        [(candidate, [])],
        {},
        run_id="run-1",
        batch_id="run-1-epoch-1",
        reviewer=reviewer,
    )

    assert result.candidates == []
    assert result.records[0].decision == "defer"
    assert result.records[0].detail_read_refs == [
        _source_key(candidate.source_refs[0])
    ]
    assert existing.id in result.warnings[0]


def test_many_candidates_are_batched_with_two_bounded_reviewers(tmp_path):
    catalog_root = tmp_path / "catalog"
    existing = _seed(catalog_root)
    candidates = [
        (
            _lifecycle_candidate(
                existing,
                candidate_id=f"runtime:batch-{index}",
                claim_key=f"method.reduction.batch_{index}",
                title=f"Batch candidate {index}",
                body=f"# Claim\n\nIndependent candidate {index}.",
            ),
            [],
        )
        for index in range(9)
    ]
    lock = threading.Lock()
    call_sizes: list[int] = []
    active = 0
    max_active = 0

    def reviewer(review_input):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            call_sizes.append(len(review_input.units))
        try:
            time.sleep(0.03)
            return KnowledgeReviewOutput(
                decisions=[
                    CandidateReviewDecision(
                        review_id=unit.review_id,
                        decision="defer",
                        merge_action="none",
                        evidence_refs=[],
                        conflict_refs=[],
                        scope_assessment="unverifiable",
                        reusable=False,
                        confidence="low",
                        reason="More evidence is required.",
                    )
                    for unit in review_input.units
                ]
            )
        finally:
            with lock:
                active -= 1

    result = review_candidate_batch(
        FilesystemCatalog(catalog_root),
        candidates,
        {},
        run_id="run-1",
        batch_id="run-1-epoch-1",
        reviewer=reviewer,
    )

    assert sorted(call_sizes) == [1, 4, 4]
    assert max_active == 2
    assert len(result.records) == 9
    assert {item.review_batch_count for item in result.records} == {3}
    assert {item.review_batch_num for item in result.records} == {1, 2, 3}


def test_character_budget_splits_large_candidate_units(tmp_path):
    catalog_root = tmp_path / "catalog"
    existing = _seed(catalog_root)
    candidates = [
        (
            _lifecycle_candidate(
                existing,
                candidate_id=f"runtime:large-{index}",
                claim_key=f"method.reduction.large_{index}",
                title=f"Large candidate {index}",
                body="# Claim\n\n" + (str(index) * 40_000),
            ),
            [],
        )
        for index in range(2)
    ]
    call_sizes: list[int] = []

    def reviewer(review_input):
        call_sizes.append(len(review_input.units))
        return KnowledgeReviewOutput(
            decisions=[
                CandidateReviewDecision(
                    review_id=unit.review_id,
                    decision="defer",
                    merge_action="none",
                    evidence_refs=[],
                    conflict_refs=[],
                    scope_assessment="unverifiable",
                    reusable=False,
                    confidence="low",
                    reason="More evidence is required.",
                )
                for unit in review_input.units
            ]
        )

    result = review_candidate_batch(
        FilesystemCatalog(catalog_root),
        candidates,
        {},
        run_id="run-1",
        batch_id="run-1-epoch-1",
        reviewer=reviewer,
    )

    assert sorted(call_sizes) == [1, 1]
    assert len(result.records) == 2


def test_one_failed_reviewer_batch_does_not_defer_other_batches(tmp_path):
    catalog_root = tmp_path / "catalog"
    existing = _seed(catalog_root)
    candidates = [
        (
            _lifecycle_candidate(
                existing,
                candidate_id=f"runtime:isolation-{index}",
                claim_key=f"method.reduction.isolation_{index}",
                title=f"Isolation candidate {index}",
                body=f"# Claim\n\nIndependent candidate {index}.",
            ),
            [],
        )
        for index in range(5)
    ]

    def reviewer(review_input):
        if len(review_input.units) == 1:
            raise RuntimeError("one isolated Reviewer call failed")
        source_ref = _source_key(
            review_input.units[0].candidates[0].source_refs[0]
        )
        output = KnowledgeReviewOutput(
            decisions=[
                CandidateReviewDecision(
                    review_id=unit.review_id,
                    decision="approve",
                    merge_action="create_new",
                    evidence_refs=[source_ref],
                    conflict_refs=[],
                    scope_assessment="compatible",
                    reusable=True,
                    confidence="medium",
                    reason="The exact Source supports this scoped claim.",
                )
                for unit in review_input.units
            ]
        )
        return ReviewerExecutionResult(
            output=output,
            retrievals=(
                _detail_read(
                    "isolation-source",
                    "get_source",
                    source_ref,
                ),
            ),
        )

    result = review_candidate_batch(
        FilesystemCatalog(catalog_root),
        candidates,
        {},
        run_id="run-1",
        batch_id="run-1-epoch-1",
        reviewer=reviewer,
    )

    assert len(result.candidates) == 4
    assert sum(item.decision == "approve" for item in result.records) == 4
    assert sum(item.reviewer_status == "fallback" for item in result.records) == 1
    assert any("isolated Reviewer call failed" in item for item in result.warnings)


def test_epoch_reviewer_reuses_one_workspace_without_mixing_audits(tmp_path):
    catalog_root = tmp_path / "catalog"
    existing = _seed(catalog_root)
    candidate = _lifecycle_candidate(
        existing,
        candidate_id="runtime:epoch-wrapper",
        claim_key="method.reduction.epoch_wrapper",
        title="Epoch wrapper candidate",
        body="# Claim\n\nA Source-supported candidate.",
    )
    source_ref = _source_key(candidate.source_refs[0])

    class NoopMaterializer:
        def materialize(self, _workspace):
            return None

    class KnowledgeStub:
        def materializer(self):
            return NoopMaterializer()

    class AuditedRuntime:
        supports_native_agents = False
        lock = threading.Lock()
        active = 0
        max_active = 0
        invocation = 0
        workspaces = []

        def __init__(self, workspace):
            self.workspace = workspace
            self.__class__.workspaces.append(Path(workspace))

        def invoke(self, _prompt, *, model="inherit", agent=None):
            with self.__class__.lock:
                self.__class__.active += 1
                self.__class__.max_active = max(
                    self.__class__.max_active,
                    self.__class__.active,
                )
                self.__class__.invocation += 1
                invocation = self.__class__.invocation
            time.sleep(0.02)
            layout = KnowledgeLayout(Path(self.workspace))
            FileRetrievalAudit(
                layout.retrieval_log,
                legacy_query_path=layout.query_log,
            ).append(
                _detail_read(
                    f"epoch-wrapper-source-{invocation}",
                    "get_source",
                    source_ref,
                )
            )
            with self.__class__.lock:
                self.__class__.active -= 1
            return json.dumps(
                {
                    "schema_version": "1.0",
                    "decisions": [
                        {
                            "review_id": "review:run-1-epoch-1:unit-0001",
                            "decision": "approve",
                            "merge_action": "create_new",
                            "target_concept_id": None,
                            "evidence_refs": [source_ref],
                            "conflict_refs": [],
                            "scope_assessment": "compatible",
                            "reusable": True,
                            "confidence": "medium",
                            "reason": "The exact Source was read and supports it.",
                        }
                    ],
                }
            )

    reviewer = _build_epoch_reviewer(
        cwd=tmp_path,
        epoch_num=1,
        runtime_factory=AuditedRuntime,
        knowledge=KnowledgeStub(),
    )
    review_input = KnowledgeReviewInput(
        batch_id="run-1-epoch-1",
        run_id="run-1",
        units=[
            CandidateReviewUnit(
                review_id="review:run-1-epoch-1:unit-0001",
                candidate_ids=[candidate.candidate_id],
                candidates=[candidate],
                automatic_checks=["candidate_schema_valid"],
            )
        ],
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reviewer, [review_input, review_input]))

    assert all(
        result.output.decisions[0].decision == "approve" for result in results
    )
    assert sorted(
        item.event_id for result in results for item in result.retrievals
    ) == [
        "retrieval:epoch-wrapper-source-1",
        "retrieval:epoch-wrapper-source-2",
    ]
    assert all(len(result.retrievals) == 1 for result in results)
    assert AuditedRuntime.max_active == 1
    reviewer_path = tmp_path / "1R" / "knowledge-reviewer"
    assert reviewer_path.is_dir()
    assert set(AuditedRuntime.workspaces) == {reviewer_path}
    assert list((tmp_path / "1R").glob("knowledge-reviewer*")) == [
        reviewer_path
    ]


def test_reviewer_modes_control_model_calls_and_catalog_merges(
    tmp_path,
    monkeypatch,
):
    for mode in KnowledgeReviewerMode:
        catalog_root = tmp_path / mode.value / "catalog"
        existing = _seed(catalog_root)
        candidate = _lifecycle_candidate(
            existing,
            candidate_id=f"runtime:mode-{mode.value}",
            claim_key=f"method.reduction.mode_{mode.value}",
            title=f"Reviewer mode {mode.value}",
            body=f"# Claim\n\nCandidate for {mode.value} mode.",
        )
        publisher = BatchPublisher(catalog_root, fact_reader=None)
        monkeypatch.setattr(
            publisher.materializer,
            "collect",
            lambda _workspaces, _run_id, item=candidate: [(None, item)],
        )
        monkeypatch.setattr(
            publisher.materializer,
            "collect_usage_observations",
            lambda _workspaces, *, run_id: {},
        )
        monkeypatch.setattr(
            publisher.materializer,
            "materialize_observations",
            lambda _fresh, *, run_id, published_observations, item=candidate: (
                {},
                [(item, [])],
                [],
                [],
            ),
        )
        reviewer_calls = []
        source_ref = _source_key(candidate.source_refs[0])

        def reviewer(review_input):
            reviewer_calls.append(review_input)
            unit = review_input.units[0]
            output = KnowledgeReviewOutput(
                decisions=[
                    CandidateReviewDecision(
                        review_id=unit.review_id,
                        decision="approve",
                        merge_action="create_new",
                        evidence_refs=[source_ref],
                        conflict_refs=[],
                        scope_assessment="compatible",
                        reusable=True,
                        confidence="high",
                        reason="The exact Source supports this scoped claim.",
                    )
                ]
            )
            return ReviewerExecutionResult(
                output=output,
                retrievals=(
                    _detail_read(
                        f"mode-{mode.value}-source",
                        "get_source",
                        source_ref,
                    ),
                ),
            )

        snapshot_before = FilesystemCatalog(catalog_root).snapshot()
        result = publisher.publish(
            [tmp_path / mode.value / "workspace"],
            run_id=f"run-{mode.value}",
            batch_id=f"run-{mode.value}-epoch-1",
            reviewer=reviewer,
            reviewer_mode=mode,
        )
        snapshot_after = FilesystemCatalog(catalog_root).snapshot()
        review_record = json.loads(
            (catalog_root / result.review_audit_ref).read_text(
                encoding="utf-8"
            )
        )
        publish_audit = json.loads(
            (catalog_root / result.audit_ref).read_text(encoding="utf-8")
        )

        assert result.reviewer_mode == mode.value
        assert publish_audit["reviewer_mode"] == mode.value
        assert publish_audit["result"]["reviewer_mode"] == mode.value
        assert review_record["reviewer_mode"] == mode.value
        if mode == KnowledgeReviewerMode.OFF:
            assert reviewer_calls == []
            assert result.status == "noop"
            assert result.approved_candidates == []
            assert result.shadow_approved_candidates == []
            assert result.deferred_candidates == [candidate.candidate_id]
            assert review_record["decision"] == "defer"
            assert review_record["reviewer_status"] == "disabled"
            assert snapshot_after == snapshot_before
        elif mode == KnowledgeReviewerMode.SHADOW:
            assert len(reviewer_calls) == 1
            assert result.status == "noop"
            assert result.approved_candidates == []
            assert result.shadow_approved_candidates == [
                candidate.candidate_id
            ]
            assert result.deferred_candidates == [candidate.candidate_id]
            assert review_record["decision"] == "approve"
            assert review_record["reviewer_status"] == "completed"
            assert snapshot_after == snapshot_before
        else:
            assert len(reviewer_calls) == 1
            assert result.status == "published"
            assert result.approved_candidates == [candidate.candidate_id]
            assert result.shadow_approved_candidates == []
            assert result.deferred_candidates == []
            assert result.processed_candidates == [candidate.candidate_id]
            assert candidate.proposed_id in result.created
            assert snapshot_after != snapshot_before


def test_legacy_publish_result_infers_whether_review_was_enforced():
    reviewed = PublishResult.model_validate(
        {
            "status": "noop",
            "deferred_candidates": ["runtime:legacy-candidate"],
        }
    )
    unreviewed = PublishResult.model_validate({"status": "noop"})

    assert reviewed.reviewer_mode == "enforce"
    assert unreviewed.reviewer_mode == "off"
