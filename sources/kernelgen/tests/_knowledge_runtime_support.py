"""Shared setup for knowledge runtime behavior tests."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
import yaml

import kernelgen.workflows.knowledge_bridge as knowledge_bridge_module
from kernelgen.knowledge.records import (
    FileQueryAudit,
    FileRetrievalAudit,
)
from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.run_facts import (
    KernelGenRunFactReader,
    _target_devices_match,
)
from kernelgen.knowledge.round_index import SQLiteRoundIndex
from kernelgen.knowledge.index import SQLiteKnowledgeIndex
from kernelgen.knowledge.run_archive import KernelGenRunArchive
from kernelgen.knowledge.publishing.merge import (
    _evidence_state,
    merge_concepts,
)
from kernelgen.knowledge.publishing.lifecycle import (
    delete_concept,
    deprecate_concept,
    revalidation_candidates,
    rollback_publish_batch,
)
from kernelgen.knowledge.publishing.publisher import BatchPublisher
from kernelgen.knowledge.metrics import build_knowledge_metrics
from kernelgen.knowledge.sources import SourceSearcher
from kernelgen.knowledge.bootstrap import build_workspace_services
from kernelgen.knowledge.config import KnowledgeConfig, KnowledgeMode
from kernelgen.knowledge.context import (
    KnowledgeWorkspaceMaterializer,
    build_operator_signature,
    build_query_context,
    build_target_context,
    workspace_provenance,
)
from kernelgen.knowledge.publishing.static_import import StaticKnowledgeImporter
from kernelgen.knowledge.models import (
    CandidateConcept,
    CandidateDraft,
    QueryContext,
    Concept,
    ConceptEvidence,
    ConceptRelation,
    KnowledgeUsageScope,
    ManagedMetadata,
    ProfileFindingContext,
    Scope,
    SourceReference,
    TargetScope,
)
from kernelgen.knowledge.contracts.runtime import (
    PublishResult,
    QueryRecord,
    WorkspaceKnowledgeState,
)
from kernelgen.knowledge.scope import match_scope
from kernelgen.knowledge.layout import CatalogLayout, KnowledgeLayout
from kernelgen.knowledge.validation import (
    canonical_concept_hash,
    source_content_checksum,
    validate_knowledge_base,
)
from kernelgen.data.ledger import Ledger
from kernelgen.data.experiment_plan import ExperimentPlan
from kernelgen.tests.helpers import round_conclusion
from kernelgen.workflows.knowledge_bridge import (
    KernelGenKnowledgeBridge,
    append_candidate_drafts,
    recent_detail_read_refs,
    validate_knowledge_uses,
)
from kernelgen.tools.kb import publish_workspace_batch, rebuild_indexes
from kernelgen.scripts.migrations.migrate_single_concept_files import (
    migrate_single_concept_files,
)


_DEFINITION = {
    "name": "sum_rows",
    "op_type": "reduction",
    "axes": {"reduction_size": {"type": "const", "value": 4096}},
    "inputs": {"x": {"shape": [16, 4096], "dtype": "float16"}},
    "outputs": {"y": {"shape": [16], "dtype": "float32"}},
    "reference": "def run(x): return x.sum(-1)",
}


def _seed(catalog_root, *, level="portable", backend=""):
    catalog = FilesystemCatalog(catalog_root)
    concept = Concept(
        id="kg:method:two-stage-reduction",
        kind="method",
        title="Two-stage reduction",
        summary="Split long reductions into partial and final stages.",
        claim_key="method.reduction.two_stage",
        domains=["optimization"],
        sources=[
            SourceReference(
                resource="https://example.invalid/reduction",
                title="Reduction note",
            )
        ],
        scope=Scope(
            target=TargetScope(level=level, backend=backend),
            operator={"motifs": ["reduction"]},
        ),
        retrieval={
            "phases": ["initial"],
            "tasks": ["architecture_selection"],
            "techniques": ["two_stage_reduction"],
            "keywords": ["reduction"],
        },
        evidence_state="source_supported",
        managed=ManagedMetadata(
            content_hash="sha256:" + "0" * 64,
            created_by="test",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        body="# Claim\n\nUse two stages for long reductions.",
    )
    concept = concept.model_copy(
        update={
            "managed": concept.managed.model_copy(
                update={"content_hash": canonical_concept_hash(concept)}
            )
        }
    )
    catalog.write_concept(concept)
    return concept


def _reference_body(claim: str) -> str:
    return (
        f"## Claim\n\n{claim}\n\n"
        "## Evidence\n\nThe cited Source documents this rule.\n\n"
        "## Applicability\n\nUse only within the structured scope.\n\n"
        "## Action\n\nApply the documented rule.\n\n"
        "## Limits\n\nDo not generalize beyond the cited Source."
    )


def _lifecycle_candidate(
    concept,
    *,
    candidate_id,
    claim_key,
    title,
    body,
    publish_action="create",
    target_concept_id=None,
    relations=None,
):
    return CandidateConcept(
        candidate_id=candidate_id,
        publish_action=publish_action,
        target_concept_id=target_concept_id,
        proposed_kind=concept.kind,
        proposed_id=f"kg:{concept.kind}:{claim_key.replace('.', '-')}",
        claim_key=claim_key,
        title=title,
        summary=title,
        domains=concept.domains,
        scope=concept.scope,
        retrieval=concept.retrieval,
        body=body,
        relations=relations or [],
        source_refs=concept.sources,
        created_by="source-ingest:test",
    )



def _service_status(*, backend="cuda", device="A100"):
    return {
        "backend": backend,
        "service_instance_id": "test",
        "target": {
            "backend": backend,
            "device": device,
        },
        "profile": {"capabilities": ["memory_counters"]},
    }


def _materialize(catalog_root, workspace, *, backend="cuda"):
    signature = build_operator_signature(_DEFINITION)
    target = build_target_context(
        target_hardware="A100",
        implementation_language="triton",
        service_status=_service_status(backend=backend),
    )
    KnowledgeWorkspaceMaterializer(
        catalog_root=catalog_root,
        mode=KnowledgeMode.READ_WRITE_V1,
        run_id="run-1",
        operator_signature=signature,
        target_context=target,
    ).materialize(workspace)


def _finalize_test_round(ledger: Ledger, round_num: int) -> None:
    record = ledger.get_round(round_num)
    baseline = record.plan.kind == "baseline"
    ledger.finalize_round(
        round_num,
        {
            "round_num": round_num,
            "expectation_status": "baseline" if baseline else "met",
            "root_cause": "host fixture completed the measured experiment",
            "perf_gap_analysis": (
                "" if baseline else "the authoritative measurement was recorded"
            ),
            "next_suggestion": "continue the lineage fixture",
            "architecture_tag": "host_fixture",
            "optimization_level": "L2_memory",
            "knowledge_assessments": [
                {
                    **(
                        {"concept_ref": use.concept_ref}
                        if use.concept_ref
                        else {"source_ref": use.source_ref.model_dump()}
                    ),
                    "assessment": "inconclusive",
                    "rationale": (
                        "host fixture does not isolate this knowledge item"
                    ),
                }
                for use in record.plan.knowledge_uses
            ],
        },
    )
