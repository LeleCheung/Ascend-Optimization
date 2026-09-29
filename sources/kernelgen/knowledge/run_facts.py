"""KernelGen ledger adapter for authority-owned Observation fields."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from kernelgen.data.ledger import Ledger
from kernelgen.knowledge.records import FileRetrievalAudit
from kernelgen.knowledge.run_archive import KernelGenRunArchive
from kernelgen.knowledge.models import (
    ObservationRecord,
    OperatorSignature,
    TargetContext,
    KnowledgeApplicationAssessment,
    KnowledgeApplication,
    KnowledgeUsageScope,
    ObservationComparison,
    ObservationLocator,
    ObservationOutcome,
    ObservationParent,
)
from kernelgen.knowledge.contracts.runtime import WorkspaceKnowledgeState
from kernelgen.knowledge.context import normalize_device
from kernelgen.knowledge.layout import KnowledgeLayout, safe_name


class KernelGenRunFactReader:
    def __init__(self, run_archive: KernelGenRunArchive | None = None):
        self.run_archive = run_archive

    def materialize_usage_observations(
        self,
        workspace: Path,
        state: WorkspaceKnowledgeState,
        signature: OperatorSignature,
        target: TargetContext,
        *,
        run_id: str,
    ) -> list[ObservationRecord]:
        """Materialize every measured round that declares knowledge use."""

        ledger = Ledger(workspace)
        return [
            self.materialize_observation(
                workspace,
                state,
                signature,
                target,
                round_num=record.round_num,
                run_id=run_id,
            )
            for record in ledger.history.rounds
            if record.plan.knowledge_uses
        ]

    def materialize_observation(
        self,
        workspace: Path,
        state: WorkspaceKnowledgeState,
        signature: OperatorSignature,
        target: TargetContext,
        *,
        round_num: int,
        run_id: str,
    ) -> ObservationRecord:
        ledger = Ledger(workspace)
        if (
            ledger.history.definition_name
            and ledger.history.definition_name != signature.definition_name
        ):
            raise ValueError("ledger definition does not match workspace context")
        if (
            ledger.history.target_hardware
            and not _target_devices_match(
                ledger.history.target_hardware,
                target.device,
            )
        ):
            raise ValueError("ledger target does not match workspace context")
        record = ledger.get_round(round_num)
        solution_sha = record.solution.sha256
        if not _is_sha256(solution_sha):
            solution_sha = hashlib.sha256(
                record.solution.code.encode("utf-8")
            ).hexdigest()
        performance_baseline_round = (
            record.evaluation.comparison.performance_baseline_round_num
        )
        experiment_parent = None
        if record.experiment_parent_round_num is not None:
            parent = ledger.get_round(record.experiment_parent_round_num)
            experiment_parent = ObservationParent(
                round_num=parent.round_num,
                status=parent.evaluation.status,
            )
        applications = [
            KnowledgeApplication.model_validate(item.model_dump(mode="python"))
            for item in record.plan.knowledge_uses
        ]
        assessments = {
            _reference_key(item): KnowledgeApplicationAssessment(
                assessment=item.assessment,
                rationale=item.rationale,
            )
            for item in (
                record.conclusion.knowledge_assessments
                if record.conclusion is not None
                else []
            )
        }
        applications = [
            item.model_copy(
                update={"agent_assessment": assessments.get(_reference_key(item))}
            )
            for item in applications
        ]
        retrieval_audit = FileRetrievalAudit(
            KnowledgeLayout(workspace).retrieval_log,
            legacy_query_path=KnowledgeLayout(workspace).query_log,
        )
        retrievals = retrieval_audit.read_all()
        applications = [
            item.model_copy(
                update={
                    "lineage_status": _lineage_status(
                        item,
                        retrieval_audit,
                        retrievals,
                    )
                }
            )
            for item in applications
        ]
        applied_count = sum(
            item.disposition in {"adopted", "adapted"}
            for item in applications
        )
        usage_mode = (
            "none"
            if applied_count == 0
            else "single"
            if applied_count == 1
            else "combined"
        )

        artifact_refs = []
        archive_ref = ""
        if self.run_archive is not None:
            archive_ref = self.run_archive.archive_workspace(
                workspace,
                state,
                signature,
                target,
                run_id=run_id,
            ).ref
        context_ref = (
            archive_ref
            or (
                f"artifact://run/{safe_name(run_id)}/"
                f"{state.workspace_id}"
            )
        )
        if record.solution.snapshot_path:
            artifact_refs.append(
                (
                    f"{archive_ref}/artifacts/{record.solution.snapshot_path}"
                    if archive_ref
                    else (
                        f"artifact://run/{safe_name(run_id)}/"
                        f"{state.workspace_id}/{record.solution.snapshot_path}"
                    )
                )
            )
        if record.profile.analysis_path:
            analysis = Path(workspace) / record.profile.analysis_path
            if analysis.is_file():
                artifact_refs.append(
                    (
                        f"{archive_ref}/artifacts/{record.profile.analysis_path}"
                        if archive_ref
                        else (
                            f"artifact://run/{safe_name(run_id)}/"
                            f"{state.workspace_id}/{record.profile.analysis_path}"
                        )
                    )
                )
        return ObservationRecord(
            id=(
                f"kg:observation:{safe_name(run_id)}:{safe_name(state.workspace_id)}:"
                f"round-{round_num}"
            ),
            run_id=run_id,
            workspace_id=state.workspace_id,
            definition_id=signature.definition_id,
            round_num=round_num,
            target_context_ref=f"{context_ref}/context/target-context.json",
            operator_signature_ref=(
                f"{context_ref}/context/operator-signature.json"
            ),
            ledger_locator=ObservationLocator(
                path=(
                    f"{archive_ref}/ledger.json"
                    if archive_ref
                    else (
                        f"artifact://run/{safe_name(run_id)}/"
                        f"{state.workspace_id}/.ledger.json"
                    )
                ),
                round_num=round_num,
            ),
            solution_sha256=solution_sha,
            evaluated_at=record.evaluation.evaluated_at,
            experiment_plan=record.plan.model_dump(mode="json"),
            code_changes=record.plan.code_changes,
            agent_conclusion=(
                record.conclusion.model_dump(mode="json")
                if record.conclusion is not None
                else {}
            ),
            experiment_parent=experiment_parent,
            evaluation_scope=KnowledgeUsageScope.from_context(
                signature,
                target,
            ),
            outcome=ObservationOutcome(
                status=record.evaluation.status,
                geo_mean_speedup=record.evaluation.geo_mean,
                workload_results_ref=(
                    (
                        f"{archive_ref}/ledger.json"
                        if archive_ref
                        else (
                            f"artifact://run/{safe_name(run_id)}/"
                            f"{state.workspace_id}/.ledger.json"
                        )
                    )
                    + f"#/rounds/{round_num - 1}/evaluation/workloads"
                ),
            ),
            comparison=ObservationComparison(
                performance_baseline_round_num=(
                    performance_baseline_round
                ),
                geo_mean_delta_pct=record.evaluation.comparison.geo_mean_delta_pct,
            ),
            applications=applications,
            usage_mode=usage_mode,
            artifact_refs=artifact_refs,
            recorded_at=datetime.now(timezone.utc),
        )


def _reference_key(item) -> tuple[str, ...]:
    if item.concept_ref:
        return ("concept", item.concept_ref)
    source = item.source_ref
    return ("source", source.resource, source.revision, source.locator)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _target_devices_match(requested: str, reported: str) -> bool:
    """Accept a stable Ascend family name and its service-reported SKU."""
    return (
        normalize_device(requested).casefold()
        == normalize_device(reported).casefold()
    )


def _lineage_status(
    application: KnowledgeApplication,
    retrieval_audit: FileRetrievalAudit,
    retrievals,
) -> str:
    query_operation = (
        "query_sources"
        if application.source_ref is not None
        else "query_knowledge"
    )
    detail_operation = (
        "get_source"
        if application.source_ref is not None
        else "get_knowledge"
    )
    query_id = application.query_event_id
    if not query_id:
        return "query_missing"
    try:
        retrieval_audit.read_query(query_id, operation=query_operation)
    except KeyError:
        return "query_missing"

    expected_ref = application.concept_ref
    if application.source_ref is not None:
        source = application.source_ref
        expected_ref = (
            f"{source.resource}@{source.revision}::{source.locator}"
        )
    for item in retrievals:
        if (
            item.query_id != query_id
            or item.operation != detail_operation
            or item.status != "success"
        ):
            continue
        returned = (
            item.returned_sources
            if application.source_ref is not None
            else item.returned_refs
        )
        if expected_ref in returned:
            return "complete"
    return "detail_not_read"
