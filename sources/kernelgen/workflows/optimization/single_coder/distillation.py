"""Legacy and V1 Knowledge distillation for SingleCoderOptimizationWorkflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from kernelgen.agents.distiller import DistillerAgent

if TYPE_CHECKING:
    from kernelgen.workflows.optimization.single_coder.workflow import (
        SingleCoderOptimizationInput,
    )


def run_distillation(
    workspace: Path,
    inp: "SingleCoderOptimizationInput",
    runtime,
    *,
    verified_geo_mean: float | None = None,
) -> Optional[Any]:
    """Read the workspace ledger and run the selected Distiller."""
    if inp.knowledge_enabled:
        return _run_knowledge_distillation(workspace, inp, runtime, verified_geo_mean=verified_geo_mean)

    ledger, rounds = _load_rounds(workspace)
    if not rounds:
        return None

    distill_input = {
        "definition_name": inp.definition.name,
        "op_type": inp.definition.op_type,
        "target_hardware": inp.target_hardware,
        "implementation_language": inp.implementation_language,
        "definition": inp.definition.model_dump(),
        "rounds": rounds,
        "best_geo_mean": verified_geo_mean if verified_geo_mean is not None else ledger.history.best_geo_mean if ledger is not None else 0.0,
        "best_round": ledger.history.best_round if ledger is not None else 0,
    }
    return DistillerAgent().run(distill_input, runtime)


def save_distillation_artifacts(
    workspace: Path,
    inp: "SingleCoderOptimizationInput",
    distill_out,
    ledger,
) -> None:
    """Persist the selected Distiller's derived reports and proposals."""
    if inp.knowledge_enabled:
        _save_knowledge_distillation_artifacts(workspace, distill_out, ledger)
    else:
        _save_legacy_distillation_artifacts(workspace, distill_out)


def read_profile_analysis_files(
    workspace: Path,
    ledger,
) -> list[dict[str, Any]]:
    """List validated analyses without exposing raw profiler artifacts."""
    root = workspace.resolve()
    files = []
    for record in ledger.history.rounds:
        relative_path = record.profile.analysis_path
        if not relative_path:
            continue
        try:
            path = (root / relative_path).resolve()
            path.relative_to(root)
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        files.append(
            {
                "round_num": record.round_num,
                "path": relative_path,
            }
        )
    return files


def _load_rounds(workspace: Path):
    try:
        from kernelgen.data.ledger import Ledger

        ledger = Ledger(workspace)
        rounds = [
            record.model_dump(mode="json")
            for record in ledger.history.rounds
        ]
    except Exception:
        return None, []
    return ledger, rounds


def _run_knowledge_distillation(
    workspace: Path,
    inp: "SingleCoderOptimizationInput",
    runtime,
    *,
    verified_geo_mean: float | None = None,
) -> Optional[Any]:
    """Build the provenance-bounded input for the V1 Knowledge Distiller."""
    ledger, rounds = _load_rounds(workspace)
    if not rounds:
        return None

    from kernelgen.agents.knowledge_distiller import KnowledgeDistillerAgent
    from kernelgen.knowledge.bootstrap import resolve_workspace_catalog
    from kernelgen.knowledge.context import workspace_provenance
    from kernelgen.knowledge.contracts import WorkspaceKnowledgeState
    from kernelgen.knowledge.layout import KnowledgeLayout
    from kernelgen.knowledge.vocabulary import Vocabulary

    available_sources, known_concepts = workspace_provenance(workspace)
    layout = KnowledgeLayout(workspace)
    vocabulary = Vocabulary()
    if layout.state.is_file():
        state = WorkspaceKnowledgeState.model_validate_json(
            layout.state.read_text(encoding="utf-8")
        )
        vocabulary = Vocabulary.from_catalog(
            resolve_workspace_catalog(workspace, state)
        )
    distill_input = {
        "definition_name": inp.definition.name,
        "op_type": inp.definition.op_type,
        "target_hardware": inp.target_hardware,
        "implementation_language": inp.implementation_language,
        "definition": inp.definition.model_dump(),
        "rounds": rounds,
        "best_geo_mean": verified_geo_mean if verified_geo_mean is not None else ledger.history.best_geo_mean if ledger is not None else 0.0,
        "best_round": ledger.history.best_round if ledger is not None else 0,
        "profile_analysis_files": read_profile_analysis_files(workspace, ledger),
        "available_source_refs": [
            item.model_dump(mode="json")
            for item in available_sources
        ],
        "known_concept_ids": known_concepts,
        "canonical_symptoms": vocabulary.canonical_terms("symptoms"),
        "canonical_techniques": vocabulary.canonical_terms("techniques"),
    }
    return KnowledgeDistillerAgent().run(distill_input, runtime)


def _save_legacy_distillation_artifacts(
    workspace: Path,
    distill_out,
) -> None:
    """Save per-agent legacy reports for Synthesizer to read."""
    if distill_out.candidate_experience:
        (workspace / ".new_experience.md").write_text(
            distill_out.candidate_experience,
            encoding="utf-8",
        )
    if distill_out.candidate_detailed:
        (workspace / ".new_detailed.md").write_text(
            distill_out.candidate_detailed,
            encoding="utf-8",
        )


def _save_knowledge_distillation_artifacts(
    workspace: Path,
    distill_out,
    ledger,
) -> None:
    """Persist V1 run reports and structured proposals for epoch publish."""
    from kernelgen.agents.knowledge_distiller.report_format import (
        render_detailed_report,
        render_experience_report,
    )
    from kernelgen.data._atomic import atomic_write_text

    if distill_out.candidate_experience:
        atomic_write_text(
            workspace / ".new_experience.md",
            render_experience_report(
                distill_out.candidate_experience,
                ledger.history,
            ),
        )
    if distill_out.candidate_detailed:
        atomic_write_text(
            workspace / ".new_detailed.md",
            render_detailed_report(
                distill_out.candidate_detailed,
                ledger.history,
            ),
        )
    if distill_out.candidate_concepts:
        from kernelgen.workflows.knowledge_bridge import append_candidate_drafts

        append_candidate_drafts(
            workspace,
            distill_out.candidate_concepts,
        )
