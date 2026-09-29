"""Prepare a fixed run contract, initial seed and shared analysis checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kernelgen.agents.analyzer import AnalyzerAgent, AnalyzerOutput
from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot, freeze_evaluation_snapshot, freeze_catalog_evaluation_snapshot, snapshot_optimization_context
from kernelgen.data.trace import infer_destination_passing_style
from kernelgen.framework.run_control import RunCancelled, RunControl, RunState, WorkspaceRunControl
from kernelgen.knowledge.config import KnowledgeConfig
from kernelgen.tools import kernelgen_server_adapter as server_adapter
from kernelgen.workflows.optimization.kernelgen import knowledge as knowledge_helpers
from kernelgen.workflows.optimization.kernelgen.contracts import EpochResult, KernelGenInput
from kernelgen.workflows.optimization.kernelgen.epoch import create_epoch_workspace, resolve_optimization_context
from kernelgen.workflows.optimization.kernelgen.recovery import load_resume_state
from kernelgen.workflows.knowledge_bridge import KernelGenKnowledgeBridge


@dataclass(frozen=True)
class PreparedRun:
    inp: KernelGenInput
    knowledge: KernelGenKnowledgeBridge | None
    runtime_factory: Callable
    analysis: AnalyzerOutput
    evaluation_snapshot: dict[str, Any] | None
    initial_seed_code: str
    best_result: EpochResult
    next_directions: list


def start_mode(inp: KernelGenInput) -> str:
    return inp.start_mode or ("resume" if inp.start_epoch > 1 else "fresh")


def resolve_server_target(inp: KernelGenInput) -> tuple[KernelGenInput, dict[str, Any]]:
    """Validate identity once; retain the Server facts for shared analysis."""
    status = server_adapter.get_service_status(inp.eval_server_url)
    target = server_adapter.require_target_context(inp, status)
    context = {key: status[key] for key in (
        "api_version", "backend", "timing", "target", "software", "metadata",
    ) if key in status}
    return inp.model_copy(update={"target_hardware": target.device}), context


def freeze_run_evaluation_contract(cwd: Path, inp: KernelGenInput):
    if inp.evaluation_snapshot is not None:
        requested = CatalogEvaluationSnapshot.model_validate(inp.evaluation_snapshot)
        if (requested.catalog_name, requested.definition_name) != (inp.catalog_name, inp.definition.name):
            raise ValueError("KernelGen input and evaluation snapshot identities differ")
        snapshot, _ = freeze_evaluation_snapshot(cwd, requested)
        definition, _ = snapshot_optimization_context(snapshot)
        return inp.model_copy(update={"definition": definition}), snapshot.model_dump(mode="json")
    if inp.catalog_name != "kernelswift":
        return inp, None
    snapshot, _ = freeze_catalog_evaluation_snapshot(cwd, inp.catalog_name, inp.definition.name)
    definition, _ = snapshot_optimization_context(snapshot)
    return inp.model_copy(update={"definition": definition}), snapshot.model_dump(mode="json")


def configured_runtime_factory(inp: KernelGenInput, factory: Callable) -> Callable:
    def create(path: str):
        runtime = factory(path)
        if hasattr(runtime, "timeout"):
            runtime.timeout = float(inp.timeout)
        return runtime
    return create


def prepare_run(
    *,
    cwd: Path,
    inp: KernelGenInput,
    knowledge_config: KnowledgeConfig | None,
    runtime_factory: Callable,
    run_control: RunControl,
) -> PreparedRun:
    if inp.start_epoch < 1 or inp.start_epoch > inp.n_epoch:
        raise ValueError("start_epoch must be between 1 and n_epoch")
    mode = start_mode(inp)
    if mode != "resume" and inp.start_epoch > 1:
        raise ValueError("start_epoch > 1 requires start_mode=resume")
    inp, evaluation_snapshot = freeze_run_evaluation_contract(cwd, inp)
    inp, server_context = resolve_server_target(inp)
    knowledge = knowledge_helpers.build_knowledge_bridge(
        config=knowledge_config, cwd=cwd, inp=inp, start_mode=mode,
    )
    runtime_factory = configured_runtime_factory(inp, runtime_factory)
    if inp.initial_seed_code and mode != "fresh":
        raise ValueError("initial_seed_code is only valid with start_mode=fresh")
    initial_seed = inp.initial_seed_code
    if initial_seed:
        print("[Seed] Using an initial candidate for epoch 1; target evaluation is still required")
    if mode == "fork":
        if knowledge is None:
            raise ValueError("fork requires a Knowledge-enabled KernelGen run")
        seed = knowledge.resolve_fork_seed()
        initial_seed = seed.code
        print(f"[Fork] Seeding from {seed.manifest.solution_ref} ({seed.manifest.geo_mean:.4f}x)")

    if inp.start_epoch == 1:
        analysis = prepare_analysis(
            cwd, inp, knowledge, runtime_factory, run_control, resume=mode == "resume",
            evaluation_snapshot=evaluation_snapshot, server_context=server_context,
        )
        best_result = EpochResult(inp.definition.name)
        next_directions = []
    else:
        analysis, synthesis, best_result = load_resume_state(cwd=cwd, inp=inp)
        next_directions = list(synthesis.next_directions)
        print(f"\n[Resume] Starting {inp.start_epoch}R from {inp.start_epoch - 1}R/synthesis/synthesis.json")
    return PreparedRun(
        inp=inp, knowledge=knowledge, runtime_factory=runtime_factory,
        analysis=analysis, evaluation_snapshot=evaluation_snapshot,
        initial_seed_code=initial_seed, best_result=best_result,
        next_directions=next_directions,
    )


def prepare_analysis(
    cwd, inp, knowledge, runtime_factory, run_control, *, resume: bool,
    evaluation_snapshot=None, server_context=None,
):
    workspace = create_epoch_workspace(cwd, "1R", knowledge,
                                       inherit_legacy_knowledge=inp.cross_epoch_knowledge)
    path = Path(workspace.allocate("shared_analysis"))
    checkpoint = path / "analysis.json"
    control = WorkspaceRunControl(path, source="workflow:kernel_gen:shared_analysis")
    if resume and checkpoint.is_file():
        try:
            analysis = AnalyzerOutput.model_validate_json(checkpoint.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"\n[Shared Analysis] Existing checkpoint is invalid; rerunning: {exc}")
        else:
            print(f"\n[Shared Analysis] Reusing checkpoint: {checkpoint}")
            control.update_progress(state=RunState.SUCCEEDED, stage="COMPLETED", message="Shared analysis checkpoint reused")
            return analysis

    run_control.update_progress(
        state=RunState.RUNNING, stage="SHARED_ANALYSIS", current_epoch=1,
        message=f"Running shared analysis for {inp.definition.name}",
    )
    run_control.record_event("ANALYSIS_STARTED", stage="SHARED_ANALYSIS")
    print(f"\n[Shared Analysis] Running centralized analysis for {inp.definition.name}…")
    try:
        definition, workloads, _ = resolve_optimization_context(inp, evaluation_snapshot)
        analysis = AnalyzerAgent().run({
            "definition": definition.model_dump(),
            "workloads": workloads,
            "server_context": server_context or {},
            "target_hardware": inp.target_hardware,
            "implementation_language": inp.implementation_language,
            "destination_passing_style": infer_destination_passing_style(definition),
            "evaluation_contract": inp.evaluation_contract.model_dump(),
        }, runtime_factory(str(path)))
        checkpoint.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
    except RunCancelled:
        control.acknowledge_cancellation(stage="CANCELLED")
        raise
    except Exception as exc:
        control.update_progress(state=RunState.FAILED, stage="FAILED", message=f"{type(exc).__name__}: {exc}")
        raise
    control.update_progress(state=RunState.SUCCEEDED, stage="COMPLETED", message="Shared analysis completed")
    run_control.record_event("ANALYSIS_COMPLETED", stage="SHARED_ANALYSIS")
    print(f"[Shared Analysis] Done. Workspace: {path}")
    print(f"[Shared Analysis] Log saved to: {path / '.kernelgen' / 'claude-runtime.log'}")
    return analysis
