"""Run one isolated epoch and derive its selection/synthesis inputs from ledgers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

from kernelgen.agents.epoch_summary import EpochSummaryAgent
from kernelgen.data.catalog import resolve_builtin_catalog_path
from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot, snapshot_optimization_context
from kernelgen.data.ledger import Ledger
from kernelgen.data.selection import pick_best
from kernelgen.data.trace import infer_destination_passing_style, load_catalog_optimization_context
from kernelgen.data.trajectory import build_synthesis_trajectory
from kernelgen.framework.mcp_config import MCP_CONFIGURATION_PATH
from kernelgen.framework.parallel import IsolatedDirectory, ParallelExecutionError, Workspace, run_parallel
from kernelgen.framework.run_control import RunControl, RunState
from kernelgen.workflows.optimization.kernelgen.contracts import AgentSummary, EpochResult, FailedAgent, KernelGenInput
from kernelgen.workflows.knowledge_bridge import KernelGenKnowledgeBridge
from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationWorkflow
from kernelgen.workflows.optimization.single_coder.inputs import build_optimizer_input


def create_epoch_workspace(
    cwd: Path,
    epoch_label: str,
    knowledge: KernelGenKnowledgeBridge | None,
    *,
    inherit_legacy_knowledge: bool = True,
) -> IsolatedDirectory:
    """Allocate real agent directories with the configured shared resources."""
    return IsolatedDirectory(
        base=cwd / epoch_label,
        claude_source=cwd / ".claude",
        mcp_source=cwd / MCP_CONFIGURATION_PATH if (cwd / MCP_CONFIGURATION_PATH).is_file() else None,
        kb_source=cwd / "kb" if inherit_legacy_knowledge and knowledge is None and (cwd / "kb").exists() else None,
        materializers=[knowledge.materializer()] if knowledge is not None else [],
    )


def resolve_optimization_context(
    inp: KernelGenInput,
    evaluation_snapshot: dict[str, Any] | None = None,
):
    """Resolve the same frozen definition/workloads for Analyzer and Coders."""
    definition = inp.definition
    if evaluation_snapshot is not None:
        native_snapshot = CatalogEvaluationSnapshot.model_validate(evaluation_snapshot)
        expected = (inp.catalog_name, inp.definition.name)
        actual = (native_snapshot.catalog_name, native_snapshot.definition_name)
        if actual != expected:
            raise ValueError(
                "KernelGen input and evaluation snapshot identities differ: "
                f"input={expected!r}, snapshot={actual!r}"
            )
        definition, workloads = snapshot_optimization_context(native_snapshot)
        evaluation_snapshot = native_snapshot.model_dump(mode="json")
    else:
        _, workloads = load_catalog_optimization_context(
            resolve_builtin_catalog_path(inp.catalog_name), inp.definition.name,
        )
    return definition, workloads, evaluation_snapshot


def build_coder_input(
    inp: KernelGenInput,
    analysis,
    *,
    seed_code: str = "",
    knowledge_enabled: bool = False,
    evaluation_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    definition, workloads, evaluation_snapshot = resolve_optimization_context(inp, evaluation_snapshot)
    return build_optimizer_input(
        inp,
        definition=definition.model_dump(),
        destination_passing_style=infer_destination_passing_style(definition),
        analysis=analysis.model_dump() if hasattr(analysis, "model_dump") else dict(analysis or {}),
        workloads=workloads,
        evaluation_snapshot=evaluation_snapshot,
        knowledge_enabled=knowledge_enabled,
        seed_code=seed_code,
        seed_is_validated_baseline=False,
    )


def build_epoch_inputs(
    inp: KernelGenInput,
    analysis,
    next_directions: list,
    seed_code: str,
    epoch_num: int,
    *,
    knowledge_enabled: bool = False,
    evaluation_snapshot: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Construct every Coder input through the same mapping."""
    base_analysis = analysis.model_dump() if hasattr(analysis, "model_dump") else dict(analysis or {})
    inputs = []
    for index in range(inp.n_parallel):
        directed_analysis = (
            {**base_analysis, "assigned_direction": next_directions[index % len(next_directions)]}
            if inp.cross_epoch_knowledge and epoch_num > 1 and next_directions else base_analysis
        )
        inputs.append(build_coder_input(
            inp, directed_analysis, seed_code=seed_code,
            knowledge_enabled=knowledge_enabled, evaluation_snapshot=evaluation_snapshot,
        ))
    if epoch_num == 1 and inp.initial_seed_is_validated_baseline and inputs:
        inputs[0]["seed_is_validated_baseline"] = True
    return inputs


def confirmed_workspace_best(path: Path) -> tuple[str, dict | None]:
    """Select only the exported confirmation bound to the current search best."""
    try:
        ledger = Ledger(path)
        best = ledger.best
        if best["round"] <= 0:
            status = ledger.history.rounds[-1].evaluation.status if ledger.history.rounds else "FAILED"
            return status if status != "PASSED" else "FAILED", None
        output = json.loads((path / "optimize_definition_output.json").read_text())
        verification = json.loads((path / ".kernelgen/final-verification.json").read_text())
        if output.get("final_verification") != verification:
            return "NEEDS_RETEST", None
        if output.get("status") != "PASSED" or verification.get("status") != "PASSED":
            status = output.get("status") if output.get("status") != "PASSED" else verification.get("status")
            return status or "NEEDS_RETEST", None
        geo = verification.get("geo_mean")
        if (
            isinstance(geo, bool) or not isinstance(geo, (float, int)) or not math.isfinite(geo) or geo <= 0
            or geo != output.get("best_geo_mean")
            or verification.get("round_num") != best["round"]
            or verification.get("solution_sha256") != hashlib.sha256(best["code"].encode()).hexdigest()
            or output.get("best_code") != best["code"]
        ):
            return "NEEDS_RETEST", None
        return "PASSED", {**best, "geo_mean": geo}
    except (OSError, ValueError, KeyError, TypeError):
        return "NEEDS_RETEST", None


def collect_epoch_result(definition_name: str, results, workspace: Workspace) -> EpochResult:
    """Keep confirmed code, score, round and identity together, including recovery."""
    per_agent = []
    candidates = []
    for report, workspace_name in results:
        path = Path(workspace.path_of(workspace_name)).resolve()
        status, best = confirmed_workspace_best(path)
        geo = best["geo_mean"] if best is not None else None
        per_agent.append(AgentSummary(
            status=status, geo_mean=geo, workspace=workspace_name,
        ))
        if geo is not None:
            candidates.append((path, best))
    winner = pick_best(candidates, key=lambda item: item[1]["geo_mean"])
    return EpochResult(
        definition_name=definition_name, per_agent=tuple(per_agent),
        best_workspace_path=winner[0] if winner else None,
        best_round=winner[1]["round"] if winner else 0,
        best_geo_mean=winner[1]["geo_mean"] if winner else None,
        best_code=winner[1]["code"] if winner else "",
    )


def select_best_result(current: EpochResult, candidate: EpochResult) -> EpochResult:
    """Keep the earliest winner on ties, retaining its epoch's public result metadata."""
    return pick_best(
        [current, candidate],
        key=lambda result: (result.best_geo_mean or 0.0, bool(result.per_agent)),
    )


def run_epoch(
    *,
    cwd: Path,
    inp: KernelGenInput,
    analysis,
    next_directions: list,
    seed_code: str,
    epoch_num: int,
    run_control: RunControl,
    runtime_factory: Callable,
    knowledge: KernelGenKnowledgeBridge | None,
    knowledge_enabled: bool,
    evaluation_snapshot: dict[str, Any] | None,
):
    """Run parallel Coder slots and retain completed results when peers fail."""
    epoch_label = f"{epoch_num}R"
    run_control.checkpoint("BEFORE_EPOCH")
    run_control.update_progress(
        state=RunState.RUNNING, stage="CODER_RUNNING", current_epoch=epoch_num,
        total_tasks=inp.n_parallel, completed_tasks=0, failed_tasks=0, cancelled_tasks=0,
        message=f"Running epoch {epoch_num}/{inp.n_epoch}",
    )
    run_control.record_event(
        "EPOCH_STARTED", stage="CODER_RUNNING",
        data={"epoch": epoch_num, "total_epochs": inp.n_epoch},
    )
    print(f"\n{'=' * 60}\nStarting {epoch_label} ({inp.n_parallel} parallel agents)\n{'=' * 60}")
    inputs = build_epoch_inputs(
        inp, analysis, next_directions, seed_code, epoch_num,
        knowledge_enabled=knowledge_enabled, evaluation_snapshot=evaluation_snapshot,
    )
    workspace = create_epoch_workspace(cwd, epoch_label, knowledge,
                                       inherit_legacy_knowledge=inp.cross_epoch_knowledge)
    failed_agents = []
    print(f"\nLaunching {inp.n_parallel} parallel agents…")
    try:
        results = run_parallel(
            SingleCoderOptimizationWorkflow, inputs, workspace=workspace,
            runtime_factory=runtime_factory, max_workers=inp.n_parallel,
            task_name=lambda index, _: f"agent{index}", cancellation_token=run_control,
        )
    except ParallelExecutionError as exc:
        results = [item for item in exc.partial_results if item is not None]
        failed_agents = [
            FailedAgent(name=failure.name, error_type=type(failure.error).__name__, error=str(failure.error)[:4000])
            for failure in exc.failures
        ]
        for failure in failed_agents:
            print(f"[Agent {failure.name}] FAILED: {failure.error_type}: {failure.error}", flush=True)
        if not results:
            raise
        print(
            f"[KernelGen] Continuing {epoch_label} with {len(results)}/{len(inputs)} completed agents; "
            "failed agents will be recorded in the epoch manifest.", flush=True,
        )
    run_control.checkpoint("AFTER_PARALLEL_AGENTS")
    run_control.update_progress(
        state=RunState.RUNNING, stage="EPOCH_FINALIZING", current_epoch=epoch_num,
        message=f"Collecting epoch {epoch_num} results",
    )
    epoch_result = collect_epoch_result(inp.definition.name, results, workspace)
    for agent in epoch_result.per_agent:
        path = Path(workspace.path_of(agent.workspace))
        print(f"\n[Agent {agent.workspace}] Log saved to: {path / '.kernelgen' / 'claude-runtime.log'}")
        print(f"[Agent {agent.workspace}] Finished: status={agent.status} geo_mean={agent.geo_mean}")
    print(f"\n{epoch_label} agents complete: status={epoch_result.status} best_geo_mean={epoch_result.best_geo_mean}")
    return results, workspace, epoch_result, failed_agents


def _workspace_id(cwd: Path, path: Path) -> str:
    path, cwd = path.resolve(), cwd.resolve()
    return str(path.relative_to(cwd)) if path.is_relative_to(cwd) else str(path)


def summarize_epoch(
    cwd: Path,
    inp: KernelGenInput,
    results,
    best_result: EpochResult,
    runtime,
    *,
    epoch_workspace: Workspace,
):
    """Compare trajectories without inferring the global winner's identity from its score."""
    agent_results = []
    for report, name in results:
        path = Path(epoch_workspace.path_of(name))
        ledger = Ledger(path)
        best = ledger.best
        experience_path = path / ".new_experience.md"
        agent_results.append({
            "agent_id": _workspace_id(cwd, path),
            "status": getattr(report, "status", "FAILED"),
            "best_geo": best["geo_mean"],
            "best_round": best["round"],
            "strategy": getattr(report, "summary", "")[:200],
            "trajectory": build_synthesis_trajectory(
                [record.model_dump(mode="json") for record in ledger.history.rounds],
                best_round=best["round"],
            ),
            "new_experience": experience_path.read_text(encoding="utf-8") if experience_path.exists() else "",
        })
    return EpochSummaryAgent().run({
        "definition_name": inp.definition.name,
        "op_type": inp.definition.op_type,
        "target_hardware": inp.target_hardware,
        "implementation_language": inp.implementation_language,
        "n_agents": len(results),
        "agent_results": agent_results,
        "fixed_best_geo": best_result.best_geo_mean,
        "fixed_best_agent": _workspace_id(cwd, best_result.best_workspace_path) if best_result.best_workspace_path else "",
        "fixed_best_round": best_result.best_round,
        "fixed_best_kernel": best_result.best_code,
    }, runtime)
