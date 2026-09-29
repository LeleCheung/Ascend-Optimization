"""Publication and synthesis for one completed KernelGen epoch."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Callable

from kernelgen.agents.epoch_summary import EpochSummaryOutput
from kernelgen.agents.knowledge_reviewer import KnowledgeReviewerAgent
from kernelgen.data._atomic import atomic_write_json, atomic_write_text
from kernelgen.framework.mcp_config import MCP_CONFIGURATION_PATH
from kernelgen.framework.parallel import IsolatedDirectory, Workspace
from kernelgen.framework.run_control import (
    RunCancelled,
    RunControl,
    RunState,
    WorkspaceRunControl,
)
from kernelgen.knowledge.config import KnowledgeConfig, KnowledgeReviewerMode
from kernelgen.knowledge.layout import KnowledgeLayout
from kernelgen.knowledge.records import FileRetrievalAudit
from kernelgen.knowledge.publishing.reviewer import ReviewerExecutionResult
from kernelgen.workflows.optimization.kernelgen.contracts import (
    EpochCompletionManifest,
    EpochResult,
    FailedAgent,
    KernelGenInput,
)
from kernelgen.workflows.knowledge_bridge import KernelGenKnowledgeBridge
from kernelgen.workflows.optimization.kernelgen import epoch, knowledge as knowledge_helpers, preparation, recovery


def finalize_epoch_outputs(
    *,
    cwd: Path,
    run_control: RunControl,
    inp: KernelGenInput,
    results,
    best_result: EpochResult,
    epoch_num: int,
    epoch_workspace: Workspace,
    runtime_factory: Callable[[str], Any],
    knowledge: KernelGenKnowledgeBridge | None,
    attempted_agents: list[str] | None = None,
    failed_agents: list[FailedAgent] | None = None,
) -> EpochSummaryOutput:
    """Publish one completed epoch and persist its required checkpoint."""
    run_control.checkpoint("BEFORE_EPOCH_PUBLICATION")
    synthesis_path = cwd / f"{epoch_num}R" / "synthesis"
    synthesis_file = synthesis_path / "synthesis.json"
    synthesis_control = WorkspaceRunControl(
        synthesis_path,
        source="workflow:kernel_gen:synthesis",
    )
    if knowledge is None and inp.cross_epoch_knowledge:
        merger_path = Path(epoch_workspace.allocate("knowledge-merger"))
        merger_control = WorkspaceRunControl(
            merger_path,
            source="workflow:kernel_gen:knowledge_merger",
        )
        merger_control.update_progress(
            state=RunState.RUNNING,
            stage="KNOWLEDGE_MERGING",
            message=f"Merging epoch {epoch_num} knowledge candidates",
        )
        merger_runtime = runtime_factory(str(merger_path))
        try:
            knowledge_helpers.merge_epoch_kb(
                cwd=cwd,
                inp=inp,
                results=results,
                runtime=merger_runtime,
                epoch_num=epoch_num,
                epoch_workspace=epoch_workspace,
            )
        except RunCancelled:
            merger_control.acknowledge_cancellation(stage="CANCELLED")
            raise
        except Exception as exc:
            merger_control.update_progress(
                state=RunState.FAILED,
                stage="FAILED",
                message=f"{type(exc).__name__}: {exc}",
            )
            raise
        merger_control.update_progress(
            state=RunState.SUCCEEDED,
            stage="COMPLETED",
            message=f"Epoch {epoch_num} knowledge merge completed",
        )
    elif knowledge is not None:
        publish_workspaces = [
            Path(epoch_workspace.path_of(workspace_name))
            for _, workspace_name in results
        ]
        reviewer = None
        if knowledge.config.reviewer_mode != KnowledgeReviewerMode.OFF:
            reviewer = _build_epoch_reviewer(
                cwd=cwd,
                epoch_num=epoch_num,
                runtime_factory=runtime_factory,
                knowledge=knowledge,
            )
        else:
            print(
                "\n[Knowledge Reviewer] mode=off; runtime Candidates "
                "will be deferred without a model call."
            )
        publish_result = knowledge.publish_epoch(
            publish_workspaces, epoch_num=epoch_num, reviewer=reviewer,
        )
        if publish_result is None:
            raise RuntimeError("knowledge publication produced no result")
        publish_status = getattr(publish_result, "status", "")
        if publish_status not in {"published", "noop"}:
            raise RuntimeError(
                "knowledge publication did not complete: "
                f"{publish_status}"
            )

    run_control.checkpoint("BEFORE_EPOCH_SYNTHESIS")
    successful_agents = [workspace_name for _, workspace_name in results]
    manifest_path = cwd / f"{epoch_num}R" / "epoch-completion.json"
    existing_manifest = None
    if attempted_agents is None and manifest_path.is_file():
        existing_manifest = EpochCompletionManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    attempted_agents = (
        list(existing_manifest.attempted_agents)
        if existing_manifest is not None
        else (attempted_agents or successful_agents)
    )
    failed_agents = (
        list(existing_manifest.failed_agents)
        if existing_manifest is not None
        else (failed_agents or [])
    )
    manifest = EpochCompletionManifest(
        attempted_agents=attempted_agents,
        successful_agents=successful_agents,
        failed_agents=failed_agents,
    )

    if len(results) <= 1:
        synthesis = EpochSummaryOutput(
            next_directions=[],
            synthesis_report=(
                f"Epoch {epoch_num} completed with one successful agent; "
                "no cross-agent comparison was available."
            ),
        )
        atomic_write_text(
            synthesis_file,
            synthesis.model_dump_json(indent=2) + "\n",
        )
        atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
        synthesis_control.update_progress(
            state=RunState.SUCCEEDED,
            stage="COMPLETED",
            message=f"Epoch {epoch_num} synthesis completed without comparison",
        )
        return synthesis

    if synthesis_file.is_file():
        synthesis = EpochSummaryOutput.model_validate_json(
            synthesis_file.read_text(encoding="utf-8")
        )
        print(f"\n[Epoch Synthesis] Reusing checkpoint: {synthesis_file}")
        atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
        synthesis_control.update_progress(
            state=RunState.SUCCEEDED,
            stage="COMPLETED",
            message=f"Epoch {epoch_num} synthesis checkpoint reused",
        )
        return synthesis

    synthesis_workspace = IsolatedDirectory(
        base=cwd / f"{epoch_num}R",
        claude_source=cwd / ".claude",
        mcp_source=(
            cwd / MCP_CONFIGURATION_PATH
            if (cwd / MCP_CONFIGURATION_PATH).is_file()
            else None
        ),
        materializers=(
            [knowledge.materializer()]
            if knowledge is not None
            else []
        ),
    )
    synthesis_path = Path(synthesis_workspace.allocate("synthesis"))
    synthesis_runtime = runtime_factory(str(synthesis_path))
    add_directory = getattr(synthesis_runtime, "add_directory", None)
    if callable(add_directory):
        add_directory(cwd / f"{epoch_num}R")
    print(f"\n[Epoch Synthesis] Comparing agents for {epoch_num}R…")
    synthesis_control.update_progress(
        state=RunState.RUNNING,
        stage="SYNTHESIZING",
        message=f"Comparing agents for epoch {epoch_num}",
    )
    try:
        synthesis = epoch.summarize_epoch(
            cwd,
            inp,
            results,
            best_result,
            synthesis_runtime,
            epoch_workspace=epoch_workspace,
        )
        synthesis_control.checkpoint("AFTER_EPOCH_SYNTHESIS")
        atomic_write_text(
            synthesis_file,
            synthesis.model_dump_json(indent=2) + "\n",
        )
        atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
    except RunCancelled:
        synthesis_control.acknowledge_cancellation(stage="CANCELLED")
        raise
    except Exception as exc:
        synthesis_control.update_progress(
            state=RunState.FAILED,
            stage="FAILED",
            message=f"{type(exc).__name__}: {exc}",
        )
        raise
    synthesis_control.update_progress(
        state=RunState.SUCCEEDED,
        stage="COMPLETED",
        message=f"Epoch {epoch_num} synthesis completed",
    )
    print(f"[Epoch Synthesis] Done. Result: {synthesis_file}")
    print(
        "[Epoch Synthesis] Log saved to: "
        f"{synthesis_path / '.kernelgen' / 'claude-runtime.log'}"
    )
    return synthesis


def finalize_completed_epoch(
    *,
    cwd: Path,
    inp: KernelGenInput,
    epoch_num: int,
    run_control: RunControl,
    runtime_factory: Callable,
    knowledge_config: KnowledgeConfig | None,
) -> EpochResult:
    """Finalize existing completed agents without Analyzer or Coder execution."""
    inp, _ = preparation.resolve_server_target(inp)
    run_control.update_progress(
        state=RunState.RUNNING, stage="EPOCH_FINALIZING", mode="kernelgen",
        current_epoch=epoch_num, total_epochs=inp.n_epoch,
        message=f"Loading completed epoch {epoch_num}",
    )
    results, workspace = recovery.load_completed_epoch(
        cwd=cwd, definition_name=inp.definition.name, target_hardware=inp.target_hardware,
        implementation_language=inp.implementation_language,
        epoch_num=epoch_num, expected_agents=inp.n_parallel,
    )
    best_result = recovery.load_completed_result(
        cwd=cwd, definition_name=inp.definition.name, target_hardware=inp.target_hardware,
        implementation_language=inp.implementation_language, through_epoch=epoch_num,
    )
    knowledge = knowledge_helpers.build_knowledge_bridge(
        config=knowledge_config, cwd=cwd, inp=inp, start_mode=preparation.start_mode(inp),
    )
    runtime_factory = preparation.configured_runtime_factory(inp, runtime_factory)
    print(f"\n[Finalize] Reusing {len(results)} completed agents from {epoch_num}R; Analyzer and Coder will not run.")
    run_control.update_progress(
        state=RunState.RUNNING, stage="SYNTHESIZING", current_epoch=epoch_num,
        message=f"Finalizing epoch {epoch_num} outputs",
    )
    finalize_epoch_outputs(
        cwd=cwd, run_control=run_control, inp=inp, results=results, best_result=best_result,
        epoch_num=epoch_num, epoch_workspace=workspace, runtime_factory=runtime_factory,
        knowledge=knowledge,
    )
    run_control.checkpoint("BEFORE_FINAL_PROMOTION")
    run_control.update_progress(
        state=RunState.RUNNING, stage="PROMOTING_BEST", current_epoch=epoch_num,
        message="Promoting the best measured solution",
    )
    knowledge_helpers.promote_best_solution(knowledge, best_result)
    print(f"\n{epoch_num}R complete.")
    return best_result


def _build_epoch_reviewer(
    *,
    cwd: Path,
    epoch_num: int,
    runtime_factory: Callable[[str], Any],
    knowledge: KernelGenKnowledgeBridge,
):
    """Reuse one serialized Reviewer workspace for all packets in an epoch."""

    workspace = IsolatedDirectory(
        base=cwd / f"{epoch_num}R",
        claude_source=cwd / ".claude",
        mcp_source=(
            cwd / MCP_CONFIGURATION_PATH
            if (cwd / MCP_CONFIGURATION_PATH).is_file()
            else None
        ),
        materializers=[knowledge.materializer()],
        include_skills=False,
    )
    review_lock = Lock()
    reviewer_path: Path | None = None

    def serialized(callback):
        def invoke(review_input):
            with review_lock:
                return callback(review_input)

        return invoke

    @serialized
    def review(review_input):
        nonlocal reviewer_path
        if reviewer_path is None:
            reviewer_path = Path(workspace.allocate("knowledge-reviewer"))
        reviewer_control = WorkspaceRunControl(
            reviewer_path,
            source="workflow:kernel_gen:knowledge_reviewer",
        )
        reviewer_control.update_progress(
            state=RunState.RUNNING,
            stage="KNOWLEDGE_REVIEW",
            message=f"Reviewing epoch {epoch_num} knowledge candidates",
        )
        reviewer_runtime = runtime_factory(str(reviewer_path))
        add_directory = getattr(reviewer_runtime, "add_directory", None)
        if callable(add_directory):
            add_directory(cwd / f"{epoch_num}R")
        layout = KnowledgeLayout(reviewer_path)
        retrieval_audit = FileRetrievalAudit(
            layout.retrieval_log,
            legacy_query_path=layout.query_log,
        )
        prior_retrieval_count = len(retrieval_audit.read_all())
        print(
            "\n[Knowledge Reviewer] Reviewing "
            f"{len(review_input.units)} candidate group(s)…"
        )
        try:
            result = KnowledgeReviewerAgent().run(
                review_input,
                reviewer_runtime,
            )
            print(
                "[Knowledge Reviewer] Done. Log saved to: "
                f"{reviewer_path / '.kernelgen' / 'claude-runtime.log'}"
            )
            retrievals = retrieval_audit.read_all()
            if len(retrievals) < prior_retrieval_count:
                raise RuntimeError("Reviewer retrieval audit was truncated")
        except RunCancelled:
            reviewer_control.acknowledge_cancellation(stage="CANCELLED")
            raise
        except Exception as exc:
            reviewer_control.update_progress(
                state=RunState.FAILED,
                stage="FAILED",
                message=f"{type(exc).__name__}: {exc}",
            )
            raise
        reviewer_control.update_progress(
            state=RunState.SUCCEEDED,
            stage="COMPLETED",
            message=f"Epoch {epoch_num} knowledge review completed",
        )
        return ReviewerExecutionResult(
            output=result,
            retrievals=tuple(retrievals[prior_retrieval_count:]),
        )

    return review


__all__ = ["finalize_epoch_outputs"]
