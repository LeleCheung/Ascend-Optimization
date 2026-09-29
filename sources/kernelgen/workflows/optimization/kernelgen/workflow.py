"""Public entry points and global epoch ordering for KernelGen."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from kernelgen.framework.run_control import RunControl, RunState, WorkspaceRunControl
from kernelgen.framework.workflow import Workflow
from kernelgen.knowledge.config import KnowledgeConfig
from kernelgen.workflows.optimization.kernelgen import epoch, finalization, knowledge, preparation
from kernelgen.workflows.optimization.kernelgen.contracts import KernelGenInput, KernelGenOutput
from kernelgen.workflows.lifecycle import run_workflow


class KernelGenWorkflow(Workflow):
    """Coordinate prepared runs, parallel epochs and one authoritative winner."""

    name = "kernel_gen"
    InputModel = KernelGenInput
    OutputModel = KernelGenOutput

    def __init__(
        self,
        *,
        cwd: str = ".",
        runtime_factory: Callable[[str], Any] = None,
        knowledge_config: KnowledgeConfig | None = None,
        run_control: RunControl | None = None,
    ):
        self._cwd = Path(cwd)
        self._runtime_factory = runtime_factory
        self._knowledge_config = knowledge_config
        self._run_control = run_control or WorkspaceRunControl(self._cwd, source="workflow:kernel_gen")

    def _execute(self, inp: KernelGenInput) -> KernelGenOutput:
        return run_workflow(
            self._run_control, name=self.name, mode="kernelgen", progress_kind="epochs",
            definition_name=inp.definition.name, max_round=inp.max_round,
            checkpoint="BEFORE_KERNEL_GEN", output_model=self.OutputModel,
            execute=lambda: self._execute_controlled(inp),
            current_epoch=inp.start_epoch, total_epochs=inp.n_epoch,
        )

    def _execute_controlled(self, inp: KernelGenInput) -> KernelGenOutput:
        prepared = preparation.prepare_run(
            cwd=self._cwd, inp=inp, knowledge_config=self._knowledge_config,
            runtime_factory=self._runtime_factory, run_control=self._run_control,
        )
        inp = prepared.inp
        best_result = prepared.best_result
        next_directions = prepared.next_directions
        for epoch_num in range(inp.start_epoch, inp.n_epoch + 1):
            results, workspace, epoch_result, failed_agents = epoch.run_epoch(
                cwd=self._cwd, inp=inp, analysis=prepared.analysis,
                next_directions=next_directions,
                seed_code=prepared.initial_seed_code if epoch_num == 1 else best_result.best_code,
                epoch_num=epoch_num, run_control=self._run_control,
                runtime_factory=prepared.runtime_factory, knowledge=prepared.knowledge,
                knowledge_enabled=bool(self._knowledge_config and self._knowledge_config.reads_v1),
                evaluation_snapshot=prepared.evaluation_snapshot,
            )
            best_result = epoch.select_best_result(best_result, epoch_result)
            self._run_control.update_progress(
                state=RunState.RUNNING, stage="SYNTHESIZING", current_epoch=epoch_num,
                message=f"Synthesizing epoch {epoch_num} results",
            )
            synthesis = finalization.finalize_epoch_outputs(
                cwd=self._cwd, run_control=self._run_control, inp=inp,
                results=results, best_result=best_result, epoch_num=epoch_num,
                epoch_workspace=workspace, runtime_factory=prepared.runtime_factory,
                knowledge=prepared.knowledge,
                attempted_agents=[f"agent{index}" for index in range(inp.n_parallel)],
                failed_agents=failed_agents,
            )
            next_directions = synthesis.next_directions or []
            print(f"\n{epoch_num}R complete.")
            self._run_control.update_progress(
                state=RunState.RUNNING, stage="EPOCH_FINALIZING", current_epoch=epoch_num,
                message=f"Epoch {epoch_num}/{inp.n_epoch} completed",
            )
            self._run_control.record_event(
                "EPOCH_COMPLETED", stage="EPOCH_FINALIZING",
                data={
                    "epoch": epoch_num, "status": epoch_result.status,
                    "best_geo_mean": epoch_result.best_geo_mean,
                },
            )
        self._run_control.checkpoint("BEFORE_FINAL_PROMOTION")
        self._run_control.update_progress(
            state=RunState.RUNNING, stage="PROMOTING_BEST", current_epoch=inp.n_epoch,
            message="Promoting the best measured solution",
        )
        knowledge.promote_best_solution(prepared.knowledge, best_result)
        return best_result.as_output()

    def finalize_completed_epoch(
        self,
        inp: KernelGenInput | dict[str, Any],
        epoch_num: int,
    ) -> KernelGenOutput:
        """Reuse completed ledgers without rerunning Analyzer or Coder."""
        inp = KernelGenInput.model_validate(inp)
        if epoch_num < 1 or epoch_num > inp.n_epoch:
            raise ValueError("finalize epoch must be between 1 and n_epoch")
        return run_workflow(
            self._run_control, name=self.name, mode="kernelgen", progress_kind="epochs",
            definition_name=inp.definition.name, max_round=inp.max_round,
            checkpoint="BEFORE_EPOCH_FINALIZATION", output_model=self.OutputModel,
            execute=lambda: finalization.finalize_completed_epoch(
                cwd=self._cwd, inp=inp, epoch_num=epoch_num, run_control=self._run_control,
                runtime_factory=self._runtime_factory, knowledge_config=self._knowledge_config,
            ).as_output(),
            current_epoch=epoch_num, total_epochs=inp.n_epoch, completed_epoch=epoch_num,
        )


__all__ = ["KernelGenWorkflow"]
