"""SingleCoderOptimizationWorkflow: optimize one prepared definition end to end.

Coder (optimize) → Distiller (generate candidate KB). This is what run_parallel
parallelizes — each agent slot runs this complete workflow in its own isolated
workspace. Epoch-level knowledge reduction is intentionally owned by
KernelGenWorkflow so parallel agents never race to write the canonical KB.

Input = SingleCoderOptimizationInput (CoderInput plus Python-owned stop controls).
Output = SingleCoderOptimizationOutput, assembled authoritatively from the workspace Ledger.
Every successful workflow return is also atomically persisted as
``optimize_definition_output.json`` for downstream workflows such as PR submission.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

from kernelgen.agents.coder import CoderInput
from kernelgen.data.evaluation_snapshot import (
    CatalogEvaluationSnapshot,
    freeze_evaluation_snapshot,
    snapshot_optimization_context,
)
from kernelgen.data.tool_context import ToolContext
from kernelgen.framework.base import AgentContractError
from kernelgen.framework.run_control import (
    RunControl,
    RunState,
    WorkspaceRunControl,
)
from kernelgen.framework.workflow import Workflow
from kernelgen.workflows.lifecycle import run_workflow
from kernelgen.workflows.optimization.single_coder.coder_loop import run_coder_loop
from kernelgen.workflows.optimization.single_coder.distillation import (
    run_distillation,
    save_distillation_artifacts,
)
from kernelgen.workflows.optimization.single_coder.profiling import ensure_best_profile
from kernelgen.tools.retest import verify_final_best
from kernelgen.data._atomic import atomic_write_json
from kernelgen.data.constants import DEFAULT_MAX_ROUNDS


KERNEL_OPTIMIZATION_OUTPUT_FILENAME = "optimize_definition_output.json"


class SingleCoderOptimizationInput(CoderInput):
    """Coder input plus orchestration controls that must not enter analysis."""

    early_stop_rounds: int = Field(default=3, ge=0)
    min_rounds: int = Field(default=2, ge=1)
    max_round: int = Field(default=DEFAULT_MAX_ROUNDS, ge=1)
    max_coder_sessions: int = Field(
        default=3,
        ge=1,
        description=(
            "Maximum Coder invocations used to reach a persisted terminal STOP "
            "verdict. Early returns must resume the same provider session"
        ),
    )


class SingleCoderOptimizationOutput(BaseModel):
    """Authoritative result of optimizing one definition."""

    definition_name: str
    op_type: str = ""
    status: str = "FAILED"
    best_geo_mean: Optional[float] = None
    best_code: str = ""
    search_best_geo_mean: Optional[float] = None
    final_verification: dict[str, Any] = Field(default_factory=dict)
    rounds: int = 0
    summary: str = ""
    workspace: str = ""


class SingleCoderOptimizationWorkflow(Workflow):
    """One agent slot: optimize kernel and emit distilled KB candidates."""
    name = "optimize_definition"
    InputModel = SingleCoderOptimizationInput
    OutputModel = SingleCoderOptimizationOutput

    def __init__(
        self,
        *,
        cwd: str = ".",
        runtime_factory: Callable[[str], Any] = None,
        run_control: RunControl | None = None,
        run_mode: str = "optimize_definition",
    ):
        self._cwd = Path(cwd)
        self._runtime_factory = runtime_factory
        self._run_control = run_control or WorkspaceRunControl(
            self._cwd,
            source="workflow:optimize_definition",
        )
        self._run_mode = run_mode

    @classmethod
    def bind(cls, path: str, runtime_factory) -> "SingleCoderOptimizationWorkflow":
        return cls(cwd=path, runtime_factory=runtime_factory)

    def _execute(self, inp: SingleCoderOptimizationInput) -> SingleCoderOptimizationOutput:
        return run_workflow(
            self._run_control, name=self.name, mode=self._run_mode,
            definition_name=inp.definition.name, max_round=inp.max_round,
            checkpoint="BEFORE_OPTIMIZATION", output_model=self.OutputModel,
            execute=lambda: self._execute_controlled(inp),
        )

    def _execute_controlled(
        self,
        inp: SingleCoderOptimizationInput,
    ) -> SingleCoderOptimizationOutput:
        evaluation_snapshot_path = None
        if inp.evaluation_snapshot is not None:
            requested_snapshot = CatalogEvaluationSnapshot.model_validate(
                inp.evaluation_snapshot
            )
            evaluation_snapshot, evaluation_snapshot_path = (
                freeze_evaluation_snapshot(self._cwd, requested_snapshot)
            )
            frozen_definition, frozen_workloads = snapshot_optimization_context(
                evaluation_snapshot
            )
            inp = inp.model_copy(
                update={
                    "definition": frozen_definition,
                    "workloads": frozen_workloads,
                    "evaluation_snapshot": evaluation_snapshot.model_dump(
                        mode="json"
                    ),
                }
            )
        target_context = self._resolve_target_context(inp)
        inp = inp.model_copy(
            update={"target_hardware": target_context.device}
        )
        main_rt = self._runtime_factory(str(self._cwd))

        # Persist the Python stop policy used by finalize_round.
        from kernelgen.data.ledger import Ledger
        from kernelgen.data.stop_policy import StopConfig
        led = Ledger(self._cwd)
        led.set_stop_config(StopConfig(
            early_stop_rounds=inp.early_stop_rounds,
            min_rounds=inp.min_rounds,
            max_round=inp.max_round,
        ))

        # Bind deterministic MCP tools to this isolated workspace. The model only
        # supplies a relative kernel path plus plan/conclusion; ledger/eval configuration
        # remains server-side and cannot be redirected to another agent workspace.
        self._build_tool_context(
            inp,
            target_context,
            evaluation_snapshot_path=evaluation_snapshot_path,
        ).write(self._cwd)
        self._materialize_validated_seed(inp)

        # 1. Coder: optimize kernel and reach a durable terminal round boundary.
        report = run_coder_loop(
            inp,
            self._cwd,
            main_rt,
            run_control=self._run_control,
        )

        self._run_control.checkpoint("BEFORE_FINAL_RETEST")
        self._run_control.update_progress(
            state=RunState.RUNNING, stage="RETESTING", message="Independently verifying final best",
        )
        verification = verify_final_best(self._cwd, run_control=self._run_control)
        atomic_write_json(self._cwd / ".kernelgen" / "final-verification.json", verification)
        if verification["status"] != "PASSED":
            result_ledger = Ledger(self._cwd)
            result = SingleCoderOptimizationOutput(
                definition_name=inp.definition.name, op_type=inp.definition.op_type,
                status=verification["status"], best_code=result_ledger.best.get("code", ""),
                search_best_geo_mean=result_ledger.history.best_geo_mean or None,
                final_verification=verification, rounds=len(result_ledger.history.rounds),
                summary=f"Final best not confirmed: {verification.get('reason_code', 'unknown')}. {report.summary}",
                workspace=str(self._cwd.resolve()),
            )
            self._save_output(result)
            return result  # Do not distill/publish an unverified winning result.

        # Distillation explains the winning kernel. Ensure the eventual best has
        # one backend-native analysis when profiling is enabled, without making
        # profiler availability a condition for keeping a valid optimized kernel.
        self._run_control.checkpoint("BEFORE_FINAL_PROFILE")
        self._run_control.update_progress(
            state=RunState.RUNNING,
            stage="PROFILING",
            message="Checking final-best profile",
        )
        completed_ledger = Ledger(self._cwd)
        ensure_best_profile(self._cwd, inp, main_rt, completed_ledger)

        # 2. Distiller: SimpleOpt keeps the legacy report-only path. A
        # Knowledge-enabled KernelGen workspace additionally emits structured
        # CandidateDraft records for the epoch Publisher.
        self._run_control.checkpoint("BEFORE_DISTILLATION")
        self._run_control.update_progress(
            state=RunState.RUNNING,
            stage="DISTILLING",
            message="Distilling optimization results",
        )
        try:
            distill_out = run_distillation(self._cwd, inp, main_rt, verified_geo_mean=verification["geo_mean"])
        except AgentContractError as exc:
            print(
                f"[Distiller] Skipped invalid output: {exc}",
                flush=True,
            )
            distill_out = None

        self._run_control.checkpoint("BEFORE_OUTPUT_FINALIZATION")
        self._run_control.update_progress(
            state=RunState.RUNNING,
            stage="FINALIZING_OUTPUT",
            message="Writing authoritative optimization output",
        )

        # 3. Save derived reports. Only Knowledge-enabled KernelGen workspaces
        # receive a structured outbox; SimpleOpt never publishes V1 KB data.
        if distill_out and not distill_out.skip_reason:
            report_ledger = Ledger(self._cwd)
            # Derived reports use the confirmed score; the search ledger remains immutable.
            report_ledger.history.best_geo_mean = verification["geo_mean"]
            save_distillation_artifacts(
                self._cwd,
                inp,
                distill_out,
                report_ledger,
            )

        # The LLM report is narrative only. Assemble authoritative result fields
        # from a fresh Ledger because eval_round updates it in separate processes.
        result_ledger = Ledger(self._cwd)
        best_geo = verification["geo_mean"]
        result = SingleCoderOptimizationOutput(
            definition_name=inp.definition.name,
            op_type=inp.definition.op_type,
            status="PASSED" if best_geo > 0 else "FAILED",
            best_geo_mean=best_geo if best_geo > 0 else None,
            best_code=result_ledger.best.get("code", ""),
            search_best_geo_mean=result_ledger.history.best_geo_mean or None,
            final_verification=verification,
            rounds=len(result_ledger.history.rounds),
            summary=report.summary,
            workspace=str(self._cwd.resolve()),
        )
        self._save_output(result)
        return result

    def _materialize_validated_seed(
        self,
        inp: SingleCoderOptimizationInput,
    ) -> None:
        """Freeze the externally validated baseline before the Coder starts."""

        if not inp.seed_is_validated_baseline:
            return
        destination = self._cwd / "tmp" / "main.py"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = destination.read_text(encoding="utf-8")
            if existing != inp.seed_code:
                raise ValueError(
                    "validated baseline destination already contains different "
                    "code: tmp/main.py"
                )
            return
        destination.write_text(inp.seed_code, encoding="utf-8")

    def _resolve_target_context(self, inp: SingleCoderOptimizationInput):
        """Resolve and validate the Server-owned target before starting Coder."""

        from kernelgen.tools.kernelgen_server_adapter import (
            get_service_status,
            require_target_context,
        )

        service_status = get_service_status(inp.eval_server_url)
        return require_target_context(inp, service_status)

    @staticmethod
    def _build_tool_context(
        inp: SingleCoderOptimizationInput,
        target_context,
        *,
        evaluation_snapshot_path: Path | None,
    ) -> ToolContext:
        """Map validated workflow input into the workspace-bound tool contract."""
        contract = inp.evaluation_contract
        if contract.required_matched_ratio != 1.0:
            raise ValueError(
                "Protocol v6.2 bound evaluation takes required_matched_ratio "
                "from the Server catalog Workload; KernelGen cannot override it"
            )
        tolerance_mode = contract.tolerance_mode or (
            "fixed"
            if contract.atol is not None and contract.rtol is not None
            else "strict"
        )
        return ToolContext(
            definition=inp.definition.name,
            target_hardware=target_context.device,
            implementation_language=inp.implementation_language,
            eval_server_url=inp.eval_server_url,
            catalog_name=inp.catalog_name,
            evaluation_snapshot_path=(
                str(evaluation_snapshot_path.resolve())
                if evaluation_snapshot_path is not None
                else ""
            ),
            destination_passing_style=inp.destination_passing_style,
            profile_enabled=inp.profile_enabled,
            warmup_ms=inp.warmup_ms,
            benchmark_ms=inp.benchmark_ms,
            num_trials=inp.num_trials,
            eval_tolerance_mode=tolerance_mode,
            eval_atol=contract.atol if contract.atol is not None else 1e-2,
            eval_rtol=contract.rtol if contract.rtol is not None else 1e-2,
            eval_timeout_seconds=inp.eval_timeout_seconds,
            eval_transport_timeout_seconds=inp.eval_transport_timeout_seconds,
            target_context=target_context,
        )

    def _save_output(self, result: SingleCoderOptimizationOutput) -> None:
        """Atomically persist the public workflow result for downstream agents."""
        self._cwd.mkdir(parents=True, exist_ok=True)
        destination = self._cwd / KERNEL_OPTIMIZATION_OUTPUT_FILENAME
        fd, temporary = tempfile.mkstemp(
            dir=str(self._cwd),
            prefix=f".{KERNEL_OPTIMIZATION_OUTPUT_FILENAME}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(result.model_dump_json(indent=2))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
