"""Historical Python Batch layout; new batches use kg run --batch-file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

from pydantic import BaseModel, Field

from kernelgen.framework.mcp_config import MCP_CONFIGURATION_PATH
from kernelgen.framework.parallel import (
    IsolatedDirectory,
    ParallelExecutionError,
    run_parallel,
)
from kernelgen.framework.run_control import (
    RunCancelled,
    RunControl,
    RunState,
    WorkspaceRunControl,
)
from kernelgen.framework.workflow import Workflow
from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationOutput
from kernelgen.workflows.legacy.simple_opt import SimpleOptInput, SimpleOptWorkflow


BATCH_SIMPLE_OPT_OUTPUT_FILENAME = "batch_simple_opt_definition_output.json"


class BatchSimpleOptDefinitionInput(BaseModel):
    definitions: List[SimpleOptInput] = Field(min_length=1)
    max_workers: int = Field(default=0, ge=0)
    launch_interval_seconds: float = Field(default=1.0, ge=0)


class BatchSimpleOptDefinitionOutput(BaseModel):
    summary: str
    results: List[SingleCoderOptimizationOutput]


class BatchSimpleOptDefinitionWorkflow(Workflow):
    """Run independent SimpleOpt workflows in parallel and summarize them."""

    name = "batch_simple_opt_definition"
    InputModel = BatchSimpleOptDefinitionInput
    OutputModel = BatchSimpleOptDefinitionOutput

    def __init__(
        self,
        *,
        cwd: str = ".",
        runtime_factory: Callable[[str], Any] = None,
        run_control: RunControl | None = None,
    ):
        self._cwd = Path(cwd)
        self._runtime_factory = runtime_factory
        self._run_control = run_control or WorkspaceRunControl(
            self._cwd,
            source="workflow:batch_simple_opt_definition",
        )

    @classmethod
    def bind(cls, path: str, runtime_factory) -> "BatchSimpleOptDefinitionWorkflow":
        return cls(cwd=path, runtime_factory=runtime_factory)

    def _execute(self, inp: BatchSimpleOptDefinitionInput) -> Dict[str, Any]:
        try:
            self._run_control.checkpoint("BEFORE_BATCH")
            self._run_control.update_progress(
                state=RunState.RUNNING,
                stage="OPTIMIZING_BATCH",
                mode="batch_simple_opt",
                message=f"Running {len(inp.definitions)} definitions",
            )
            self._run_control.record_event(
                "WORKFLOW_STARTED",
                stage="OPTIMIZING_BATCH",
                data={
                    "workflow": self.name,
                    "definitions": len(inp.definitions),
                },
            )
            output = self._execute_controlled(inp)
        except RunCancelled:
            self._run_control.acknowledge_cancellation(stage="CANCELLED")
            raise
        except Exception as exc:
            self._run_control.update_progress(
                state=RunState.FAILED,
                stage="FAILED",
                message=f"{type(exc).__name__}: {exc}",
            )
            self._run_control.record_event(
                "WORKFLOW_FAILED",
                message=f"{type(exc).__name__}: {exc}",
                stage="FAILED",
                level="ERROR",
                data={"workflow": self.name},
            )
            raise

        passed = sum(result.status == "PASSED" for result in output.results)
        state = (
            RunState.SUCCEEDED
            if passed == len(output.results)
            else RunState.FAILED
        )
        self._run_control.update_progress(
            state=state,
            stage="COMPLETED",
            message=output.summary,
        )
        self._run_control.record_event(
            "WORKFLOW_COMPLETED",
            message=output.summary,
            stage="COMPLETED",
            level="INFO" if state == RunState.SUCCEEDED else "WARNING",
            data={
                "workflow": self.name,
                "passed": passed,
                "total": len(output.results),
            },
        )
        return output

    def _execute_controlled(
        self,
        inp: BatchSimpleOptDefinitionInput,
    ) -> BatchSimpleOptDefinitionOutput:
        self._cwd.mkdir(parents=True, exist_ok=True)
        workspace = IsolatedDirectory(
            base=self._cwd / "definitions",
            claude_source=self._cwd / ".claude",
            mcp_source=(
                self._cwd / MCP_CONFIGURATION_PATH
                if (self._cwd / MCP_CONFIGURATION_PATH).is_file()
                else None
            ),
            include_skills=False,
        )
        batch_error: Exception | None = None
        try:
            parallel_results = run_parallel(
                SimpleOptWorkflow,
                inp.definitions,
                workspace=workspace,
                runtime_factory=self._runtime_factory,
                max_workers=inp.max_workers,
                task_name="definition_name",
                launch_interval_seconds=inp.launch_interval_seconds,
                cancellation_token=self._run_control,
            )
            results = [result for result, _ in parallel_results]
        except ParallelExecutionError as exc:
            batch_error = exc
            failures_by_index = {
                failure.index: failure for failure in exc.failures
            }
            results = []
            for index, item in enumerate(inp.definitions):
                completed = exc.partial_results[index]
                if completed is not None:
                    results.append(completed[0])
                    continue
                failure = failures_by_index.get(index)
                error = failure.error if failure is not None else exc
                results.append(self._read_or_build_failed_result(item, error))
        except RunCancelled:
            raise
        except Exception as exc:
            # Compatibility for alternate run_parallel implementations and
            # failures that happen before per-task collection is available.
            batch_error = exc
            results = [
                self._read_or_build_failed_result(item, exc)
                for item in inp.definitions
            ]

        passed = sum(result.status == "PASSED" for result in results)
        summary = f"{passed}/{len(results)} definitions passed"
        if batch_error is not None:
            failed = sum(result.status != "PASSED" for result in results)
            summary += f"; {failed} task(s) failed"
        print(f"[BatchSimpleOpt] {summary}", flush=True)

        output = BatchSimpleOptDefinitionOutput(
            summary=summary,
            results=results,
        )
        (self._cwd / BATCH_SIMPLE_OPT_OUTPUT_FILENAME).write_text(
            output.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        return output

    def _read_or_build_failed_result(
        self,
        item: SimpleOptInput,
        error: Exception,
    ) -> SingleCoderOptimizationOutput:
        item_dir = self._cwd / "definitions" / item.definition_name
        output_path = item_dir / "optimize_definition_output.json"
        try:
            return SingleCoderOptimizationOutput.model_validate_json(
                output_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            pass

        rounds = 0
        best_geo_mean = None
        best_code = ""
        try:
            ledger = json.loads(
                (item_dir / ".ledger.json").read_text(encoding="utf-8")
            )
            rounds = len(ledger.get("rounds", []))
            best = ledger.get("best_geo_mean")
            best_geo_mean = best if isinstance(best, (int, float)) and best > 0 else None
            best_code = str(ledger.get("best_code") or "")
        except (OSError, ValueError, TypeError):
            pass

        return SingleCoderOptimizationOutput(
            definition_name=item.definition_name,
            status="FAILED",
            best_geo_mean=best_geo_mean,
            best_code=best_code,
            rounds=rounds,
            summary=f"{type(error).__name__}: {error}",
            workspace=str(item_dir.resolve()),
        )
