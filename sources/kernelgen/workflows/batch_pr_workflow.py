"""BatchPRWorkflow: run PRWorkflow in parallel for all PASSED results in a batch.

Uses run_parallel so each operator gets an isolated workspace. The fork remote
is set up once before parallel execution to avoid concurrent git-config races.

Usage:
    wf = BatchPRWorkflow(cwd=".", runtime_factory=make_rt)
    out = wf.run({
        "batch_dir": "/path/to/batch/v4",
        "flaggems_dir": "/path/to/FlagGems-Experimental",
        "vendor": "ascend",
        "target_repo": "flagos-ai/FlagGems-Experimental",
    })
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field

from kernelgen.framework.parallel import Directory, run_parallel
from kernelgen.framework.workflow import Workflow
from kernelgen.framework.models import DefinitionModel
from kernelgen.workflows.legacy.batch_simple_opt_definition import (
    BatchSimpleOptDefinitionOutput,
    BATCH_SIMPLE_OPT_OUTPUT_FILENAME,
)
from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationOutput
from kernelgen.workflows.pr_workflow import PRWorkflow, PRWorkflowInput, PRWorkflowOutput


BATCH_PR_OUTPUT_FILENAME = "batch_pr_output.json"


class PRResult(BaseModel):
    definition_name: str
    status: str
    pr_url: Optional[str] = None
    skip_reason: str = ""


class BatchPRInput(BaseModel):
    batch_dir: str = Field(description="Directory containing batch_simple_opt_definition_output.json")
    flaggems_dir: str = Field(description="Path to the FlagGems repo clone")
    vendor: str = Field(description="Target vendor, e.g. 'ascend', 'kunlunxin'")
    target_repo: str = "flagos-ai/FlagGems-Experimental"
    base_branch: str = "master"
    draft: bool = True
    max_workers: int = Field(default=0, ge=0)


class BatchPROutput(BaseModel):
    summary: str
    results: List[PRResult]


class BatchPRWorkflow(Workflow):
    """Run PRWorkflow in parallel for all PASSED kernels in a batch output directory."""

    name = "batch_pr"
    InputModel = BatchPRInput
    OutputModel = BatchPROutput

    def __init__(self, *, cwd: str = ".", runtime_factory: Callable[[str], Any] = None):
        self._cwd = Path(cwd)
        self._runtime_factory = runtime_factory

    @classmethod
    def bind(cls, path: str, runtime_factory) -> "BatchPRWorkflow":
        return cls(cwd=path, runtime_factory=runtime_factory)

    def _execute(self, inp: BatchPRInput):
        batch_dir = Path(inp.batch_dir).resolve()
        flaggems_dir = Path(inp.flaggems_dir).resolve()

        # Load batch results
        output_file = batch_dir / BATCH_SIMPLE_OPT_OUTPUT_FILENAME
        batch_out = BatchSimpleOptDefinitionOutput.model_validate_json(
            output_file.read_text(encoding="utf-8")
        )

        # Set up fork remote once before parallel execution
        try:
            PRWorkflow.ensure_fork_remote(flaggems_dir)
        except RuntimeError as e:
            raise RuntimeError(
                f"Fork remote setup failed: {e}\n\n"
                f"Please set up the fork remote manually:\n"
                f"  cd {flaggems_dir}\n"
                f"  git remote set-url origin https://github.com/<your-fork>/FlagGems-Experimental.git\n"
                f"  git remote add upstream https://github.com/flagos-ai/FlagGems-Experimental.git"
            ) from e

        # Build inputs for PASSED results only
        pr_inputs: List[PRWorkflowInput] = []
        skipped: List[PRResult] = []
        for result in batch_out.results:
            if result.status != "PASSED":
                skipped.append(PRResult(
                    definition_name=result.definition_name,
                    status="SKIPPED",
                    skip_reason=f"status={result.status}",
                ))
                continue

            definition = self._load_definition(batch_dir, result.definition_name)
            pr_inputs.append(PRWorkflowInput(
                opt_result=result,
                definition=definition,
                vendor=inp.vendor,
                flaggems_dir=str(flaggems_dir),
                base_branch=inp.base_branch,
                target_repo=inp.target_repo,
                draft=inp.draft,
            ))

        # Run in parallel using run_parallel
        workspace = Directory(base=self._cwd / "pr_runs")
        workers = inp.max_workers or min(len(pr_inputs), 8)

        parallel_results = run_parallel(
            PRWorkflow,
            pr_inputs,
            workspace=workspace,
            runtime_factory=self._runtime_factory,
            max_workers=workers,
            task_name=lambda i, x: x.opt_result.definition_name,
        )

        pr_results: List[PRResult] = list(skipped)
        for pr_inp, (wf_out, _) in zip(pr_inputs, parallel_results):
            pr_url = wf_out.pr_report.pr_url if wf_out.pr_report else None
            pr_results.append(PRResult(
                definition_name=pr_inp.opt_result.definition_name,
                status=wf_out.status,
                pr_url=pr_url,
                skip_reason=wf_out.skip_reason,
            ))

        submitted = sum(r.status == "SUBMITTED" for r in pr_results)
        summary = f"{submitted}/{len(pr_results)} PRs submitted"
        print(f"[BatchPR] {summary}", flush=True)

        output = BatchPROutput(summary=summary, results=pr_results)
        (self._cwd / BATCH_PR_OUTPUT_FILENAME).write_text(
            output.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        return output.model_dump()

    def _load_definition(self, batch_dir: Path, definition_name: str) -> DefinitionModel:
        evals_dir = batch_dir / "definitions" / definition_name / ".kernelgen" / "evals"
        candidates = sorted(evals_dir.glob("round-*/definition.json"))
        if not candidates:
            raise FileNotFoundError(f"No definition.json found for {definition_name}")
        return DefinitionModel.model_validate_json(candidates[-1].read_text())
