"""ExtractOptWorkflow: extract a PyTorch operator, then optimize its definitions.

This is the extraction-oriented workflow.  It turns one PyTorch operator into one
or more definition/workload pairs, materializes them as a trace set, and runs one
SingleCoderOptimizationWorkflow per definition in parallel.

The checked-in examples currently use pre-extracted JSON from
``tmp/{operator}/output.json``.  The PyTorchV5ExtractorAgent call is kept as the
next integration step.

Usage:
    cd /data/akg_kernel_bench_lite
    PYTHONPATH=. python3 kernelgen/examples/extract_opt/run_example.py \
        --operator add --eval-server http://localhost:8000
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationWorkflow
from kernelgen.agents.extractor.pytorch import (
    PyTorchV5ExtractorAgent,
    PyTorchExtractorOutput,
    ExtractorResult,
)
from kernelgen.framework.parallel import IsolatedDirectory, run_parallel
from kernelgen.framework.workflow import Workflow
from kernelgen.data.trace import infer_destination_passing_style


# ---------------------------------------------------------------------------
# I/O contract
# ---------------------------------------------------------------------------

class ExtractOptInput(BaseModel):
    operator: str = Field(description="PyTorch operator name, e.g. 'add', 'gelu', 'softmax'")
    target_hardware: str = Field(default="", description="e.g. 'Ascend910B', 'A100'")
    eval_server_url: str = Field(default="http://localhost:8000")
    trace_root: str = Field(default="", description="Where to store extracted definitions")
    trace_set_key: str = Field(default="")
    early_stop_rounds: int = Field(default=3)
    min_rounds: int = Field(default=2)
    max_round: int = Field(default=15, ge=1)


class ExtractedDefinitionResult(BaseModel):
    definition_name: str = ""
    status: str = "FAILED"
    best_geo_mean: Optional[float] = None
    summary: str = ""
    workspace: str = ""


class ExtractOptOutput(BaseModel):
    operator: str = ""
    num_definitions: int = 0
    results: List[ExtractedDefinitionResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class ExtractOptWorkflow(Workflow):
    """Extract definitions → parallel optimize each one."""

    name = "extract_opt"
    InputModel = ExtractOptInput
    OutputModel = ExtractOptOutput

    def __init__(self, *, cwd: str = ".", runtime_factory: Callable[[str], Any] = None):
        self._cwd = Path(cwd)
        self._runtime_factory = runtime_factory

    @classmethod
    def bind(cls, path: str, runtime_factory) -> "ExtractOptWorkflow":
        return cls(cwd=path, runtime_factory=runtime_factory)

    def _execute(self, inp: ExtractOptInput) -> Dict[str, Any]:
        # --- Phase 1: Extract definitions ---
        extractor_results = self._extract(inp)
        if not extractor_results:
            return {"operator": inp.operator, "num_definitions": 0, "results": []}

        print(f"\n  Extracted {len(extractor_results)} definition(s):")
        for r in extractor_results:
            print(f"    - {r.definition.name} ({r.definition.op_type})")

        # --- Phase 2: Determine trace_root from extractor output paths ---
        # Extractor already wrote files; we just need to tell eval_round where they are.
        # trace_root is derived from the first result's definition_path:
        #   definition_path = {trace_root}/definitions/{op_type}/{name}.json
        #   → trace_root = definition_path.parents[2] (go up: name.json → op_type → definitions → root)
        trace_root = self._resolve_trace_root(extractor_results, inp)

        # --- Phase 3: Parallel optimize each definition ---
        coder_inputs = self._build_coder_inputs(extractor_results, inp, trace_root)

        # Each definition gets its own isolated workspace
        ws = IsolatedDirectory(
            base=self._cwd / "agents",
            claude_source=self._cwd / ".claude",
        )

        results = run_parallel(
            SingleCoderOptimizationWorkflow,
            coder_inputs,
            workspace=ws,
            runtime_factory=self._runtime_factory,
            task_name=lambda i, inp_dict: extractor_results[i].definition.name,
        )

        # --- Phase 4: Collect results ---
        per_def = []
        for i, (report, ws_name) in enumerate(results):
            defn_name = extractor_results[i].definition.name
            per_def.append({
                "definition_name": defn_name,
                "status": report.status,
                "best_geo_mean": report.best_geo_mean,
                "summary": report.summary,
                "workspace": report.workspace,
            })

        return {
            "operator": inp.operator,
            "num_definitions": len(extractor_results),
            "results": per_def,
        }

    # -- Phase 1: Extract --

    def _extract(self, inp: ExtractOptInput) -> List[ExtractorResult]:
        """Call PyTorchV5ExtractorAgent to extract definition(s) from PyTorch."""
        # --- Real extraction ---
        # main_rt = self._runtime_factory(str(self._cwd))
        # extractor_out = PyTorchV5ExtractorAgent().run(
        #     {"operator": inp.operator}, main_rt
        # )
        # return extractor_out.results

        # --- Dummy: load from pre-extracted data ---
        dummy = self._cwd / "tmp" / inp.operator / "output.json"
        return self._load_from_json(str(dummy))

    def _load_from_json(self, path: str) -> List[ExtractorResult]:
        """Load pre-extracted results from a JSON file."""
        p = Path(path)
        if not p.exists():
            print(f"  ⚠ Data not found: {p}", flush=True)
            return []
        data = json.loads(p.read_text())
        out = PyTorchExtractorOutput.model_validate(data)
        return out.results

    # -- Phase 2: Resolve trace root --

    def _resolve_trace_root(self, results: List[ExtractorResult], inp: ExtractOptInput) -> Path:
        """Ensure definition + workloads exist at trace_root for eval_round.

        trace_root is a shared, persistent directory (default: kernelgen/trace_data/)
        that lives outside any single run's workspace. This allows:
        - One extraction, multiple optimization runs
        - Easy overview of all extracted operators
        """
        # If user specified trace_root explicitly, use it
        if inp.trace_root:
            trace_root = Path(inp.trace_root).expanduser().resolve()
        else:
            # Default: kernelgen/trace_data (project-level, not per-run)
            trace_root = (Path(__file__).resolve().parents[1] / "trace_data").resolve()

        # Ensure trace_data has the right structure for eval_round
        # (extractor may write to a different layout)
        base = trace_root / inp.trace_set_key if inp.trace_set_key else trace_root
        for r in results:
            op_type = r.definition.op_type or "other"
            name = r.definition.name

            expected_def = base / "definitions" / op_type / f"{name}.json"
            expected_def.parent.mkdir(parents=True, exist_ok=True)
            expected_def.write_text(
                json.dumps(r.definition.model_dump(exclude_none=True), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            expected_wl = base / "workloads" / op_type / f"{name}.jsonl"
            expected_wl.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for w in r.workloads:
                wl_entry = {
                    "definition": name,
                    "workload": w.model_dump(exclude_none=True),
                    "solution": None,
                    "evaluation": None,
                }
                lines.append(json.dumps(wl_entry, ensure_ascii=False))
            expected_wl.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print(f"  ✅ Trace data ready at {base}")
        return trace_root

    # -- Phase 3: Build coder inputs --

    def _build_coder_inputs(self, results: List[ExtractorResult], inp: ExtractOptInput,
                            trace_root: Path) -> List[Dict[str, Any]]:
        """Build one CoderInput per definition."""
        inputs = []
        for r in results:
            d = r.definition
            is_dps = infer_destination_passing_style(d)

            coder_inp = {
                "definition": d.model_dump(exclude_none=True),
                "destination_passing_style": is_dps,
                "target_hardware": inp.target_hardware,
                "analysis": {},
                "workloads": [
                    {
                        "phase": phase,
                        **workload.model_dump(mode="json", exclude_none=True),
                    }
                    for phase, phase_workloads in (
                        ("correctness", r.correctness_workloads),
                        ("timing", r.timing_workloads),
                    )
                    for workload in phase_workloads
                ],
                "early_stop_rounds": inp.early_stop_rounds,
                "min_rounds": inp.min_rounds,
                "max_round": inp.max_round,
                "eval_server_url": inp.eval_server_url,
                "trace_root": str(trace_root),
                "trace_set_key": inp.trace_set_key,
            }
            inputs.append(coder_inp)
        return inputs
