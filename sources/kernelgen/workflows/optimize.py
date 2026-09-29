"""OptimizeWorkflow: iterative kernel optimization with Python-controlled loop.

Each iteration:
1. Run OptimizeAgent (one round: analyze history → pick direction → edit → test → benchmark)
2. Python saves version + updates PERFORMANCE.md in workspace
3. Check stop condition (target speedup reached or max iterations)
4. Next iteration's agent sees updated history

This is a Workflow (Runnable), so it can be used with run_parallel to optimize
multiple operators concurrently — each in its own workspace.

Usage:
    wf = OptimizeWorkflow(cwd="/workspace/softmax", runtime_factory=make_rt)
    out = wf.run({"operator": "softmax", "max_iters": 20, "target_speedup": 1.5})
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from kernelgen.agents.optimize import OptimizeAgent
from kernelgen.framework.workflow import Workflow


# ---------------------------------------------------------------------------
# I/O contract
# ---------------------------------------------------------------------------

class OptimizeWorkflowInput(BaseModel):
    """Input for the optimization loop."""
    operator: str = Field(description="Operator name")
    max_iters: int = Field(default=20, description="Max optimization iterations")
    target_speedup: float = Field(default=0.8, description="Stop when speedup >= target")


class OptimizeWorkflowOutput(BaseModel):
    """Output: best result across all iterations."""
    operator: str
    status: str = "failed"
    best_speedup: Optional[float] = None
    iterations_run: int = 0
    target_reached: bool = False
    history: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class OptimizeWorkflow(Workflow):
    """Iterative kernel optimization: Python loop × OptimizeAgent rounds.

    The agent does one round per invoke; Python handles:
    - Version persistence (versions/vN/)
    - Performance history (PERFORMANCE.md)
    - Stop condition (target speedup / max iters)
    """

    name = "optimize_loop"
    InputModel = OptimizeWorkflowInput
    OutputModel = OptimizeWorkflowOutput

    def __init__(self, *, cwd: str = ".", runtime_factory: Callable[[str], Any] = None):
        self._cwd = Path(cwd)
        self._runtime_factory = runtime_factory

    @classmethod
    def bind(cls, path: str, runtime_factory) -> "OptimizeWorkflow":
        return cls(cwd=path, runtime_factory=runtime_factory)

    def _execute(self, inp: OptimizeWorkflowInput) -> Dict[str, Any]:
        rt = self._runtime_factory(str(self._cwd))
        agent = OptimizeAgent()
        history: List[Dict[str, Any]] = []
        best_speedup: Optional[float] = None

        for i in range(inp.max_iters):
            # Run one optimization round
            result = agent.run({"operator": inp.operator}, rt)
            result_dict = result.model_dump()

            speedup = result_dict.get("speedup")
            test_passed = result_dict.get("test_passed")
            print(f"  [{inp.operator}] iter {i + 1}/{inp.max_iters}: "
                  f"speedup={speedup}, test_passed={test_passed}")

            # Save version + update history
            self._save_version(i, result_dict)
            self._update_performance_md()

            history.append({
                "iteration": i + 1,
                "speedup": speedup,
                "test_passed": test_passed,
                "optimization_direction": result_dict.get("optimization_direction", ""),
                "status": result_dict.get("status"),
            })

            if speedup is not None and (best_speedup is None or speedup > best_speedup):
                best_speedup = speedup

            # Stop condition
            if speedup is not None and speedup >= inp.target_speedup:
                print(f"  [{inp.operator}] target reached! ({speedup:.4f} >= {inp.target_speedup})")
                return {
                    "operator": inp.operator,
                    "status": "success",
                    "best_speedup": best_speedup,
                    "iterations_run": i + 1,
                    "target_reached": True,
                    "history": history,
                }

        return {
            "operator": inp.operator,
            "status": "success" if best_speedup and best_speedup > 0 else "failed",
            "best_speedup": best_speedup,
            "iterations_run": inp.max_iters,
            "target_reached": False,
            "history": history,
        }

    # -- helpers: version tracking (same logic as old optimize_loop) --------

    def _save_version(self, iteration: int, result: dict) -> None:
        """Save kernel version and metadata to workspace/versions/vN/."""
        version_dir = self._cwd / "versions" / f"v{iteration + 1}"
        version_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "iteration": iteration + 1,
            "speedup": result.get("speedup"),
            "test_passed": result.get("test_passed"),
            "optimization_direction": result.get("optimization_direction", ""),
            "status": result.get("status"),
        }
        (version_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        kernel_path = result.get("kernel_path")
        if kernel_path:
            src = self._cwd / kernel_path
            if src.exists():
                shutil.copy2(src, version_dir / "kernel.py")

    def _update_performance_md(self) -> None:
        """Regenerate PERFORMANCE.md from all saved versions."""
        versions_dir = self._cwd / "versions"
        if not versions_dir.exists():
            return

        rows = []
        best_version = None
        best_speedup = 0.0

        for vdir in sorted(versions_dir.iterdir(), key=lambda p: p.name):
            meta_path = vdir / "meta.json"
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            v = vdir.name
            status = "PASS" if meta.get("test_passed") else "FAIL"
            speedup = meta.get("speedup")
            speedup_str = f"{speedup:.4f}" if speedup is not None else "-"
            direction = meta.get("optimization_direction", "")[:50]
            rows.append(f"| {v} | {status} | {speedup_str} | {direction} |")

            if meta.get("test_passed") and speedup is not None and speedup > best_speedup:
                best_speedup = speedup
                best_version = v

        md = f"""# Performance History

Best: {best_version or 'none'} (speedup: {best_speedup:.4f})

| Version | Status | Speedup | Direction |
|---------|--------|---------|-----------|
"""
        md += "\n".join(rows) + "\n"
        (self._cwd / "PERFORMANCE.md").write_text(md)
