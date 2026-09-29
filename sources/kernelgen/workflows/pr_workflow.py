"""PRWorkflow: read optimization results, prepare inputs, and call PRSubmitterAgent
to integrate each passing kernel into a FlagGems vendor backend and open PRs.

Usage (from an outer orchestrator):
    pr_wf = PRWorkflow(cwd=".", runtime_factory=make_rt)
    out = pr_wf.run({
        "opt_result": opt_output.model_dump(),
        "definition": definition.model_dump(),
        "vendor": "kunlunxin",
        "flaggems_dir": "/path/to/FlagGems",
    })
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_worktree_lock = threading.Lock()

from pydantic import BaseModel, Field

from kernelgen.agents.pr_submitter import PRSubmitterAgent, PRSubmitterInput, PRSubmitterReport
from kernelgen.framework.agent_roles import materialize_agent_role
from kernelgen.framework.models import DefinitionModel
from kernelgen.framework.workflow import Workflow
from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationOutput


# ---------------------------------------------------------------------------
# I/O contract
# ---------------------------------------------------------------------------

class PRWorkflowInput(BaseModel):
    opt_result: SingleCoderOptimizationOutput        # upstream optimization result (has best_code)
    definition: DefinitionModel                 # operator definition (for context/PR body)
    vendor: str = Field(description="Target vendor, e.g. 'kunlunxin', 'ascend'")
    flaggems_dir: str = Field(description="Path to the FlagGems repo (worktree will be created here)")
    base_branch: str = "master"
    target_repo: str = "flagos-ai/FlagGems"
    draft: bool = True


class PRWorkflowOutput(BaseModel):
    status: str = "SKIPPED"                     # SUBMITTED / SKIPPED / FAILED
    pr_report: Optional[PRSubmitterReport] = None
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

class PRWorkflow(Workflow):
    """Take an optimization result, integrate into FlagGems vendor backend, and open a PR."""

    name = "pr_workflow"
    InputModel = PRWorkflowInput
    OutputModel = PRWorkflowOutput

    def __init__(self, *, cwd: str = ".", runtime_factory: Callable[[str], Any] = None):
        self._cwd = Path(cwd)
        self._runtime_factory = runtime_factory

    @classmethod
    def bind(cls, path: str, runtime_factory) -> "PRWorkflow":
        return cls(cwd=path, runtime_factory=runtime_factory)

    def _execute(self, inp: PRWorkflowInput) -> Dict[str, Any]:
        result = inp.opt_result

        # --- Guard: only submit passing results with code ---
        if result.status != "PASSED":
            return {"status": "SKIPPED", "skip_reason": f"status={result.status}"}

        if not result.best_code:
            return {"status": "SKIPPED", "skip_reason": "no kernel code in result"}

        # --- Create FlagGems worktree ---
        flaggems = Path(inp.flaggems_dir).resolve()
        operator = inp.definition.name.removeprefix("pytorch_").removeprefix("flaggems_")
        branch_name = f"kernelgen/{inp.vendor}/{operator}"
        wt_path = self._create_worktree(flaggems, branch_name, inp.base_branch)
        if not wt_path:
            return {"status": "FAILED", "skip_reason": "failed to create worktree"}

        # --- Build PRSubmitterInput ---
        pr_input = PRSubmitterInput(
            operator=operator,
            kernel_code=result.best_code,
            definition=inp.definition,
            vendor=inp.vendor,
            geo_mean=result.best_geo_mean,
            base_branch=inp.base_branch,
            target_repo=inp.target_repo,
            draft=inp.draft,
        )

        # --- Run PRSubmitterAgent ---
        try:
            rt = self._runtime_factory(str(wt_path))
            report = PRSubmitterAgent().run(pr_input.model_dump(), rt)
            return {
                "status": report.status,
                "pr_report": report.model_dump(),
            }
        except Exception as e:
            return {
                "status": "FAILED",
                "skip_reason": f"PRSubmitterAgent raised: {e}",
            }
        finally:
            self._remove_worktree(flaggems, wt_path)

    # -- Helpers --

    @classmethod
    def ensure_fork_remote(cls, flaggems_dir: Path) -> None:
        """Verify 'origin' points to a fork the current gh user can push to.

        Raises RuntimeError if origin is not pushable. Users must set up the
        fork remote manually before running the workflow:
            git remote set-url origin https://github.com/<your-fork>/FlagGems-Experimental.git
            git remote add upstream https://github.com/flagos-ai/FlagGems-Experimental.git
        """
        import re
        r = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(flaggems_dir), capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"No 'origin' remote found in {flaggems_dir}")
        origin_url = r.stdout.strip()

        m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", origin_url)
        if not m:
            return
        repo_slug = m.group(1)

        push_check = subprocess.run(
            ["gh", "api", f"repos/{repo_slug}", "--jq", ".permissions.push"],
            capture_output=True, text=True,
        )
        if push_check.stdout.strip() != "true":
            raise RuntimeError(
                f"No push access to origin '{repo_slug}'. "
                f"Set origin to your fork before running:\n"
                f"  git -C {flaggems_dir} remote set-url origin https://github.com/<your-fork>/{repo_slug.split('/')[-1]}.git"
            )

    def _create_worktree(self, flaggems_dir: Path, branch: str, base_branch: str) -> Optional[Path]:
        wt_path = flaggems_dir / ".worktrees" / branch.replace("/", "-")
        try:
            # Clean up stale branch/worktree from a previous failed run
            if wt_path.exists():
                subprocess.run(
                    ["git", "worktree", "remove", str(wt_path), "--force"],
                    cwd=str(flaggems_dir), capture_output=True, text=True,
                )
            subprocess.run(
                ["git", "branch", "-D", branch],
                cwd=str(flaggems_dir), capture_output=True, text=True,
            )
            for candidate in [f"upstream/{base_branch}", f"origin/{base_branch}", base_branch]:
                r = subprocess.run(
                    ["git", "rev-parse", "--verify", candidate],
                    cwd=str(flaggems_dir), capture_output=True, text=True,
                )
                if r.returncode == 0:
                    base_ref = candidate
                    break
            else:
                base_ref = base_branch
            with _worktree_lock:
                subprocess.run(
                    ["git", "worktree", "add", "-b", branch, str(wt_path), base_ref],
                    cwd=str(flaggems_dir),
                    capture_output=True,
                    text=True,
                    check=True,
                )
            # Materialize the provider-neutral PR role into the worktree.
            agents_src = Path(__file__).resolve().parents[1] / ".kernelgen" / "agents" / "kernel-pr-submitter.md"
            if agents_src.exists():
                materialize_agent_role(agents_src, wt_path)
            return wt_path
        except subprocess.CalledProcessError as e:
            print(f"  ⚠ Failed to create worktree: {e.stderr.strip()}", flush=True)
            return None

    def _remove_worktree(self, flaggems_dir: Path, wt_path: Path) -> None:
        """Remove the worktree after PR submission (branch stays)."""
        try:
            subprocess.run(
                ["git", "worktree", "remove", str(wt_path), "--force"],
                cwd=str(flaggems_dir),
                capture_output=True,
                text=True,
            )
        except Exception:
            pass
