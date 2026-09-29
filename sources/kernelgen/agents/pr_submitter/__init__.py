"""PRSubmitterAgent: integrate a generated kernel into a FlagGems vendor backend
worktree, run static checks, and open a GitHub PR.

Scenario: a generic FlagGems operator doesn't run well (or at all) on a specific
chip. CoderAgent has produced a specialized kernel. This agent integrates it as a
vendor-specific override:
  1. Write kernel to backend/_{vendor}/ops/{op}.py
  2. Register in backend/_{vendor}/ops/__init__.py (import + __all__)
  3. Static checks (sort_registrations --vendor-init + pre-commit)
  4. Branch → commit → push → gh pr create

Kernel correctness is assumed already verified upstream (eval server), so this
agent does NOT run GPU accuracy tests — only static checks.
"""

from __future__ import annotations

import textwrap
from typing import List, Optional

from pydantic import BaseModel, Field

from kernelgen.framework.base import BaseAgent
from kernelgen.framework.models import DefinitionModel


# ---------------------------------------------------------------------------
# I/O models
# ---------------------------------------------------------------------------

class PRSubmitterInput(BaseModel):
    operator: str                                   # e.g. "gelu"
    kernel_code: str                                # best kernel source from upstream
    definition: DefinitionModel                     # operator definition (for context/PR body)
    vendor: str                                     # target vendor, e.g. "kunlunxin", "ascend"
    geo_mean: Optional[float] = None                # coarse perf summary for the PR body
    base_branch: str = "master"                     # PR target branch
    target_repo: str = "flagos-ai/FlagGems"         # upstream repo for `gh pr create`
    draft: bool = True                              # open as draft PR


class PRSubmitterReport(BaseModel):
    status: str = Field(description="SUBMITTED / CHECKS_FAILED / FAILED")
    pr_url: Optional[str] = Field(default=None, description="URL of the created PR")
    branch: str = Field(default="", description="branch that was pushed")
    files_changed: List[str] = Field(default_factory=list)
    checks_passed: bool = False
    tested_locally: bool = Field(default=False, description="Whether local correctness test was attempted")
    test_skip_reason: str = Field(default="", description="device_mismatch / test_not_found / empty if tested")
    test_passed: Optional[bool] = Field(default=None, description="None=not tested, True=passed, False=failed")
    summary: str = Field(description="1-3 sentence summary of what happened")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PRSubmitterAgent(BaseAgent):
    """Integrate a verified kernel and submit its FlagGems pull request.

    Neutral role: ``.kernelgen/agents/kernel-pr-submitter.md``.
    """

    name = "pr_submitter"
    InputModel = PRSubmitterInput
    OutputModel = PRSubmitterReport

    def preprocess(self, inp: PRSubmitterInput, runtime) -> str:
        from kernelgen.framework.contract import render_contract

        role = self._role_for(runtime)

        geo_str = f"{inp.geo_mean:.4f}" if inp.geo_mean is not None else "N/A"

        context_block = textwrap.dedent(f"""\
            <task>
            Operator: {inp.operator}
            Vendor (target chip): {inp.vendor}
            Op type: {inp.definition.op_type}
            Geo-mean speedup (from upstream eval): {geo_str}
            PR target: {inp.target_repo} (base branch: {inp.base_branch})
            Draft PR: {inp.draft}
            Vendor ops directory: src/flag_gems/runtime/backend/_{inp.vendor}/ops
            Kernel destination: src/flag_gems/runtime/backend/_{inp.vendor}/ops/{inp.operator}.py
            Registration file: src/flag_gems/runtime/backend/_{inp.vendor}/ops/__init__.py
            Branch: kernelgen/{inp.vendor}/{inp.operator}
            </task>""")

        kernel_block = textwrap.dedent(f"""\
            <kernel_code>
            This is the vendor-specialized kernel to integrate (already verified for
            correctness upstream). Adapt it to FlagGems vendor conventions as needed:
            ```python
            {inp.kernel_code}
            ```
            </kernel_code>""")

        definition_block = textwrap.dedent(f"""\
            <definition>
            Name: {inp.definition.name}
            Type: {inp.definition.op_type}
            Reference (generic implementation):
            {inp.definition.reference}
            </definition>""")

        return "\n\n".join(filter(None, [
            role,
            context_block,
            kernel_block,
            definition_block,
            f"--- FINAL REPORT CONTRACT ---\n{render_contract(self.OutputModel)}",
        ]))

    def postprocess(self, raw: str, runtime) -> PRSubmitterReport:
        from kernelgen.framework.contract import extract_json
        return self.OutputModel.model_validate(extract_json(raw))
