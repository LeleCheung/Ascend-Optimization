#!/usr/bin/env python3
"""Dry-run test for PRWorkflow + PRSubmitterAgent.

Exercises the full chain with fake upstream data — no real LLM, git, or
FlagGems repo needed. Validates data flow end-to-end.

    cd /share-evpfs/tj/workspace/code-integration
    python kernelgen/tests/run_pr_dry_run.py
"""

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernelgen.framework.runtime.base import FakeRuntime
from kernelgen.framework.models import DefinitionModel
from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationOutput
from kernelgen.workflows.pr_workflow import PRWorkflow, PRWorkflowInput, PRWorkflowOutput
from kernelgen.agents.pr_submitter import PRSubmitterReport


# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

DUMMY_KERNEL = textwrap.dedent("""\
    import logging
    import torch
    import triton
    import triton.language as tl

    logger = logging.getLogger(__name__)

    @triton.jit
    def _gelu_fwd_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        out = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
        tl.store(out_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=mask)

    def gelu(self, *, approximate="none"):
        logger.debug("GEMS_KUNLUNXIN GELU_FORWARD")
        output = torch.empty_like(self)
        n = self.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
        _gelu_fwd_kernel[grid](self, output, n, BLOCK=1024)
        return output
""")

DUMMY_DEFINITION = DefinitionModel(
    name="pytorch_gelu",
    op_type="pointwise",
    axes={"shape_id": {"type": "var", "description": "Index into SHAPES list"}},
    inputs={"input_0": {"shape": ["shape_id"], "dtype": "float32"}},
    outputs={"output": {"shape": ["shape_id"], "dtype": "float32"}},
    reference="import torch\ndef run(x, out): out.copy_(torch.nn.functional.gelu(x))\n",
    custom_inputs_entrypoint="gen_inputs",
)

FAKE_PR_REPORT = {
    "status": "SUBMITTED",
    "pr_url": "https://github.com/flagos-ai/FlagGems/pull/42",
    "branch": "kernelgen/kunlunxin/gelu",
    "files_changed": [
        "src/flag_gems/runtime/backend/_kunlunxin/ops/gelu.py",
        "src/flag_gems/runtime/backend/_kunlunxin/ops/__init__.py",
    ],
    "checks_passed": True,
    "summary": "Integrated gelu for kunlunxin. sort_registrations and pre-commit passed. Opened draft PR #42.",
}


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class DryRunPRWorkflow(PRWorkflow):
    """Stub git operations for dry-run."""

    def _create_worktree(self, flaggems_dir, branch, base_branch):
        print(f"  [DRY-RUN] Would create worktree: {flaggems_dir}/.worktrees/{branch.replace('/', '-')}")
        print(f"            git worktree add -b {branch} ... {base_branch}")
        return Path("/tmp/dry_run_wt")

    def _remove_worktree(self, flaggems_dir, wt_path):
        print(f"  [DRY-RUN] Would remove worktree: {wt_path}")


def make_fake_rt_factory(pr_report=None):
    """Return a runtime_factory that gives FakeRuntime with a scripted PR report."""
    report_json = json.dumps(pr_report or FAKE_PR_REPORT)
    def factory(path):
        return FakeRuntime([f"```json\n{report_json}\n```"])
    return factory


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def run_scenario(title, opt_status, best_code, geo_mean, expect_status):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    opt_result = SingleCoderOptimizationOutput(
        definition_name="pytorch_gelu",
        op_type="pointwise",
        status=opt_status,
        best_geo_mean=geo_mean,
        best_code=best_code,
        rounds=3,
        summary="converged" if opt_status == "PASSED" else "did not converge",
        workspace="/tmp/fake_opt_ws",
    )

    wf_input = PRWorkflowInput(
        opt_result=opt_result,
        definition=DUMMY_DEFINITION,
        vendor="kunlunxin",
        flaggems_dir="/tmp/fake_flaggems",
        base_branch="master",
        target_repo="flagos-ai/FlagGems",
        draft=True,
    )

    print(f"\n  Input:")
    print(f"    operator:    {DUMMY_DEFINITION.name}")
    print(f"    vendor:      kunlunxin")
    print(f"    opt status:  {opt_status}")
    print(f"    best_code:   {'<{} chars>'.format(len(best_code)) if best_code else '<empty>'}")
    print(f"    geo_mean:    {geo_mean}")

    wf = DryRunPRWorkflow(cwd=".", runtime_factory=make_fake_rt_factory())
    out = wf.run(wf_input.model_dump())

    print(f"\n  Output:")
    print(f"    status:      {out.status}")
    if out.pr_report:
        print(f"    pr_url:      {out.pr_report.pr_url}")
        print(f"    branch:      {out.pr_report.branch}")
        print(f"    checks:      {'✅' if out.pr_report.checks_passed else '❌'}")
        print(f"    files:       {out.pr_report.files_changed}")
        print(f"    summary:     {out.pr_report.summary}")
    if out.skip_reason:
        print(f"    skip_reason: {out.skip_reason}")

    ok = out.status == expect_status
    print(f"\n  {'✅ PASS' if ok else '❌ FAIL'}: expected status={expect_status}, got={out.status}")
    return ok


def main():
    print("=" * 70)
    print("  PRWorkflow Dry-Run Test")
    print("  Tests the full chain: PRWorkflowInput → guards → PRSubmitterAgent → output")
    print("  No real LLM, git, or FlagGems repo — all stubbed")
    print("=" * 70)

    results = []

    # Scenario 1: Happy path — PASSED + code → SUBMITTED
    results.append(run_scenario(
        title="Scenario 1: Happy path (PASSED + kernel code)",
        opt_status="PASSED",
        best_code=DUMMY_KERNEL,
        geo_mean=1.85,
        expect_status="SUBMITTED",
    ))

    # Scenario 2: Skip — optimization failed
    results.append(run_scenario(
        title="Scenario 2: Skip (optimization FAILED)",
        opt_status="FAILED",
        best_code=DUMMY_KERNEL,
        geo_mean=None,
        expect_status="SKIPPED",
    ))

    # Scenario 3: Skip — no kernel code
    results.append(run_scenario(
        title="Scenario 3: Skip (no kernel code)",
        opt_status="PASSED",
        best_code="",
        geo_mean=1.5,
        expect_status="SKIPPED",
    ))

    print(f"\n{'='*70}")
    passed = sum(results)
    print(f"  Summary: {passed}/{len(results)} scenarios passed")
    print(f"{'='*70}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
