#!/usr/bin/env python3
"""E2E test for PRSubmitterAgent with a real LLM.

Creates a FlagGems worktree, runs PRSubmitterAgent with a real LLM API to
integrate a dummy 'mish' kernel into the kunlunxin backend, then opens a
draft PR. Clean up manually after inspection:

    gh pr close <PR_URL> --delete-branch

Usage:
    cd /share-evpfs/tj/workspace/code-integration
    python kernelgen/tests/run_pr_e2e.py

Requires:
    - ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN env vars
    - gh CLI authenticated (gh auth status)
    - FlagGems repo at /share-evpfs/tj/workspace/code-integration/FlagGems
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernelgen.framework.runtime.claude import ClaudeRuntime
from kernelgen.agents.pr_submitter import PRSubmitterAgent, PRSubmitterInput
from kernelgen.framework.models import DefinitionModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FLAGGEMS_DIR = Path("/share-evpfs/tj/workspace/code-integration/FlagGems")
OPERATOR = "mish"
VENDOR = "kunlunxin"
BASE_BRANCH = "master"
TARGET_REPO = "Schopenhauer-loves-Hegel/FlagGems"  # your fork, not upstream
BRANCH_NAME = f"kernelgen/{VENDOR}/{OPERATOR}"

# Dummy kernel — a simple mish implementation for testing
DUMMY_KERNEL = textwrap.dedent("""\
    import logging
    import torch
    import triton
    import triton.language as tl

    logger = logging.getLogger(__name__)

    @triton.jit
    def _mish_fwd_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        # mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        sp = tl.log(1.0 + tl.exp(x))
        out = x * tl.math.tanh(sp)
        tl.store(out_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=mask)

    def mish(self):
        logger.debug("GEMS_KUNLUNXIN MISH_FORWARD")
        output = torch.empty_like(self)
        n = self.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
        _mish_fwd_kernel[grid](self, output, n, BLOCK=1024)
        return output
""")

DEFINITION = DefinitionModel(
    name="pytorch_mish",
    op_type="pointwise",
    axes={"shape_id": {"type": "var"}},
    inputs={"input_0": {"shape": ["shape_id"], "dtype": "float32"}},
    outputs={"output": {"shape": ["shape_id"], "dtype": "float32"}},
    reference="import torch\ndef run(x, out): out.copy_(torch.nn.functional.mish(x))\n",
    custom_inputs_entrypoint="gen_inputs",
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  PRSubmitterAgent E2E Test (real LLM, real git, draft PR)")
    print("=" * 70)

    # --- Pre-checks ---
    print("\n[1/5] Pre-checks...")
    assert FLAGGEMS_DIR.exists(), f"FlagGems not found at {FLAGGEMS_DIR}"

    gh_check = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if gh_check.returncode != 0:
        print("  ❌ gh not authenticated. Run: gh auth login")
        sys.exit(1)
    print("  ✅ gh authenticated")

    for var in ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"]:
        if not os.environ.get(var):
            print(f"  ❌ {var} not set")
            sys.exit(1)
    print("  ✅ LLM API env vars present")

    # Check mish doesn't already exist in kunlunxin
    existing = FLAGGEMS_DIR / "src/flag_gems/runtime/backend/_kunlunxin/ops/mish.py"
    if existing.exists():
        print(f"  ⚠ {existing} already exists — test may conflict. Proceeding anyway.")

    # --- Create worktree ---
    print("\n[2/5] Creating FlagGems worktree...")
    wt_path = FLAGGEMS_DIR / ".worktrees" / BRANCH_NAME.replace("/", "-")

    # Clean up stale worktree/branch if exists
    if wt_path.exists():
        subprocess.run(
            ["git", "worktree", "remove", str(wt_path), "--force"],
            cwd=str(FLAGGEMS_DIR), capture_output=True,
        )
    subprocess.run(
        ["git", "branch", "-D", BRANCH_NAME],
        cwd=str(FLAGGEMS_DIR), capture_output=True,
    )

    result = subprocess.run(
        ["git", "worktree", "add", "-b", BRANCH_NAME, str(wt_path), BASE_BRANCH],
        cwd=str(FLAGGEMS_DIR), capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ❌ Failed: {result.stderr.strip()}")
        sys.exit(1)
    print(f"  ✅ Worktree at {wt_path}")

    # --- Build runtime ---
    print("\n[3/5] Setting up ClaudeRuntime...")
    rt = ClaudeRuntime(
        workspace=str(wt_path),
        model=os.environ.get("ANTHROPIC_MODEL", "ep-20260313004351-feb36"),
        base_url=os.environ["ANTHROPIC_BASE_URL"],
        auth_token=os.environ["ANTHROPIC_AUTH_TOKEN"],
        timeout=600,
        idle_timeout=300,
    )
    print("  ✅ Runtime ready")

    # --- Build input ---
    print("\n[4/5] Running PRSubmitterAgent...")
    pr_input = PRSubmitterInput(
        operator=OPERATOR,
        kernel_code=DUMMY_KERNEL,
        definition=DEFINITION,
        vendor=VENDOR,
        geo_mean=1.42,
        base_branch=BASE_BRANCH,
        target_repo=TARGET_REPO,
        draft=True,
    )

    print(f"  operator:    {OPERATOR}")
    print(f"  vendor:      {VENDOR}")
    print(f"  target_repo: {TARGET_REPO}")
    print(f"  draft:       True")
    print(f"  kernel:      {len(DUMMY_KERNEL)} chars")

    try:
        report = PRSubmitterAgent().run(pr_input.model_dump(), rt)
    except Exception as e:
        print(f"\n  ❌ Agent failed: {e}")
        # Cleanup
        subprocess.run(
            ["git", "worktree", "remove", str(wt_path), "--force"],
            cwd=str(FLAGGEMS_DIR), capture_output=True,
        )
        sys.exit(1)

    # --- Report ---
    print(f"\n[5/5] Result:")
    print(f"  status:      {report.status}")
    print(f"  pr_url:      {report.pr_url}")
    print(f"  branch:      {report.branch}")
    print(f"  checks:      {'✅' if report.checks_passed else '❌'}")
    print(f"  files:       {report.files_changed}")
    print(f"  summary:     {report.summary}")

    # --- Cleanup worktree (keep branch for PR) ---
    subprocess.run(
        ["git", "worktree", "remove", str(wt_path), "--force"],
        cwd=str(FLAGGEMS_DIR), capture_output=True,
    )
    print(f"\n  ✅ Worktree cleaned up (branch '{BRANCH_NAME}' preserved for PR)")

    if report.pr_url:
        print(f"\n  🔗 Draft PR: {report.pr_url}")
        print(f"  To clean up after review:")
        print(f"    gh pr close {report.pr_url} --delete-branch")

    print(f"\n{'='*70}")
    ok = report.status == "SUBMITTED"
    print(f"  {'✅ E2E PASSED' if ok else '❌ E2E FAILED'}")
    print(f"{'='*70}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
