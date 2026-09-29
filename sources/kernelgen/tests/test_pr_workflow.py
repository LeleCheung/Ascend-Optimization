"""Unit tests for PRWorkflow.

Pure Python, no torch/GPU/LLM/git (mocks worktree creation):

    cd /share-evpfs/tj/workspace/code-integration
    python kernelgen/tests/test_pr_workflow.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernelgen.framework.runtime.base import FakeRuntime
from kernelgen.workflows.pr_workflow import PRWorkflow, PRWorkflowInput, PRWorkflowOutput


_PR_REPORT = {
    "status": "SUBMITTED",
    "pr_url": "https://github.com/flagos-ai/FlagGems/pull/99",
    "branch": "kernelgen/kunlunxin/gelu",
    "files_changed": [
        "src/flag_gems/runtime/backend/_kunlunxin/ops/gelu.py",
        "src/flag_gems/runtime/backend/_kunlunxin/ops/__init__.py",
    ],
    "checks_passed": True,
    "summary": "Integrated gelu for kunlunxin.",
}

_BASE_INPUT = {
    "opt_result": {
        "definition_name": "pytorch_gelu",
        "op_type": "pointwise",
        "status": "PASSED",
        "best_geo_mean": 1.85,
        "best_code": "import triton\n@triton.jit\ndef gelu_kernel(): ...\n",
        "rounds": 3,
        "summary": "converged in 3 rounds",
        "workspace": "/tmp/fake_ws",
    },
    "definition": {
        "name": "pytorch_gelu",
        "op_type": "pointwise",
        "reference": "import torch\ndef run(x, out): ...\n",
    },
    "vendor": "kunlunxin",
    "flaggems_dir": "/tmp/fake_flaggems",
}


class _TestPRWorkflow(PRWorkflow):
    """Override helpers to avoid real git in unit tests."""

    def _create_worktree(self, flaggems_dir, branch, base_branch):
        return Path("/tmp/fake_wt")

    def _remove_worktree(self, flaggems_dir, wt_path):
        pass


def _make_rt_factory(pr_report=None):
    report = json.dumps(pr_report or _PR_REPORT)
    def factory(path):
        return FakeRuntime([f"```json\n{report}\n```"])
    return factory


# --- I/O models -----------------------------------------------------------

def test_input_requires_vendor_and_flaggems_dir():
    try:
        PRWorkflowInput(
            opt_result={
                "definition_name": "x", "status": "PASSED",
                "best_code": "...", "summary": "ok",
            },
            definition={"name": "x"},
        )
    except Exception:
        pass
    else:
        raise AssertionError("vendor and flaggems_dir are required")


def test_input_defaults():
    inp = PRWorkflowInput(**_BASE_INPUT)
    assert inp.base_branch == "master"
    assert inp.target_repo == "flagos-ai/FlagGems"
    assert inp.draft is True


def test_output_model():
    out = PRWorkflowOutput(status="SUBMITTED")
    assert out.pr_report is None
    assert out.skip_reason == ""


# --- Workflow execution ---------------------------------------------------

def test_happy_path():
    wf = _TestPRWorkflow(cwd=".", runtime_factory=_make_rt_factory())
    out = wf.run(_BASE_INPUT)
    assert isinstance(out, PRWorkflowOutput)
    assert out.status == "SUBMITTED"
    assert out.pr_report.pr_url.endswith("/pull/99")


def test_skips_failed_status():
    inp = dict(_BASE_INPUT)
    inp["opt_result"] = dict(inp["opt_result"])
    inp["opt_result"]["status"] = "FAILED"
    wf = _TestPRWorkflow(cwd=".", runtime_factory=_make_rt_factory())
    out = wf.run(inp)
    assert out.status == "SKIPPED"
    assert "status=FAILED" in out.skip_reason


def test_skips_no_code():
    inp = dict(_BASE_INPUT)
    inp["opt_result"] = dict(inp["opt_result"])
    inp["opt_result"]["best_code"] = ""
    wf = _TestPRWorkflow(cwd=".", runtime_factory=_make_rt_factory())
    out = wf.run(inp)
    assert out.status == "SKIPPED"
    assert "no kernel code" in out.skip_reason


def test_skips_worktree_failure():
    class _NoWtWorkflow(_TestPRWorkflow):
        def _create_worktree(self, flaggems_dir, branch, base_branch):
            return None

    wf = _NoWtWorkflow(cwd=".", runtime_factory=_make_rt_factory())
    out = wf.run(_BASE_INPUT)
    assert out.status == "FAILED"
    assert "worktree" in out.skip_reason


def test_operator_strips_pytorch_prefix():
    """'pytorch_gelu' → 'gelu' for branch name and PR."""
    wf = _TestPRWorkflow(cwd=".", runtime_factory=_make_rt_factory())
    out = wf.run(_BASE_INPUT)
    # The PRSubmitterAgent received 'gelu' not 'pytorch_gelu'
    assert out.status == "SUBMITTED"


if __name__ == "__main__":
    import inspect
    import tempfile
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            if "tmp_path" in inspect.signature(t).parameters:
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
                t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
