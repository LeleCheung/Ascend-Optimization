"""Unit tests for PRSubmitterAgent.

Pure Python, no torch/GPU/LLM/git (uses FakeRuntime):

    cd /share-evpfs/tj/workspace/code-integration
    python kernelgen/tests/test_pr_submitter.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernelgen.framework.runtime.base import FakeRuntime
from kernelgen.agents.pr_submitter import (
    PRSubmitterAgent,
    PRSubmitterInput,
    PRSubmitterReport,
)

_VALID_OUTPUT = {
    "status": "SUBMITTED",
    "pr_url": "https://github.com/flagos-ai/FlagGems/pull/123",
    "branch": "kernelgen/kunlunxin/gelu",
    "files_changed": [
        "src/flag_gems/runtime/backend/_kunlunxin/ops/gelu.py",
        "src/flag_gems/runtime/backend/_kunlunxin/ops/__init__.py",
    ],
    "checks_passed": True,
    "summary": "Integrated gelu for kunlunxin, all static checks passed, opened draft PR #123.",
}

_BASE_INPUT = {
    "operator": "gelu",
    "kernel_code": "import torch\nimport triton\n@triton.jit\ndef gelu_kernel(): ...\n",
    "definition": {
        "name": "pytorch_gelu",
        "op_type": "pointwise",
        "reference": "import torch\ndef run(x, out): out.copy_(torch.nn.functional.gelu(x))\n",
    },
    "vendor": "kunlunxin",
    "geo_mean": 1.85,
}


def _rt(output=None):
    payload = json.dumps(output or _VALID_OUTPUT)
    return FakeRuntime([f"```json\n{payload}\n```"])


# --- I/O models -----------------------------------------------------------

def test_input_requires_operator_code_vendor():
    try:
        PRSubmitterInput()
    except Exception:
        pass
    else:
        raise AssertionError("operator, kernel_code, vendor are required")


def test_input_defaults():
    inp = PRSubmitterInput(
        operator="gelu",
        kernel_code="...",
        definition={"name": "pytorch_gelu"},
        vendor="kunlunxin",
    )
    assert inp.geo_mean is None
    assert inp.base_branch == "master"
    assert inp.target_repo == "flagos-ai/FlagGems"
    assert inp.draft is True


def test_output_model_validates():
    out = PRSubmitterReport.model_validate(_VALID_OUTPUT)
    assert out.status == "SUBMITTED"
    assert out.pr_url.endswith("/pull/123")
    assert out.checks_passed is True
    assert len(out.files_changed) == 2


def test_output_defaults():
    out = PRSubmitterReport(status="FAILED", summary="push failed")
    assert out.pr_url is None
    assert out.branch == ""
    assert out.files_changed == []
    assert out.checks_passed is False


# --- Agent.run via FakeRuntime --------------------------------------------

def test_run_happy_path():
    rt = _rt()
    out = PRSubmitterAgent().run(_BASE_INPUT, rt)
    assert isinstance(out, PRSubmitterReport)
    assert out.status == "SUBMITTED"
    assert out.branch == "kernelgen/kunlunxin/gelu"


def test_native_runtime_uses_canonical_agent_and_dynamic_task():
    class NativeFakeRuntime(FakeRuntime):
        supports_native_agents = True

    rt = NativeFakeRuntime([f"```json\n{json.dumps(_VALID_OUTPUT)}\n```"])
    PRSubmitterAgent().run(_BASE_INPUT, rt)
    call = rt.calls[0]
    assert call["agent"] == "kernel-pr-submitter"
    assert "You integrate a vendor-specialized GPU kernel" not in call["prompt"]
    assert "Operator: gelu" in call["prompt"]
    assert "Vendor (target chip): kunlunxin" in call["prompt"]
    assert "PR target: flagos-ai/FlagGems" in call["prompt"]


def test_prompt_contains_operator_and_vendor():
    rt = _rt()
    PRSubmitterAgent().run(_BASE_INPUT, rt)
    prompt = rt.calls[0]["prompt"]
    assert "gelu" in prompt
    assert "kunlunxin" in prompt
    assert "{{OPERATOR}}" not in prompt
    assert "{{VENDOR}}" not in prompt


def test_prompt_contains_vendor_path():
    rt = _rt()
    PRSubmitterAgent().run(_BASE_INPUT, rt)
    prompt = rt.calls[0]["prompt"]
    # should mention the vendor backend path
    assert "_kunlunxin/ops/" in prompt


def test_prompt_contains_target_repo():
    rt = _rt()
    PRSubmitterAgent().run(_BASE_INPUT, rt)
    prompt = rt.calls[0]["prompt"]
    assert "flagos-ai/FlagGems" in prompt
    assert "<TARGET_REPO>" not in prompt
    assert "<BASE_BRANCH>" not in prompt


def test_prompt_contains_kernel_code():
    rt = _rt()
    PRSubmitterAgent().run(_BASE_INPUT, rt)
    prompt = rt.calls[0]["prompt"]
    assert "gelu_kernel" in prompt
    assert "@triton.jit" in prompt


def test_prompt_contains_geo_mean():
    rt = _rt()
    PRSubmitterAgent().run(_BASE_INPUT, rt)
    assert "1.8500" in rt.calls[0]["prompt"]


def test_prompt_geo_mean_na_when_absent():
    rt = _rt()
    inp = dict(_BASE_INPUT)
    inp.pop("geo_mean")
    PRSubmitterAgent().run(inp, rt)
    assert "N/A" in rt.calls[0]["prompt"]


def test_prompt_contains_contract():
    rt = _rt()
    PRSubmitterAgent().run(_BASE_INPUT, rt)
    assert "FINAL REPORT CONTRACT" in rt.calls[0]["prompt"]


def test_prompt_no_aten_ops_field():
    """Vendor specialization does not need aten op registration."""
    rt = _rt()
    PRSubmitterAgent().run(_BASE_INPUT, rt)
    prompt = rt.calls[0]["prompt"]
    assert "aten_ops" not in prompt
    # operators.yaml is mentioned as "do NOT touch" — that's fine
    assert "_FULL_CONFIG" not in prompt  # no aten-level registration


def test_run_repairs_then_succeeds():
    rt = FakeRuntime([
        "not valid json",
        f"```json\n{json.dumps(_VALID_OUTPUT)}\n```",
    ])
    out = PRSubmitterAgent().run(_BASE_INPUT, rt)
    assert out.status == "SUBMITTED"
    assert len(rt.calls) == 2
    assert "FAILED VALIDATION" in rt.calls[1]["prompt"]


def test_run_bad_input_raises_before_invoke():
    rt = FakeRuntime([])
    try:
        PRSubmitterAgent().run({}, rt)
    except Exception:
        assert rt.calls == []
    else:
        raise AssertionError("missing required fields should raise")


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
