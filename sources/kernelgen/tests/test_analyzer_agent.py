"""Unit tests for AnalyzerAgent (ADR-3 task #6).

Pure Python, no torch/GPU/LLM (uses FakeRuntime):

    cd /data/akg_kernel_bench_lite
    python -m pytest tests/test_analyzer_agent.py -v
    # or:
    python tests/test_analyzer_agent.py
"""

import json
import sys
from pathlib import Path


from kernelgen.framework import FakeRuntime  # noqa: E402
from kernelgen.agents.analyzer import (  # noqa: E402
    AnalyzerAgent,
    AnalyzerInput,
    AnalyzerOutput,
)

_BASE_INP = dict(
    target_hardware="Ascend910B",
    evaluation_contract={
        "tolerance_mode": "fixed",
        "atol": 1e-2,
        "rtol": 1e-2,
        "required_matched_ratio": 1.0,
        "consider_reduced_precision": True,
    },
    definition=dict(
        name="abl_t2_moe_topk_softmax",
        op_type="moe",
        axes={"input_0_d0": {"type": "const", "value": 1024}},
        inputs={"input_0": {"shape": [1024, 8], "dtype": "float32"}},
        outputs={"output_0": {"shape": [1024, 2], "dtype": "float32"},
                 "output_1": {"shape": [1024, 2], "dtype": "int64"}},
        reference="def run(input_0, output_0, output_1): ...",
    ),
)

_VALID_OUTPUT = {
    "core_math": "probs = softmax; topk + renorm",
    "program_grid": "grid = (VEC_CORE_NUM,) = (40,)",
    "inner_loop": "interleaved loop over tokens",
    "softmax_strategy": "online 3-pass",
    "data_layout": "contiguous [1024,8] float32",
    "key_pitfalls": ["DPS: run() must have 3 params", "use tl.argmax not tl.sort"],
    "workload_dispatch_strategy": "single fused kernel, 40 VEC cores",
}


def _rt(output=None):
    payload = json.dumps(output or _VALID_OUTPUT)
    return FakeRuntime([f"```json\n{payload}\n```"])


def test_cold_start_run_returns_output_model():
    rt = _rt()
    out = AnalyzerAgent().run(_BASE_INP, rt)
    assert isinstance(out, AnalyzerOutput)
    assert out.program_grid == "grid = (VEC_CORE_NUM,) = (40,)"
    assert len(out.key_pitfalls) == 2


def test_prompt_contains_role_definition_contract_tool():
    rt = _rt()
    AnalyzerAgent().run(_BASE_INP, rt)
    prompt = rt.calls[0]["prompt"]
    assert "GPU kernel architect" in prompt        # role .md loaded
    assert "abl_t2_moe_topk_softmax" in prompt    # definition injected
    assert "--- OUTPUT CONTRACT ---" in prompt     # contract injected
    assert "$kernelgen-knowledge" in prompt        # current knowledge skill
    assert 'phase: "initial"' in prompt             # Analyzer query context
    assert "<implementation_profile>" in prompt
    assert "Language: triton" in prompt
    assert "<evaluation_contract>" in prompt
    assert "0.01 + 0.01 * |reference|" in prompt
    assert "FP16, BF16" in prompt
    assert "eval_round is authoritative" in prompt
    assert "gluon" not in prompt.lower()


def test_native_runtime_selects_agent_and_omits_inline_role():
    class _NativeFakeRuntime(FakeRuntime):
        supports_native_agents = True

    rt = _NativeFakeRuntime([f"```json\n{json.dumps(_VALID_OUTPUT)}\n```"])
    AnalyzerAgent().run(_BASE_INP, rt)

    call = rt.calls[0]
    assert call["agent"] == "kernel-analyzer"
    assert "GPU kernel architect" not in call["prompt"]
    assert "abl_t2_moe_topk_softmax" in call["prompt"]
    assert "--- OUTPUT CONTRACT ---" in call["prompt"]


def test_prompt_contains_hardware():
    rt = _rt()
    AnalyzerAgent().run(_BASE_INP, rt)
    prompt = rt.calls[0]["prompt"]
    assert "Ascend910B" in prompt
    assert "required submission target is Ascend910B2" in prompt


def test_prompt_contains_ascend_b4_to_b2_portability_rules():
    rt = _rt()
    AnalyzerAgent().run(
        dict(_BASE_INP, target_hardware="Ascend910B4-1"),
        rt,
    )
    prompt = rt.calls[0]["prompt"]
    assert "<target_compatibility>" in prompt
    assert "required submission target is Ascend910B2" in prompt
    assert "tl.num_programs(0)" in prompt


def test_prompt_contains_metax_c550_to_c500_portability_rules():
    rt = _rt()
    AnalyzerAgent().run(dict(_BASE_INP, target_hardware="MetaX C550"), rt)
    prompt = rt.calls[0]["prompt"]
    assert "<target_compatibility>" in prompt
    assert "MetaX C500 portability requirement" in prompt
    assert "C550 autotune winner" in prompt


def test_prompt_omits_target_compatibility_rules_for_a100():
    rt = _rt()
    AnalyzerAgent().run(dict(_BASE_INP, target_hardware="A100"), rt)
    assert "<target_compatibility>" not in rt.calls[0]["prompt"]


def test_ascend_core_count_rule_is_target_scoped():
    ascend_rt = _rt()
    AnalyzerAgent().run(
        dict(_BASE_INP, target_hardware="Ascend910B3"),
        ascend_rt,
    )
    ascend_prompt = ascend_rt.calls[0]["prompt"]
    assert "<target_compatibility>" in ascend_prompt
    assert "num_aicore" in ascend_prompt
    assert "num_vectorcore" in ascend_prompt
    assert "tl.num_programs(0)" in ascend_prompt

    a100_rt = _rt()
    AnalyzerAgent().run(dict(_BASE_INP, target_hardware="A100"), a100_rt)
    assert "<target_compatibility>" not in a100_rt.calls[0]["prompt"]


def test_catalog_signature_preserves_keyword_only_abi_in_prompt():
    definition = dict(
        _BASE_INP["definition"],
        run_signature="(self, mat1, mat2, *, beta=1, alpha=1)",
    )
    rt = _rt()

    AnalyzerAgent().run(
        dict(_BASE_INP, definition=definition),
        rt,
    )

    prompt = rt.calls[0]["prompt"]
    assert "Exact run ABI:" in prompt
    assert "run(self, mat1, mat2, *, beta=1, alpha=1)" in prompt
    assert "5 positional parameters" not in prompt


def test_definition_block_indents_every_multiline_value():
    definition = dict(
        _BASE_INP["definition"],
        axes={
            "batch_size": {"type": "const", "value": 16},
            "num_q_heads": {"type": "const", "value": 128},
        },
        reference="import torch\ndef run(x):\n    return x",
    )
    rt = _rt()
    AnalyzerAgent().run(
        dict(_BASE_INP, definition=definition),
        rt,
    )
    block = rt.calls[0]["prompt"].split("<definition>", 1)[1].split(
        "</definition>",
        1,
    )[0]

    assert "\nAxes:\n  batch_size: const = 16\n  num_q_heads: const = 128" in block
    assert (
        "\nReference Implementation:\n"
        "  import torch\n"
        "  def run(x):\n"
        "      return x"
    ) in block


def test_defaults_for_optional_fields():
    out = AnalyzerOutput.model_validate(_VALID_OUTPUT)
    assert out.head_mapping == "N/A"
    assert out.masking == "N/A"
    assert out.lse_formula == "N/A"
    assert out.memory_traffic_analysis == {}
    assert out.numerical_strategy == ""
    assert out.reduced_precision_candidates == []


def test_repair_retry_on_bad_json():
    rt = FakeRuntime(["not json at all",
                      f"```json\n{json.dumps(_VALID_OUTPUT)}\n```"])
    out = AnalyzerAgent().run(_BASE_INP, rt)
    assert out.program_grid == _VALID_OUTPUT["program_grid"]
    assert len(rt.calls) == 2
    assert "FAILED VALIDATION" in rt.calls[1]["prompt"]


def test_definition_requires_name():
    inp = dict(target_hardware="Ascend910B",
               definition={"op_type": "moe"})   # no "name"
    try:
        AnalyzerAgent().run(inp, _rt())
    except Exception:
        pass
    else:
        raise AssertionError("definition without name should fail validation")


def test_dps_derived_from_definition():
    # dps_param_count = len(inputs)+len(outputs), derived by DefinitionModel property
    rt = _rt()
    AnalyzerAgent().run(_BASE_INP, rt)
    # base has 1 input + 2 outputs = 3
    assert "run() MUST have exactly 3 positional" in rt.calls[0]["prompt"]


def test_legacy_triton_grid_input_remains_readable():
    out = AnalyzerOutput.model_validate({"triton_grid": "legacy-grid"})
    assert out.program_grid == "legacy-grid"
    assert out.triton_grid == "legacy-grid"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
