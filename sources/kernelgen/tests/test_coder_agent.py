"""Unit tests for CoderAgent (ADR-3 task #7).

Pure Python, no torch/GPU/LLM (FakeRuntime). The inner loop itself runs inside the
real LLM invoke (exercised end-to-end in docker); here we test the prompt assembly
and the CoderReport contract.

    cd /data/akg_kernel_bench_lite
    python -m pytest tests/test_coder_agent.py -v
    # or:
    python tests/test_coder_agent.py
"""

import json
import sys
from pathlib import Path


from kernelgen.framework import FakeRuntime  # noqa: E402
from kernelgen.agents.coder import (  # noqa: E402
    CoderAgent,
    CoderInput,
    CoderReport,
)

_BASE = dict(
    definition=dict(
        name="abl_t2_moe_topk_softmax",
        op_type="moe",
        axes={"input_0_d0": {"type": "const", "value": 1024}},
        inputs={"input_0": {"shape": [1024, 8], "dtype": "float32"}},
        outputs={"output_0": {"shape": [1024, 2], "dtype": "float32"},
                 "output_1": {"shape": [1024, 2], "dtype": "int64"}},
        reference="def run(input_0, output_0, output_1): ...",
    ),
    target_hardware="Ascend910B",
    evaluation_contract={
        "tolerance_mode": "fixed",
        "atol": 1e-2,
        "rtol": 1e-2,
        "required_matched_ratio": 1.0,
        "consider_reduced_precision": True,
    },
    ledger_dir="/wt/agent0",
    eval_server_url="http://localhost:8000",
    catalog_name="simple-v6-test",
)

_REPORT = json.dumps({"status": "PASSED", "summary": "fused vec kernel, 8.9x geo_mean"})


def _rt():
    return FakeRuntime([f"```json\n{_REPORT}\n```"])


def _assert_device_incident_order(prompt: str):
    incident = prompt[prompt.index(
        "If `eval_round` returns `SUSPECTED_DEVICE_ERROR`"
    ):]
    assert incident.index("First finalize") < incident.index(
        "call `get_server_status` exactly once"
    )
    assert incident.index("call `get_server_status` exactly once") < (
        incident.index("Then stop.")
    )


def test_run_returns_coder_report():
    rt = _rt()
    out = CoderAgent().run(_BASE, rt)
    assert isinstance(out, CoderReport)
    assert out.status == "PASSED"
    assert "8.9x" in out.summary


def test_input_rejects_removed_trace_fields():
    for legacy_field in ("trace_root", "trace_set_key"):
        try:
            CoderInput.model_validate({**_BASE, legacy_field: "legacy"})
        except ValueError as exc:
            assert legacy_field in str(exc)
        else:
            raise AssertionError(f"removed field was accepted: {legacy_field}")


def test_prompt_has_role_loop_and_tools():
    rt = _rt()
    CoderAgent().run(_BASE, rt)
    p = rt.calls[0]["prompt"]
    assert "LOOP FOREVER" in p                       # role md (AutoKernel style)
    assert "mcp__kernelgen__get_server_status" in p
    assert "exactly once at the beginning of each" in p
    assert "For every target-environment fact needed for an implementation decision" in p
    assert "Target environment introspection" in p
    assert "Precise candidate diagnosis" in p
    assert "rounding boundary that introduces a numerical error" in p
    assert "Never inspect or rely on the Agent host's" in p
    assert "mcp__kernelgen__preflight_kernel" in p
    assert "mcp__kernelgen__submit_debug_job" in p
    assert "mcp__kernelgen__get_debug_job" not in p
    assert "mcp__kernelgen__cancel_debug_job" not in p
    assert "mcp__kernelgen__eval_round" in p
    assert "mcp__kernelgen__finalize_round" in p
    assert "$kernelgen-knowledge" not in p
    assert "query_knowledge" not in p
    assert "query_sources" not in p
    assert "workspace-relative path `tmp/main.py`" in p
    assert "virtual and is not visible to MCP tools" in p
    assert "mcp__kernelgen__next" not in p
    assert "`finalize_round` returns the persisted CONTINUE/STOP verdict" in p
    assert "candidate_action" in p
    assert "timing_skipped" in p
    assert "CORRECTNESS_FAILED" in p
    assert 'status: "TIMEOUT"' in p
    assert "SUSPECTED_DEVICE_ERROR" in p
    assert "do not claim that a card is broken" in p
    assert "scheduler.broken > 0" in p
    _assert_device_incident_order(p)
    assert "<implementation_profile>" in p
    assert "Language: triton" in p
    assert "gluon" not in p.lower()
    assert "knowledge_assessments" in p
    assert "required submission target is Ascend910B2" in p

    assert "<evaluation_contract>" in p
    assert "0.01 + 0.01 * |reference|" in p
    assert "FP16, BF16" in p
    assert "conversion costs" in p

def test_ascend_core_count_rule_is_target_scoped():
    ascend_rt = _rt()
    CoderAgent().run(
        dict(_BASE, target_hardware="Ascend910B2"),
        ascend_rt,
    )
    ascend_prompt = ascend_rt.calls[0]["prompt"]
    assert "<target_compatibility>" in ascend_prompt
    assert "num_aicore" in ascend_prompt
    assert "num_vectorcore" in ascend_prompt
    assert "tl.num_programs(0)" in ascend_prompt

    a100_rt = _rt()
    CoderAgent().run(dict(_BASE, target_hardware="A100"), a100_rt)
    assert "<target_compatibility>" not in a100_rt.calls[0]["prompt"]


def test_prompt_hides_tool_config_and_has_definition():
    rt = _rt()
    CoderAgent().run(_BASE, rt)
    p = rt.calls[0]["prompt"]
    assert "/wt/agent0" not in p
    assert "http://localhost:8000" not in p
    assert "abl_t2_moe_topk_softmax" in p
    assert "run() MUST have exactly 3 positional parameters" in p
    assert "positional count does not authorize stripping defaults" in p


def test_prompt_contains_ascend_b4_to_b2_portability_rules():
    rt = _rt()
    CoderAgent().run(dict(_BASE, target_hardware="Ascend910B4-1"), rt)
    prompt = rt.calls[0]["prompt"]
    assert "<target_compatibility>" in prompt
    assert "required submission target is Ascend910B2" in prompt
    assert "num_aicore" in prompt
    assert "num_vectorcore" in prompt
    assert "B2 no-regression" not in prompt


def test_prompt_contains_metax_c550_to_c500_portability_rules():
    rt = _rt()
    CoderAgent().run(dict(_BASE, target_hardware="MetaX C550"), rt)
    prompt = rt.calls[0]["prompt"]
    assert "<target_compatibility>" in prompt
    assert "MetaX C500 portability requirement" in prompt
    assert "device family in dispatch and autotune-cache keys" in prompt
    assert "C500 no-regression" not in prompt


def test_prompt_omits_target_compatibility_rules_for_a100():
    rt = _rt()
    CoderAgent().run(dict(_BASE, target_hardware="A100"), rt)
    assert "<target_compatibility>" not in rt.calls[0]["prompt"]


def test_dps_flag_off():
    inp = dict(_BASE, destination_passing_style=False)
    rt = _rt()
    CoderAgent().run(inp, rt)
    prompt = rt.calls[0]["prompt"]
    assert "run() MUST have exactly 1 positional parameter" in prompt
    assert "Evaluation mode: value-returning (no DPS)" in prompt


def test_catalog_signature_preserves_keyword_only_abi_in_prompt():
    definition = dict(
        _BASE["definition"],
        run_signature="(self, mat1, mat2, *, beta=1, alpha=1)",
    )
    rt = _rt()

    CoderAgent().run(dict(_BASE, definition=definition), rt)

    prompt = rt.calls[0]["prompt"]
    assert (
        "run() MUST use this exact Python signature: "
        "run(self, mat1, mat2, *, beta=1, alpha=1)"
    ) in prompt
    assert "5 positional parameters" not in prompt


def test_native_runtime_selects_coder_and_omits_inline_role():
    class _NativeFakeRuntime(FakeRuntime):
        supports_native_agents = True

    rt = _NativeFakeRuntime([f"```json\n{_REPORT}\n```"])
    CoderAgent().run(_BASE, rt)
    assert rt.calls[0]["agent"] == "kernel-coder"
    assert "LOOP FOREVER" not in rt.calls[0]["prompt"]
    assert "abl_t2_moe_topk_softmax" in rt.calls[0]["prompt"]


def test_runtime_managed_role_selects_coder_and_omits_inline_role():
    class _RoleRuntime(FakeRuntime):
        supports_agent_roles = True

    rt = _RoleRuntime([f"```json\n{_REPORT}\n```"])
    CoderAgent().run(_BASE, rt)

    assert rt.calls[0]["agent"] == "kernel-coder"
    assert "LOOP FOREVER" not in rt.calls[0]["prompt"]


def test_knowledge_prompt_has_bounded_on_demand_dual_retrieval():
    inp = dict(_BASE, knowledge_enabled=True)
    rt = _rt()

    CoderAgent().run(inp, rt)

    prompt = rt.calls[0]["prompt"]
    assert "$kernelgen-knowledge" in prompt
    assert "On-demand Dual Retrieval" in prompt
    assert "Retrieval is independent of ExperimentPlan kind" in prompt
    assert "one `query_knowledge` call" in prompt
    assert "one `query_sources` call" in prompt
    assert "`max_results=4`" in prompt
    assert "at most 200 lines" in prompt
    assert "carry that use forward without querying again" in prompt
    assert "partially_confirmed" in prompt
    assert "combination-level evidence" in prompt
    assert "do not claim independent causality" in prompt
    _assert_device_incident_order(prompt)


def test_native_runtime_selects_knowledge_coder_when_enabled():
    class _NativeFakeRuntime(FakeRuntime):
        supports_native_agents = True

    rt = _NativeFakeRuntime([f"```json\n{_REPORT}\n```"])

    CoderAgent().run(dict(_BASE, knowledge_enabled=True), rt)

    assert rt.calls[0]["agent"] == "kernel-knowledge-coder"
    assert "On-demand Dual Retrieval" not in rt.calls[0]["prompt"]


def test_continue_session_preserves_knowledge_coder_role():
    class _ResumableNativeRuntime:
        supports_native_agents = True
        last_session_id = "session-1"

        def __init__(self):
            self.calls = []

        def resume(
            self,
            prompt,
            *,
            model="inherit",
            agent=None,
            session_id=None,
        ):
            self.calls.append({
                "prompt": prompt,
                "model": model,
                "agent": agent,
                "session_id": session_id,
            })
            return _REPORT

    rt = _ResumableNativeRuntime()
    out = CoderAgent().continue_session(
        "continue after round 1",
        rt,
        knowledge_enabled=True,
    )

    assert out.status == "PASSED"
    assert rt.calls[0]["agent"] == "kernel-knowledge-coder"
    assert rt.calls[0]["prompt"] == "continue after round 1"


def test_analysis_and_seed_and_history_injected():
    inp = dict(_BASE,
               analysis={"triton_grid": "grid=(40,)", "key_pitfalls": ["p1"]},
               seed_code="import torch\nimport triton\n\n\ndef run():\n    pass  # seed",
               history_summary="Round 1: RUNTIME_ERROR ...")
    rt = _rt()
    CoderAgent().run(inp, rt)
    p = rt.calls[0]["prompt"]
    assert "grid=(40,)" in p                          # analysis json
    assert "# seed" in p                              # seed code
    assert (
        "<seed_code>\n"
        "Start from this best-so-far solution; improve on it:\n"
        "import torch\n"
        "import triton\n\n\n"
        "def run():\n"
        "    pass  # seed\n"
        "</seed_code>"
    ) in p
    assert "Round 1: RUNTIME_ERROR" in p              # history
    assert "<final_report_contract>" not in p         # contract is plain-delimited
    assert "FINAL REPORT CONTRACT" in p


def test_validated_baseline_seed_requires_unchanged_eval_and_profile_first():
    rt = _rt()
    CoderAgent().run(
        dict(
            _BASE,
            seed_code="def run(input):\n    return input",
            seed_is_validated_baseline=True,
        ),
        rt,
    )

    prompt = rt.calls[0]["prompt"]
    assert "externally validated Native baseline" in prompt
    assert "materialized byte-for-byte at tmp/main.py" in prompt
    assert "Before any code change" in prompt
    assert "Do not use Write or Edit before the baseline eval" in prompt
    assert "complete the required profile analysis" in prompt
    assert "def run(input):\n    return input" in prompt


def test_optional_reference_code_is_injected_with_adaptation_boundaries():
    source = (
        "@triton.jit\n"
        "def source_kernel(x, BLOCK: tl.constexpr):\n"
        "    # CUDA-only tuning assumption\n"
        "    return\n"
    )
    rt = _rt()
    CoderAgent().run(
        dict(_BASE, reference_code_source=source),
        rt,
    )
    prompt = rt.calls[0]["prompt"]

    assert "<reference_code>" in prompt
    assert source in prompt
    assert "optional, read-only design evidence" in prompt
    assert "may fail compilation or correctness" in prompt
    assert "Definition, workloads, reference semantics, and validator are" in prompt
    assert "Do not treat this source as the initial candidate" in prompt
    assert "hardware assumptions are\n  not transferable" in prompt
    assert "normal remote" in prompt
    assert "preflight and eval\n  lifecycle" in prompt


def test_reference_code_prompt_describes_provenance_within_boundaries():
    source = "@triton.jit\ndef source_kernel(x):\n    return\n"
    guidance = (
        "This source ran on Ascend 910B4 under the historical native evaluator.\n"
        "Reuse its tiling idea, but revalidate the current public ABI."
    )
    rt = _rt()

    CoderAgent().run(
        dict(
            _BASE,
            reference_code_source=source,
            reference_code_prompt=guidance,
        ),
        rt,
    )

    prompt = rt.calls[0]["prompt"]
    assert "<reference_code_guidance>" in prompt
    assert guidance in prompt
    assert "historical speedups are not comparable" in prompt
    assert "Never modify or replace Torch/FlagGems APIs" in prompt
    reference_start = prompt.index("<reference_code>")
    assert prompt.index("<reference_code_guidance>", reference_start) < (
        prompt.index("\n<source>\n", reference_start)
    )


def test_reference_code_prompt_requires_source():
    try:
        CoderInput.model_validate(
            dict(_BASE, reference_code_prompt="hardware provenance")
        )
    except ValueError as exc:
        assert "reference_code_prompt requires reference_code_source" in str(exc)
    else:
        raise AssertionError("reference guidance without source was accepted")


def test_workload_literals_are_injected_as_authoritative_values():
    inp = dict(
        _BASE,
        definition={
            **_BASE["definition"],
            "inputs": {
                "input_0": {"shape": [1024, 8], "dtype": "float32"},
                "approximate": {"shape": None, "dtype": "dynamic"},
            },
            "reference": "def run(input_0, approximate): ...",
        },
        workloads=[
            {
                "phase": "correctness",
                "uuid": "gelu-tanh",
                "axes": {},
                "inputs": {
                    "input_0": {"type": "random"},
                    "approximate": {"type": "literal", "value": "tanh"},
                },
            }
        ],
    )
    rt = _rt()
    CoderAgent().run(inp, rt)
    prompt = rt.calls[0]["prompt"]
    assert "approximate: non-tensor (dynamic)" in prompt
    assert '"value": "tanh"' in prompt
    assert "authoritative evaluation inputs" in prompt
    assert "does not mean its runtime value is Python None" in prompt


def test_prompt_samples_at_most_ten_workloads_across_phases():
    workloads = [
        {
            "phase": phase,
            "uuid": f"{phase}-{index}",
            "axes": {},
            "inputs": {
                "input_0": {
                    "type": "random",
                    "shape": [index + 1, 8],
                    "dtype": "float32",
                }
            },
        }
        for phase in ("correctness", "timing")
        for index in range(20)
    ]
    rt = _rt()

    CoderAgent().run(dict(_BASE, workloads=workloads), rt)

    prompt = rt.calls[0]["prompt"]
    workload_json = prompt.split("<workloads>\n", 1)[1].split(
        "\n</workloads>", 1
    )[0]
    shown = json.loads(workload_json[workload_json.index("["):])
    assert len(shown) == 10
    assert [item["phase"] for item in shown].count("correctness") == 5
    assert [item["phase"] for item in shown].count("timing") == 5
    assert shown[0]["uuid"] == "correctness-0"
    assert shown[4]["uuid"] == "correctness-19"
    assert shown[5]["uuid"] == "timing-0"
    assert shown[9]["uuid"] == "timing-19"
    assert "Showing 10 deterministically sampled workloads out of 40 total" in prompt
    assert "Server still evaluates the full authoritative workload set" in prompt
    shown_uuids = {item["uuid"] for item in shown}
    assert "correctness-1" not in shown_uuids
    assert "timing-1" not in shown_uuids


def test_cold_start_omits_optional_blocks():
    rt = _rt()
    CoderAgent().run(_BASE, rt)  # no analysis/seed/history
    p = rt.calls[0]["prompt"]
    assert "<analysis>" not in p
    assert "<workloads>" not in p
    assert "<reference_code>" not in p
    assert "<seed_code>" not in p
    assert "<optimization_history>" not in p


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


def test_gems_protection_uses_catalog_evaluator_not_framework_or_name(monkeypatch):
    import kernelgen.agents.coder as module
    for evaluator in ("native", "flaggems"):
        monkeypatch.setattr(module, "load_catalog_manifest", lambda name, kind=evaluator: {
            "name": "flaggems-native", "framework": "flaggems", "evaluator": kind,
        })
        rt = _rt()
        CoderAgent().run({**_BASE, "catalog_name": "flaggems-native"}, rt)
        prompt = rt.calls[0]["prompt"]
        assert ("Gems adapter protection:" in prompt) == (evaluator == "flaggems")
        assert "never modify Torch source/registrations" in prompt
        assert "compiler/JIT internals" in prompt


def test_native_snapshot_omits_gems_protection_without_reloading_catalog(monkeypatch):
    import kernelgen.agents.coder as module
    def unexpected(name):
        raise AssertionError("native snapshot must not re-resolve evaluator")
    monkeypatch.setattr(module, "load_catalog_manifest", unexpected)
    rt = _rt()
    CoderAgent().run({**_BASE, "evaluation_snapshot": {"catalog_name": "flaggems-native"}}, rt)
    assert "Gems adapter protection:" not in rt.calls[0]["prompt"]
