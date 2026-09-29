"""Host tests for AutoGenAgent + OptimizeAgent + OptimizeWorkflow.

Pure Python, no LLM/GPU — uses FakeRuntime + Directory.

    cd /data/akg_kernel_bench_lite
    PYTHONPATH=. python3 kernelgen/tests/test_auto_gen.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernelgen.agents.auto_gen import AutoGenAgent, AutoGenInput, AutoGenOutput
from kernelgen.agents.optimize import OptimizeAgent, OptimizeInput, OptimizeOutput
from kernelgen.framework.runtime.base import FakeRuntime
from kernelgen.framework.parallel import Directory, run_parallel
from kernelgen.workflows.optimize import OptimizeWorkflow, OptimizeWorkflowInput, OptimizeWorkflowOutput


# ---------------------------------------------------------------------------
# AutoGenAgent tests
# ---------------------------------------------------------------------------

def _fake_autogen_output(operator="relu", status="success"):
    return json.dumps({
        "operator": operator,
        "status": status,
        "accuracy_passed": True,
        "error_message": None,
        "files_created": [f"src/flag_gems/ops/{operator}.py"],
        "files_modified": ["src/flag_gems/ops/__init__.py"],
        "aten_ops_registered": [f"aten.{operator}.default"],
        "implementation_mode": "pointwise_dynamic",
        "test_results": {"total": 8, "passed": 8, "failed": 0},
        "benchmark_results": {},
        "notes": "",
    })


def test_autogen_input_model():
    inp = AutoGenInput(operator="relu")
    assert inp.operator == "relu"


def test_autogen_output_model():
    data = json.loads(_fake_autogen_output("silu"))
    out = AutoGenOutput.model_validate(data)
    assert out.operator == "silu"
    assert out.accuracy_passed is True


def test_autogen_preprocess_template():
    agent = AutoGenAgent()
    inp = AutoGenInput(operator="sigmoid")
    prompt = agent.preprocess(inp, None)
    assert "{{OPERATOR}}" not in prompt
    assert "sigmoid" in prompt
    assert "资深 FlagGems 与 Triton 算子实现工程师" in prompt


def test_autogen_preprocess_contract():
    agent = AutoGenAgent()
    inp = AutoGenInput(operator="abs")
    prompt = agent.preprocess(inp, None)
    assert "operator" in prompt
    assert "status" in prompt


def test_autogen_run():
    rt = FakeRuntime([_fake_autogen_output("relu")])
    out = AutoGenAgent().run({"operator": "relu"}, rt)
    assert isinstance(out, AutoGenOutput)
    assert out.operator == "relu"
    assert out.status == "success"


def test_autogen_native_agent_keeps_static_role_out_of_prompt():
    class NativeFakeRuntime(FakeRuntime):
        supports_native_agents = True

    rt = NativeFakeRuntime([_fake_autogen_output("relu")])
    AutoGenAgent().run({"operator": "relu"}, rt)
    call = rt.calls[0]
    assert call["agent"] == "kernel-auto-gen"
    assert "资深 FlagGems 与 Triton 算子实现工程师" not in call["prompt"]
    assert '"operator": "relu"' in call["prompt"]


def test_autogen_run_failed():
    rt = FakeRuntime([json.dumps({"operator": "bad", "status": "failed", "error_message": "err"})])
    out = AutoGenAgent().run({"operator": "bad"}, rt)
    assert out.status == "failed"
    assert out.error_message == "err"


def test_autogen_bind():
    factory = lambda path: FakeRuntime([_fake_autogen_output("ceil")])
    agent = AutoGenAgent.bind("/tmp/ws", factory)
    out = agent.run({"operator": "ceil"})
    assert out.operator == "ceil"


def test_autogen_run_parallel(tmp_path):
    """run_parallel with AutoGenAgent — no Workflow needed."""
    inputs = [{"operator": "relu"}, {"operator": "gelu"}]

    def factory(path):
        # Determine which operator from path name
        if "task0" in str(path):
            return FakeRuntime([_fake_autogen_output("relu")])
        return FakeRuntime([_fake_autogen_output("gelu")])

    ws = Directory(tmp_path)
    results = run_parallel(AutoGenAgent, inputs, workspace=ws, runtime_factory=factory, max_workers=2)
    assert len(results) == 2
    ops = {r.operator for r, _ in results}
    assert "relu" in ops or "gelu" in ops


# ---------------------------------------------------------------------------
# OptimizeAgent tests
# ---------------------------------------------------------------------------

def _fake_optimize_output(operator="softmax", speedup=1.2, test_passed=True):
    return json.dumps({
        "operator": operator,
        "status": "success",
        "speedup": speedup,
        "test_passed": test_passed,
        "kernel_path": f"src/flag_gems/ops/{operator}.py",
        "optimization_direction": "vectorize inner loop",
        "error": None,
    })


def test_optimize_input_model():
    inp = OptimizeInput(operator="softmax")
    assert inp.operator == "softmax"


def test_optimize_output_model():
    data = json.loads(_fake_optimize_output())
    out = OptimizeOutput.model_validate(data)
    assert out.speedup == 1.2
    assert out.test_passed is True


def test_optimize_preprocess_template():
    agent = OptimizeAgent()
    inp = OptimizeInput(operator="layernorm")
    prompt = agent.preprocess(inp, None)
    assert "{{OPERATOR}}" not in prompt
    assert "layernorm" in prompt
    assert "资深 FlagGems 与 Triton 性能优化工程师" in prompt


def test_optimize_run():
    rt = FakeRuntime([_fake_optimize_output("softmax", 1.5)])
    out = OptimizeAgent().run({"operator": "softmax"}, rt)
    assert out.operator == "softmax"
    assert out.speedup == 1.5


def test_optimize_native_agent_keeps_static_role_out_of_prompt():
    class NativeFakeRuntime(FakeRuntime):
        supports_native_agents = True

    rt = NativeFakeRuntime([_fake_optimize_output("softmax", 1.5)])
    OptimizeAgent().run({"operator": "softmax"}, rt)
    call = rt.calls[0]
    assert call["agent"] == "kernel-optimize"
    assert "资深 FlagGems 与 Triton 性能优化工程师" not in call["prompt"]
    assert '"operator": "softmax"' in call["prompt"]


# ---------------------------------------------------------------------------
# OptimizeWorkflow tests
# ---------------------------------------------------------------------------

def test_optimize_workflow_target_reached(tmp_path):
    """Workflow stops early when target speedup is reached."""
    def factory(path):
        # 2 replies: first round 0.5, second round 1.0 (reaches target 0.9)
        return FakeRuntime([
            _fake_optimize_output("softmax", 0.5),
            _fake_optimize_output("softmax", 1.0),
        ])

    wf = OptimizeWorkflow(cwd=str(tmp_path), runtime_factory=factory)
    out = wf.run({"operator": "softmax", "max_iters": 10, "target_speedup": 0.9})

    assert isinstance(out, OptimizeWorkflowOutput)
    assert out.target_reached is True
    assert out.iterations_run == 2
    assert out.best_speedup == 1.0


def test_optimize_workflow_max_iters(tmp_path):
    """Workflow stops at max_iters if target not reached."""
    def factory(path):
        # 3 replies for 3 iterations
        return FakeRuntime([
            _fake_optimize_output("softmax", 0.3),
            _fake_optimize_output("softmax", 0.3),
            _fake_optimize_output("softmax", 0.3),
        ])

    wf = OptimizeWorkflow(cwd=str(tmp_path), runtime_factory=factory)
    out = wf.run({"operator": "softmax", "max_iters": 3, "target_speedup": 2.0})

    assert out.target_reached is False
    assert out.iterations_run == 3
    assert out.best_speedup == 0.3


def test_optimize_workflow_versions_saved(tmp_path):
    """Workflow creates versions/ and PERFORMANCE.md."""
    def factory(path):
        # 2 replies for 2 iterations
        return FakeRuntime([
            _fake_optimize_output("softmax", 0.8),
            _fake_optimize_output("softmax", 0.8),
        ])

    wf = OptimizeWorkflow(cwd=str(tmp_path), runtime_factory=factory)
    wf.run({"operator": "softmax", "max_iters": 2, "target_speedup": 5.0})

    assert (tmp_path / "versions" / "v1" / "meta.json").exists()
    assert (tmp_path / "versions" / "v2" / "meta.json").exists()
    assert (tmp_path / "PERFORMANCE.md").exists()
    perf = (tmp_path / "PERFORMANCE.md").read_text()
    assert "v1" in perf
    assert "0.8000" in perf


def test_optimize_workflow_bind(tmp_path):
    """Workflow.bind works for run_parallel."""
    def factory(path):
        return FakeRuntime([_fake_optimize_output("abs", 1.5)])

    wf = OptimizeWorkflow.bind(str(tmp_path), factory)
    out = wf.run({"operator": "abs", "max_iters": 1, "target_speedup": 1.0})
    assert out.target_reached is True


def test_optimize_workflow_run_parallel(tmp_path):
    """Multiple operators optimized in parallel via run_parallel(OptimizeWorkflow)."""
    def factory(path):
        return FakeRuntime([_fake_optimize_output("op", 1.0)])

    ws = Directory(tmp_path)
    inputs = [
        {"operator": "softmax", "max_iters": 1, "target_speedup": 0.5},
        {"operator": "layernorm", "max_iters": 1, "target_speedup": 0.5},
    ]
    results = run_parallel(OptimizeWorkflow, inputs, workspace=ws, runtime_factory=factory, max_workers=2)
    assert len(results) == 2
    for result, ws_name in results:
        assert result.target_reached is True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            if "tmp_path" in t.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as td:
                    t(Path(td))
            else:
                t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
