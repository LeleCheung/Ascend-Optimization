"""Unit tests for run_parallel + Workspace + KernelGenWorkflow (ADR-3 #9a).

Host-testable with Directory workspace + FakeRuntime (no torch/GPU/LLM). Verifies
analyze once -> N isolated Coders -> authoritative selection and epoch recovery.
Runtime fixtures write real measured ledgers; production readers are not replaced.

    cd /data/akg_kernel_bench_lite
    python -m pytest tests/test_kernel_gen.py -v
    # or:
    python tests/test_kernel_gen.py
"""

import json
import hashlib
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace


import pytest
from pydantic import BaseModel  # noqa: E402

import kernelgen.framework.parallel as parallel_module
import kernelgen.workflows.optimization.kernelgen.finalization as finalization_module
import kernelgen.workflows.optimization.kernelgen.epoch as epoch_module
import kernelgen.workflows.optimization.kernelgen.knowledge as knowledge_module
import kernelgen.workflows.optimization.kernelgen.preparation as preparation_module
import kernelgen.workflows.optimization.kernelgen.recovery as recovery_module
from kernelgen.workflows.optimization.kernelgen.contracts import EpochResult
import kernelgen.workflows.knowledge_bridge as knowledge_bridge_module
from kernelgen.data.target_context import build_target_context
from kernelgen.framework import (  # noqa: E402
    Runnable,
    Workflow,
    Workspace,
    Directory,
    ParallelExecutionError,
    run_parallel,
    BaseAgent,
    FakeRuntime,
)
from kernelgen.data.ledger import Ledger
from kernelgen.data.stop_policy import StopConfig
from kernelgen.tests.helpers import experiment_plan, round_conclusion
from kernelgen.data.implementation import ImplementationLanguage
from kernelgen.data.trace import load_catalog_optimization_context
from kernelgen.framework.models import DefinitionModel
from kernelgen.framework.parallel import ParallelTaskFailure
from kernelgen.framework.run_control import (
    RunCancelled,
    RunState,
    WorkspaceRunControl,
)
from kernelgen.knowledge.config import KnowledgeConfig, KnowledgeReviewerMode
from kernelgen.workflows.optimization.kernelgen import KernelGenWorkflow, KernelGenInput, KernelGenOutput
from kernelgen.agents.coder import CoderReport
from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationWorkflow
from kernelgen_server import builtin_catalog_path


@pytest.fixture(autouse=True)
def _stub_eval_service_target(monkeypatch):
    """Keep orchestration tests host-only while satisfying the target contract."""
    from kernelgen.workflows.optimization.single_coder import workflow as optimize_module
    # This suite simulates device measurements; test_retest exercises the real
    # verifier and independently checks that epoch selection rejects missing evidence.
    monkeypatch.setattr(optimize_module, "verify_final_best", lambda workspace, **kw: _test_verification(workspace))

    def resolve_target_context(self, inp):
        backend = (
            "ascend"
            if "ascend" in inp.target_hardware.lower()
            or "910" in inp.target_hardware.lower()
            else "cuda"
        )
        return build_target_context(
            target_hardware=inp.target_hardware,
            implementation_language=inp.implementation_language.value,
            service_status={
                "backend": backend,
                "target": {
                    "backend": backend,
                    "device": inp.target_hardware,
                },
            },
        )

    monkeypatch.setattr(
        SingleCoderOptimizationWorkflow,
        "_resolve_target_context",
        resolve_target_context,
    )
    monkeypatch.setattr(
        preparation_module,
        "resolve_server_target",
        lambda inp: (inp, {}),
    )

    real_load = load_catalog_optimization_context

    def load_test_context(root, definition_name):
        if definition_name == "abl_t1_gelu":
            return DefinitionModel.model_validate(_DEFN), []
        return real_load(root, definition_name)

    monkeypatch.setattr(
        epoch_module,
        "load_catalog_optimization_context",
        load_test_context,
    )


# --- run_parallel + Directory --------------------------------------------

def test_kernel_gen_enables_profile_by_default():
    assert KernelGenInput.model_fields["profile_enabled"].default is True
    assert KernelGenInput.model_fields["timeout"].default == 3600
    assert "trace_root" not in KernelGenInput.model_fields
    assert "trace_set_key" not in KernelGenInput.model_fields


def test_kernel_gen_loads_workloads_from_canonical_catalog(tmp_path):
    catalog_root = builtin_catalog_path()
    definition, expected_workloads = load_catalog_optimization_context(
        catalog_root,
        "identity",
    )
    workflow = KernelGenWorkflow(cwd=str(tmp_path), runtime_factory=lambda _: None)
    inp = KernelGenInput(
        definition=definition,
        target_hardware="A100",
        catalog_name="simple-v6-test",
    )

    coder_input = epoch_module.build_coder_input(inp, {}, seed_code="")

    assert coder_input["workloads"] == expected_workloads
    assert coder_input["knowledge_enabled"] is False
    assert {item["phase"] for item in coder_input["workloads"]} == {
        "correctness",
        "timing",
    }


def test_kernel_gen_enables_knowledge_role_for_v1_read_write_mode(tmp_path):
    catalog_root = builtin_catalog_path()
    definition, _ = load_catalog_optimization_context(
        catalog_root,
        "identity",
    )
    inp = KernelGenInput(
        definition=definition,
        target_hardware="A100",
        catalog_name="simple-v6-test",
    )
    workflow = KernelGenWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: None,
        knowledge_config=KnowledgeConfig(
            catalog_root=tmp_path / "kb",
        ),
    )

    coder_input = epoch_module.build_coder_input(
        inp, {}, seed_code="", knowledge_enabled=workflow._knowledge_config.reads_v1,
    )

    assert coder_input["knowledge_enabled"] is True


def test_kernel_gen_uses_catalog_benchmark_identity(tmp_path, monkeypatch):
    definition, _ = load_catalog_optimization_context(
        builtin_catalog_path(),
        "identity",
    )
    monkeypatch.setattr(
        knowledge_bridge_module,
        "_service_status",
        lambda _: {
            "backend": "cuda",
            "target": {
                "backend": "cuda",
                "device": "A100",
            },
        },
    )
    workflow = KernelGenWorkflow(
        cwd=str(tmp_path / "run"),
        runtime_factory=lambda _: None,
        knowledge_config=KnowledgeConfig(
            catalog_root=tmp_path / "knowledge",
        ),
    )
    bridge = knowledge_module.build_knowledge_bridge(config=workflow._knowledge_config, cwd=workflow._cwd, start_mode="fresh", inp=
        KernelGenInput(
            definition=definition,
            target_hardware="A100",
            catalog_name="simple-v6-test",
        )
    )

    assert bridge is not None
    assert bridge.benchmark_id == "simple-v6-test-v6.0"


def test_kernel_gen_uses_explicit_knowledge_run_id_with_workspace_fallback(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        knowledge_bridge_module,
        "_service_status",
        lambda _: {
            "backend": "cuda",
            "target": {
                "backend": "cuda",
                "device": "A100",
            },
        },
    )
    workflow = KernelGenWorkflow(
        cwd=str(tmp_path / "run"),
        runtime_factory=lambda _: None,
        knowledge_config=KnowledgeConfig(
            catalog_root=tmp_path / "knowledge",
        ),
    )

    fallback = knowledge_module.build_knowledge_bridge(config=workflow._knowledge_config, cwd=workflow._cwd, start_mode="fresh", inp=
        KernelGenInput(definition=_DEFN, target_hardware="A100")
    )
    explicit = knowledge_module.build_knowledge_bridge(config=workflow._knowledge_config, cwd=workflow._cwd, start_mode="fresh", inp=
        KernelGenInput(
            definition=_DEFN,
            target_hardware="A100",
            knowledge_run_id="campaign-a--flaggems_rsqrt",
        )
    )

    assert fallback is not None
    assert fallback.run_id == "run"
    assert explicit is not None
    assert explicit.run_id == "campaign-a--flaggems_rsqrt"


def test_fork_seed_is_passed_to_every_first_epoch_agent(tmp_path):
    definition, _ = load_catalog_optimization_context(
        builtin_catalog_path(),
        "identity",
    )
    workflow = KernelGenWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: None,
    )
    inp = KernelGenInput(
        definition=definition,
        target_hardware="A100",
        catalog_name="simple-v6-test",
        n_parallel=2,
        start_mode="fork",
    )

    inputs = epoch_module.build_epoch_inputs(
        inp,
        analysis={},
        next_directions=[],
        seed_code="EXACT_SOLUTION_SEED",
        epoch_num=1,
    )

    assert [item["seed_code"] for item in inputs] == [
        "EXACT_SOLUTION_SEED",
        "EXACT_SOLUTION_SEED",
    ]


def test_validated_initial_seed_is_measured_by_only_first_agent(tmp_path):
    definition, _ = load_catalog_optimization_context(
        builtin_catalog_path(),
        "identity",
    )
    workflow = KernelGenWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: None,
    )
    inp = KernelGenInput(
        definition=definition,
        target_hardware="A100",
        catalog_name="simple-v6-test",
        n_parallel=3,
        initial_seed_code="VALIDATED_NATIVE_BASELINE",
        initial_seed_is_validated_baseline=True,
    )

    inputs = epoch_module.build_epoch_inputs(
        inp,
        analysis={},
        next_directions=[],
        seed_code=inp.initial_seed_code,
        epoch_num=1,
    )

    assert [item["seed_code"] for item in inputs] == [
        "VALIDATED_NATIVE_BASELINE",
        "VALIDATED_NATIVE_BASELINE",
        "VALIDATED_NATIVE_BASELINE",
    ]
    assert [item["seed_is_validated_baseline"] for item in inputs] == [
        True,
        False,
        False,
    ]


def test_validated_seed_is_materialized_without_overwriting(tmp_path):
    workflow = SingleCoderOptimizationWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: None,
    )
    inp = SingleCoderOptimizationWorkflow.InputModel.model_validate(
        {
            "definition": _DEFN,
            "target_hardware": "A100",
            "seed_code": "def run(input):\n    return input\n",
            "seed_is_validated_baseline": True,
        }
    )

    workflow._materialize_validated_seed(inp)
    candidate = tmp_path / "tmp/main.py"
    assert candidate.read_text(encoding="utf-8") == inp.seed_code
    workflow._materialize_validated_seed(inp)

    candidate.write_text("different\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already contains different code"):
        workflow._materialize_validated_seed(inp)


def test_next_epoch_without_directions_keeps_authoritative_seed(tmp_path):
    workflow = KernelGenWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: None,
    )
    inp = KernelGenInput(
        definition=DefinitionModel.model_validate(_DEFN),
        target_hardware="A100",
        n_parallel=2,
    )

    inputs = epoch_module.build_epoch_inputs(
        inp,
        analysis={},
        next_directions=[],
        seed_code="CURRENT_VALIDATED_BEST",
        epoch_num=2,
    )

    assert [item["seed_code"] for item in inputs] == [
        "CURRENT_VALIDATED_BEST",
        "CURRENT_VALIDATED_BEST",
    ]


def test_fork_requires_knowledge_configuration(tmp_path):
    definition, _ = load_catalog_optimization_context(
        builtin_catalog_path(),
        "identity",
    )
    workflow = KernelGenWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: FakeRuntime([]),
    )

    with pytest.raises(ValueError, match="Knowledge-enabled"):
        workflow.run(
            {
                "definition": definition,
                "target_hardware": "A100",
                "start_mode": "fork",
            }
        )


class _EchoIn(BaseModel):
    v: int


class _EchoOut(BaseModel):
    doubled: int


class _EchoAgent(BaseAgent):
    name = "echo"
    md_path = ""
    InputModel = _EchoIn
    OutputModel = _EchoOut


def _fake_rt_factory(replies_by_path=None):
    # each workspace path gets a runtime replying based on its input
    def factory(path):
        # reply is filled per-call; here we return a runtime that doubles v
        return _DoublingRuntime()
    return factory


class _DoublingRuntime:
    """A fake runtime that reads nothing but returns a doubled value — actually we
    can't see input here, so tests script replies explicitly instead."""
    def __init__(self, reply='{"doubled": 0}'):
        self.reply = reply
    def invoke(self, prompt, *, model="inherit"):
        return self.reply


def test_run_parallel_directory_basic(tmp_path):
    inputs = [{"v": 1}, {"v": 2}, {"v": 3}]
    # runtime_factory returns a runtime scripted to echo doubled(v) — but the fake
    # can't see v, so we make each call deterministic by workspace name via closure.
    call_order = []

    def factory(path):
        call_order.append(path)
        return FakeRuntime(['{"doubled": 42}'])

    results = run_parallel(_EchoAgent, inputs, workspace=Directory(base=tmp_path), runtime_factory=factory, max_workers=3)
    assert len(results) == 3
    # results are (out, ws_name) in input order
    for i, (out, name) in enumerate(results):
        assert isinstance(out, _EchoOut)
        assert out.doubled == 42
        assert name == f"task{i}"
    # three distinct workspaces were allocated
    assert len(set(call_order)) == 3


def test_run_parallel_preserves_input_order(tmp_path):
    inputs = [{"v": i} for i in range(5)]

    def factory(path):
        # reply encodes which task via the trailing digit of the path
        n = int(path.rstrip("/").split("task")[-1])
        return FakeRuntime(['{"doubled": %d}' % (n * 10)])

    results = run_parallel(_EchoAgent, inputs, workspace=Directory(base=tmp_path), runtime_factory=factory, max_workers=5)
    # order preserved: task i -> doubled == i*10
    for i, (out, name) in enumerate(results):
        assert out.doubled == i * 10
        assert name == f"task{i}"


def test_run_parallel_empty():
    assert run_parallel(_EchoAgent, [], workspace=Directory(), runtime_factory=lambda p: None) == []


def test_run_parallel_spaces_task_submissions(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(parallel_module.time, "sleep", sleeps.append)

    results = run_parallel(
        _EchoAgent,
        [{"v": 1}, {"v": 2}, {"v": 3}],
        workspace=Directory(base=tmp_path),
        runtime_factory=lambda path: FakeRuntime(['{"doubled": 42}']),
        max_workers=3,
        launch_interval_seconds=1.0,
    )

    assert len(results) == 3
    assert sleeps == [1.0, 1.0]


def test_run_parallel_propagates_exception(tmp_path):

    def factory(path):
        return FakeRuntime(["not json"])   # postprocess fails -> AgentContractError

    try:
        run_parallel(_EchoAgent, [{"v": 1}], workspace=Directory(base=tmp_path), runtime_factory=factory,
                     max_workers=1)
    except Exception:
        pass
    else:
        raise AssertionError("a failing task should propagate")


def test_run_parallel_collects_each_failure_and_completed_output(tmp_path):
    class RaisingRuntime:
        def __init__(self, message):
            self.message = message

        def invoke(self, prompt, *, model="inherit"):
            raise TimeoutError(self.message)

    def factory(path):
        task = Path(path).name
        if task == "task0":
            return FakeRuntime(['{"doubled": 42}'])
        if task == "task1":
            return RaisingRuntime("eval timed out")
        return RaisingRuntime("SSH gateway unavailable")

    with pytest.raises(ParallelExecutionError) as caught:
        run_parallel(
            _EchoAgent,
            [{"v": 0}, {"v": 1}, {"v": 2}],
            workspace=Directory(base=tmp_path),
            runtime_factory=factory,
            max_workers=3,
        )

    error = caught.value
    assert [failure.index for failure in error.failures] == [1, 2]
    assert [failure.name for failure in error.failures] == ["task1", "task2"]
    assert [str(failure.error) for failure in error.failures] == [
        "eval timed out",
        "SSH gateway unavailable",
    ]
    assert error.partial_results[0][0].doubled == 42
    assert error.partial_results[1:] == (None, None)


# --- KernelGenWorkflow is a Runnable/Workflow -----------------------------

def test_kernelgen_is_runnable():
    assert issubclass(KernelGenWorkflow, Workflow)
    assert issubclass(KernelGenWorkflow, Runnable)


# --- KernelGenWorkflow end-to-end (fake) ----------------------------------

_DEFN = {
    "name": "abl_t1_gelu",
    "op_type": "elementwise",
    "axes": {"input_0_d0": {"type": "const", "value": 32}},
    "inputs": {"input_0": {"shape": [32, 512], "dtype": "float32"}},
    "outputs": {"output": {"shape": [32, 512], "dtype": "float32"}},
    "reference": "def run(input_0):\n    return gelu(input_0)",
}

_ANALYSIS = json.dumps({
    "core_math": "gelu", "program_grid": "grid=(40,)", "inner_loop": "loop",
    "softmax_strategy": "N/A", "data_layout": "contiguous", "key_pitfalls": [],
    "workload_dispatch_strategy": "single", "directions": [], "recommended_direction": "",
})
_REPORT = json.dumps({"status": "PASSED", "summary": "best gelu 1.1x"})
_SYNTHESIS = json.dumps({
    "next_directions": [
        {"direction": "try vectorized loads", "expected_gain": "0.10x"},
    ],
    "synthesis_report": "epoch done",
})


def _test_verification(workspace):
    best = Ledger(workspace).best
    return {"status": "PASSED" if best["geo_mean"] > 0 else "FAILED", "geo_mean": best["geo_mean"] or None,
            "round_num": best["round"], "solution_sha256": hashlib.sha256(best["code"].encode()).hexdigest()}


def _write_confirmed_output(workspace):
    workspace = Path(workspace)
    verification = _test_verification(workspace)
    (workspace / ".kernelgen").mkdir(exist_ok=True)
    (workspace / ".kernelgen/final-verification.json").write_text(json.dumps(verification))
    (workspace / "optimize_definition_output.json").write_text(json.dumps({
        "status": verification["status"], "best_geo_mean": verification["geo_mean"],
        "best_code": Ledger(workspace).best["code"], "final_verification": verification,
    }))


class _StubKernelGen(KernelGenWorkflow):
    """Simulate the Coder's measured tool writes, keeping all production readers real."""
    def __init__(self, geos, **kw):
        factory = kw.pop("runtime_factory")

        def measured_runtime(path):
            runtime = factory(path)
            name = Path(path).name
            geo = geos.get(name)
            if geo is not None:
                runtime._replies.append(json.dumps({
                    "candidate_experience": "", "candidate_detailed": "", "skip_reason": "host fixture",
                }))
                invoke = runtime.invoke

                def invoke_and_record(*args, **kwargs):
                    output = invoke(*args, **kwargs)
                    ledger = Ledger(path)
                    if not ledger.history.rounds:
                        ledger.record_eval(
                            {"status": "PASSED", "geo_mean": geo}, f"# code for {name}", experiment_plan(1),
                            definition_name=self._test_input.definition.name,
                            target_hardware=self._test_input.target_hardware,
                            implementation_language=self._test_input.implementation_language.value,
                        )
                        ledger.finalize_round(1, round_conclusion(1), StopConfig(max_round=1))
                    return output

                runtime.invoke = invoke_and_record
            return runtime

        super().__init__(runtime_factory=measured_runtime, **kw)

    def _execute(self, inp):
        self._test_input = inp
        return super()._execute(inp)


def test_kernelgen_cold_start_flow(tmp_path):
    analysis_paths = []
    synthesis_runtimes = []

    # runtime_factory: analysis space -> analysis JSON; coder spaces -> report JSON
    def factory(path):
        if Path(path).name == "shared_analysis":
            analysis_paths.append(path)
            return FakeRuntime([f"```json\n{_ANALYSIS}\n```"])
        if Path(path).name == "synthesis":
            runtime = FakeRuntime([f"```json\n{_SYNTHESIS}\n```"])
            synthesis_runtimes.append(runtime)
            return runtime
        if Path(path).name.startswith("agent"):
            return FakeRuntime([f"```json\n{_REPORT}\n```"])
        return FakeRuntime([])

    # agent0 geo=1.1, agent1 geo=1.5 (best), agent2 geo=0.9
    wf = _StubKernelGen(
        geos={"agent0": 1.1, "agent1": 1.5, "agent2": 0.9},
        cwd=str(tmp_path), runtime_factory=factory,
    )
    inp = {
        "definition": _DEFN, "target_hardware": "Ascend910B",
        "n_parallel": 3, "eval_server_url": "http://x:8000",
    }
    out = wf.run(inp, None)
    assert isinstance(out, KernelGenOutput)
    assert out.definition_name == "abl_t1_gelu"
    assert out.status == "PASSED"
    assert out.best_geo_mean == 1.5              # authoritative best across agents
    assert out.best_code == "# code for agent1"   # from the best agent
    assert out.num_agents == 3
    assert len(out.per_agent) == 3
    analysis_dir = tmp_path / "1R" / "shared_analysis"
    assert analysis_paths == [str(analysis_dir)]
    assert json.loads((analysis_dir / "analysis.json").read_text())["core_math"] == "gelu"
    assert (tmp_path / "1R" / "agent0").is_dir()
    synthesis_dir = tmp_path / "1R" / "synthesis"
    synthesis = json.loads((synthesis_dir / "synthesis.json").read_text())
    assert synthesis["next_directions"][0]["direction"] == "try vectorized loads"
    assert len(synthesis_runtimes) == 1
    synthesis_prompt = synthesis_runtimes[0].calls[0]["prompt"]
    assert "best_geo=1.500x" in synthesis_prompt
    assert synthesis_prompt.count("# code for agent1") == 1
    assert synthesis_prompt.count("<authoritative_best_kernel>") == 1
    assert Ledger(tmp_path / "1R" / "agent1").best["geo_mean"] == 1.5
    progress = WorkspaceRunControl(tmp_path).progress()
    assert progress.state == RunState.SUCCEEDED
    assert progress.stage == "COMPLETED"
    assert progress.scopes["1R/shared_analysis"].state == RunState.SUCCEEDED
    assert progress.scopes["1R/shared_analysis"].stage == "COMPLETED"
    assert progress.scopes["1R/knowledge-merger"].state == RunState.SUCCEEDED
    assert progress.scopes["1R/knowledge-merger"].stage == "COMPLETED"
    assert progress.scopes["1R/synthesis"].state == RunState.SUCCEEDED
    assert progress.scopes["1R/synthesis"].stage == "COMPLETED"
    assert all(
        scope.state != RunState.RUNNING
        for scope in progress.scopes.values()
    )


def test_kernelgen_reports_fine_grained_epoch_stages(tmp_path, monkeypatch):
    control = WorkspaceRunControl(tmp_path)
    observed = {}

    def current_stage():
        progress = control.progress()
        return progress.stage, progress.current_epoch, progress.total_epochs

    def factory(runtime_path):
        runtime_path = Path(runtime_path)
        name = runtime_path.name
        if name == "shared_analysis":
            observed["shared_analysis"] = current_stage()
            return FakeRuntime([f"```json\n{_ANALYSIS}\n```"])
        if name.startswith("agent"):
            epoch = int(runtime_path.parent.name.removesuffix("R"))
            observed.setdefault(f"epoch_{epoch}_coder", current_stage())
            return FakeRuntime([f"```json\n{_REPORT}\n```"])
        if name == "knowledge-merger":
            epoch = int(runtime_path.parent.name.removesuffix("R"))
            observed[f"epoch_{epoch}_knowledge_merger"] = current_stage()
            return FakeRuntime([])
        if name == "synthesis":
            epoch = int(runtime_path.parent.name.removesuffix("R"))
            observed[f"epoch_{epoch}_synthesis"] = current_stage()
            return FakeRuntime([f"```json\n{_SYNTHESIS}\n```"])
        return FakeRuntime([])

    workflow = _StubKernelGen(
        geos={"agent0": 1.1, "agent1": 1.2},
        cwd=str(tmp_path),
        runtime_factory=factory,
    )
    collect = epoch_module.collect_epoch_result

    def observe_collect(definition_name, results, workspace=None):
        epoch = int(Path(workspace.base).name.removesuffix("R"))
        observed[f"epoch_{epoch}_finalizing"] = current_stage()
        return collect(definition_name, results, workspace)

    monkeypatch.setattr(epoch_module, "collect_epoch_result", observe_collect)
    monkeypatch.setattr(
        knowledge_module,
        "promote_best_solution",
        lambda knowledge, best_result: observed.update(
            promoting=current_stage()
        ),
    )

    output = workflow.run(
        {
            "definition": _DEFN,
            "target_hardware": "H800",
            "n_parallel": 2,
            "n_epoch": 2,
        }
    )

    assert output.status == "PASSED"
    assert observed == {
        "shared_analysis": ("SHARED_ANALYSIS", 1, 2),
        "epoch_1_coder": ("CODER_RUNNING", 1, 2),
        "epoch_1_finalizing": ("EPOCH_FINALIZING", 1, 2),
        "epoch_1_knowledge_merger": ("SYNTHESIZING", 1, 2),
        "epoch_1_synthesis": ("SYNTHESIZING", 1, 2),
        "epoch_2_coder": ("CODER_RUNNING", 2, 2),
        "epoch_2_finalizing": ("EPOCH_FINALIZING", 2, 2),
        "epoch_2_knowledge_merger": ("SYNTHESIZING", 2, 2),
        "epoch_2_synthesis": ("SYNTHESIZING", 2, 2),
        "promoting": ("PROMOTING_BEST", 2, 2),
    }


def test_kernelgen_shared_analysis_failure_closes_scope(tmp_path):
    class FailingRuntime:
        supports_native_agents = False

        @staticmethod
        def invoke(prompt, *, model="inherit", agent=None):
            raise RuntimeError("analysis provider failed")

    def factory(runtime_path):
        if Path(runtime_path).name == "shared_analysis":
            return FailingRuntime()
        return FakeRuntime([])

    workflow = _StubKernelGen(
        geos={},
        cwd=str(tmp_path),
        runtime_factory=factory,
    )

    with pytest.raises(RuntimeError, match="analysis provider failed"):
        workflow.run(
            {
                "definition": _DEFN,
                "target_hardware": "H800",
                "n_parallel": 1,
            }
        )

    progress = WorkspaceRunControl(tmp_path).progress()
    assert progress.state == RunState.FAILED
    assert progress.stage == "FAILED"
    assert progress.scopes["1R/shared_analysis"].state == RunState.FAILED
    assert progress.scopes["1R/shared_analysis"].stage == "FAILED"


def test_kernelgen_shared_analysis_cancellation_closes_scope(tmp_path):
    class CancellingRuntime:
        supports_native_agents = False

        @staticmethod
        def invoke(prompt, *, model="inherit", agent=None):
            control = WorkspaceRunControl(tmp_path / "1R" / "shared_analysis")
            control.request_cancel("stop shared analysis")
            control.checkpoint("AFTER_MODEL_INVOCATION")
            raise AssertionError("cancellation checkpoint must raise")

    def factory(runtime_path):
        if Path(runtime_path).name == "shared_analysis":
            return CancellingRuntime()
        return FakeRuntime([])

    workflow = _StubKernelGen(
        geos={},
        cwd=str(tmp_path),
        runtime_factory=factory,
    )

    with pytest.raises(RunCancelled, match="stop shared analysis"):
        workflow.run(
            {
                "definition": _DEFN,
                "target_hardware": "H800",
                "n_parallel": 1,
            }
        )

    progress = WorkspaceRunControl(tmp_path).progress()
    assert progress.state == RunState.CANCELLED
    assert progress.stage == "CANCELLED"
    assert progress.scopes["1R/shared_analysis"].state == RunState.CANCELLED
    assert progress.scopes["1R/shared_analysis"].stage == "CANCELLED"


def test_kernelgen_knowledge_merge_cancellation_closes_scope(
    tmp_path,
    monkeypatch,
):
    def factory(runtime_path):
        name = Path(runtime_path).name
        if name == "shared_analysis":
            return FakeRuntime([f"```json\n{_ANALYSIS}\n```"])
        if name.startswith("agent"):
            return FakeRuntime([f"```json\n{_REPORT}\n```"])
        return FakeRuntime([])

    workflow = _StubKernelGen(
        geos={"agent0": 1.1},
        cwd=str(tmp_path),
        runtime_factory=factory,
    )

    def cancel_merge(*args, **kwargs):
        control = WorkspaceRunControl(tmp_path / "1R" / "knowledge-merger")
        control.request_cancel("stop knowledge merge")
        control.checkpoint("AFTER_MODEL_INVOCATION")

    monkeypatch.setattr(knowledge_module, "merge_epoch_kb", cancel_merge)

    with pytest.raises(RunCancelled, match="stop knowledge merge"):
        workflow.run(
            {
                "definition": _DEFN,
                "target_hardware": "H800",
                "n_parallel": 1,
            }
        )

    progress = WorkspaceRunControl(tmp_path).progress()
    assert progress.state == RunState.CANCELLED
    assert progress.stage == "CANCELLED"
    merger = progress.scopes["1R/knowledge-merger"]
    assert merger.state == RunState.CANCELLED
    assert merger.stage == "CANCELLED"


def test_kernelgen_same_epoch_resume_reuses_shared_analysis(tmp_path):
    analysis_dir = tmp_path / "1R" / "shared_analysis"
    analysis_dir.mkdir(parents=True)
    analysis_checkpoint = analysis_dir / "analysis.json"
    analysis_checkpoint.write_text(_ANALYSIS, encoding="utf-8")

    def factory(path):
        name = Path(path).name
        if name == "shared_analysis":
            raise AssertionError("same-epoch resume must reuse shared analysis")
        if name == "synthesis":
            return FakeRuntime([f"```json\n{_SYNTHESIS}\n```"])
        if name.startswith("agent"):
            return FakeRuntime([f"```json\n{_REPORT}\n```"])
        return FakeRuntime([])

    wf = _StubKernelGen(
        geos={"agent0": 1.1},
        cwd=str(tmp_path),
        runtime_factory=factory,
    )
    out = wf.run({
        "definition": _DEFN,
        "target_hardware": "A100",
        "n_parallel": 1,
        "start_epoch": 1,
        "start_mode": "resume",
    })

    assert out.status == "PASSED"
    assert out.best_geo_mean == 1.1
    assert json.loads(analysis_checkpoint.read_text())["core_math"] == "gelu"


def test_kernelgen_same_epoch_resume_reruns_invalid_shared_analysis(tmp_path):
    analysis_dir = tmp_path / "1R" / "shared_analysis"
    analysis_dir.mkdir(parents=True)
    analysis_checkpoint = analysis_dir / "analysis.json"
    analysis_checkpoint.write_text("{invalid", encoding="utf-8")
    analysis_paths = []

    def factory(path):
        name = Path(path).name
        if name == "shared_analysis":
            analysis_paths.append(path)
            return FakeRuntime([f"```json\n{_ANALYSIS}\n```"])
        if name == "synthesis":
            return FakeRuntime([f"```json\n{_SYNTHESIS}\n```"])
        if name.startswith("agent"):
            return FakeRuntime([f"```json\n{_REPORT}\n```"])
        return FakeRuntime([])

    wf = _StubKernelGen(
        geos={"agent0": 1.1},
        cwd=str(tmp_path),
        runtime_factory=factory,
    )
    out = wf.run({
        "definition": _DEFN,
        "target_hardware": "A100",
        "n_parallel": 1,
        "start_epoch": 1,
        "start_mode": "resume",
    })

    assert out.status == "PASSED"
    assert analysis_paths == [str(analysis_dir)]
    assert json.loads(analysis_checkpoint.read_text())["core_math"] == "gelu"


def test_kernelgen_all_fail(tmp_path):

    def factory(path):
        if Path(path).name == "shared_analysis":
            return FakeRuntime([f"```json\n{_ANALYSIS}\n```"])
        if Path(path).name == "synthesis":
            return FakeRuntime([f"```json\n{_SYNTHESIS}\n```"])
        if Path(path).name.startswith("agent"):
            return FakeRuntime([f"```json\n{_REPORT}\n```"])
        return FakeRuntime([])

    wf = _StubKernelGen(geos={}, cwd=str(tmp_path), runtime_factory=factory)  # no passing geo
    inp = {"definition": _DEFN, "target_hardware": "A100", "n_parallel": 2}
    out = wf.run(inp, None)
    assert out.status == "FAILED"
    assert out.best_geo_mean is None
    assert out.num_agents == 2
    assert (tmp_path / "1R" / "synthesis" / "synthesis.json").is_file()


def test_kernelgen_continues_when_one_parallel_agent_fails(
    tmp_path,
    monkeypatch,
):
    import kernelgen.workflows.optimization.kernelgen.epoch as workflow_module

    def partial_run(*args, workspace, **kwargs):
        workspace.allocate("agent0")
        workspace.allocate("agent1")
        ledger = Ledger(workspace.path_of("agent1"))
        ledger.record_eval(
            {"status": "PASSED", "geo_mean": 1.5}, "# code for agent1", experiment_plan(1),
            definition_name="abl_t1_gelu", target_hardware="A100",
        )
        ledger.finalize_round(1, round_conclusion(1), StopConfig(max_round=1))
        _write_confirmed_output(workspace.path_of("agent1"))
        report = CoderReport(status="PASSED", summary="agent1 completed")
        raise ParallelExecutionError(
            failures=[
                ParallelTaskFailure(
                    index=0,
                    name="agent0",
                    error=TimeoutError("provider timed out"),
                )
            ],
            partial_results=[None, (report, "agent1")],
        )

    monkeypatch.setattr(workflow_module, "run_parallel", partial_run)

    def factory(path):
        if Path(path).name == "shared_analysis":
            return FakeRuntime([f"```json\n{_ANALYSIS}\n```"])
        return FakeRuntime([])

    workflow = _StubKernelGen(
        geos={"agent1": 1.5},
        cwd=str(tmp_path),
        runtime_factory=factory,
    )
    monkeypatch.setattr(
        knowledge_module,
        "merge_epoch_kb",
        lambda *args, **kwargs: None,
    )

    output = workflow.run(
        {
            "definition": _DEFN,
            "target_hardware": "A100",
            "n_parallel": 2,
        }
    )

    assert output.status == "PASSED"
    assert output.num_agents == 1
    assert output.best_geo_mean == 1.5
    manifest = json.loads(
        (tmp_path / "1R" / "epoch-completion.json").read_text()
    )
    assert manifest["attempted_agents"] == ["agent0", "agent1"]
    assert manifest["successful_agents"] == ["agent1"]
    assert manifest["failed_agents"] == [
        {
            "name": "agent0",
            "error_type": "TimeoutError",
            "error": "provider timed out",
        }
    ]
    synthesis = json.loads(
        (tmp_path / "1R" / "synthesis" / "synthesis.json").read_text()
    )
    assert synthesis["next_directions"] == []


def test_kernelgen_raises_when_every_parallel_agent_fails(
    tmp_path,
    monkeypatch,
):
    import kernelgen.workflows.optimization.kernelgen.epoch as workflow_module

    def failed_run(*args, workspace, **kwargs):
        workspace.allocate("agent0")
        workspace.allocate("agent1")
        raise ParallelExecutionError(
            failures=[
                ParallelTaskFailure(0, "agent0", TimeoutError("first")),
                ParallelTaskFailure(1, "agent1", RuntimeError("second")),
            ],
            partial_results=[None, None],
        )

    monkeypatch.setattr(workflow_module, "run_parallel", failed_run)

    def factory(path):
        if Path(path).name == "shared_analysis":
            return FakeRuntime([f"```json\n{_ANALYSIS}\n```"])
        return FakeRuntime([])

    workflow = _StubKernelGen(
        geos={},
        cwd=str(tmp_path),
        runtime_factory=factory,
    )
    with pytest.raises(ParallelExecutionError):
        workflow.run(
            {
                "definition": _DEFN,
                "target_hardware": "A100",
                "n_parallel": 2,
            }
        )

    assert not (tmp_path / "1R" / "epoch-completion.json").exists()


def test_kernelgen_validates_input(tmp_path):
    wf = _StubKernelGen(geos={}, cwd=str(tmp_path),
                        runtime_factory=lambda p: FakeRuntime([]))
    try:
        wf.run({"definition": _DEFN}, None)   # missing target_hardware
    except Exception:
        pass
    else:
        raise AssertionError("should validate input (missing target_hardware)")


def test_evaluation_contract_reaches_analyzer_and_coder_inputs(tmp_path):
    inp = KernelGenInput.model_validate(
        {
            "definition": _DEFN,
            "target_hardware": "Ascend910B",
            "evaluation_contract": {
                "tolerance_mode": "fixed",
                "atol": 1e-2,
                "rtol": 1e-2,
                "required_matched_ratio": 1.0,
                "consider_reduced_precision": True,
            },
        }
    )
    runtime = FakeRuntime([f"```json\n{_ANALYSIS}\n```"])
    wf = KernelGenWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda path: runtime,
    )

    analysis = preparation_module.prepare_analysis(tmp_path, inp, None, lambda _: runtime, WorkspaceRunControl(tmp_path), resume=False)
    analyzer_prompt = runtime.calls[0]["prompt"]
    assert "Tolerance mode: fixed" in analyzer_prompt
    assert "0.01 + 0.01 * |reference|" in analyzer_prompt

    coder_input = epoch_module.build_coder_input(inp, analysis)
    assert coder_input["evaluation_contract"] == {
        "tolerance_mode": "fixed",
        "atol": 0.01,
        "rtol": 0.01,
        "required_matched_ratio": 1.0,
        "check_output_dtype": True,
        "reject_non_finite": True,
        "consider_reduced_precision": True,
    }


def test_evaluation_contract_requires_atol_rtol_pair():
    try:
        KernelGenInput.model_validate(
            {
                "definition": _DEFN,
                "target_hardware": "Ascend910B",
                "evaluation_contract": {"atol": 1e-2},
            }
        )
    except Exception:
        pass
    else:
        raise AssertionError("a partial evaluator tolerance contract must fail")


def test_epoch_synthesis_uses_the_global_best_code_anchor(tmp_path):
    runtime = FakeRuntime([f"```json\n{_SYNTHESIS}\n```"])
    workspace = Directory(base=tmp_path)
    workspace.allocate("agent0")
    workspace.allocate("agent1")
    inp = KernelGenInput.model_validate({
        "definition": _DEFN,
        "target_hardware": "Ascend910B",
        "n_parallel": 2,
    })
    reports = [
        (type("Report", (), {"status": "PASSED", "summary": "current agent0"})(), "agent0"),
        (type("Report", (), {"status": "PASSED", "summary": "current agent1"})(), "agent1"),
    ]

    epoch_module.summarize_epoch(
        tmp_path,
        inp,
        reports,
        EpochResult(inp.definition.name, best_workspace_path=tmp_path / "prior" / "agent1",
                    best_round=3, best_geo_mean=1.5, best_code="GLOBAL_BEST_FROM_PRIOR_EPOCH"),
        runtime,
        epoch_workspace=workspace,
    )

    prompt = runtime.calls[0]["prompt"]
    assert prompt.count("GLOBAL_BEST_FROM_PRIOR_EPOCH") == 1
    assert "Authoritative best: geo=1.500x" in prompt


def test_epoch_synthesis_uses_actual_successful_agent_count(tmp_path):
    runtime = FakeRuntime([f"```json\n{_SYNTHESIS}\n```"])
    workspace = Directory(base=tmp_path)
    workspace.allocate("agent0")
    workspace.allocate("agent2")
    inp = KernelGenInput.model_validate(
        {
            "definition": _DEFN,
            "target_hardware": "A100",
            "n_parallel": 3,
        }
    )
    reports = [
        (CoderReport(status="PASSED", summary="first"), "agent0"),
        (CoderReport(status="PASSED", summary="third"), "agent2"),
    ]

    epoch_module.summarize_epoch(
        tmp_path,
        inp,
        reports,
        EpochResult(inp.definition.name, best_workspace_path=tmp_path / "agent0",
                    best_round=1, best_geo_mean=1.2, best_code="BEST"),
        runtime,
        epoch_workspace=workspace,
    )

    assert "Agents: 2" in runtime.calls[0]["prompt"]



def test_kernelgen_multi_epoch(tmp_path):
    """Multi-epoch: epoch 1 cold-start → EpochSummary → epoch 2 with directions."""
    call_log = []
    analysis_paths = []
    synthesis_paths = []
    agent_runtimes = {}

    def factory(path):
        call_log.append(path)
        if Path(path).name == "shared_analysis":
            analysis_paths.append(path)
            return FakeRuntime([f"```json\n{_ANALYSIS}\n```"])
        if Path(path).name == "synthesis":
            synthesis_paths.append(path)
            return FakeRuntime([f"```json\n{_SYNTHESIS}\n```"])
        runtime = FakeRuntime([f"```json\n{_REPORT}\n```"])
        agent_runtimes[path] = runtime
        return runtime

    # epoch 1: agent0 geo=1.1, agent1 geo=1.5
    # epoch 2: agent0 geo=1.8 (improved!), agent1 geo=1.6
    epoch_geos = {"agent0": 1.5, "agent1": 1.1}  # simplified: same geos both epochs

    wf = _StubKernelGen(
        geos=epoch_geos,
        cwd=str(tmp_path), runtime_factory=factory,
    )
    inp = {
        "definition": _DEFN, "target_hardware": "Ascend910B",
        "n_parallel": 2, "n_epoch": 2,
    }
    out = wf.run(inp, None)
    assert out.status == "PASSED"
    assert out.best_geo_mean == 1.5
    # Verify epoch dirs were created
    assert (tmp_path / "1R").is_dir()
    assert (tmp_path / "2R").is_dir()
    # Verify both epochs ran agents (agent0, agent1 in each)
    assert (tmp_path / "1R" / "agent0").is_dir()
    assert (tmp_path / "2R" / "agent0").is_dir()
    assert analysis_paths == [str(tmp_path / "1R" / "shared_analysis")]
    assert not (tmp_path / "2R" / "shared_analysis").exists()
    assert json.loads(
        (tmp_path / "1R" / "shared_analysis" / "analysis.json").read_text()
    )["program_grid"] == "grid=(40,)"
    assert synthesis_paths == [
        str(tmp_path / "1R" / "synthesis"),
        str(tmp_path / "2R" / "synthesis"),
    ]
    assert json.loads(
        (tmp_path / "1R" / "synthesis" / "synthesis.json").read_text()
    )["synthesis_report"] == "epoch done"
    assert json.loads(
        (tmp_path / "2R" / "synthesis" / "synthesis.json").read_text()
    )["synthesis_report"] == "epoch done"
    second_round_prompt = agent_runtimes[
        str(tmp_path / "2R" / "agent0")
    ].calls[0]["prompt"]
    assert '"assigned_direction"' in second_round_prompt
    assert "try vectorized loads" in second_round_prompt


def test_kernelgen_resumes_at_second_epoch_from_checkpoints(tmp_path):
    analysis_dir = tmp_path / "1R" / "shared_analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "analysis.json").write_text(_ANALYSIS, encoding="utf-8")
    synthesis_dir = tmp_path / "1R" / "synthesis"
    synthesis_dir.mkdir(parents=True)
    (synthesis_dir / "synthesis.json").write_text(_SYNTHESIS, encoding="utf-8")

    _write_completed_agents(tmp_path)

    agent_runtimes = {}

    def factory(path):
        name = Path(path).name
        if name == "shared_analysis":
            raise AssertionError("resume must not rerun cold analysis")
        if name == "synthesis":
            return FakeRuntime([f"```json\n{_SYNTHESIS}\n```"])
        if name.startswith("agent"):
            runtime = FakeRuntime([f"```json\n{_REPORT}\n```"])
            agent_runtimes[path] = runtime
            return runtime
        return FakeRuntime([])

    wf = KernelGenWorkflow(cwd=str(tmp_path), runtime_factory=factory)
    out = wf.run({
        "definition": _DEFN,
        "target_hardware": "Ascend910B",
        "n_parallel": 2,
        "start_epoch": 2,
        "n_epoch": 2,
    })

    assert out.status == "PASSED"
    assert out.best_geo_mean == 1.5
    assert out.best_code == "# prior best from agent1"
    assert (tmp_path / "2R" / "synthesis" / "synthesis.json").is_file()
    prompt = agent_runtimes[str(tmp_path / "2R" / "agent0")].calls[0]["prompt"]
    assert "# prior best from agent1" in prompt
    assert '"assigned_direction"' in prompt
    assert "try vectorized loads" in prompt


class _NoopMaterializer:
    def materialize(self, workspace):
        marker = Path(workspace) / ".knowledge-materialized"
        marker.write_text("ok", encoding="utf-8")


class _PublishedKnowledge:
    def __init__(self, status="noop"):
        self.status = status
        self.config = SimpleNamespace(reviewer_mode=KnowledgeReviewerMode.OFF)
        self.publish_calls = []

    def materializer(self):
        return _NoopMaterializer()

    def publish_epoch(self, workspaces, *, epoch_num, reviewer=None):
        self.publish_calls.append((list(workspaces), epoch_num))
        return type("PublishResult", (), {"status": self.status})()

    def promote_solution(self, workspace):
        return None


def test_preparation_resolves_fork_seed_once_and_keeps_runtime_timeout(tmp_path, monkeypatch):
    seeds = []

    class Knowledge(_PublishedKnowledge):
        def resolve_fork_seed(self):
            seeds.append("resolve")
            return SimpleNamespace(code="EXACT_FORK_SEED", manifest=SimpleNamespace(solution_ref="solution://parent", geo_mean=1.5))

    monkeypatch.setattr(knowledge_module, "build_knowledge_bridge", lambda **_: Knowledge())
    runtime = FakeRuntime([f"```json\n{_ANALYSIS}\n```"])
    runtime.timeout = 1
    config = KnowledgeConfig(catalog_root=tmp_path / "catalog")
    inp = KernelGenInput(definition=_DEFN, target_hardware="A100", start_mode="fork", n_parallel=2)
    prepared = preparation_module.prepare_run(
        cwd=tmp_path, inp=inp, knowledge_config=config,
        runtime_factory=lambda _: runtime, run_control=WorkspaceRunControl(tmp_path),
    )
    inputs = epoch_module.build_epoch_inputs(
        prepared.inp, prepared.analysis, prepared.next_directions, prepared.initial_seed_code, 1,
        knowledge_enabled=config.reads_v1, evaluation_snapshot=prepared.evaluation_snapshot,
    )
    assert seeds == ["resolve"]
    assert runtime.timeout == inp.timeout
    assert [item["seed_code"] for item in inputs] == ["EXACT_FORK_SEED", "EXACT_FORK_SEED"]
    assert all(item["knowledge_enabled"] for item in inputs)
    assert not any(item["seed_is_validated_baseline"] for item in inputs)


def _write_completed_agents(tmp_path, *, count=2):
    for index in range(count):
        geo = 1.1 + index * 0.4
        agent_dir = tmp_path / "1R" / f"agent{index}"
        agent_dir.mkdir(parents=True)
        ledger = Ledger(agent_dir)
        ledger.record_eval(
            {"status": "PASSED", "geo_mean": geo},
            f"# prior best from agent{index}",
            experiment_plan(1),
            definition_name="abl_t1_gelu",
            target_hardware="Ascend910B",
            implementation_language="triton",
        )
        ledger.finalize_round(1, round_conclusion(1), StopConfig(max_round=1))
        _write_confirmed_output(agent_dir)


def test_finalize_completed_epoch_reuses_agents_and_checkpoint(
    tmp_path,
    monkeypatch,
):
    _write_completed_agents(tmp_path)
    runtime_paths = []
    knowledge = _PublishedKnowledge()

    def factory(path):
        runtime_paths.append(path)
        name = Path(path).name
        if name in {"shared_analysis", "agent0", "agent1"}:
            raise AssertionError("finalize must not invoke Analyzer or Coder")
        if name == "synthesis":
            assert (Path(path) / ".knowledge-materialized").is_file()
            return FakeRuntime([f"```json\n{_SYNTHESIS}\n```"])
        return FakeRuntime([])

    inp = KernelGenInput.model_validate(
        {
            "definition": _DEFN,
            "target_hardware": "Ascend910B",
            "n_parallel": 2,
            "n_epoch": 1,
        }
    )
    workflow = KernelGenWorkflow(
        cwd=str(tmp_path),
        runtime_factory=factory,
    )
    monkeypatch.setattr(
        knowledge_module,
        "build_knowledge_bridge",
        lambda **_: knowledge,
    )

    first = workflow.finalize_completed_epoch(inp, 1)
    second = workflow.finalize_completed_epoch(inp, 1)

    assert first.status == second.status == "PASSED"
    assert first.best_geo_mean == second.best_geo_mean == 1.5
    assert (tmp_path / "1R" / "synthesis" / "synthesis.json").is_file()
    assert [Path(path).name for path in runtime_paths].count("synthesis") == 1
    assert len(knowledge.publish_calls) == 2
    assert all(call[1] == 1 for call in knowledge.publish_calls)
    progress = WorkspaceRunControl(tmp_path).progress()
    assert progress.state == RunState.SUCCEEDED
    assert progress.stage == "COMPLETED"
    assert progress.current_epoch == 1
    assert progress.scopes["1R/synthesis"].state == RunState.SUCCEEDED


def test_finalize_completed_epoch_publishes_real_knowledge_batch(
    tmp_path,
    monkeypatch,
):
    _write_completed_agents(tmp_path)
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    monkeypatch.setattr(
        knowledge_bridge_module,
        "_service_status",
        lambda _: {
            "backend": "ascend",
            "target": {
                "backend": "ascend",
                "device": "Ascend910B",
            },
        },
    )
    def reviewer_must_not_be_built(**_):
        raise AssertionError("Reviewer must not be built in off mode")

    monkeypatch.setattr(
        finalization_module,
        "_build_epoch_reviewer",
        reviewer_must_not_be_built,
    )

    def factory(path):
        if Path(path).name == "synthesis":
            state = Path(path) / ".kernelgen" / "knowledge" / "state.json"
            assert state.is_file()
            return FakeRuntime([f"```json\n{_SYNTHESIS}\n```"])
        return FakeRuntime([])

    inp = KernelGenInput.model_validate(
        {
            "definition": _DEFN,
            "target_hardware": "Ascend910B",
            "n_parallel": 2,
            "n_epoch": 1,
            "eval_server_url": "http://eval.invalid",
        }
    )
    workflow = KernelGenWorkflow(
        cwd=str(tmp_path),
        runtime_factory=factory,
        knowledge_config=KnowledgeConfig(catalog_root=catalog),
    )
    bridge = knowledge_module.build_knowledge_bridge(config=workflow._knowledge_config, cwd=workflow._cwd, start_mode="fresh", inp=inp)
    assert bridge is not None
    for index in range(2):
        bridge.materializer().materialize(
            tmp_path / "1R" / f"agent{index}"
        )

    output = workflow.finalize_completed_epoch(inp, 1)

    assert output.status == "PASSED"
    for index in range(2):
        publish_result = (
            tmp_path
            / "1R"
            / f"agent{index}"
            / ".kernelgen"
            / "knowledge"
            / "publish-result.json"
        )
        assert json.loads(publish_result.read_text())["status"] in {
            "published",
            "noop",
        }


def test_finalize_completed_epoch_requires_every_agent(tmp_path):
    _write_completed_agents(tmp_path, count=1)
    workflow = KernelGenWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda path: FakeRuntime([]),
    )

    with pytest.raises(
        ValueError,
        match="1R has 1 completed agent ledgers; expected 2",
    ):
        workflow.finalize_completed_epoch(
            {
                "definition": _DEFN,
                "target_hardware": "Ascend910B",
                "n_parallel": 2,
                "n_epoch": 1,
            },
            1,
        )


def test_load_completed_epoch_uses_partial_success_manifest(tmp_path):
    _write_completed_agents(tmp_path, count=1)
    (tmp_path / "1R" / "epoch-completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "attempted_agents": ["agent0", "agent1"],
                "successful_agents": ["agent0"],
                "failed_agents": [
                    {
                        "name": "agent1",
                        "error_type": "TimeoutError",
                        "error": "provider timed out",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    workflow = KernelGenWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda path: FakeRuntime([]),
    )

    results, workspace = recovery_module.load_completed_epoch(
        cwd=tmp_path,
        definition_name="abl_t1_gelu",
        target_hardware="Ascend910B",
        implementation_language=ImplementationLanguage.TRITON,
        epoch_num=1,
        expected_agents=2,
    )

    assert [name for _, name in results] == ["agent0"]
    assert Path(workspace.path_of("agent0")) == tmp_path / "1R" / "agent0"


def test_epoch_finalize_rejects_incomplete_publication(
    tmp_path,
    monkeypatch,
):
    _write_completed_agents(tmp_path)
    knowledge = _PublishedKnowledge(status="rejected")
    workflow = KernelGenWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda path: FakeRuntime([]),
    )
    monkeypatch.setattr(
        knowledge_module,
        "build_knowledge_bridge",
        lambda **_: knowledge,
    )

    with pytest.raises(
        RuntimeError,
        match="knowledge publication did not complete: rejected",
    ):
        workflow.finalize_completed_epoch(
            {
                "definition": _DEFN,
                "target_hardware": "Ascend910B",
                "n_parallel": 2,
                "n_epoch": 1,
            },
            1,
        )

    assert not (tmp_path / "1R" / "synthesis" / "synthesis.json").exists()


def test_epoch_finalize_requires_synthesis_checkpoint(tmp_path, monkeypatch):
    _write_completed_agents(tmp_path)
    knowledge = _PublishedKnowledge()
    workflow = KernelGenWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda path: FakeRuntime([]),
    )
    monkeypatch.setattr(
        knowledge_module,
        "build_knowledge_bridge",
        lambda **_: knowledge,
    )
    def fail_synthesis(*args, **kwargs):
        raise RuntimeError("synthesis provider unavailable")

    monkeypatch.setattr(epoch_module.EpochSummaryAgent, "run", fail_synthesis)

    with pytest.raises(
        RuntimeError,
        match="synthesis provider unavailable",
    ):
        workflow.finalize_completed_epoch(
            {
                "definition": _DEFN,
                "target_hardware": "Ascend910B",
                "n_parallel": 2,
                "n_epoch": 1,
            },
            1,
        )


@pytest.mark.parametrize("count", [1, 2])
def test_cancelled_finalize_never_resolves_target_or_publishes(tmp_path, monkeypatch, count):
    _write_completed_agents(tmp_path, count=count)
    control = WorkspaceRunControl(tmp_path)
    workflow = KernelGenWorkflow(cwd=str(tmp_path), run_control=control)

    def unexpected(*args, **kwargs):
        raise AssertionError("cancelled finalize must not launch further work")

    monkeypatch.setattr(preparation_module, "resolve_server_target", unexpected)
    monkeypatch.setattr(knowledge_module, "promote_best_solution", unexpected)
    control.request_cancel("cancel before finalize")
    with pytest.raises(RunCancelled, match="cancel before finalize"):
        workflow.finalize_completed_epoch({
            "definition": _DEFN, "target_hardware": "Ascend910B", "n_parallel": count,
        }, 1)
    assert control.progress().state == RunState.CANCELLED
    assert not (tmp_path / "1R" / "epoch-completion.json").exists()


@pytest.mark.parametrize("outcome", ["cancel_in_runtime", "cancel_after_output", "error"])
def test_synthesis_outcome_reaches_scope_and_root_unchanged(tmp_path, monkeypatch, outcome):
    _write_completed_agents(tmp_path)
    full_output = f"```json\n{_SYNTHESIS}\n```"
    completed_outputs = []
    knowledge = _PublishedKnowledge()
    control = WorkspaceRunControl(tmp_path)

    class Runtime(FakeRuntime):
        def invoke(self, *args, **kwargs):
            if outcome == "error":
                raise RuntimeError("original synthesis failure")
            result = super().invoke(*args, **kwargs)
            completed_outputs.append(result)
            scope = WorkspaceRunControl(tmp_path / "1R" / "synthesis")
            scope.request_cancel("cancel synthesis after full output")
            if outcome == "cancel_in_runtime":
                scope.checkpoint("AFTER_COMPLETE_MODEL_OUTPUT")
            return result

    workflow = KernelGenWorkflow(
        cwd=str(tmp_path), run_control=control,
        runtime_factory=lambda _: Runtime([full_output]),
    )
    monkeypatch.setattr(knowledge_module, "build_knowledge_bridge", lambda **_: knowledge)
    promotions = []
    monkeypatch.setattr(knowledge_module, "promote_best_solution", lambda *args: promotions.append(args))
    expected_error = RuntimeError if outcome == "error" else RunCancelled
    match = "original synthesis failure" if outcome == "error" else "cancel synthesis after full output"
    with pytest.raises(expected_error, match=match):
        workflow.finalize_completed_epoch({
            "definition": _DEFN, "target_hardware": "Ascend910B", "n_parallel": 2,
        }, 1)
    state = RunState.FAILED if outcome == "error" else RunState.CANCELLED
    assert control.progress().state == state
    assert control.progress().scopes["1R/synthesis"].state == state
    assert completed_outputs == ([] if outcome == "error" else [full_output])
    assert not promotions
    assert not (tmp_path / "1R" / "epoch-completion.json").exists()
    assert not (tmp_path / "1R" / "synthesis" / "synthesis.json").exists()
    assert Ledger(tmp_path / "1R" / "agent1").best["geo_mean"] == 1.5


@pytest.mark.parametrize("cached", [False, True])
def test_cancellation_after_publication_prevents_checkpoint_and_promotion(tmp_path, monkeypatch, cached):
    count = 2 if cached else 1
    _write_completed_agents(tmp_path, count=count)
    control = WorkspaceRunControl(tmp_path)
    checkpoint = tmp_path / "1R" / "synthesis" / "synthesis.json"
    if cached:
        checkpoint.parent.mkdir()
        checkpoint.write_text(_SYNTHESIS, encoding="utf-8")

    class Knowledge(_PublishedKnowledge):
        def publish_epoch(self, *args, **kwargs):
            result = super().publish_epoch(*args, **kwargs)
            control.request_cancel("cancel after publication")
            return result

    workflow = KernelGenWorkflow(cwd=str(tmp_path), run_control=control, runtime_factory=lambda _: FakeRuntime([]))
    monkeypatch.setattr(knowledge_module, "build_knowledge_bridge", lambda **_: Knowledge())
    promotions = []
    monkeypatch.setattr(knowledge_module, "promote_best_solution", lambda *args: promotions.append(args))
    with pytest.raises(RunCancelled, match="cancel after publication"):
        workflow.finalize_completed_epoch({
            "definition": _DEFN, "target_hardware": "Ascend910B", "n_parallel": count,
        }, 1)
    assert control.progress().state == RunState.CANCELLED
    assert not promotions
    assert not (tmp_path / "1R" / "epoch-completion.json").exists()
    assert checkpoint.exists() is cached
    if cached:
        assert checkpoint.read_text() == _SYNTHESIS


def test_finalize_checks_cancellation_before_final_promotion(tmp_path, monkeypatch):
    _write_completed_agents(tmp_path, count=1)
    control = WorkspaceRunControl(tmp_path)
    workflow = KernelGenWorkflow(cwd=str(tmp_path), run_control=control, runtime_factory=lambda _: FakeRuntime([]))
    monkeypatch.setattr(knowledge_module, "build_knowledge_bridge", lambda **_: _PublishedKnowledge())
    monkeypatch.setattr(finalization_module, "finalize_epoch_outputs", lambda *args, **kwargs: control.request_cancel("cancel before promotion"))
    promotions = []
    monkeypatch.setattr(knowledge_module, "promote_best_solution", lambda *args: promotions.append(args))
    with pytest.raises(RunCancelled, match="cancel before promotion"):
        workflow.finalize_completed_epoch({
            "definition": _DEFN, "target_hardware": "Ascend910B", "n_parallel": 1,
        }, 1)
    assert not promotions
    assert control.progress().state == RunState.CANCELLED


if __name__ == "__main__":
    import inspect
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
