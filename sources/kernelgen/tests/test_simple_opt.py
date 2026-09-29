"""Host tests for SimpleOptWorkflow, ExtractOptWorkflow, and trace helpers."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

from kernelgen.data.ledger import Ledger
from kernelgen.data.catalog import (
    DEFAULT_CATALOG_NAME,
    resolve_builtin_catalog_path,
)
from kernelgen.data.stop_policy import StopConfig
from kernelgen.data.target_context import TargetContext
from kernelgen.tests.helpers import experiment_plan, round_conclusion
from kernelgen.data.trace import (
    find_workload_paths,
    find_workloads_path,
    infer_destination_passing_style,
    load_definition,
    load_workload_context,
)
from kernelgen.agents.extractor.pytorch import (
    Definition as ExtractedDefinition,
    ExtractorResult,
)
from kernelgen.framework import AgentContractError, FakeRuntime
from kernelgen.framework.models import DefinitionModel
from kernelgen.framework.run_control import WorkspaceRunControl
from kernelgen.workflows.extract_opt import ExtractOptInput, ExtractOptWorkflow
from kernelgen.workflows.legacy.simple_opt import (
    SimpleOptInput,
    SimpleOptWorkflow,
)
from kernelgen.workflows.optimization.single_coder import (
    KERNEL_OPTIMIZATION_OUTPUT_FILENAME,
    SingleCoderOptimizationOutput,
    SingleCoderOptimizationWorkflow,
)


_RETURN_DEFINITION = {
    "name": "demo_add",
    "op_type": "elementwise",
    "axes": {"N": {"type": "const", "value": 16}},
    "inputs": {
        "x": {"shape": ["N"], "dtype": "float32"},
        "y": {"shape": ["N"], "dtype": "float32"},
    },
    "outputs": {"output": {"shape": ["N"], "dtype": "float32"}},
    "reference": "def run(x, y):\n    return x + y\n",
}


@pytest.fixture(autouse=True)
def _server_target(monkeypatch):
    # These tests isolate Coder lifecycle and distillation with synthetic legacy
    # ledger entries. Real snapshot/retest/output integration lives in test_retest.
    from kernelgen.workflows.optimization.single_coder import workflow as optimize_module
    monkeypatch.setattr(optimize_module, 'verify_final_best', lambda workspace, **kwargs: {
        'status': 'PASSED' if Ledger(workspace).history.best_round else 'FAILED',
        'geo_mean': Ledger(workspace).history.best_geo_mean or None,
    })
    from kernelgen.tools import kernelgen_server_adapter
    from kernelgen.agents import coder as coder_module

    monkeypatch.setattr(coder_module, "load_catalog_manifest", lambda name: {"evaluator": "native"})
    from kernelgen.workflows.legacy.simple_opt import preparation as simple_opt_module
    from kernelgen.workflows.optimization import knowledge as simple_opt_knowledge
    from kernelgen.workflows import knowledge_bridge as knowledge_bridge_module

    def load_test_context(root, definition_name):
        definition, _ = load_definition(root, definition_name)
        workloads = load_workload_context(
            find_workload_paths(root, definition_name),
            definition_name,
        )
        return definition, workloads

    monkeypatch.setattr(
        simple_opt_module,
        "resolve_builtin_catalog_path",
        lambda catalog_name: Path(catalog_name).resolve(),
    )
    monkeypatch.setattr(
        simple_opt_module,
        "load_catalog_optimization_context",
        load_test_context,
    )
    monkeypatch.setattr(
        simple_opt_knowledge,
        "catalog_benchmark_id",
        lambda catalog_name: "test-catalog-v5.1",
    )

    service_status = {
        "status": "ok",
        "api_version": "v5.1",
        "backend": "cuda",
        "timing": "triton",
        "target": {
            "backend": "cuda",
            "vendor": "nvidia",
            "architecture": "ampere",
            "device": "NVIDIA A100-SXM4-40GB",
        },
        "software": {"language": "triton"},
    }
    monkeypatch.setattr(
        kernelgen_server_adapter,
        "get_service_status",
        lambda server_url: service_status,
    )
    monkeypatch.setattr(
        knowledge_bridge_module,
        "_service_status",
        lambda server_url: service_status,
    )


def _write_trace(root: Path, definition=None):
    definition = definition or _RETURN_DEFINITION
    op_type = definition["op_type"]
    name = definition["name"]
    definition_path = root / "definitions" / op_type / f"{name}.json"
    definition_path.parent.mkdir(parents=True)
    definition_path.write_text(json.dumps(definition), encoding="utf-8")

    workloads_path = root / "workloads" / op_type / f"{name}.jsonl"
    workloads_path.parent.mkdir(parents=True)
    workloads_path.write_text(
        json.dumps({
            "definition": name,
            "workload": {
                "uuid": "w0",
                "axes": {},
                "inputs": {"x": {"type": "random"}, "y": {"type": "random"}},
            },
            "solution": None,
            "evaluation": None,
        }) + "\n",
        encoding="utf-8",
    )


def _write_phased_trace(root: Path, definition=None):
    definition = definition or _RETURN_DEFINITION
    op_type = definition["op_type"]
    name = definition["name"]
    definition_path = root / "definitions" / op_type / f"{name}.json"
    definition_path.parent.mkdir(parents=True)
    definition_path.write_text(json.dumps(definition), encoding="utf-8")

    phased = root / "phased_workloads" / op_type
    phased.mkdir(parents=True)
    for phase, uuid in (("correctness", "c0"), ("timing", "t0")):
        (phased / f"{name}.{phase}.jsonl").write_text(
            json.dumps({
                "definition": name,
                "workload": {
                    "uuid": uuid,
                    "axes": {},
                    "inputs": {
                        "x": {"type": "random"},
                        "y": {"type": "random"},
                    },
                },
                "solution": None,
                "evaluation": None,
            }) + "\n",
            encoding="utf-8",
        )


@pytest.mark.parametrize("best_export", ["intact", "missing", "stale"])
def test_simple_opt_loads_definition_name_and_collects_authoritative_result(tmp_path, best_export):
    trace_root = tmp_path / "trace"
    workspace = tmp_path / "run"
    reference_code = tmp_path / "reference.py"
    reference_code_prompt = tmp_path / "reference.md"
    _write_trace(trace_root)
    reference_code.write_text(
        "@triton.jit\ndef reference_kernel(x):\n    return\n",
        encoding="utf-8",
    )
    reference_code_prompt.write_text(
        "Ran on Ascend 910B4 with the historical native evaluator.",
        encoding="utf-8",
    )

    Ledger(workspace).record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.25,
            "min_speedup": 1.25,
            "latency_ms": 0.1,
            "per_workload": [{"uuid": "w0", "speedup": 1.25}],
        },
        "def run(x, y): return x + y",
        experiment_plan(1),
        definition_name="demo_add",
        target_hardware="A100",
    )
    Ledger(workspace).finalize_round(
        1,
        round_conclusion(1),
        StopConfig(max_round=1),
    )
    export_path = Ledger(workspace).best_kernel_path
    if best_export == "missing":
        export_path.unlink()
    elif best_export == "stale":
        export_path.write_text("stale_unmeasured_code", encoding="utf-8")
    legacy_kb = (
        workspace
        / "kb"
        / "experience"
        / "by_definition"
        / "elementwise"
        / "demo_add"
        / "A100"
        / "experience.md"
    )
    legacy_kb.parent.mkdir(parents=True)
    legacy_kb.write_text("must-not-enter-distiller-prompt", encoding="utf-8")

    runtime = FakeRuntime([
        json.dumps({
            "candidate_experience": "",
            "candidate_detailed": "",
            "skip_reason": "host test",
        }),
    ])
    workflow = SimpleOptWorkflow(
        cwd=str(workspace),
        runtime_factory=lambda _: runtime,
    )
    out = workflow.run({
        "definition_name": "demo_add",
        "catalog_name": str(trace_root),
        "target_hardware": "A100",
        "eval_server_url": "http://localhost:8000",
        "early_stop_rounds": 5,
        "min_rounds": 3,
        "profile_enabled": False,
        "reference_code_path": reference_code,
        "reference_code_prompt_path": reference_code_prompt,
        "eval_timeout_seconds": 900,
    })

    assert isinstance(out, SingleCoderOptimizationOutput)
    assert out.definition_name == "demo_add"
    assert out.status == "PASSED"
    assert out.best_geo_mean == 1.25
    assert out.best_code == "def run(x, y): return x + y"
    assert out.rounds == 1
    saved_output_path = workspace / KERNEL_OPTIMIZATION_OUTPUT_FILENAME
    saved_output = SingleCoderOptimizationOutput.model_validate_json(
        saved_output_path.read_text(encoding="utf-8")
    )
    assert saved_output == out
    lifecycle_events = WorkspaceRunControl(workspace).read_events()
    assert len([event for event in lifecycle_events if event.event_type == "WORKFLOW_STARTED"]) == 1
    assert len([event for event in lifecycle_events if event.event_type == "WORKFLOW_COMPLETED"]) == 1
    assert not list(workspace.glob(f".{KERNEL_OPTIMIZATION_OUTPUT_FILENAME}.*.tmp"))
    # The terminal authoritative ledger is resumed without another Coder call;
    # the only model invocation is the report-only Distiller.
    assert len(runtime.calls) == 1
    distiller_prompt = runtime.calls[0]["prompt"]
    assert "senior GPU optimization knowledge synthesizer" in distiller_prompt
    assert "demo_add" in distiller_prompt
    assert "<trajectory>" in distiller_prompt
    assert "Best: geo_mean=1.250x at round 1" in distiller_prompt
    assert str(trace_root) not in distiller_prompt
    assert "_early_stop_rounds" not in distiller_prompt
    assert "_min_rounds" not in distiller_prompt
    assert "_max_round" not in distiller_prompt
    assert "definition_path" not in distiller_prompt
    assert "workloads_path" not in distiller_prompt
    assert "<reference_code>" not in distiller_prompt
    assert "<reference_code_guidance>" not in distiller_prompt
    assert str(reference_code) not in distiller_prompt
    assert str(reference_code_prompt) not in distiller_prompt
    assert "must-not-enter-distiller-prompt" not in distiller_prompt
    assert "<existing_kb>" not in distiller_prompt
    assert "<existing_detailed>" not in distiller_prompt

    tool_context = json.loads(
        (workspace / ".kernelgen" / "tool-context.json").read_text()
    )
    assert tool_context["definition"] == "demo_add"
    assert tool_context["target_hardware"] == "A100"
    assert tool_context["target_context"]["device"] == "A100"
    assert tool_context["target_context"]["source"] == "eval_service"
    assert tool_context["catalog_name"] == str(trace_root)
    assert tool_context["destination_passing_style"] is False
    assert tool_context["eval_timeout_seconds"] == 900
    assert tool_context["eval_transport_timeout_seconds"] == 1200

    stop_config = json.loads((workspace / ".stop_config.json").read_text())
    assert stop_config == {
        "early_stop_rounds": 5,
        "min_rounds": 3,
        "max_round": 10,
        "soft_stop_disabled": False,
    }


def test_simple_opt_distiller_contract_failure_keeps_authoritative_result(
    tmp_path,
    monkeypatch,
    capsys,
):
    from kernelgen.workflows.optimization.single_coder import workflow as workflow_module

    trace_root = tmp_path / "trace"
    workspace = tmp_path / "run"
    _write_trace(trace_root)

    ledger = Ledger(workspace)
    ledger.record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.25,
            "min_speedup": 1.25,
            "latency_ms": 0.1,
            "per_workload": [{"uuid": "w0", "speedup": 1.25}],
        },
        "def run(x, y): return x + y",
        experiment_plan(1),
        definition_name="demo_add",
        target_hardware="A100",
    )
    ledger.finalize_round(
        1,
        round_conclusion(1),
        StopConfig(max_round=1),
    )

    def fail_distillation(workspace, inp, runtime, **kwargs):
        raise AgentContractError("invalid report bundle")

    monkeypatch.setattr(
        workflow_module,
        "run_distillation",
        fail_distillation,
    )

    out = SimpleOptWorkflow(
        cwd=str(workspace),
        runtime_factory=lambda _: FakeRuntime(
            [json.dumps({"status": "PASSED", "summary": "best add kernel"})]
        ),
    ).run(
        {
            "definition_name": "demo_add",
            "catalog_name": str(trace_root),
            "target_hardware": "A100",
            "profile_enabled": False,
        }
    )

    assert out.status == "PASSED"
    assert out.best_geo_mean == 1.25
    assert out.best_code == "def run(x, y): return x + y"
    assert "Skipped invalid output" in capsys.readouterr().out


def test_simple_opt_accepts_phased_workloads(tmp_path):
    trace_root = tmp_path / "trace"
    _write_phased_trace(trace_root)
    runtime = FakeRuntime([
        json.dumps({"status": "FAILED", "summary": "host-only validation"}),
    ])

    out = SimpleOptWorkflow(
        cwd=str(tmp_path / "run"),
        runtime_factory=lambda _: runtime,
    ).run({
        "definition_name": "demo_add",
        "catalog_name": str(trace_root),
        "target_hardware": "A100",
        "profile_enabled": False,
    })

    paths = find_workload_paths(trace_root, "demo_add")
    assert paths.mode == "phased"
    assert len(paths.correctness) == 1
    assert len(paths.timing) == 1
    assert out.definition_name == "demo_add"
    prompt = runtime.calls[0]["prompt"]
    assert '"phase": "correctness"' in prompt
    assert '"uuid": "c0"' in prompt
    assert '"phase": "timing"' in prompt
    assert '"uuid": "t0"' in prompt


def test_workload_context_preserves_v4_tolerance_and_seed(tmp_path):
    trace_root = tmp_path / "trace"
    _write_phased_trace(trace_root)
    path = (
        trace_root
        / "phased_workloads"
        / "elementwise"
        / "demo_add.correctness.jsonl"
    )
    entry = json.loads(path.read_text().splitlines()[0])
    entry["workload"]["tolerance"] = {"max_atol": 0.01}
    entry["workload"]["seed"] = 7
    path.write_text(json.dumps(entry) + "\n")

    contexts = load_workload_context(
        find_workload_paths(trace_root, "demo_add"),
        "demo_add",
    )
    correctness = next(item for item in contexts if item["phase"] == "correctness")
    assert correctness["tolerance"] == {"max_atol": 0.01}
    assert correctness["seed"] == 7


def test_optimize_workflow_rejects_nonterminal_continue_verdict(tmp_path):
    trace_root = tmp_path / "trace"
    workspace = tmp_path / "run"
    _write_trace(trace_root)
    ledger = Ledger(workspace)
    ledger.record_eval(
        {"status": "PASSED", "geo_mean": 1.0, "per_workload": []},
        "good",
        experiment_plan(1),
        definition_name="demo_add",
        target_hardware="A100",
    )
    ledger.finalize_round(1, round_conclusion(1))
    runtime = FakeRuntime([
        json.dumps({"status": "PASSED", "summary": "returned too early"}),
    ])

    try:
        SimpleOptWorkflow(
            cwd=str(workspace),
            runtime_factory=lambda _: runtime,
        ).run({
            "definition_name": "demo_add",
            "catalog_name": str(trace_root),
            "target_hardware": "A100",
            "profile_enabled": False,
            "max_coder_sessions": 1,
        })
    except RuntimeError as exc:
        assert "terminal STOP verdict is required" in str(exc)
    else:
        raise AssertionError("workflow accepted a nonterminal CONTINUE verdict")


def _record_two_continuing_best_rounds(workspace: Path) -> None:
    ledger = Ledger(workspace)
    for round_num, geo_mean in ((1, 1.0), (2, 1.2)):
        ledger.record_eval(
            {
                "status": "PASSED",
                "geo_mean": geo_mean,
                "per_workload": [],
            },
            f"round_{round_num}_code",
            experiment_plan(round_num),
            definition_name="demo_add",
            target_hardware="A100",
        )
        ledger.finalize_round(
            round_num,
            round_conclusion(round_num),
        )


def test_optimize_workflow_preserves_best_at_coder_session_limit(
    tmp_path,
    monkeypatch,
):
    trace_root = tmp_path / "trace"
    workspace = tmp_path / "run"
    _write_trace(trace_root)
    _record_two_continuing_best_rounds(workspace)
    runtime = FakeRuntime([
        json.dumps({"status": "PASSED", "summary": "validated plateau"}),
    ])
    from kernelgen.workflows.optimization.single_coder import workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "run_distillation",
        lambda workspace, inp, runtime, **kwargs: None,
    )

    out = SimpleOptWorkflow(
        cwd=str(workspace),
        runtime_factory=lambda _: runtime,
    ).run({
        "definition_name": "demo_add",
        "catalog_name": str(trace_root),
        "target_hardware": "A100",
        "profile_enabled": False,
        "min_rounds": 2,
        "max_coder_sessions": 1,
    })

    stopped = Ledger(workspace).get_round(2).next_verdict
    assert out.status == "PASSED"
    assert out.best_geo_mean == 1.2
    assert out.best_code == "round_2_code"
    assert stopped is not None
    assert stopped.should_continue is False
    assert stopped.code == "coder_session_limit_reached"
    assert "preserving validated best R2" in stopped.reason


def test_optimize_workflow_stops_without_best_at_coder_session_limit(
    tmp_path,
    monkeypatch,
):
    trace_root = tmp_path / "trace"
    workspace = tmp_path / "run"
    _write_trace(trace_root)
    ledger = Ledger(workspace)
    for round_num in (1, 2):
        ledger.record_eval(
            {
                "status": "PARTIAL_PASS",
                "geo_mean": None,
                "per_workload": [],
            },
            f"incorrect_round_{round_num}",
            experiment_plan(round_num),
            definition_name="demo_add",
            target_hardware="A100",
        )
        ledger.finalize_round(round_num, round_conclusion(round_num))
    runtime = FakeRuntime([
        json.dumps({
            "status": "INCORRECT_NUMERICAL",
            "summary": "no candidate passed correctness",
        }),
    ])
    from kernelgen.workflows.optimization.single_coder import workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "run_distillation",
        lambda workspace, inp, runtime, **kwargs: None,
    )

    out = SimpleOptWorkflow(
        cwd=str(workspace),
        runtime_factory=lambda _: runtime,
    ).run({
        "definition_name": "demo_add",
        "catalog_name": str(trace_root),
        "target_hardware": "A100",
        "profile_enabled": False,
        "min_rounds": 2,
        "max_coder_sessions": 1,
    })

    stopped = Ledger(workspace).get_round(2).next_verdict
    assert out.status == "FAILED"
    assert out.best_geo_mean is None
    assert out.best_code == ""
    assert stopped is not None
    assert stopped.should_continue is False
    assert stopped.code == "coder_session_limit_reached"
    assert "no validated PASSED candidate after 2 measured round(s)" in stopped.reason


def test_optimize_workflow_uses_session_limit_when_coder_makes_no_round_progress(
    tmp_path,
    monkeypatch,
):
    trace_root = tmp_path / "trace"
    workspace = tmp_path / "run"
    _write_trace(trace_root)
    _record_two_continuing_best_rounds(workspace)

    class _NoProgressRuntime:
        supports_native_agents = False

        def __init__(self):
            self.last_session_id = None
            self.calls = []

        def invoke(self, prompt, *, model="inherit", agent=None):
            self.calls.append("invoke")
            self.last_session_id = "coder-session-1"
            return json.dumps({
                "status": "PASSED",
                "summary": "validated plateau",
            })

        def resume(
            self,
            prompt,
            *,
            model="inherit",
            agent=None,
            session_id=None,
        ):
            self.calls.append("resume")
            return json.dumps({
                "status": "PASSED",
                "summary": "still at validated plateau",
            })

    runtime = _NoProgressRuntime()
    from kernelgen.workflows.optimization.single_coder import workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "run_distillation",
        lambda workspace, inp, runtime, **kwargs: None,
    )

    out = SimpleOptWorkflow(
        cwd=str(workspace),
        runtime_factory=lambda _: runtime,
    ).run({
        "definition_name": "demo_add",
        "catalog_name": str(trace_root),
        "target_hardware": "A100",
        "profile_enabled": False,
        "min_rounds": 2,
        "max_coder_sessions": 3,
    })

    stopped = Ledger(workspace).get_round(2).next_verdict
    assert runtime.calls == ["invoke", "resume", "resume"]
    assert out.status == "PASSED"
    assert stopped is not None
    assert stopped.should_continue is False
    assert stopped.code == "coder_session_limit_reached"
    assert "maximum Coder invocation limit reached" in stopped.reason
    assert "preserving validated best R2" in stopped.reason


def test_optimize_workflow_resumes_same_coder_session_until_terminal_stop(
    tmp_path,
    monkeypatch,
):
    trace_root = tmp_path / "trace"
    workspace = tmp_path / "run"
    _write_trace(trace_root)

    class _ResumableRuntime:
        supports_native_agents = False

        def __init__(self):
            self.last_session_id = None
            self.calls = []

        def invoke(self, prompt, *, model="inherit", agent=None):
            self.calls.append(("invoke", prompt))
            self.last_session_id = "coder-session-1"
            first = Ledger(workspace)
            first.record_eval(
                {
                    "status": "PASSED",
                    "geo_mean": 1.0,
                    "per_workload": [],
                },
                "round_1_code",
                experiment_plan(1),
                definition_name="demo_add",
                target_hardware="A100",
            )
            first.finalize_round(1, round_conclusion(1))
            return json.dumps({
                "status": "PASSED",
                "summary": "returned too early",
            })

        def resume(
            self,
            prompt,
            *,
            model="inherit",
            agent=None,
            session_id=None,
        ):
            self.calls.append(("resume", prompt))
            resumed = Ledger(workspace)
            resumed.record_eval(
                {
                    "status": "PASSED",
                    "geo_mean": 1.2,
                    "per_workload": [],
                },
                "round_2_code",
                experiment_plan(2),
                definition_name="demo_add",
                target_hardware="A100",
            )
            resumed.finalize_round(
                2,
                round_conclusion(2),
                StopConfig(max_round=2),
            )
            return json.dumps({
                "status": "PASSED",
                "summary": "continued to terminal stop",
            })

    runtime = _ResumableRuntime()
    from kernelgen.workflows.optimization.single_coder import workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "run_distillation",
        lambda workspace, inp, runtime, **kwargs: None,
    )

    out = SimpleOptWorkflow(
        cwd=str(workspace),
        runtime_factory=lambda _: runtime,
    ).run({
        "definition_name": "demo_add",
        "catalog_name": str(trace_root),
        "target_hardware": "A100",
        "profile_enabled": False,
        "max_coder_sessions": 2,
    })

    assert [kind for kind, _ in runtime.calls] == ["invoke", "resume"]
    assert "OPTIMIZATION HISTORY" not in runtime.calls[0][1]
    assert "same conversation" in runtime.calls[1][1]
    assert "OPTIMIZATION HISTORY" not in runtime.calls[1][1]
    assert out.status == "PASSED"
    assert out.rounds == 2
    assert out.best_geo_mean == 1.2
    assert out.best_code == "round_2_code"


def test_optimize_workflow_rejects_continue_without_resumable_session(
    tmp_path,
    monkeypatch,
):
    from kernelgen.agents.coder import CoderReport
    from kernelgen.workflows.optimization import single_coder as optimize_module

    trace_root = tmp_path / "trace"
    workspace = tmp_path / "run"
    _write_trace(trace_root)
    ledger = Ledger(workspace)
    ledger.record_eval(
        {
            "status": "PASSED",
            "geo_mean": 1.0,
            "per_workload": [],
        },
        "round_1_code",
        experiment_plan(1),
        definition_name="demo_add",
        target_hardware="A100",
    )
    ledger.finalize_round(1, round_conclusion(1))

    coder_inputs = []

    def fake_coder_run(self, inp, runtime=None):
        coder_inputs.append(inp)
        return CoderReport(status="PASSED", summary="fixture report")

    monkeypatch.setattr(optimize_module.CoderAgent, "run", fake_coder_run)
    from kernelgen.workflows.optimization.single_coder import workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "run_distillation",
        lambda workspace, inp, runtime, **kwargs: None,
    )

    try:
        SimpleOptWorkflow(
            cwd=str(workspace),
            runtime_factory=lambda _: FakeRuntime([]),
        ).run({
            "definition_name": "demo_add",
            "catalog_name": str(trace_root),
            "target_hardware": "A100",
            "profile_enabled": False,
            "max_coder_sessions": 2,
        })
    except RuntimeError as exc:
        assert "provider session cannot be resumed" in str(exc)
    else:
        raise AssertionError("workflow started a fresh Coder after CONTINUE")

    assert len(coder_inputs) == 1
    workflow_only_fields = {
        "early_stop_rounds",
        "min_rounds",
        "max_round",
        "max_coder_sessions",
    }
    assert all(
        workflow_only_fields.isdisjoint(coder_input)
        for coder_input in coder_inputs
    )
    assert "OPTIMIZATION HISTORY" in coder_inputs[0]["history_summary"]
    assert "Do not submit plan.kind='baseline'" in coder_inputs[0]["history_summary"]
    assert coder_inputs[0]["seed_code"] == "round_1_code"


def test_simple_opt_missing_definition_fails_fast(tmp_path):
    trace_root = tmp_path / "trace"
    (trace_root / "definitions").mkdir(parents=True)
    workflow = SimpleOptWorkflow(
        cwd=str(tmp_path / "run"),
        runtime_factory=lambda _: FakeRuntime([]),
    )
    try:
        workflow.run({
            "definition_name": "missing",
            "catalog_name": str(trace_root),
        })
    except FileNotFoundError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing definition should fail before invoking an agent")


def test_simple_opt_rejects_missing_server_device(tmp_path, monkeypatch):
    from kernelgen.data.target_context import TargetContextError
    from kernelgen.tools import kernelgen_server_adapter

    trace_root = tmp_path / "trace"
    _write_trace(trace_root)
    monkeypatch.setattr(
        kernelgen_server_adapter,
        "get_service_status",
        lambda server_url: {
            "status": "ok",
            "backend": "cuda",
            "target": {"backend": "cuda"},
        },
    )

    with pytest.raises(TargetContextError, match="TARGET_DEVICE_MISSING"):
        SimpleOptWorkflow(
            cwd=str(tmp_path / "run"),
            runtime_factory=lambda _: FakeRuntime([]),
        ).run({
            "definition_name": "demo_add",
            "catalog_name": str(trace_root),
            "target_hardware": "A100",
        })


def test_simple_opt_rejects_target_hardware_mismatch(tmp_path, monkeypatch):
    from kernelgen.data.target_context import TargetContextError
    from kernelgen.tools import kernelgen_server_adapter

    trace_root = tmp_path / "trace"
    _write_trace(trace_root)
    monkeypatch.setattr(
        kernelgen_server_adapter,
        "get_service_status",
        lambda server_url: {
            "status": "ok",
            "backend": "npu",
            "target": {
                "backend": "ascend",
                "device": "Ascend 910B",
            },
        },
    )

    with pytest.raises(TargetContextError, match="TARGET_HARDWARE_MISMATCH"):
        SimpleOptWorkflow(
            cwd=str(tmp_path / "run"),
            runtime_factory=lambda _: FakeRuntime([]),
        ).run({
            "definition_name": "demo_add",
            "catalog_name": str(trace_root),
            "target_hardware": "A100",
        })


def test_simple_opt_reuses_optimize_definition_output_contract():
    assert SimpleOptWorkflow.OutputModel is SingleCoderOptimizationOutput
    assert SingleCoderOptimizationWorkflow.OutputModel is SingleCoderOptimizationOutput


def test_simple_opt_rejects_missing_knowledge_catalog(tmp_path):
    trace_root = tmp_path / "trace"
    _write_trace(trace_root)

    with pytest.raises(
        ValueError,
        match="knowledge_catalog_path does not exist",
    ):
        SimpleOptWorkflow(
            cwd=str(tmp_path / "run"),
            runtime_factory=lambda _: FakeRuntime([]),
        ).run({
            "definition_name": "demo_add",
            "catalog_name": str(trace_root),
            "target_hardware": "A100",
            "knowledge_catalog_path": tmp_path / "missing",
        })


def test_simple_opt_materializes_knowledge_and_enables_coder(
    tmp_path,
    monkeypatch,
):
    from kernelgen.knowledge.contracts import (
        OperatorSignature,
        TargetContext,
        WorkspaceKnowledgeState,
    )
    from kernelgen.knowledge.layout import KnowledgeLayout

    trace_root = tmp_path / "trace"
    workspace = tmp_path / "run"
    knowledge_catalog = tmp_path / "knowledge"
    _write_trace(trace_root)
    knowledge_catalog.mkdir()
    captured = {}

    def run_optimize(_workflow, coder_input):
        captured.update(coder_input)
        return SingleCoderOptimizationOutput(
            definition_name="demo_add",
            op_type="elementwise",
            status="FAILED",
            workspace=str(workspace),
        )

    monkeypatch.setattr(
        SingleCoderOptimizationWorkflow,
        "run",
        run_optimize,
    )

    out = SimpleOptWorkflow(
        cwd=str(workspace),
        runtime_factory=lambda _: FakeRuntime([]),
    ).run({
        "definition_name": "demo_add",
        "catalog_name": str(trace_root),
        "target_hardware": "A100",
        "eval_server_url": "http://localhost:8000",
        "knowledge_catalog_path": knowledge_catalog,
    })

    layout = KnowledgeLayout(workspace)
    state = WorkspaceKnowledgeState.model_validate_json(
        layout.state.read_text(encoding="utf-8")
    )
    signature = OperatorSignature.model_validate_json(
        layout.operator_signature.read_text(encoding="utf-8")
    )
    target = TargetContext.model_validate_json(
        layout.target_context.read_text(encoding="utf-8")
    )

    assert out.definition_name == "demo_add"
    assert captured["knowledge_enabled"] is True
    assert Path(state.catalog_ref) == knowledge_catalog.resolve()
    assert signature.definition_name == "demo_add"
    assert target.backend == "cuda"
    assert target.device == "A100"
    assert not layout.publish_result.exists()

    other_catalog = tmp_path / "other-knowledge"
    other_catalog.mkdir()
    with pytest.raises(
        ValueError,
        match="Knowledge state does not match",
    ):
        SimpleOptWorkflow(
            cwd=str(workspace),
            runtime_factory=lambda _: FakeRuntime([]),
        ).run({
            "definition_name": "demo_add",
            "catalog_name": str(trace_root),
            "target_hardware": "A100",
            "eval_server_url": "http://localhost:8000",
            "knowledge_catalog_path": other_catalog,
        })
    preserved_state = WorkspaceKnowledgeState.model_validate_json(
        layout.state.read_text(encoding="utf-8")
    )
    assert Path(preserved_state.catalog_ref) == knowledge_catalog.resolve()


def test_default_kernelgen_server_catalog_is_available():
    from kernelgen_server import Catalog

    defaults = SimpleOptInput(definition_name="addmm_")
    assert "trace_root" not in SimpleOptInput.model_fields
    assert "trace_set_key" not in SimpleOptInput.model_fields
    assert "eval_transport_timeout_seconds" not in SimpleOptInput.model_fields
    assert defaults.eval_timeout_seconds == 1500
    catalog = Catalog(resolve_builtin_catalog_path(defaults.catalog_name))
    operator = catalog.load(defaults.definition_name)

    assert defaults.catalog_name == DEFAULT_CATALOG_NAME
    assert defaults.target_hardware == "Ascend910B"
    assert defaults.profile_enabled is False
    assert defaults.reference_code_path is None
    assert defaults.reference_code_prompt_path is None
    assert defaults.knowledge_catalog_path is None
    assert operator.definition.name == "addmm_"
    assert operator.evaluator == "flaggems"
    assert operator.correctness_workloads == []
    assert operator.timing_workloads == []


def test_reference_code_legacy_input_alias_and_prompt_dependency(tmp_path):
    reference = tmp_path / "reference.py"
    legacy = SimpleOptInput(
        definition_name="addmm_",
        reference_code=reference,
    )
    assert legacy.reference_code_path == reference
    assert legacy.reference_code == reference

    with pytest.raises(
        ValueError,
        match="reference_code_prompt_path requires reference_code_path",
    ):
        SimpleOptInput(
            definition_name="addmm_",
            reference_code_prompt_path=tmp_path / "reference.md",
        )


def test_flaggems_adapter_catalog_preserves_public_abi():
    from kernelgen_server import Catalog, builtin_catalog_path

    operator = Catalog(builtin_catalog_path("flaggems-adapter-definitions")).load("addmm_")
    parameters = {item.name: item for item in operator.definition.parameters}

    assert list(parameters) == ["self", "mat1", "mat2", "beta", "alpha"]
    assert parameters["beta"].kind.value == "keyword_only"
    assert parameters["beta"].default == 1
    assert parameters["alpha"].kind.value == "keyword_only"
    assert parameters["alpha"].default == 1


def test_dps_inference_uses_run_parameter_count():
    value_returning = DefinitionModel.model_validate(_RETURN_DEFINITION)
    assert infer_destination_passing_style(value_returning) is False

    dps_definition = dict(
        _RETURN_DEFINITION,
        reference="def run(x, y, output):\n    output.copy_(x + y)\n",
    )
    assert infer_destination_passing_style(
        DefinitionModel.model_validate(dps_definition)
    ) is True


def test_extract_opt_keeps_operator_oriented_contract(tmp_path):
    workflow = ExtractOptWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: FakeRuntime([]),
    )
    inp = ExtractOptInput(operator="add", trace_root=str(tmp_path / "trace"))
    assert workflow.name == "extract_opt"
    assert inp.operator == "add"


def test_extract_opt_keeps_stop_controls_out_of_analysis(tmp_path):
    workflow = ExtractOptWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: FakeRuntime([]),
    )
    inp = ExtractOptInput(
        operator="add",
        trace_root=str(tmp_path / "trace"),
        early_stop_rounds=7,
        min_rounds=4,
        max_round=6,
    )
    extracted = ExtractorResult(
        group="pointwise",
        definition=ExtractedDefinition(
            name="demo_add",
            inputs=["x", "y"],
            outputs=["output"],
            reference=_RETURN_DEFINITION["reference"],
        ),
        correctness_workloads=[],
        timing_workloads=[],
    )

    coder_input = workflow._build_coder_inputs(
        [extracted], inp, tmp_path / "trace"
    )[0]

    assert coder_input["analysis"] == {}
    assert coder_input["workloads"] == []
    assert coder_input["early_stop_rounds"] == 7
    assert coder_input["min_rounds"] == 4
    assert coder_input["max_round"] == 6


def test_bound_evaluation_rejects_a_kernelgen_matched_ratio_override(tmp_path):
    workflow = SingleCoderOptimizationWorkflow(
        cwd=str(tmp_path),
        runtime_factory=lambda _: FakeRuntime([]),
    )
    inp = workflow.InputModel.model_validate(
        {
            "definition": _RETURN_DEFINITION,
            "target_hardware": "A100",
            "catalog_name": "fixture",
            "evaluation_contract": {"required_matched_ratio": 0.9},
        }
    )
    target = TargetContext(backend="cuda", device="A100")

    with pytest.raises(ValueError, match="Server catalog Workload"):
        workflow._build_tool_context(
            inp,
            target,
            evaluation_snapshot_path=None,
        )


if __name__ == "__main__":
    import inspect
    import traceback

    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_")]
    passed = 0
    for test in tests:
        try:
            if "tmp_path" in inspect.signature(test).parameters:
                with tempfile.TemporaryDirectory() as directory:
                    test(Path(directory))
            else:
                test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except Exception:
            print(f"  ✗ {test.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
