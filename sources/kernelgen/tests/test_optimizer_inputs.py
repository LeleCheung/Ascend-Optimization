"""Input equivalence at the shared optimizer boundary; no Agent or KGS calls."""

from copy import deepcopy

import pytest
from pydantic import BaseModel

from kernelgen.framework.models import DefinitionModel
from kernelgen.workflows.legacy import simple_opt
from kernelgen.workflows.legacy.simple_opt import preparation
from kernelgen.workflows.optimization.kernelgen import epoch
from kernelgen.workflows.optimization.kernelgen.contracts import KernelGenInput
from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationInput, SingleCoderOptimizationWorkflow
from kernelgen.workflows.optimization.single_coder.inputs import build_optimizer_input


@pytest.fixture
def context():
    definition = DefinitionModel.model_validate({
        "name": "identity", "op_type": "elementwise", "axes": {},
        "inputs": {"x": {"shape": [16], "dtype": "float32"}},
        "outputs": {"out": {"shape": [16], "dtype": "float32"}},
        "reference": "def run(x):\n    return x\n",
    })
    workloads = [{"uuid": "c", "phase": "correctness"}, {"uuid": "t", "phase": "timing"}]
    return definition, workloads


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("dps", [None, False, True])
def test_simple_opt_preserves_exact_optimizer_arguments(tmp_path, monkeypatch, context, custom, dps):
    definition, workloads = context
    monkeypatch.setattr(preparation, "resolve_builtin_catalog_path", lambda _: tmp_path)
    monkeypatch.setattr(preparation, "load_catalog_optimization_context", lambda *_: context)
    monkeypatch.setattr(SingleCoderOptimizationWorkflow, "run", lambda _, args: args)
    monkeypatch.setattr(preparation, "materialize_knowledge", lambda *_: None)
    source = tmp_path / "reference.py"
    prompt = tmp_path / "reference.md"
    source.write_text("def reference(x): return x\n")
    prompt.write_text("Read-only reference evidence")
    options = dict(
        target_hardware="A100", eval_server_url="http://kgs:8001",
        profile_enabled=True, warmup_ms=0, benchmark_ms=321, num_trials=4,
        eval_timeout_seconds=91, early_stop_rounds=0, min_rounds=1,
        max_round=7, max_coder_sessions=2, reference_code_path=source,
        reference_code_prompt_path=prompt, knowledge_catalog_path=tmp_path / "kb",
    ) if custom else {}
    inp = simple_opt.SimpleOptInput(
        definition_name=definition.name, destination_passing_style=dps, **options,
    )
    before = inp.model_dump()
    args = simple_opt.SimpleOptWorkflow(cwd=str(tmp_path / "run"))._execute_controlled(inp)
    # Deliberately freeze the pre-refactor mapping, including absent/default fields.
    assert args == {
        "definition": definition.model_dump(exclude_none=True),
        "destination_passing_style": dps if dps is not None else False,
        "target_hardware": inp.target_hardware,
        "implementation_language": inp.implementation_language,
        "analysis": {}, "workloads": workloads, "evaluation_snapshot": None,
        "reference_code_source": source.read_text() if custom else "",
        "reference_code_prompt": prompt.read_text() if custom else "",
        "early_stop_rounds": inp.early_stop_rounds, "min_rounds": inp.min_rounds,
        "max_round": inp.max_round, "max_coder_sessions": inp.max_coder_sessions,
        "eval_server_url": inp.eval_server_url, "catalog_name": inp.catalog_name,
        "knowledge_enabled": custom, "profile_enabled": inp.profile_enabled,
        "warmup_ms": inp.warmup_ms, "benchmark_ms": inp.benchmark_ms,
        "num_trials": inp.num_trials, "eval_timeout_seconds": inp.eval_timeout_seconds,
        "eval_transport_timeout_seconds": inp.eval_timeout_seconds + 300,
    }
    SingleCoderOptimizationInput.model_validate(args)
    assert inp.model_dump() == before


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("dps", [False, True])
def test_kernel_gen_preserves_exact_optimizer_arguments(monkeypatch, context, custom, dps):
    definition, workloads = context
    monkeypatch.setattr(epoch, "resolve_builtin_catalog_path", lambda _: None)
    monkeypatch.setattr(epoch, "load_catalog_optimization_context", lambda *_: context)
    monkeypatch.setattr(epoch, "infer_destination_passing_style", lambda _: dps)
    inp = KernelGenInput(
        definition=definition, target_hardware="A100", n_parallel=4, n_epoch=2,
        timeout=77, initial_seed_code="original seed", initial_seed_is_validated_baseline=True,
        **(dict(profile_enabled=False, early_stop_rounds=0, min_rounds=1, max_round=6,
                evaluation_contract={"atol": 0.01, "rtol": 0.02}) if custom else {}),
    )
    before = inp.model_dump()
    analysis = {"assigned_direction": "tile size"}
    args = epoch.build_coder_input(inp, analysis, seed_code="epoch seed", knowledge_enabled=custom)
    assert args == {
        "definition": definition.model_dump(), "destination_passing_style": dps,
        "target_hardware": inp.target_hardware, "implementation_language": inp.implementation_language,
        "evaluation_contract": inp.evaluation_contract.model_dump(), "analysis": analysis,
        "workloads": workloads, "evaluation_snapshot": None,
        "eval_server_url": inp.eval_server_url, "catalog_name": inp.catalog_name,
        "knowledge_enabled": custom, "profile_enabled": inp.profile_enabled,
        "early_stop_rounds": inp.early_stop_rounds, "min_rounds": inp.min_rounds,
        "max_round": inp.max_round, "seed_code": "epoch seed", "seed_is_validated_baseline": False,
        "warmup_ms": inp.warmup_ms, "benchmark_ms": inp.benchmark_ms,
        "num_trials": inp.num_trials, "eval_timeout_seconds": inp.eval_timeout_seconds,
        "eval_transport_timeout_seconds": 1800, "max_coder_sessions": inp.max_coder_sessions,
        "reference_code_source": "", "reference_code_prompt": "",
    }
    validated = SingleCoderOptimizationInput.model_validate(args)
    assert validated.eval_timeout_seconds == 1500
    assert validated.eval_transport_timeout_seconds == 1800
    assert inp.model_dump() == before


def test_prepared_bundle_context_is_forwarded_without_loading_catalog(context, monkeypatch):
    definition, workloads = context
    monkeypatch.setattr(preparation, "resolve_builtin_catalog_path", lambda _: pytest.fail("must not reload Catalog"))
    inp = simple_opt.SimpleOptInput(definition_name=definition.name)
    snapshot = {"bundle_id": "sha256:" + "a" * 64, "definition": definition.model_dump()}
    before = deepcopy(snapshot)
    args = build_optimizer_input(
        inp, definition=definition.model_dump(), workloads=workloads,
        evaluation_snapshot=snapshot, catalog_name="uploaded-a", destination_passing_style=False,
    )
    assert args["catalog_name"] == "uploaded-a"
    assert args["evaluation_snapshot"] == before
    assert snapshot == before
    assert args["workloads"] == workloads


def test_prepared_analysis_accepts_model_in_kernel_gen(monkeypatch, context):
    class Analysis(BaseModel):
        summary: str = "analysis"

    monkeypatch.setattr(epoch, "resolve_builtin_catalog_path", lambda _: None)
    monkeypatch.setattr(epoch, "load_catalog_optimization_context", lambda *_: context)
    inp = KernelGenInput(definition=context[0], target_hardware="A100")
    assert epoch.build_coder_input(inp, Analysis())["analysis"] == {"summary": "analysis"}
