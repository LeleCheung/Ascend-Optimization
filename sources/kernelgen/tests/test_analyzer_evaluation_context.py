"""Shared analysis consumes frozen evaluation inputs and remote facts, host-only."""

import json

import pytest

from kernelgen.agents.analyzer import AnalyzerAgent, AnalyzerOutput
from kernelgen.agents._workload_prompt import sample_workloads_for_prompt
from kernelgen.data.evaluation_snapshot import CatalogEvaluationSnapshot, snapshot_optimization_context
from kernelgen.framework import FakeRuntime
from kernelgen.framework.run_control import WorkspaceRunControl
from kernelgen.workflows.optimization.kernelgen import KernelGenInput
from kernelgen.workflows.optimization.kernelgen import epoch, preparation


@pytest.fixture
def snapshot():
    return CatalogEvaluationSnapshot(
        catalog_name="uploaded-fixture", catalog_api_version="v6.2",
        definition_name="attention", bundle_id="sha256:" + "a" * 64,
        benchmark_fingerprint="frozen-fingerprint",
        definition={
            "name": "attention", "parameters": [{"name": "q", "required": True}],
            "outputs": ["out"], "reference": "def run(q): return q",
        },
        correctness_workloads=[{
            "name": "corr", "seed": 17,
            "inputs": {"q": {"type": "random", "shape": [32, 16, 16384, 64], "dtype": "float16"}},
        }],
        timing_workloads=[{
            "name": "time", "seed": 19,
            "inputs": {"q": {"type": "random", "shape": [1, 16, 2048, 64], "dtype": "float16"}},
        }],
    )


def make_input(snapshot):
    definition, _ = snapshot_optimization_context(snapshot)
    return KernelGenInput(
        definition=definition, target_hardware="Ascend910B4-1",
        catalog_name=snapshot.catalog_name,
        evaluation_snapshot=snapshot.model_dump(mode="json"),
    )


@pytest.mark.parametrize("cross_epoch_knowledge", [False, True])
def test_analyzer_and_coders_use_identical_frozen_workloads(tmp_path, monkeypatch, snapshot, cross_epoch_knowledge):
    def unexpected(*args, **kwargs):
        raise AssertionError("must not reload the local Catalog for a frozen bundle")
    monkeypatch.setattr(epoch, "load_catalog_optimization_context", unexpected)
    inp = make_input(snapshot).model_copy(update={"cross_epoch_knowledge": cross_epoch_knowledge})
    create_workspace = preparation.create_epoch_workspace
    inherited = []

    def tracked_workspace(*args, **kwargs):
        inherited.append(kwargs["inherit_legacy_knowledge"])
        return create_workspace(*args, **kwargs)

    monkeypatch.setattr(preparation, "create_epoch_workspace", tracked_workspace)
    runtime = FakeRuntime([AnalyzerOutput(core_math="frozen attention").model_dump_json()])
    remote = {"target": {"device": "Ascend910B4-1", "devices": [{"num_sm": 20}]}}
    analysis = preparation.prepare_analysis(
        tmp_path, inp, None, lambda path: runtime, WorkspaceRunControl(tmp_path),
        resume=False, evaluation_snapshot=inp.evaluation_snapshot, server_context=remote,
    )
    prompt = runtime.calls[0]["prompt"]
    payload = prompt.split("<workloads>\n", 1)[1].split("\n</workloads>", 1)[0]
    shown = json.loads(payload.split("\n", 1)[1])
    coder = epoch.build_coder_input(inp, analysis, evaluation_snapshot=inp.evaluation_snapshot)
    assert shown == coder["workloads"]
    assert [item["phase"] for item in shown] == ["correctness", "timing"]
    assert shown[0]["inputs"]["q"]["shape"] == [32, 16, 16384, 64]
    assert [item["seed"] for item in shown] == [17, 19]
    assert json.dumps(remote, ensure_ascii=False) in prompt
    assert "Missing metadata is unknown" in prompt
    assert "Never import or probe the Agent host's Torch" in prompt
    assert inherited == [cross_epoch_knowledge]


def test_server_facts_are_from_the_same_validation_request(monkeypatch, snapshot):
    inp = make_input(snapshot)
    status = {
        "api_version": "v6.2", "backend": "npu", "timing": "profiler",
        "target": {"backend": "ascend", "device": "Ascend910B4-1", "devices": [{"num_sm": 20}]},
        "metadata": {"complete": False, "missing": ["software.language_version"]},
        "scheduler": {"active": 5}, "debug": {"workspace_root": "/private"},
    }
    calls = []
    def get_status(url):
        calls.append(url)
        return status
    monkeypatch.setattr(preparation.server_adapter, "get_service_status", get_status)
    resolved, facts = preparation.resolve_server_target(inp)
    assert len(calls) == 1
    assert resolved.target_hardware == "Ascend910B"
    assert facts["target"]["device"] == "Ascend910B4-1"
    assert facts["metadata"] == status["metadata"]
    assert "scheduler" not in facts and "debug" not in facts
    status["backend"] = "cuda"
    with pytest.raises(ValueError, match="backend"):
        preparation.resolve_server_target(inp)


def test_resume_reuses_analysis_without_resolving_new_inputs(tmp_path, monkeypatch, snapshot):
    checkpoint = tmp_path / "1R/shared_analysis/analysis.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text(AnalyzerOutput(core_math="original checkpoint").model_dump_json())
    def unexpected(*args, **kwargs):
        raise AssertionError("valid resume checkpoint must not be regenerated")
    monkeypatch.setattr(preparation, "resolve_optimization_context", unexpected)
    analysis = preparation.prepare_analysis(
        tmp_path, make_input(snapshot), None, unexpected, WorkspaceRunControl(tmp_path), resume=True,
    )
    assert analysis.core_math == "original checkpoint"


def test_absent_context_is_explicit_and_does_not_probe_local_runtime(snapshot):
    inp = make_input(snapshot)
    runtime = FakeRuntime([AnalyzerOutput().model_dump_json()])
    AnalyzerAgent().run({"definition": inp.definition, "target_hardware": inp.target_hardware}, runtime)
    prompt = runtime.calls[0]["prompt"]
    assert "Empty means unavailable, not unrestricted" in prompt
    assert "Missing metadata is unknown" in prompt
    assert "\n[]\n</workloads>" in prompt


def test_changed_snapshot_identity_is_rejected_before_analysis(tmp_path, snapshot):
    inp = make_input(snapshot).model_copy(update={"catalog_name": "different"})
    with pytest.raises(ValueError, match="identities differ"):
        preparation.prepare_analysis(
            tmp_path, inp, None, lambda path: FakeRuntime([]), WorkspaceRunControl(tmp_path),
            resume=False, evaluation_snapshot=snapshot.model_dump(mode="json"),
        )


def test_prompt_sampling_is_bounded_phase_balanced_and_non_mutating():
    items = [{"phase": phase, "uuid": f"{phase}-{i}"}
             for phase, count in (("correctness", 80), ("timing", 40)) for i in range(count)]
    before = json.dumps(items)
    shown = sample_workloads_for_prompt(items)
    assert len(shown) == 10
    assert [item["uuid"] for item in shown if item["phase"] == "correctness"] == [
        "correctness-0", "correctness-20", "correctness-40", "correctness-59", "correctness-79",
    ]
    assert shown[-1]["uuid"] == "timing-39"
    assert json.dumps(items) == before
