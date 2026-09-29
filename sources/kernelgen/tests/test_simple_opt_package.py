"""Legacy boundaries preserve reference guards and historical run ownership."""

import pytest

from kernelgen.framework.run_control import RunState, WorkspaceRunControl
from kernelgen.workflows.legacy.simple_opt import SimpleOptInput, SimpleOptWorkflow
from kernelgen.workflows.legacy.simple_opt import preparation, workflow
from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationOutput


def test_legacy_entrypoints_use_same_archived_class():
    from kernelgen.cli import legacy_simple_opt as run_example
    from kernelgen.workflows.legacy import batch_simple_opt_definition
    from kernelgen.workflows.optimization.options import SimpleOptInput as Contract

    assert SimpleOptInput is Contract
    assert run_example.SimpleOptWorkflow is batch_simple_opt_definition.SimpleOptWorkflow is SimpleOptWorkflow
    assert SimpleOptWorkflow.InputModel is SimpleOptInput
    assert SimpleOptWorkflow.OutputModel is SingleCoderOptimizationOutput
    assert SimpleOptWorkflow.name == "simple_opt"


@pytest.mark.parametrize("kind,message", [
    ("missing", "does not exist"), ("directory", "regular file"),
    ("large", "prompt limit"), ("encoding", "UTF-8"), ("empty", "is empty"),
])
def test_reference_validation_precedes_catalog_and_workspace(tmp_path, monkeypatch, kind, message):
    path = tmp_path / "reference.py"
    if kind == "directory":
        path.mkdir()
    elif kind == "large":
        path.write_bytes(b"x" * 1_000_001)
    elif kind == "encoding":
        path.write_bytes(b"\xff")
    elif kind == "empty":
        path.write_text(" \n")
    monkeypatch.setattr(preparation, "resolve_builtin_catalog_path", lambda _: pytest.fail("reference must be checked first"))
    workspace = tmp_path / "run"
    with pytest.raises(ValueError, match=message):
        preparation.prepare_optimization(SimpleOptInput(definition_name="identity", reference_code_path=path), workspace)
    assert not workspace.exists()


def test_orchestrator_forwards_workspace_runtime_and_control_once(tmp_path, monkeypatch):
    control = WorkspaceRunControl(tmp_path)
    factory = object()
    prepared = {"prepared": True}
    calls = []

    def prepare(inp, workspace):
        calls.append((inp.definition_name, workspace))
        return prepared

    class Optimizer:
        def __init__(self, **kwargs):
            assert kwargs == dict(cwd=str(tmp_path), runtime_factory=factory, run_control=control, run_mode="simple_opt")

        def run(self, args):
            assert args is prepared
            return SingleCoderOptimizationOutput(definition_name="identity", status="FAILED", workspace=str(tmp_path))

    monkeypatch.setattr(workflow, "prepare_optimization", prepare)
    monkeypatch.setattr(workflow, "SingleCoderOptimizationWorkflow", Optimizer)
    output = SimpleOptWorkflow(cwd=str(tmp_path), runtime_factory=factory, run_control=control).run({"definition_name": "identity"})
    assert output.status == "FAILED"
    assert calls == [("identity", tmp_path)]
    assert control.progress().state == RunState.FAILED
    assert [event.event_type for event in control.read_events()].count("WORKFLOW_COMPLETED") == 1
