"""Workflow build composes typed child workflows and plain function calls."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from kernelgen.framework.workflow import Workflow, WorkflowResult, WorkflowSummary
from kernelgen.framework.workflow_result import WorkflowReceipt


class Composition(Workflow):
    OutputModel = WorkflowResult[WorkflowSummary]

    def __init__(self, root, calls):
        self.root, self.calls = root, calls

    def build(self, inp):
        return self.calls

    def execute(self, *, resume=False, plan=None):
        inp = SimpleNamespace(operator="add", dummy=False, resume=resume)
        plan = plan or {"operator": "add", "simulated": False, "stages": [name for name, _ in self.calls]}
        return self.run_build(inp, plan=plan, mode="test")


@pytest.mark.parametrize("state", ["FAILED", "WAITING", "CANCELLED"])
def test_completed_calls_skip_and_incomplete_calls_restart(tmp_path, state):
    calls = []
    class Prepared(BaseModel):
        value: int

    class Prepare(Workflow):
        class InputModel(BaseModel):
            pass

        OutputModel = WorkflowResult[Prepared]

        def _execute(self, inp):
            return WorkflowResult[Prepared](output=Prepared(value=42))

    def prepare(context):
        calls.append((context.stage, context.attempt))
        result = Prepare().run({})
        assert isinstance(result.output, Prepared)
        return result

    def review(context):
        calls.append((context.stage, context.attempt))
        assert context.outputs["prepare"] == {"value": 42}
        return WorkflowResult(state=state if context.attempt == 1 else "SUCCEEDED", output={"reviewed": True})

    workflow = Composition(tmp_path, (("prepare", prepare), ("review", review)))
    first = workflow.execute()
    assert first.state == state
    original = {path: path.read_bytes() for path in map(Path, first.output.reports)}
    resumed = workflow.execute(resume=True)
    assert resumed.state == "SUCCEEDED"
    assert calls == [("prepare", 1), ("review", 1), ("review", 2)]
    assert all(path.read_bytes() == content for path, content in original.items())
    assert Path(resumed.output.reports[-1]).parent.name == "02"
    calls.clear()
    assert workflow.execute(resume=True).state == "SUCCEEDED"
    assert not calls


def test_existing_receipt_codec_preserves_resume_contract():
    stored = dict(schema_version="1.0", operator="add", stage="prepare", attempt=1,
                  input_sha256="abc", result=dict(simulated=False, outcome="COMPLETED", message="ready", data={"value": 42}))
    receipt = WorkflowReceipt.model_validate(stored)
    returned = receipt.returned()
    assert returned.state == "SUCCEEDED" and returned.output == {"value": 42}
    assert WorkflowReceipt.from_returned(returned=returned, **{k: v for k, v in stored.items() if k != "result"}).model_dump() == stored


@pytest.mark.parametrize("names", [(), ("a", "a"), ("../escape",), ("a/b",)])
def test_invalid_build_does_not_create_state(tmp_path, names):
    with pytest.raises(ValueError):
        Composition(tmp_path, tuple((name, lambda ctx: WorkflowResult(output={})) for name in names)).execute()
    assert not list(tmp_path.iterdir())


def test_build_must_match_persisted_plan(tmp_path):
    calls = (("first", lambda ctx: WorkflowResult(output={})),)
    with pytest.raises(ValueError, match="must match the persisted plan"):
        Composition(tmp_path, calls).execute(plan={"operator": "add", "simulated": False, "stages": ["other"]})
    assert not list(tmp_path.iterdir())


def test_plain_return_is_not_silently_successful(tmp_path):
    with pytest.raises(TypeError, match="must return WorkflowResult"):
        Composition(tmp_path, (("bad", lambda ctx: {"state": "FAILED"}),)).execute()
