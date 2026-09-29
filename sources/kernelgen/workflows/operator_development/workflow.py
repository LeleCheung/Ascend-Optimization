"""Operator development build, executed by the common Workflow machinery."""

from pathlib import Path

from kernelgen.framework.workflow import Workflow
from kernelgen.framework.workflow_execution import cancel_lifecycle, lifecycle_status

from .contracts import (
    STAGES, DummyWorkflowCall, OperatorDevelopmentInput, OperatorDevelopmentOutput,
    OperatorWorkflowContext,
)


class OperatorDevelopmentWorkflow(Workflow):
    name = "operator_lifecycle"
    InputModel = OperatorDevelopmentInput
    OutputModel = OperatorDevelopmentOutput

    def __init__(self, *, cwd=".", runtime_factory=None, workflow_call=None):
        self.root = Path(cwd).expanduser().resolve()
        self.workflow_call = workflow_call or DummyWorkflowCall()

    def build(self, inp):
        # The pilot uses one dummy adapter; real business stages supply their own callables.
        def invoke(context):
            return self.workflow_call(OperatorWorkflowContext(
                **vars(context), optimize=inp.optimize if context.stage == "optimize" else None,
            ))

        return tuple((name, invoke) for name in STAGES if name in inp.stages)

    def _execute(self, inp):
        if not inp.dummy:
            raise NotImplementedError("operator_lifecycle only supports explicit dummy=True")
        plan = {"schema_version": "1.1", "operator": inp.operator, **inp.stage_plan(), "simulated": inp.dummy}
        return self.run_build(inp, plan=plan, mode="dummy")
