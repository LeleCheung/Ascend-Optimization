"""Historical flat-workspace SimpleOpt execution; not a new-run entry."""

from pathlib import Path
from typing import Any, Callable

from kernelgen.framework.workflow import Workflow
from kernelgen.framework.run_control import RunControl, WorkspaceRunControl
from kernelgen.workflows.lifecycle import run_workflow
from kernelgen.workflows.optimization.single_coder import SingleCoderOptimizationOutput, SingleCoderOptimizationWorkflow
from kernelgen.workflows.optimization.options import SimpleOptInput
from kernelgen.workflows.legacy.simple_opt.preparation import prepare_optimization


class SimpleOptWorkflow(Workflow):
    """Load one built-in Catalog Definition and run its optimization lifecycle."""

    name = "simple_opt"
    InputModel = SimpleOptInput
    OutputModel = SingleCoderOptimizationOutput

    def __init__(
        self,
        *,
        cwd: str = ".",
        runtime_factory: Callable[[str], Any] = None,
        run_control: RunControl | None = None,
    ):
        self._cwd = Path(cwd)
        self._runtime_factory = runtime_factory
        self._run_control = run_control or WorkspaceRunControl(
            self._cwd,
            source="workflow:simple_opt",
        )

    @classmethod
    def bind(cls, path: str, runtime_factory) -> "SimpleOptWorkflow":
        return cls(cwd=path, runtime_factory=runtime_factory)

    def _execute(self, inp: SimpleOptInput) -> SingleCoderOptimizationOutput:
        return run_workflow(
            self._run_control, name=self.name, mode="simple_opt",
            definition_name=inp.definition_name, max_round=inp.max_round,
            checkpoint="BEFORE_SIMPLE_OPT", output_model=self.OutputModel,
            execute=lambda: self._execute_controlled(inp),
        )

    def _execute_controlled(self, inp: SimpleOptInput) -> SingleCoderOptimizationOutput:
        coder_input = prepare_optimization(inp, self._cwd)
        return SingleCoderOptimizationWorkflow(
            cwd=str(self._cwd),
            runtime_factory=self._runtime_factory,
            run_control=self._run_control,
            run_mode="simple_opt",
        ).run(coder_input)
