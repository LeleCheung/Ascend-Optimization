"""Workflow: an orchestrated graph of Runnables, itself a Runnable (ADR-3 #9).

A Workflow IS a Runnable: it has an InputModel/OutputModel and is called via
``run(inp) -> validated output`` — identical to a single agent. Its ``_execute``
orchestrates inner Runnables (agents, tools, or OTHER workflows) plus plain Python
(pick_best, loops, conditionals). Because it satisfies the same contract, a whole
workflow can be nested inside a bigger one, and I/O is validated at every boundary
— which is what makes modules (kernel-gen / pr / kernel-extract) composable.

Design principle (aligned with KGRunner "Python is the orchestration language"):
- Heavy steps that need scheduling (agents; parallel/GPU/worktree Python) go
  through the executor (run / run_parallel).
- Light pure functions (pick_best, pick_seed, should_stop) are called DIRECTLY as
  plain Python inside _execute — NOT wrapped as nodes. Wrapping a same-process pure
  function adds ceremony with no benefit.

Subclasses set InputModel/OutputModel and implement ``_execute(inp, runtime)``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from kernelgen.framework.runnable import Runnable
from kernelgen.framework.workflow_result import WorkflowContext, WorkflowResult, WorkflowSummary


class Workflow(Runnable):
    """Base class for orchestrated, I/O-validated, composable modules.

    Subclass responsibilities:
      - set InputModel / OutputModel (the module's contract),
      - implement bind(path, runtime_factory) classmethod (how to construct from workspace),
      - implement _execute(inp, runtime): orchestrate inner Runnables + plain Python,
        return an OutputModel instance or a dict OutputModel can validate.
    """

    name: str = ""

    def build(self, inp):
        """Declare named child calls only for workflows using durable composition.

        Each callable receives WorkflowContext and returns WorkflowResult. Plain
        workflows may keep their existing _execute implementation without build.
        """
        raise NotImplementedError(f"{type(self).__name__} does not declare a composition")

    def run_build(self, inp, *, plan, mode, validate_artifacts=None):
        """Execute this workflow's declared calls with shared ownership/recovery."""
        from kernelgen.framework.workflow_execution import _WorkflowExecution

        calls = tuple(self.build(inp))
        plan = {"stages": [name for name, _ in calls], **plan}
        return _WorkflowExecution(
            cwd=self.root, calls=calls, mode=mode,
            progress_kinds=getattr(self, "call_progress_kinds", {}),
            validate_artifacts=validate_artifacts,
        ).run(operator=inp.operator, plan=plan, simulated=inp.dummy, resume=inp.resume)

    @classmethod
    def bind(cls, path: str, runtime_factory) -> "Workflow":
        """Default: construct with cwd + runtime_factory. Subclasses override if
        their __init__ takes different params."""
        return cls(cwd=path, runtime_factory=runtime_factory)

    def _execute(self, inp: BaseModel) -> Any:  # pragma: no cover - abstract-ish
        raise NotImplementedError(
            f"{type(self).__name__} must implement _execute(inp)"
        )
