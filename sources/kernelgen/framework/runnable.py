"""Runnable: the shared contract for Agents AND Workflows (ADR-3 #9).

A ``Runnable`` is anything with a declared input/output Pydantic contract that can
be ``run(inp, runtime) -> validated output``. Both a single ``BaseAgent`` (one LLM
invoke) and a ``Workflow`` (an orchestrated graph of other Runnables) satisfy it —
so a whole workflow can be called, I/O-validated, and composed exactly like a
single agent. This is what makes modules (kernel-gen / pr / kernel-extract)
composable: each is a Runnable; a workflow can nest another workflow.

``run()`` is a template method: it validates input, delegates to the subclass's
``_execute``, and validates output. Subclasses implement ONLY ``_execute``:
- BaseAgent._execute: build prompt -> runtime.invoke -> parse/validate + repair-retry
- Workflow._execute:  orchestrate inner Runnables + plain Python (pick_best, etc.)

The second argument ``runtime`` is the LLM-invocation handle (Runtime protocol).
Agents use it to invoke; workflows typically ignore it (they build their own
runtimes internally via runtime_factory). It defaults to None so workflows can be
called without a runtime: ``workflow.run(inp)`` — the orchestrator has its own.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type

from pydantic import BaseModel


class Runnable(ABC):
    """Shared I/O-validated execution contract for Agents and Workflows."""

    InputModel: Type[BaseModel] = BaseModel
    OutputModel: Type[BaseModel] = BaseModel

    @classmethod
    def bind(cls, path: str, runtime_factory) -> "Runnable":
        """Construct an instance bound to a workspace path. Subclasses implement
        this so run_parallel can allocate → bind → run without knowing whether
        the Runnable is an Agent or Workflow (polymorphism, not if/else)."""
        raise NotImplementedError(f"{cls.__name__} must implement bind(path, runtime_factory)")

    def run(self, inp: Any, runtime: Any = None) -> BaseModel:
        """Validate input -> _execute -> validate output. The single entry point
        callers use, identical for an agent or a whole workflow."""
        validated_in = self.InputModel.model_validate(inp)
        if runtime is not None and hasattr(self, '_runtime'):
            self._runtime = runtime   # explicit runtime overrides bind()'s
        elif runtime is not None:
            self._runtime = runtime
        out = self._execute(validated_in)
        # _execute may already return a validated OutputModel (agents do); otherwise
        # it returns a dict/obj we validate here (workflows returning a plain dict).
        if isinstance(out, self.OutputModel):
            return out
        return self.OutputModel.model_validate(out)

    @abstractmethod
    def _execute(self, inp: BaseModel) -> Any:
        """Do the work. Return an OutputModel instance or something OutputModel
        can validate (dict). Subclass-specific: agent = LLM invoke; workflow =
        orchestrate inner Runnables."""
        ...
