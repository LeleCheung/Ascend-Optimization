"""Unified operator optimization and its single-/multi-Coder implementations.

Keep package import lightweight so engines do not import their parent workflow.
"""

__all__ = ["OperatorOptimizeInput", "OperatorOptimizeWorkflow"]


def __getattr__(name):
    if name in __all__:
        from . import workflow
        return getattr(workflow, name)
    raise AttributeError(name)
