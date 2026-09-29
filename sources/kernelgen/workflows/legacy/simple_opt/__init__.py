"""Legacy SimpleOptWorkflow for historical flat-workspace campaigns only.

New tasks use OperatorOptimizeWorkflow. Historical execution remains:

    definition_name -> load definition/workloads -> SingleCoderOptimizationWorkflow

There is no extraction, analyzer, fan-out, cross-agent selection, or multi-epoch
coordination.  The workflow directory is the optimizer's workspace, so the
authoritative ledger and best kernel are written directly beneath ``cwd``.
V1 Knowledge is opt-in: an enabled run materializes query context in the same
workspace and selects the Knowledge Coder, but does not publish an epoch or
promote a Solution.
"""

from kernelgen.workflows.optimization.options import SimpleOptInput
from kernelgen.workflows.legacy.simple_opt.workflow import SimpleOptWorkflow

__all__ = ["SimpleOptInput", "SimpleOptWorkflow"]
