"""OptimizeAgent: single-round kernel performance optimization.

Given an operator name (and workspace with performance history), the agent:
1. Reads PERFORMANCE.md + past versions to understand history
2. Picks ONE new optimization direction
3. Modifies the kernel code
4. Runs accuracy tests (fix until pass)
5. Runs benchmark, records speedup
6. Outputs structured JSON result

The agent does exactly ONE optimization round. Python controls the outer loop
(OptimizeWorkflow calls this agent repeatedly, saving versions + updating
PERFORMANCE.md between rounds, stopping when target speedup is reached or
max iterations exhausted).

Its static role lives in ``.kernelgen/agents/kernel-optimize.md``; Python injects
the operator input and output contract for each invocation.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from kernelgen.framework.base import BaseAgent


# ---------------------------------------------------------------------------
# I/O contract
# ---------------------------------------------------------------------------

class OptimizeInput(BaseModel):
    """Input: which operator to optimize this round."""
    operator: str = Field(description="Operator name, e.g. 'softmax', 'layernorm'")


class OptimizeOutput(BaseModel):
    """Structured output from one optimization round."""
    operator: str = Field(description="Operator name")
    status: str = Field(description="'success' or 'failed'")
    speedup: Optional[float] = Field(default=None, description="Speedup achieved this round")
    test_passed: Optional[bool] = Field(default=None, description="Whether accuracy tests passed")
    kernel_path: Optional[str] = Field(default=None, description="Kernel file path relative to workspace")
    optimization_direction: Optional[str] = Field(default=None, description="What optimization was tried")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class OptimizeAgent(BaseAgent):
    """Single-round kernel optimization agent.

    Claude uses the native role; other runtimes receive its body as fallback.
    Each invocation does ONE optimization step. The workspace persists between
    rounds (PERFORMANCE.md, versions/), so the agent sees its own history.
    """

    name = "optimize"
    InputModel = OptimizeInput
    OutputModel = OptimizeOutput
