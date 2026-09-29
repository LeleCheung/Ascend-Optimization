"""AutoGenAgent: generates a new Triton operator for FlagGems in one session.

Given an operator name, the agent:
1. Understands the operator semantics (PyTorch API + aten schema)
2. Implements the operator in FlagGems style (pointwise_dynamic / manual kernel)
3. Registers it (ops/__init__.py + __init__.py _FULL_CONFIG)
4. Writes accuracy tests + runs them
5. Runs pre-commit + benchmark
6. Outputs structured JSON result

This is a "long-session" agent (like CoderAgent): one invoke does everything.
Its static role lives in ``.kernelgen/agents/kernel-auto-gen.md``; Python injects
the operator input and output contract for each invocation.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from kernelgen.framework.base import BaseAgent


# ---------------------------------------------------------------------------
# I/O contract
# ---------------------------------------------------------------------------

class AutoGenInput(BaseModel):
    """Input: which operator to generate."""
    operator: str = Field(description="Operator name, e.g. 'relu', 'gelu', 'silu'")


class AutoGenOutput(BaseModel):
    """Structured output the agent must produce at the end."""
    operator: str = Field(description="Operator name")
    status: str = Field(description="'success' or 'failed'")
    accuracy_passed: Optional[bool] = Field(default=None, description="Whether accuracy tests passed")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    files_created: List[str] = Field(default_factory=list, description="Files created")
    files_modified: List[str] = Field(default_factory=list, description="Files modified")
    aten_ops_registered: List[str] = Field(default_factory=list, description="Registered aten ops")
    implementation_mode: Optional[str] = Field(
        default=None,
        description="'pointwise_dynamic' or 'manual_kernel' or 'autograd_function'"
    )
    test_results: Optional[dict] = Field(default=None, description="Test results summary")
    benchmark_results: Optional[dict] = Field(default=None, description="Benchmark results")
    notes: str = Field(default="", description="Additional notes")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AutoGenAgent(BaseAgent):
    """FlagGems operator generation agent.

    Claude uses the native role; other runtimes receive its body as fallback.
    The agent runs autonomously in a workspace: implements, tests, benchmarks.
    """

    name = "auto_gen"
    InputModel = AutoGenInput
    OutputModel = AutoGenOutput
