"""Convert one FlagGems operator's pytest suites to the KernelGen contract.

The agent intentionally handles exactly one operator per invocation.  Serial
coordination, repository scope checks, and human review gates live in
``examples/pytest_converter/run_one.py``.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from kernelgen.framework.base import BaseAgent


class PytestConverterInput(BaseModel):
    """Authoritative scope for one conversion."""

    source_operator: str = Field(
        description="Operator spelling from kernel_todo_v2/20260826.csv"
    )
    operator: str = Field(
        description="Canonical FlagGems catalog id and gems_op override key"
    )
    pytest_mark: str = Field(
        description="Exact pytest marker selecting this operator's test functions"
    )
    accuracy_files: List[str] = Field(
        description="Repository-relative correctness pytest files allowed to change"
    )
    benchmark_files: List[str] = Field(
        description="Repository-relative performance pytest files allowed to change"
    )
    shared_files: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Target files also used by other catalog operators",
    )
    review_feedback: List[str] = Field(
        default_factory=list,
        description="Issues found by the human reviewer on the previous attempt",
    )
    standard_path: str = Field(
        default="docs/content/zh-cn/testing/kernelgen-integration.md",
        description="Repository-relative authoritative conversion standard",
    )
    run_tests: bool = Field(
        default=True,
        description="Whether to run the target accuracy and benchmark checks",
    )


class PytestCommandResult(BaseModel):
    """One command actually attempted by the conversion agent."""

    command: str
    exit_code: Optional[int] = None
    outcome: Literal["passed", "failed", "skipped"]
    summary: str = ""


class PytestConverterOutput(BaseModel):
    """Structured handoff for the mandatory human review."""

    source_operator: str
    operator: str
    status: Literal["converted", "already_compliant", "blocked", "failed"]
    files_modified: List[str] = Field(default_factory=list)
    commands: List[PytestCommandResult] = Field(default_factory=list)
    remaining_issues: List[str] = Field(default_factory=list)
    summary: str = ""


class PytestConverterAgent(BaseAgent):
    """Convert one existing accuracy/benchmark pair without changing semantics."""

    name = "pytest_converter"
    InputModel = PytestConverterInput
    OutputModel = PytestConverterOutput


__all__ = [
    "PytestCommandResult",
    "PytestConverterAgent",
    "PytestConverterInput",
    "PytestConverterOutput",
]
