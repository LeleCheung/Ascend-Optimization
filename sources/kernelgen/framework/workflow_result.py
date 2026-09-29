"""Results and invocation context for durable Workflow composition."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .run_control import WorkspaceRunControl

Output = TypeVar("Output")


class WorkflowResult(BaseModel, Generic[Output]):
    """Execution disposition plus the callable's own business output.

    RUNNING progress belongs to RunControl, not a returned result. A successful
    review with blocking findings returns WAITING with its complete report.
    """

    model_config = ConfigDict(extra="forbid")
    state: Literal["SUCCEEDED", "FAILED", "WAITING", "CANCELLED"] = "SUCCEEDED"
    output: Output
    message: str = ""
    simulated: bool = False


@dataclass(frozen=True)
class WorkflowContext:
    """A named invocation; workspace is fresh unless the optimizer owns recovery."""

    operator: str
    stage: str  # Persisted scope name; not a separate business object.
    workspace: Path
    attempt: int
    inputs: tuple[Path, ...]
    control: WorkspaceRunControl

    @property
    def outputs(self) -> dict[str, Any]:
        """Read successful upstream outputs without a second persisted index."""
        receipts = (WorkflowReceipt.model_validate_json(path.read_text()) for path in self.inputs)
        return {receipt.stage: receipt.returned().output for receipt in receipts}


class WorkflowSummary(BaseModel):
    operator: str
    workspace: str
    reports: list[str] = Field(default_factory=list)


class _StoredResult(BaseModel):
    """Disk codec only: preserve committed receipt bytes and hashes on resume."""

    model_config = ConfigDict(extra="forbid")
    simulated: bool = True
    outcome: Literal["COMPLETED", "FAILED", "WAITING", "CANCELLED"] = "COMPLETED"
    message: str = "Dummy result; no business validation performed"
    data: Any = Field(default_factory=dict)


class WorkflowReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    operator: str
    stage: str
    attempt: int = Field(ge=1)
    input_sha256: str
    result: _StoredResult

    def returned(self) -> WorkflowResult:
        result = self.result
        return WorkflowResult(state="SUCCEEDED" if result.outcome == "COMPLETED" else result.outcome,
                              output=result.data, message=result.message, simulated=result.simulated)

    @classmethod
    def from_returned(cls, *, returned: WorkflowResult, **identity):
        serialized = returned.model_dump(mode="json")
        return cls(**identity, result=_StoredResult(
            outcome="COMPLETED" if returned.state == "SUCCEEDED" else returned.state,
            data=serialized["output"], message=returned.message, simulated=returned.simulated,
        ))
