"""Typed stage-owned views of the existing control and ledger facts.

These models are projections, not a second progress store. Producers explicitly
select a kind; adapters never infer a stage's semantics from its name or values.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProgressKind = Literal["basic", "rounds", "epochs", "tasks"]


class BasicProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoundProgress(BasicProgress):
    current_round: int | None = Field(default=None, ge=0)
    completed_rounds: int = Field(default=0, ge=0)
    max_round: int | None = Field(default=None, ge=1)
    best_round: int = Field(default=0, ge=0)
    best_geo_mean: float | None = None
    last_evaluation_status: str = ""


class TaskProgress(BasicProgress):
    total_tasks: int | None = Field(default=None, ge=0)
    completed_tasks: int = Field(default=0, ge=0)
    failed_tasks: int = Field(default=0, ge=0)
    cancelled_tasks: int = Field(default=0, ge=0)


class EpochProgress(TaskProgress):
    current_epoch: int | None = Field(default=None, ge=1)
    total_epochs: int | None = Field(default=None, ge=1)
    best_geo_mean: float | None = None


PROGRESS_MODELS = {
    "basic": BasicProgress, "rounds": RoundProgress,
    "tasks": TaskProgress, "epochs": EpochProgress,
}
COMMON_FIELDS = ("state", "stage", "mode", "stop_reason", "message", "updated_at")


def stage_progress(snapshot: dict) -> dict:
    kind = snapshot.get("progress_kind", "basic")
    model = PROGRESS_MODELS[kind]
    details = model.model_validate({k: snapshot[k] for k in model.model_fields if k in snapshot}).model_dump(mode="json")
    return {**{k: snapshot[k] for k in COMMON_FIELDS if k in snapshot},
            "progress_kind": kind, "progress": details}


def progress_v2(snapshot: dict) -> dict:
    return {**stage_progress(snapshot), "schema_version": "2.0",
            "revision": snapshot["revision"], "cancel_requested": snapshot["cancel_requested"],
            "scopes": {scope: stage_progress(record) for scope, record in snapshot.get("scopes", {}).items()}}
