"""Compact, read-only optimization history for the ``kg`` CLI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from kernelgen.cli.batch import batch_request_path
from kernelgen.cli.models import RunMode
from kernelgen.cli.state import load_request, request_path
from kernelgen.data.ledger import LEDGER_FILENAME
from kernelgen.data.optimization_history import OptimizationHistory


class HistoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoundHistoryRecord(HistoryModel):
    round: int = Field(gt=0)
    status: str = Field(min_length=1)
    geo_mean: float | None = None
    min_speedup: float | None = None
    best_geo_mean_so_far: float | None = None
    is_new_best: bool = False
    is_final_best: bool = False
    is_hack: bool = False
    evaluated_at: datetime | None = None


class HistorySeriesRecord(HistoryModel):
    scope: str = Field(min_length=1)
    ledger_path: str = Field(min_length=1)
    definition: str
    target_hardware: str
    implementation_language: str
    best_round: int | None = Field(default=None, gt=0)
    best_geo_mean: float | None = None
    round_count: int = Field(ge=0)
    rounds: list[RoundHistoryRecord] = Field(default_factory=list)


class RunHistoryResponse(HistoryModel):
    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["run_history"] = "run_history"
    run_id: str = Field(min_length=1)
    mode: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    target_hardware: str | None = Field(min_length=1)
    workspace: str = Field(min_length=1)
    best_scope: str | None = None
    best_round: int | None = Field(default=None, gt=0)
    best_geo_mean: float | None = None
    series_count: int = Field(ge=0)
    round_count: int = Field(ge=0)
    series: list[HistorySeriesRecord] = Field(default_factory=list)


def _relative_path(path: Path, workspace: Path) -> str:
    return path.relative_to(workspace).as_posix()


def _ledger_paths(workspace: Path, mode: RunMode, *, catalog_workflow=False) -> list[Path]:
    if catalog_workflow:
        workspace = workspace / "stages" / "optimize" / "work"
    if not workspace.is_dir():
        return []
    if mode == RunMode.SIMPLE_OPT:
        ledger = workspace / LEDGER_FILENAME
        return [ledger] if ledger.is_file() else []

    paths = []
    for epoch in workspace.iterdir():
        if not epoch.is_dir() or not epoch.name.endswith("R"):
            continue
        epoch_number = epoch.name[:-1]
        if not epoch_number.isdigit() or int(epoch_number) < 1:
            continue
        for agent in epoch.iterdir():
            agent_number = agent.name.removeprefix("agent")
            ledger = agent / LEDGER_FILENAME
            if agent.is_dir() and agent_number.isdigit() and ledger.is_file():
                paths.append(ledger)
    return sorted(
        paths,
        key=lambda path: (
            int(path.relative_to(workspace).parts[0][:-1]),
            int(path.parent.name.removeprefix("agent")),
        ),
    )


def _load_series(ledger_path: Path, workspace: Path) -> HistorySeriesRecord:
    relative_ledger = _relative_path(ledger_path, workspace)
    try:
        history = OptimizationHistory.load(ledger_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read ledger {relative_ledger}: {exc}") from exc

    scope_path = ledger_path.parent.relative_to(workspace)
    scope = "." if not scope_path.parts else scope_path.as_posix()
    best_round = history.best_round or None
    best_geo_mean = history.best_geo_mean if best_round is not None else None
    running_best: float | None = None
    rounds = []
    for record in history.rounds:
        evaluation = record.evaluation
        valid = (
            evaluation.status == "PASSED"
            and not evaluation.is_hack
            and evaluation.geo_mean is not None
        )
        is_new_best = valid and (
            running_best is None or evaluation.geo_mean > running_best
        )
        if is_new_best:
            running_best = evaluation.geo_mean
        rounds.append(
            RoundHistoryRecord(
                round=record.round_num,
                status=evaluation.status,
                geo_mean=evaluation.geo_mean,
                min_speedup=evaluation.min_speedup,
                best_geo_mean_so_far=running_best,
                is_new_best=is_new_best,
                is_final_best=valid and record.round_num == best_round,
                is_hack=evaluation.is_hack,
                evaluated_at=evaluation.evaluated_at,
            )
        )
    return HistorySeriesRecord(
        scope=scope,
        ledger_path=relative_ledger,
        definition=history.definition_name,
        target_hardware=history.target_hardware,
        implementation_language=history.implementation_language,
        best_round=best_round,
        best_geo_mean=best_geo_mean,
        round_count=len(rounds),
        rounds=rounds,
    )


def load_run_history(workspace: str | Path) -> dict:
    """Load a compact projection of all authoritative ledgers in one run."""

    path = Path(workspace).expanduser().resolve()
    if batch_request_path(path).is_file():
        raise ValueError(
            "history requires a single-run workspace; use kg status <batch> "
            "to find a child workspace"
        )
    if not request_path(path).is_file():
        raise FileNotFoundError(f"kg run not found: {path}")
    request = load_request(path)
    series = [
        _load_series(ledger, path) for ledger in _ledger_paths(path, request.mode, catalog_workflow=request.schema_version == "2.0")
    ]

    best_scope = None
    best_round = None
    best_geo_mean = None
    for item in series:
        if item.best_geo_mean is not None and (
            best_geo_mean is None or item.best_geo_mean > best_geo_mean
        ):
            best_scope = item.scope
            best_round = item.best_round
            best_geo_mean = item.best_geo_mean

    response = RunHistoryResponse(
        run_id=request.run_id,
        mode=request.mode.value,
        definition=request.definition,
        target_hardware=request.target_hardware,
        workspace=str(path),
        best_scope=best_scope,
        best_round=best_round,
        best_geo_mean=best_geo_mean,
        series_count=len(series),
        round_count=sum(item.round_count for item in series),
        series=series,
    )
    return response.model_dump(mode="json")


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.12g}"


def project_progress(progress: dict, history: dict) -> dict:
    """Use the same ledger series for status, without joining Coder round axes."""
    by_scope = {series["scope"]: series for series in history["series"]}

    def measured(scope):
        ledger_scope = "stages/optimize/work" if scope == "stages/optimize" else scope
        series = by_scope.get(ledger_scope)
        rounds = series["rounds"] if series else []
        child_best = max((item["best_geo_mean"] for path, item in by_scope.items()
                          if path.startswith(ledger_scope + "/") and item["best_geo_mean"] is not None), default=None)
        return {
            "current_round": rounds[-1]["round"] if rounds else None,
            "completed_rounds": len(rounds),
            "best_round": (series["best_round"] or 0) if series else 0,
            "best_geo_mean": series["best_geo_mean"] if series else child_best,
            "last_evaluation_status": rounds[-1]["status"] if rounds else "",
        }

    return {
        **progress, **measured("."),
        "best_round": history["best_round"] or 0,
        "best_geo_mean": history["best_geo_mean"],
        "completed_rounds": history["round_count"],
        "scopes": {
            scope: {**record, **measured(scope)}
            for scope, record in progress.get("scopes", {}).items()
        },
    }


def print_run_history(history: dict) -> None:
    """Print a compact human-readable history without losing series identity."""

    print(f"definition: {history['definition']}")
    print(f"workspace: {history['workspace']}")
    print(f"best_geo_mean: {_format_metric(history['best_geo_mean'])}")
    if history["best_scope"] is not None:
        print(f"best: {history['best_scope']} round {history['best_round']}")
    print(f"series: {history['series_count']}")
    print(f"rounds: {history['round_count']}")
    for series in history["series"]:
        for record in series["rounds"]:
            flags = []
            if record["is_new_best"]:
                flags.append("new_best")
            if record["is_final_best"]:
                flags.append("final_best")
            if record["is_hack"]:
                flags.append("hack")
            flag_text = f" flags={','.join(flags)}" if flags else ""
            print(
                f"{series['scope']} R{record['round']} {record['status']} "
                f"geo_mean={_format_metric(record['geo_mean'])} "
                f"best_so_far={_format_metric(record['best_geo_mean_so_far'])}"
                f"{flag_text}"
            )
