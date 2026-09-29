"""One lifecycle owner when optimization workflows share the same RunControl."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Callable, TypeVar

from pydantic import BaseModel

from kernelgen.framework.run_control import RunCancelled, RunControl, RunState
from kernelgen.framework.progress_schema import ProgressKind

_Output = TypeVar("_Output", bound=BaseModel)
_active_controls: ContextVar[tuple[RunControl, ...]] = ContextVar("workflow_controls", default=())


def run_workflow(
    control: RunControl,
    *,
    name: str,
    mode: str,
    definition_name: str,
    max_round: int,
    checkpoint: str,
    execute: Callable[[], object],
    output_model: type[_Output],
    current_epoch: int | None = None,
    total_epochs: int | None = None,
    completed_epoch: int | None = None,
    progress_kind: ProgressKind = "rounds",
) -> _Output:
    """Own start/terminal events; nested use of the identical control delegates.

    SimpleOpt passes its control to KernelOptimization, so its outer boundary
    owns both preparation and optimization failures. Independent Coder controls
    retain their own scopes. Ownership is call-local, never persisted.
    """
    active = _active_controls.get()
    if any(owner is control for owner in active):
        control.checkpoint(checkpoint)
        return output_model.model_validate(execute())

    token = _active_controls.set((*active, control))
    source = f"workflow:{name}"
    epoch_progress = (
        {"current_epoch": current_epoch, "total_epochs": total_epochs}
        if current_epoch is not None else {}
    )
    try:
        control.checkpoint(checkpoint)
        control.update_progress(
            state=RunState.RUNNING, stage="PREPARING", mode=mode, progress_kind=progress_kind,
            max_round=max_round, message=f"Preparing {definition_name}",
            **epoch_progress,
        )
        control.record_event(
            "WORKFLOW_STARTED", stage="PREPARING", source=source,
            data={"workflow": name, "definition_name": definition_name, **(
                {"start_epoch": current_epoch, "total_epochs": total_epochs}
                if current_epoch is not None else {}
            )},
        )
        result = output_model.model_validate(execute())
        control.checkpoint("BEFORE_WORKFLOW_COMPLETION")
        state = RunState.SUCCEEDED if result.status == "PASSED" else RunState.FAILED
        message = getattr(result, "summary", "") or f"{name} finished with {result.status}"
        control.update_progress(
            state=state, stage="COMPLETED", message=message,
            **({"current_epoch": completed_epoch or total_epochs} if total_epochs is not None else {}),
        )
        control.record_event(
            "WORKFLOW_COMPLETED", stage="COMPLETED", source=source, message=message,
            level="INFO" if state == RunState.SUCCEEDED else "WARNING",
            data={"workflow": name, **result.model_dump(include={"status", "rounds", "best_geo_mean"})},
        )
        return result
    except RunCancelled:
        control.acknowledge_cancellation(stage="CANCELLED")
        raise
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        control.update_progress(state=RunState.FAILED, stage="FAILED", message=message)
        control.record_event(
            "WORKFLOW_FAILED", stage="FAILED", source=source, message=message,
            level="ERROR", data={"workflow": name},
        )
        raise
    finally:
        _active_controls.reset(token)
