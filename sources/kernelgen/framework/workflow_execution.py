"""Sequential stage execution, content-bound resume, ownership and cancellation."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

from kernelgen.framework.local_state import file_lock, process_is_alive, process_start_identity
from kernelgen.data._atomic import atomic_write_json
from kernelgen.framework.cancellation import cooperative_sigint, request_run_cancellation
from kernelgen.framework.run_control import RunCancelled, RunState, WorkspaceRunControl
from kernelgen.framework.progress_schema import ProgressKind, progress_v2

from .workflow_result import WorkflowContext, WorkflowReceipt, WorkflowResult, WorkflowSummary


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _input_digest(plan: dict, inputs: list[Path]) -> str:
    return _digest(json.dumps(
        {"plan": plan, "inputs": [_digest(path.read_bytes()) for path in inputs]},
        sort_keys=True, separators=(",", ":"),
    ).encode())


def _owner_alive(path: Path) -> bool:
    if not path.is_file():
        return False
    owner = json.loads(path.read_text())
    return process_is_alive(owner["pid"], owner["process_start"])


@contextmanager
def _exclusive_owner(root: Path):
    """Claim briefly under a file lock, retaining PID-reuse-safe ownership until exit."""
    state = root / ".kernelgen"
    owner_path = state / "lifecycle-owner.json"
    lock = state / "submission.lock"
    with file_lock(lock):
        if _owner_alive(owner_path):
            raise RuntimeError(f"lifecycle already has an active owner: {root}")
        identity = process_start_identity(os.getpid())
        if identity is None:
            raise RuntimeError("cannot determine lifecycle process identity")
        owner = {"pid": os.getpid(), "process_start": identity}
        atomic_write_json(owner_path, owner)
    try:
        yield
    finally:
        with file_lock(lock):
            if owner_path.is_file() and json.loads(owner_path.read_text()) == owner:
                owner_path.unlink()


def _stage_control(root: Path, stage: str) -> WorkspaceRunControl:
    return WorkspaceRunControl(root / "stages" / stage, root_workspace=root,
                               scope=f"stages/{stage}", source=f"lifecycle:{stage}")


def lifecycle_status(workspace: str | Path) -> dict:
    """Fresh-process projection, with stale RUNNING derived as INTERRUPTED."""
    root = Path(workspace).expanduser().resolve()
    plan = json.loads((root / ".kernelgen/operator-lifecycle.json").read_text())
    control = WorkspaceRunControl(root, root_workspace=root, scope="")
    snapshot = control.progress()
    alive = _owner_alive(root / ".kernelgen/lifecycle-owner.json")
    state = snapshot.state.value
    if not alive and snapshot.state in {RunState.RUNNING, RunState.CANCEL_REQUESTED}:
        state = "INTERRUPTED"
    return {"schema_version": "2.0", "simulated": plan["simulated"], "operator": plan["operator"],
            "workspace": str(root), "state": state, "process_alive": alive,
            "progress": progress_v2(snapshot.model_dump(mode="json"))}


def cancel_lifecycle(workspace: str | Path, reason: str = "operator lifecycle cancellation") -> dict:
    """Cancel one operator, also allowing an idle waiting lifecycle to terminate."""
    root = Path(workspace).expanduser().resolve()
    if not (root / ".kernelgen/operator-lifecycle.json").is_file():
        raise ValueError("cancel requires an existing single-operator lifecycle workspace")
    # Do not acknowledge an idle cancellation while another caller claims resume.
    with file_lock(root / ".kernelgen/submission.lock"):
        status = lifecycle_status(root)
        if status["state"] in {"SUCCEEDED", "FAILED", "CANCELLED", "INFRASTRUCTURE_ERROR"}:
            raise ValueError("lifecycle is already terminal")
        control = WorkspaceRunControl(root, root_workspace=root, scope="")
        cancellation, _, _ = request_run_cancellation(control, reason)
        result = cancellation.model_dump(mode="json")
        if not status["process_alive"]:
            for scope, record in control.progress().scopes.items():
                if record.state in {RunState.RUNNING, RunState.CANCEL_REQUESTED} or record.stage.startswith("WAITING_"):
                    WorkspaceRunControl(root, root_workspace=root, scope=scope).acknowledge_cancellation(stage=record.stage)
            control.acknowledge_cancellation(stage="CANCELLED")
        return result


class _WorkflowExecution:
    """Resume the plan, not each child: skip completed calls, retry incomplete ones.

    Every executed call gets a fresh attempt. Only the business adapter knows
    whether its underlying workflow should continue persisted internal work.
    """

    def __init__(self, *, cwd, calls: tuple[tuple[str, Callable], ...], mode: str, validate_artifacts: Callable[[WorkflowResult], None] | None = None, progress_kinds: dict[str, ProgressKind] | None = None):
        names = tuple(name for name, _ in calls)
        if not names or len(set(names)) != len(names):
            raise ValueError("stages must be nonempty and unique")
        if any(not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name) for name in names):
            raise ValueError("stage names must be safe workspace components")
        self.root = Path(cwd).expanduser().resolve()
        self.calls = dict(calls)
        self.stage_names = names
        self.mode = mode
        self.progress_kinds = progress_kinds or {}
        self.validate_artifacts = validate_artifacts or (lambda result: None)

    def run(self, *, operator: str, plan: dict, simulated: bool, resume: bool = False) -> WorkflowResult[WorkflowSummary]:
        if (tuple(plan.get("stages", ())) != self.stage_names
                or plan.get("operator") != operator or plan.get("simulated") != simulated):
            raise ValueError("execution stages and identity must match the persisted plan")
        self.operator, self.simulated, self.resume = operator, simulated, resume
        root = self.root
        plan_path = root / ".kernelgen/operator-lifecycle.json"
        if self.resume and not plan_path.exists():
            raise FileNotFoundError("no lifecycle exists to resume")
        # Never claim an existing optimization workspace, nor overwrite arbitrary inputs.
        user_entries = [entry for entry in root.iterdir() if entry.name != ".kernelgen"] if root.exists() else []
        if not plan_path.exists() and user_entries:
            raise ValueError(f"new lifecycle requires an empty workspace: {root}")
        with _exclusive_owner(root):
            if plan_path.exists():
                if json.loads(plan_path.read_text()) != plan:
                    raise ValueError("lifecycle plan changed; choose a new workspace")
                if not self.resume:
                    raise ValueError("lifecycle already exists; use explicit resume")
            elif self.resume:
                raise FileNotFoundError("no lifecycle exists to resume")
            else:
                atomic_write_json(plan_path, plan)
            control = WorkspaceRunControl(root, root_workspace=root, scope="", source="workflow:operator_lifecycle")
            # Real adapters reconcile remote work; dummy stages cannot claim it.
            if not self.simulated and control.active_server_operations():
                from kernelgen.framework.cancellation import reconcile_server_operations
                reconcile_server_operations(control)
            if control.active_server_operations():
                raise RuntimeError(f"{self.mode} lifecycle cannot resume with active KGS operations")
            if self.resume:
                cancellation = control.cancellation_state()
                if cancellation.requested:
                    control.clear_cancellation(expected_generation=cancellation.generation)
            for stage in self.stage_names:
                current = _stage_control(root, stage)
                if f"stages/{stage}" not in control.progress().scopes:
                    current.update_progress(state=RunState.PENDING, stage=stage, mode=self.mode,
                                            progress_kind=self.progress_kinds.get(stage, "basic"))
                elif stage in self.progress_kinds:
                    current.update_progress(progress_kind=self.progress_kinds[stage])
            with cooperative_sigint(control):
                result = self._run_calls(plan, control)
            # cooperative_sigint drains its notification thread on exit. Reconcile
            # a late SIGINT instead of returning success with CANCEL_REQUESTED state.
            if control.is_cancellation_requested() and result.state != "CANCELLED":
                control.acknowledge_cancellation(stage=control.progress().stage)
                return result.model_copy(update={"state": "CANCELLED"})
            return result

    def _run_calls(self, plan, control):
        inputs: list[Path] = []
        committed = {
            event.data["report"]: event.data["sha256"]
            for event in control.read_events()
            if event.event_type == "LIFECYCLE_STAGE_RESULT"
        }
        current = None
        stage = "PREPARING"
        control.update_progress(state=RunState.RUNNING, stage=stage, mode=self.mode, progress_kind="tasks",
                                total_tasks=len(self.stage_names), completed_tasks=0,
                                failed_tasks=0, cancelled_tasks=0, stop_reason="", message="Simulation only" if self.simulated else "Starting lifecycle")
        control.record_event("LIFECYCLE_STARTED", stage=stage,
                             data={"simulated": self.simulated, "resume": self.resume, "operator": self.operator})
        try:
            for stage in self.stage_names:
                current = _stage_control(self.root, stage)
                control.checkpoint(f"BEFORE_{stage.upper()}")
                control.update_progress(stage=stage)
                path = self._reuse_completed_call(stage, plan, control, current, inputs, committed)
                if path is not None:
                    self._complete(control, current, stage, path, inputs, reused=True)
                    continue
                path, result = self._execute_call(stage, self.calls[stage], plan, control, current, inputs)
                if result.state != "SUCCEEDED":
                    return self._finish_incomplete_call(control, current, stage, path, result, inputs)
                self._complete(control, current, stage, path, inputs, reused=False)
            control.checkpoint("BEFORE_LIFECYCLE_COMPLETION")
            control.update_progress(state=RunState.SUCCEEDED, stage="COMPLETED", message="Dummy lifecycle completed; not business acceptance" if self.simulated else "Lifecycle completed")
            control.record_event("LIFECYCLE_COMPLETED", stage="COMPLETED", data={"simulated": self.simulated})
            return self._output("SUCCEEDED", inputs)
        except RunCancelled:
            if current is not None and control.progress().scopes[current.scope].state != RunState.SUCCEEDED:
                current.acknowledge_cancellation(stage=stage)
            control.acknowledge_cancellation(stage=stage)
            control.update_progress(cancelled_tasks=1)
            return self._output("CANCELLED", inputs)
        except Exception as exc:
            if current is not None:
                current.update_progress(state=RunState.FAILED, stage=stage, message=str(exc))
            control.update_progress(state=RunState.FAILED, stage=stage, failed_tasks=1, message=str(exc))
            control.record_event("LIFECYCLE_FAILED", stage=stage, level="ERROR", message=str(exc), data={"simulated": self.simulated})
            raise

    def _finish_incomplete_call(self, control, current, stage, path, result, inputs):
        waiting = result.state == "WAITING"
        if result.state == "CANCELLED":
            current.acknowledge_cancellation(stage=stage)
            control.acknowledge_cancellation(stage=stage)
            control.update_progress(cancelled_tasks=1)
            return self._output("CANCELLED", inputs + [path])
        state = RunState.PENDING if waiting else RunState.FAILED
        display = f"WAITING_{stage.upper()}" if waiting else stage
        current.update_progress(state=state, stage=display, message=result.message)
        control.update_progress(state=state, stage=display, message=result.message,
                                failed_tasks=0 if waiting else 1)
        current.record_event("LIFECYCLE_STAGE_WAITING" if waiting else "LIFECYCLE_STAGE_FAILED",
                             stage=stage, data={"simulated": self.simulated, "report": str(path)})
        return self._output("WAITING" if waiting else "FAILED", inputs + [path])

    def _complete(self, control, current, stage, path, inputs, *, reused):
        inputs.append(path)
        current.update_progress(state=RunState.SUCCEEDED, stage=stage, stop_reason="", message="Stage receipt committed")
        control.update_progress(completed_tasks=len(inputs))
        current.record_event("LIFECYCLE_STAGE_REUSED" if reused else "LIFECYCLE_STAGE_COMPLETED",
                             stage=stage, data={"simulated": self.simulated, "report": str(path), "sha256": _digest(path.read_bytes())})

    def _reuse_completed_call(self, stage, plan, control, current, inputs, committed):
        attempts = self.root / "stages" / stage / "attempts"
        for committed_path in committed:
            path = Path(committed_path)
            if path.is_relative_to(attempts) and not path.is_file():
                raise ValueError(f"committed stage receipt is missing: {path}")
        previous = sorted(attempts.glob("[0-9]*/result.json"), key=lambda path: int(path.parent.name))
        last = previous[-1] if previous else None
        fingerprint = _input_digest(plan, inputs)
        if last is not None:
            expected_hash = committed.get(str(last))
            if expected_hash is not None and _digest(last.read_bytes()) != expected_hash:
                raise ValueError(f"stage receipt content changed: {last}")
            receipt = WorkflowReceipt.model_validate_json(last.read_text())
            if (receipt.operator != self.operator or receipt.stage != stage
                    or receipt.attempt != int(last.parent.name)
                    or receipt.input_sha256 != fingerprint):
                raise ValueError(f"stale or foreign stage receipt: {last}")
            if expected_hash is not None and receipt.returned().state == "SUCCEEDED":
                self.validate_artifacts(receipt.returned())
                return last
        elif control.progress().scopes[current.scope].state == RunState.SUCCEEDED:
            raise ValueError(f"successful stage receipt is missing: {stage}")
        return None

    def _execute_call(self, stage, invoke, plan, control, current, inputs):
        attempts = self.root / "stages" / stage / "attempts"
        fingerprint = _input_digest(plan, inputs)
        attempt = max((int(path.name) for path in attempts.glob("[0-9]*") if path.is_dir()), default=0) + 1
        workspace = attempts / f"{attempt:02d}"
        workspace.mkdir(parents=True, exist_ok=False)
        linked = control.link_workspace(workspace, scope=current.scope)
        current.update_progress(state=RunState.RUNNING, stage=stage, stop_reason="", message="Simulation only" if self.simulated else "Executing stage")
        current.record_event("LIFECYCLE_STAGE_STARTED", stage=stage,
                             data={"simulated": self.simulated, "attempt": attempt, "workspace": str(workspace)})
        result = invoke(WorkflowContext(
            self.operator, stage, workspace, attempt, tuple(inputs), linked,
        ))
        if not isinstance(result, WorkflowResult):
            raise TypeError(f"{stage} must return WorkflowResult")
        if result.simulated != self.simulated:
            raise ValueError("stage simulation mode differs from the execution plan")
        for upstream in inputs:
            self.validate_artifacts(WorkflowReceipt.model_validate_json(upstream.read_text()).returned())
        self.validate_artifacts(result)
        # A cancellation inside a stage is observed only after its full output.
        path = workspace / "result.json"
        receipt = WorkflowReceipt.from_returned(operator=self.operator, stage=stage, attempt=attempt,
                                               input_sha256=fingerprint, returned=result)
        atomic_write_json(path, receipt.model_dump(mode="json"))
        current.record_event("LIFECYCLE_STAGE_RESULT", stage=stage,
                             data={"simulated": self.simulated, "report": str(path), "sha256": _digest(path.read_bytes())})
        current.checkpoint(f"AFTER_{stage.upper()}")
        return path, result

    def _output(self, status, paths):
        return WorkflowResult[WorkflowSummary](
            state=status, simulated=self.simulated,
            output=WorkflowSummary(operator=self.operator, workspace=str(self.root),
                                   reports=[str(path) for path in paths]),
        )
