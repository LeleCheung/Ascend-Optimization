"""User-facing ``kg`` command line."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from kernelgen.cli.batch import (
    batch_request_path,
    load_batch_file,
    load_batch_request,
    save_batch_request,
)
from kernelgen.cli.history import load_run_history, print_run_history
from kernelgen.cli.lifecycle import add_lifecycle_run_arguments, run_lifecycle_command
from kernelgen.cli.models import (
    BatchChildRecord,
    BatchRequestRecord,
    RunMode,
)
from kernelgen.cli.runner import submit_request
from kernelgen.cli.state import (
    active_process,
    cli_home,
    load_request,
    request_path,
    runner_log_path,
)
from kernelgen.framework.worker_pool import WorkerLeasePool, set_max_workers
from kernelgen.cli.api import (
    build_request as _new_request_from_options,
    cancel_run,
    list_run_statuses,
    status as _status,
    submit_run,
)
from kernelgen.framework.cancellation import request_run_cancellation, forward_server_cancellation
from kernelgen.framework.run_control import WorkspaceRunControl
from kernelgen.framework.run_options import (
    add_optimization_arguments, optimization_parser,
    resolve_run_options as _resolve_run_options,
)


def _explicit_run_options(args) -> dict:
    names = {"mode", *(action.dest for action in optimization_parser()._actions)}
    return {name: getattr(args, name) for name in names if getattr(args, name, None) is not None}


def _json_dump(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _default_workspace(definition: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in definition)
    return cli_home() / "runs" / f"{stamp}-{safe}-{os.urandom(4).hex()}"


# _run_status / _batch_status / _status now live in kernelgen.cli.api and are
# imported above under their original private names for backward compatibility.


def _print_status(status: dict) -> None:
    if status.get("kind") == "batch":
        print(f"state: {status['state']}")
        print("kind: batch")
        print(f"workspace: {status['workspace']}")
        print(f"tasks: {status['total_tasks']}")
        counts = ", ".join(
            f"{name}={count}" for name, count in status["counts"].items()
        )
        print(f"counts: {counts}")
        for item in status["runs"]:
            progress = item.get("progress") or {}
            detail = progress.get("stage") or ""
            print(f"{item['state']:<20} {item['definition']:<32} {detail}")
        return
    record = status["progress"]
    progress = {**record, **record["progress"]}
    print(f"state: {status['state']}")
    print(f"mode: {status['mode']}")
    print(f"definition: {status['definition']}")
    print(f"workspace: {status['workspace']}")
    if progress.get("stage"):
        print(f"stage: {progress['stage']}")
    if progress.get("current_epoch") is not None:
        print(
            f"epoch: {progress['current_epoch']}/{progress.get('total_epochs') or '?'}"
        )
    if progress.get("current_round") is not None:
        print(
            f"round: {progress['current_round']}/{progress.get('max_round') or '?'}"
        )
    if progress.get("total_tasks") is not None:
        print(f"tasks: {progress['completed_tasks']}/{progress['total_tasks']}")
    metrics = [node["progress"].get("best_geo_mean") for node in (record, *record.get("scopes", {}).values())]
    best = max((value for value in metrics if value is not None), default=None)
    if best is not None:
        print(f"best_geo_mean: {best}")
    if progress.get("message"):
        print(f"message: {progress['message']}")


# _new_request_from_options (build_request) and _validate_input_paths now live
# in kernelgen.cli.api and are imported above under their original names.


def _batch_child_name(definition: str) -> str:
    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in definition)
    if not name or name in {".", ".."}:
        raise ValueError(f"definition cannot form a workspace name: {definition!r}")
    return name


def _command_batch_run(args) -> int:
    if args.foreground:
        raise ValueError("--foreground is not supported for Batch runs")
    source, defaults, operators, manifest_workspace = load_batch_file(args.batch_file)
    workspace = (
        args.workspace.expanduser().resolve()
        if args.workspace is not None
        else manifest_workspace or _default_workspace("batch")
    )
    if workspace.exists():
        raise ValueError(f"Batch workspace already exists: {workspace}")

    cli_options = _explicit_run_options(args)
    requests = []
    child_names: set[str] = set()
    for operator in operators:
        definition = operator["definition"]
        item_options = {key: value for key, value in operator.items() if key != "definition"}
        values = _resolve_run_options(defaults, cli_options, item_options)
        child_name = _batch_child_name(definition)
        if child_name in child_names:
            raise ValueError(f"Batch definitions have colliding workspace names: {definition}")
        child_names.add(child_name)
        child_workspace = workspace / "definitions" / child_name
        requests.append(
            _new_request_from_options(
                values,
                definition=definition,
                workspace=child_workspace,
                batch_workspace=workspace,
            )
        )

    batch = BatchRequestRecord(
        batch_id=uuid.uuid4().hex,
        workspace=workspace,
        source_path=source,
        children=[
            BatchChildRecord(
                definition=request.definition,
                mode=request.mode,
                workspace=request.workspace,
                run_id=request.run_id,
            )
            for request in requests
        ],
    )
    save_batch_request(batch)
    submitted = []
    try:
        for request in requests:
            submitted.append((request, submit_request(request, foreground=False)))
    except Exception as exc:
        for request, _ in submitted:
            if active_process(request.workspace) is not None:
                request_run_cancellation(
                    WorkspaceRunControl(request.workspace, source="cli:batch-submit"),
                    "Batch submission did not complete",
                )
        raise RuntimeError(
            f"Batch submission failed after {len(submitted)}/{len(requests)} tasks: {exc}"
        ) from exc

    print(f"workspace: {workspace}")
    print(f"batch_id: {batch.batch_id}")
    print(f"tasks: {len(submitted)}")
    print("status: SUBMITTED")
    for request, record in submitted:
        print(f"{request.definition}: pid={record.pid} workspace={request.workspace}")
    return 0


def _command_run(args) -> int:
    if args.mode == "lifecycle":
        if args.batch_file is not None:
            raise ValueError("lifecycle does not support --batch-file; use --operators")
        unsupported = _explicit_run_options(args)
        unsupported.pop("mode", None)
        if unsupported:
            flags = ", ".join(f"--{name.replace('_', '-')}" for name in sorted(unsupported))
            raise ValueError(f"optimization options are not supported by dummy lifecycle: {flags}")
        return run_lifecycle_command(
            args, operators=args.operators or [args.definition],
            workspace=args.workspace if args.workspace is not None else _default_workspace("lifecycle"),
        )
    if args.operators is not None or any(getattr(args, name) is not None for name in args.lifecycle_arguments):
        raise ValueError("--operators and lifecycle options require --mode lifecycle")
    if args.batch_file is not None:
        return _command_batch_run(args)
    values = _resolve_run_options(_explicit_run_options(args))
    workspace = (
        args.workspace.expanduser().resolve()
        if args.workspace is not None
        else _default_workspace(args.definition)
    )
    request, record = submit_run(
        values,
        definition=args.definition,
        workspace=workspace,
        foreground=args.foreground,
    )
    print(f"workspace: {request.workspace}")
    print(f"pid: {record.pid}")
    print(f"status: {record.state.value if args.foreground else 'SUBMITTED'}")
    return (record.exit_code or 0) if args.foreground else 0


def _command_status(args) -> int:
    status = _status(args.workspace)
    _json_dump(status) if args.detail else _print_status(status)
    return 0


def _command_history(args) -> int:
    history = load_run_history(args.workspace)
    _json_dump(history) if args.json else print_run_history(history)
    return 0


def _format_event(event) -> str:
    scope = f" {event.scope}" if event.scope else ""
    message = f": {event.message}" if event.message else ""
    return (
        f"[{event.sequence}] {event.recorded_at.isoformat()} "
        f"{event.level}{scope} {event.event_type}{message}"
    )


def _follow_raw(path: Path, *, follow: bool, lines: int) -> int:
    if not path.exists():
        if not follow:
            raise FileNotFoundError(f"log not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in content[-lines:]:
        print(line)
    if not follow:
        return 0
    position = path.stat().st_size
    while True:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(position)
            chunk = handle.read()
            position = handle.tell()
        if chunk:
            print(chunk, end="", flush=True)
        if active_process(path.parents[1]) is None:
            return 0
        time.sleep(0.5)


def _command_logs(args) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    if batch_request_path(workspace).is_file():
        raise ValueError(
            "Batch logs are stored per operator; use kg status <batch> to list child workspaces"
        )
    if not request_path(workspace).is_file():
        raise FileNotFoundError(f"kg run not found: {workspace}")
    if args.raw:
        return _follow_raw(runner_log_path(workspace), follow=args.follow, lines=args.lines)
    control = WorkspaceRunControl(workspace, source="cli:logs")
    cursor = args.after
    while True:
        events = control.read_events(after_sequence=cursor)
        for event in events:
            print(_format_event(event))
            cursor = event.sequence
        if not args.follow or active_process(workspace) is None:
            return 0
        time.sleep(0.5)


def _command_cancel(args) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    if batch_request_path(workspace).is_file():
        batch = load_batch_request(workspace)
        requested = 0
        already_terminal = 0
        server_cancel_requested = 0
        server_cancel_failed = 0
        for child in batch.children:
            process = active_process(child.workspace)
            control = WorkspaceRunControl(child.workspace, source="cli:batch-cancel")
            if process is None:
                already_terminal += 1
                child_server_requested, child_server_failed = forward_server_cancellation(control)
            else:
                _, child_server_requested, child_server_failed = request_run_cancellation(control, args.reason)
                requested += 1
            server_cancel_requested += child_server_requested
            server_cancel_failed += child_server_failed
        if requested == server_cancel_requested == server_cancel_failed == 0:
            raise RuntimeError(f"Batch has no active processes: {workspace}")
        print(f"workspace: {workspace}")
        print("status: CANCEL_REQUESTED")
        print(f"tasks_cancel_requested: {requested}")
        print(f"tasks_already_terminal: {already_terminal}")
        print(f"server_operations_cancel_requested: {server_cancel_requested}")
        print(f"server_operations_cancel_failed: {server_cancel_failed}")
        return 0
    result = cancel_run(workspace, args.reason, source="cli:cancel")
    print(f"workspace: {result['workspace']}")
    print(f"status: {result['status']}")
    if result["status"] == "CANCEL_REQUESTED":
        print(f"generation: {result['generation']}")
    print(f"server_operations_cancel_requested: {result['server_operations_cancel_requested']}")
    print(f"server_operations_cancel_failed: {result['server_operations_cancel_failed']}")
    if result["status"] == "NO_ACTIVE_PROCESS":
        return 0 if result["server_operations_cancel_failed"] == 0 else 2
    return 0


def _command_resume(args) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    if batch_request_path(workspace).is_file():
        raise ValueError(
            "Batch resume is not supported yet; resume an interrupted child workspace"
        )
    request = load_request(workspace)
    record = submit_request(request, foreground=args.foreground, resume=True)
    print(f"workspace: {workspace}")
    print(f"pid: {record.pid}")
    print(f"status: {record.state.value if args.foreground else 'SUBMITTED'}")
    return (record.exit_code or 0) if args.foreground else 0


def _command_list(args) -> int:
    statuses = list_run_statuses()
    if args.json:
        _json_dump({"schema_version": "2.0", "runs": statuses})
    else:
        for item in statuses:
            if item.get("kind") == "batch":
                print(
                    f"{item['state']:<20} {'batch':<10} "
                    f"{item['total_tasks']:<28} {item['workspace']}"
                )
            else:
                print(
                    f"{item['state']:<20} {item['mode']:<10} "
                    f"{item['definition']:<28} {item['workspace']}"
                )
    return 0


def _command_config(args) -> int:
    if args.key.startswith("extract."):
        from kernelgen.agents.extractor.flaggems.collection_config import set_extraction_setting
        set_extraction_setting(args.key.removeprefix("extract."), args.value)
        print(f"{args.key}: {args.value}")
        return 0
    if args.key != "run.max-workers":
        raise ValueError("supported keys: run.max-workers, extract.host, extract.container, extract.python")
    try:
        value = int(args.value)
    except ValueError as exc:
        raise ValueError("run.max-workers must be a positive integer") from exc
    set_max_workers(args.eval_server, value)
    snapshot = WorkerLeasePool(args.eval_server).snapshot()
    print(f"eval_server: {args.eval_server}")
    print(f"worker_pool: {snapshot['worker_pool']}")
    print(f"run.max-workers: {value}")
    if snapshot["used_workers"] > value:
        print(
            "note: active runs exceed the new limit; they will not be interrupted",
            file=sys.stderr,
        )
    return 0


def _add_run_parser(subparsers) -> None:
    parser = subparsers.add_parser("run", help="submit optimization or run a foreground dummy lifecycle", allow_abbrev=False)
    parser.add_argument("--mode", choices=[*(item.value for item in RunMode), "lifecycle"])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--definition", "--definition-name", "-d", "-n")
    source.add_argument("--batch-file", type=Path)
    source.add_argument("--operators", nargs="+", help="operator list for lifecycle only")
    parser.add_argument("--workspace", "-w", type=Path)
    parser.add_argument("--foreground", action="store_true")
    add_optimization_arguments(parser, sparse=True)
    parser.set_defaults(handler=_command_run, lifecycle_arguments=add_lifecycle_run_arguments(parser))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kg", description="KernelGen local CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    from kernelgen.cli.extract import add_extract_parser
    add_extract_parser(subparsers)
    from kernelgen.cli.definition import add_definition_parser
    add_definition_parser(subparsers)

    status = subparsers.add_parser("status", help="show one run's progress")
    status.add_argument("workspace")
    status.add_argument("--detail", action="store_true", help="show complete structured status (JSON)")
    status.set_defaults(handler=_command_status)

    history = subparsers.add_parser(
        "history", help="show per-round optimization performance"
    )
    history.add_argument("workspace")
    history.add_argument("--json", action="store_true")
    history.set_defaults(handler=_command_history)

    logs = subparsers.add_parser("logs", help="show structured or raw run logs")
    logs.add_argument("workspace")
    logs.add_argument("--follow", "-f", action="store_true")
    logs.add_argument("--after", type=int, default=0)
    logs.add_argument("--raw", action="store_true")
    logs.add_argument("--lines", type=int, default=200)
    logs.set_defaults(handler=_command_logs)

    cancel = subparsers.add_parser("cancel", help="request cooperative cancellation")
    cancel.add_argument("workspace")
    cancel.add_argument("--reason", default="cancelled by user")
    cancel.set_defaults(handler=_command_cancel)

    resume = subparsers.add_parser("resume", help="resume an interrupted or cancelled run")
    resume.add_argument("workspace")
    resume.add_argument("--foreground", action="store_true")
    resume.set_defaults(handler=_command_resume)

    listing = subparsers.add_parser("list", help="list locally submitted runs")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(handler=_command_list)

    config = subparsers.add_parser("config", help="configure Coder concurrency or the extraction environment")
    config_subparsers = config.add_subparsers(dest="config_command", required=True)
    config_set = config_subparsers.add_parser("set")
    config_set.add_argument("key")
    config_set.add_argument("value")
    config_set.add_argument(
        "--eval-server",
        "--kgs-url",
        default=os.environ.get("FIB_EVAL_SERVER", "http://localhost:8000"),
        help="KGS endpoint whose local Coder concurrency is being configured",
    )
    config_set.set_defaults(handler=_command_config)

    from kernelgen.cli.server import add_server_parser

    add_server_parser(subparsers)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"kg: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
