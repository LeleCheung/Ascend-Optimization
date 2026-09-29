"""Monitor a running BatchSimpleOptDefinitionWorkflow without changing it.

Example:
    python3 -m kernelgen.tools.monitor_batch_simple_opt \
        --workspace runs/batch_simple_opt_v4 \
        -n flaggems_linear -n flaggems_var \
        --server-url http://127.0.0.1:18083
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from kernelgen.workflows.legacy.batch_simple_opt_definition import (
    BATCH_SIMPLE_OPT_OUTPUT_FILENAME,
)
from kernelgen.workflows.optimization.single_coder import (
    KERNEL_OPTIMIZATION_OUTPUT_FILENAME,
)


@dataclass(frozen=True)
class DefinitionProgress:
    name: str
    state: str
    rounds: int
    best_geo_mean: Optional[float]
    last_eval_status: str
    last_activity_seconds: Optional[int]


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _coder_exited_with_continue_verdict(
    item_dir: Path,
    ledger: Optional[dict[str, Any]],
) -> bool:
    rounds = ledger.get("rounds", []) if ledger else []
    if not rounds:
        return False
    verdict = rounds[-1].get("next_verdict")
    if not isinstance(verdict, dict) or verdict.get("should_continue") is not True:
        return False

    try:
        runtime_log = (
            item_dir / ".kernelgen" / "claude-runtime.log"
        ).read_text(encoding="utf-8")
    except OSError:
        return False
    last_start = runtime_log.rfind("[claude] model=")
    last_done = runtime_log.rfind("[claude] done")
    return last_done > last_start >= 0


def _definition_names(workspace: Path, requested: Iterable[str]) -> list[str]:
    names = list(dict.fromkeys(requested))
    if names:
        return names

    batch_output = _read_json(workspace / BATCH_SIMPLE_OPT_OUTPUT_FILENAME)
    if batch_output:
        names.extend(
            result["definition_name"]
            for result in batch_output.get("results", [])
            if result.get("definition_name")
        )

    definitions_dir = workspace / "definitions"
    if definitions_dir.is_dir():
        names.extend(
            path.name
            for path in sorted(definitions_dir.iterdir())
            if path.is_dir()
        )
    return list(dict.fromkeys(names))


def collect_progress(
    workspace: Path,
    requested_names: Iterable[str] = (),
) -> list[DefinitionProgress]:
    now = time.time()
    progress = []
    for name in _definition_names(workspace, requested_names):
        item_dir = workspace / "definitions" / name
        output = _read_json(item_dir / KERNEL_OPTIMIZATION_OUTPUT_FILENAME)
        ledger = _read_json(item_dir / ".ledger.json")

        if output:
            state = str(output.get("status", "DONE"))
            rounds = int(output.get("rounds", 0))
            best_geo_mean = output.get("best_geo_mean")
        elif _coder_exited_with_continue_verdict(item_dir, ledger):
            state = "FAILED"
            rounds = len(ledger.get("rounds", []))
            best_geo_mean = ledger.get("best_geo_mean")
        elif ledger:
            state = "RUNNING"
            rounds = len(ledger.get("rounds", []))
            best_geo_mean = ledger.get("best_geo_mean")
        elif item_dir.is_dir():
            state = "STARTING"
            rounds = 0
            best_geo_mean = None
        else:
            state = "PENDING"
            rounds = 0
            best_geo_mean = None

        ledger_rounds = ledger.get("rounds", []) if ledger else []
        last_evaluation = (
            ledger_rounds[-1].get("evaluation") if ledger_rounds else None
        )
        last_eval_status = (
            str(last_evaluation.get("status", "-"))
            if isinstance(last_evaluation, dict)
            else "-"
        )

        activity_paths = [
            item_dir / ".kernelgen" / "claude-runtime.log",
            item_dir / ".ledger.json",
            item_dir / KERNEL_OPTIMIZATION_OUTPUT_FILENAME,
        ]
        mtimes = [
            path.stat().st_mtime
            for path in activity_paths
            if path.is_file()
        ]
        activity_age = int(max(0, now - max(mtimes))) if mtimes else None
        progress.append(
            DefinitionProgress(
                name=name,
                state=state,
                rounds=rounds,
                best_geo_mean=best_geo_mean,
                last_eval_status=last_eval_status,
                last_activity_seconds=activity_age,
            )
        )
    return progress


def _format_server_status(payload: dict) -> str:
    devices = payload.get("devices") or payload.get("gpus") or []
    if devices:
        if all(isinstance(device, dict) for device in devices):
            busy = sum(
                str(device.get("status", "")).lower() == "busy"
                for device in devices
            )
            return f"healthy, {busy}/{len(devices)} devices busy"
        details = [f"{len(devices)} devices"]
        if payload.get("workers") is not None:
            details.append(f"{payload['workers']} workers")
        if payload.get("timing"):
            details.append(f"timing={payload['timing']}")
        return f"healthy, {', '.join(details)}"
    return str(payload.get("status") or payload.get("version") or "healthy")


def _server_status(server_url: str) -> str:
    if not server_url:
        return "not configured"
    url = f"{server_url.rstrip('/')}/status"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.load(response)
    except Exception as exc:
        return f"unreachable ({type(exc).__name__}: {exc})"
    return _format_server_status(payload)


def _format_age(seconds: Optional[int]) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def render_snapshot(
    workspace: Path,
    progress: list[DefinitionProgress],
    server_url: str = "",
) -> str:
    batch = _read_json(workspace / BATCH_SIMPLE_OPT_OUTPUT_FILENAME)
    lines = [
        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}]",
        f"workspace: {workspace}",
        f"server: {_server_status(server_url)}",
        f"batch: {batch.get('summary') if batch else 'running'}",
        "",
        f"{'DEFINITION':42} {'STATE':10} {'ROUNDS':>6} "
        f"{'BEST_GEO':>12} {'LAST_EVAL':18} {'IDLE':>6}",
    ]
    for item in progress:
        best = (
            f"{item.best_geo_mean:.6g}"
            if item.best_geo_mean is not None
            else "-"
        )
        lines.append(
            f"{item.name:42} {item.state:10} {item.rounds:6d} "
            f"{best:>12} {item.last_eval_status[:18]:18} "
            f"{_format_age(item.last_activity_seconds):>6}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monitor BatchSimpleOptDefinitionWorkflow workspaces",
    )
    parser.add_argument("--workspace", "-w", type=Path, required=True)
    parser.add_argument(
        "--definition",
        "-n",
        action="append",
        default=[],
        help="Expected definition name; repeat to show pending tasks",
    )
    parser.add_argument("--server-url", default="")
    parser.add_argument("--interval", type=float, default=180.0)
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.expanduser().resolve()
    started_at = time.monotonic()
    previous = None
    while True:
        progress = collect_progress(workspace, args.definition)
        batch_output = _read_json(workspace / BATCH_SIMPLE_OPT_OUTPUT_FILENAME)
        signature = (
            tuple(
                (
                    item.name,
                    item.state,
                    item.rounds,
                    item.best_geo_mean,
                    item.last_eval_status,
                )
                for item in progress
            ),
            json.dumps(batch_output, sort_keys=True),
        )
        if signature != previous:
            print(
                render_snapshot(workspace, progress, args.server_url),
                flush=True,
            )
            previous = signature

        if batch_output is not None:
            statuses = [
                result.get("status")
                for result in batch_output.get("results", [])
            ]
            return 0 if statuses and all(s == "PASSED" for s in statuses) else 1
        if progress and all(
            item.state in {"PASSED", "FAILED"} for item in progress
        ):
            return 1
        if args.once:
            return 0
        if args.timeout and time.monotonic() - started_at >= args.timeout:
            print(f"monitor timeout after {args.timeout:g}s", flush=True)
            return 2
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
