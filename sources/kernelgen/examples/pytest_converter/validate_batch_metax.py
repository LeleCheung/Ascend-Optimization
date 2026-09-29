#!/usr/bin/env python3
"""Validate one reviewed conversion batch across isolated devices."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any


KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = KERNELGEN_ROOT / "kernel_todo_v2" / "pytest_conversion_state.json"
DEFAULT_FLAGGEMS_REPO = KERNELGEN_ROOT.parent / "FlagGems-master"
DEFAULT_MANIFEST = KERNELGEN_ROOT / "kernel_todo_v2/pytest_conversion_inventory.json"
DEFAULT_APPROVED_FILES = (
    KERNELGEN_ROOT / "kernel_todo_v2/pytest_conversion_approved_files"
)
DEFAULT_RUNS = KERNELGEN_ROOT / "kernel_todo_v2/pytest_conversion_device_runs"
METAX_VALIDATOR = Path(__file__).with_name("validate_one_metax.py")
NVIDIA_VALIDATOR = Path(__file__).with_name("validate_one_nvidia.py")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _merge_batch_results(
    batch_path: Path, result_by_operator: dict[str, dict[str, Any]]
) -> None:
    lock_name = hashlib.sha256(str(batch_path).encode()).hexdigest() + ".lock"
    lock_path = Path(tempfile.gettempdir()) / "kernelgen-pytest-batch-locks" / lock_name
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        latest_batch = _read_json(batch_path)
        for item in latest_batch["items"]:
            result = result_by_operator.get(item["source_operator"])
            if result:
                item["device_validation"] = result
        _write_json(batch_path, latest_batch)
        fcntl.flock(lock, fcntl.LOCK_UN)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _approved_operators(state: dict[str, Any]) -> set[str]:
    decisions: dict[str, str] = {}
    for review in state.get("history", []):
        decisions[review["source_operator"]] = review["decision"]
    return {
        operator
        for operator, decision in decisions.items()
        if decision == "approved"
    }


def _active_from_item(item: dict[str, Any], batch_path: Path) -> dict[str, Any]:
    return {
        "source_operator": item["source_operator"],
        "operator": item["operator"],
        "review_status": "review_pending",
        "run_record": item["run_record"],
        "target_paths": item["target_paths"],
        "changed_paths": item["changed_paths"],
        "out_of_scope_paths": item["out_of_scope_paths"],
        "command_audit_issues": item["command_audit_issues"],
        "target_hashes": item["reviewed_hashes"],
        "scope_violation_baseline": {},
        "batch_manifest": str(batch_path),
        "batch_review_notes": item["review_notes"],
    }


def _run_one(
    item: dict[str, Any],
    *,
    device: str,
    state: dict[str, Any],
    batch_path: Path,
    validation_root: Path,
    validator: Path,
    validator_args: list[str],
) -> dict[str, Any]:
    safe = _safe_name(item["source_operator"])
    item_root = validation_root / safe
    item_root.mkdir(parents=True, exist_ok=True)
    state_path = item_root / "state.json"
    run_state = dict(state)
    run_state["active_review"] = _active_from_item(item, batch_path)
    _write_json(state_path, run_state)
    log_path = item_root / "validate.log"
    command = [
        sys.executable,
        str(validator),
        "--state",
        str(state_path),
        "--device",
        device,
        *validator_args,
    ]
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=KERNELGEN_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    output = log_path.read_text(encoding="utf-8")
    matches = re.findall(r"^Device run record: (.+)$", output, flags=re.MULTILINE)
    record_path = matches[-1].strip() if matches else None
    return {
        "operator": item["source_operator"],
        "device": device,
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "log": str(log_path),
        "record": record_path,
    }


def _run_device_queue(
    device: str,
    item_queue: Queue,
    *,
    state: dict[str, Any],
    batch_path: Path,
    validation_root: Path,
    validator: Path,
    validator_args: list[str],
) -> list[dict[str, Any]]:
    results = []
    while True:
        try:
            item = item_queue.get_nowait()
        except Empty:
            return results
        try:
            result = _run_one(
                item,
                device=device,
                state=state,
                batch_path=batch_path,
                validation_root=validation_root,
                validator=validator,
                validator_args=validator_args,
            )
            results.append(result)
            _merge_batch_results(batch_path, {result["operator"]: result})
            print(
                f"[{result['status']}] {result['operator']} on device "
                f"{result['device']} log={result['log']}",
                flush=True,
            )
        finally:
            item_queue.task_done()


def main(*, default_local_nvidia: bool = False) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--flaggems-repo", type=Path, default=DEFAULT_FLAGGEMS_REPO)
    parser.add_argument(
        "--approved-files", type=Path, default=DEFAULT_APPROVED_FILES
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--operators", nargs="*")
    parser.add_argument(
        "--local-nvidia",
        action="store_true",
        default=default_local_nvidia,
    )
    parser.add_argument(
        "--devices",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated visible device IDs; one worker per device.",
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--login")
    parser.add_argument("--identity-file", type=Path)
    parser.add_argument("--container")
    parser.add_argument("--remote-template-root")
    parser.add_argument("--remote-python")
    parser.add_argument("--session-python")
    parser.add_argument("--with-site", action="store_true")
    parser.add_argument("--device-env")
    parser.add_argument("--remote-site-packages")
    parser.add_argument("--connect-timeout", type=int)
    parser.add_argument("--control-persist", type=int)
    parser.add_argument("--command-timeout", type=int)
    args = parser.parse_args()

    batch_path = args.batch.resolve()
    batch = _read_json(batch_path)
    state = _read_json(args.state.resolve())
    approved = _approved_operators(state)
    requested = set(args.operators or [])
    items = [
        item
        for item in batch["items"]
        if item["status"] in ("reviewed", "reviewed_with_fixes")
        and item["source_operator"] not in approved
        and (not requested or item["source_operator"] in requested)
    ]
    if requested != {item["source_operator"] for item in items} and requested:
        missing = sorted(requested - {item["source_operator"] for item in items})
        raise RuntimeError(f"operators are not ready for batch validation: {missing}")
    if not items:
        raise RuntimeError("no reviewed, unapproved batch items selected")
    paths = [path for item in items for path in item["target_paths"]]
    if len(paths) != len(set(paths)):
        raise RuntimeError("selected batch items must have disjoint target files")
    devices = [device.strip() for device in args.devices.split(",") if device.strip()]
    if not devices:
        raise RuntimeError("at least one device is required")

    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    validator = NVIDIA_VALIDATOR if args.local_nvidia else METAX_VALIDATOR
    validator_args = [
        "--manifest",
        str(args.manifest.resolve()),
        "--flaggems-repo",
        str(args.flaggems_repo.resolve()),
        "--approved-files",
        str(args.approved_files.resolve()),
        "--runs-dir",
        str(args.runs_dir.resolve()),
    ]
    if not args.local_nvidia:
        for option, value in (
            ("--host", args.host),
            ("--port", args.port),
            ("--login", args.login),
            ("--identity-file", args.identity_file),
            ("--container", args.container),
            ("--remote-template-root", args.remote_template_root),
            ("--remote-python", args.remote_python),
            ("--session-python", args.session_python),
            ("--device-env", args.device_env),
            ("--remote-site-packages", args.remote_site_packages),
            ("--connect-timeout", args.connect_timeout),
            ("--control-persist", args.control_persist),
            ("--command-timeout", args.command_timeout),
        ):
            if value is not None:
                validator_args.extend([option, str(value)])
        if args.with_site:
            validator_args.append("--with-site")
    validation_root = batch_path.parent / "device_validation" / timestamp
    results: list[dict[str, Any]] = []
    item_queue: Queue = Queue()
    for item in items:
        item_queue.put(item)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [
            pool.submit(
                _run_device_queue,
                device=device,
                item_queue=item_queue,
                state=state,
                batch_path=batch_path,
                validation_root=validation_root,
                validator=validator,
                validator_args=validator_args,
            )
            for device in devices
        ]
        for future in as_completed(futures):
            results.extend(future.result())

    result_by_operator = {result["operator"]: result for result in results}
    # Reviews or primary fixes may be recorded while device workers are still
    # running. Reload before merging results so a stale batch snapshot cannot
    # overwrite newer reviewed hashes, notes, or decisions.
    _merge_batch_results(batch_path, result_by_operator)
    summary_path = validation_root / "summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": "kernelgen.pytest-conversion-batch-device-run/v1",
            "batch": str(batch_path),
            "devices": devices,
            "results": sorted(results, key=lambda result: result["operator"]),
        },
    )
    print(f"Batch device summary: {summary_path}")
    if any(result["status"] != "passed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
