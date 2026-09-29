#!/usr/bin/env python3
"""Run up to thirty pytest-converter Agents concurrently in isolated worktrees.

This command does not edit the shared FlagGems worktree or the serial review
state.  It snapshots each target file from the shared worktree, runs one Agent
per detached temporary worktree, and stores before/after files plus a patch for
the primary reviewer.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any


KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLAGGEMS_REPO = KERNELGEN_ROOT.parent / "FlagGems-master"
DEFAULT_MANIFEST = (
    KERNELGEN_ROOT / "kernel_todo_v2" / "pytest_conversion_inventory.json"
)
DEFAULT_STATE = KERNELGEN_ROOT / "kernel_todo_v2" / "pytest_conversion_state.json"
DEFAULT_BATCHES = KERNELGEN_ROOT / "kernel_todo_v2" / "pytest_conversion_batches"
RUN_ONE = Path(__file__).with_name("run_one.py")
DEFAULT_BATCH_SIZE = 30
MAX_BATCH_SIZE = 30
PENDING_BATCH_STATUSES = {
    "ready_for_review",
    "reviewed",
    "reviewed_with_fixes",
}


def _agent_result_is_reviewable(status: str | None) -> bool:
    return status in ("converted", "already_compliant")


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


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_paths(item: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys([*item["accuracy_files"], *item["benchmark_files"]])
    )


def _approved_operators(state: dict[str, Any]) -> set[str]:
    latest: dict[str, str] = {}
    for entry in state.get("history", []):
        latest[entry["source_operator"]] = entry.get("decision", "")
    return {
        operator for operator, decision in latest.items() if decision == "approved"
    }


def _pending_batch_operators(batches_dir: Path) -> set[str]:
    pending: set[str] = set()
    if not batches_dir.is_dir():
        return pending
    for manifest_path in sorted(batches_dir.glob("*/manifest.json")):
        manifest = _read_json(manifest_path)
        for item in manifest.get("items", []):
            if item.get("status") in PENDING_BATCH_STATUSES:
                pending.add(item["source_operator"])
    return pending


def _select_items(
    inventory: dict[str, Any],
    state: dict[str, Any],
    size: int,
    requested: list[str] | None,
    pending_batch_operators: set[str] | None = None,
) -> list[dict[str, Any]]:
    items = inventory["operators"]
    if requested:
        selected = []
        for name in requested:
            matches = [
                item
                for item in items
                if name in (item["source_operator"], item["operator"])
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"operator {name!r} has {len(matches)} inventory matches"
                )
            selected.append(matches[0])
    else:
        approved = _approved_operators(state)
        protocol_gaps = set(state.get("protocol_gaps", {}))
        validation_blockers = set(state.get("validation_blockers", {}))
        pending_batches = pending_batch_operators or set()
        active = state.get("active_review") or {}
        active_source = active.get("source_operator")
        selected = []
        occupied: set[str] = set()
        for item in items:
            if item["source_operator"] in approved:
                continue
            if item["source_operator"] in protocol_gaps:
                continue
            if item["source_operator"] in validation_blockers:
                continue
            if item["source_operator"] in pending_batches:
                continue
            if item["source_operator"] == active_source:
                continue
            paths = set(_target_paths(item))
            if not paths or paths & occupied:
                continue
            selected.append(item)
            occupied.update(paths)
            if len(selected) == size:
                break
    if len(selected) > size:
        raise ValueError(f"requested {len(selected)} operators exceeds batch size {size}")
    occupied: dict[str, str] = {}
    for item in selected:
        for path in _target_paths(item):
            if path in occupied:
                raise ValueError(
                    f"operators {occupied[path]!r} and "
                    f"{item['source_operator']!r} share target {path}"
                )
            occupied[path] = item["source_operator"]
    return selected


def _copy_targets(source: Path, destination: Path, paths: list[str]) -> None:
    for relative_path in paths:
        source_path = source / relative_path
        if not source_path.is_file():
            raise RuntimeError(f"target file is missing: {source_path}")
        destination_path = destination / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def _snapshot_targets(source: Path, destination: Path, paths: list[str]) -> None:
    for relative_path in paths:
        destination_path = destination / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative_path, destination_path)


def _make_patch(
    before_root: Path, after_root: Path, paths: list[str]
) -> str:
    parts: list[str] = []
    for relative_path in paths:
        before = (before_root / relative_path).read_text(encoding="utf-8")
        after = (after_root / relative_path).read_text(encoding="utf-8")
        parts.extend(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{relative_path}",
                tofile=f"b/{relative_path}",
            )
        )
    return "".join(parts)


def _run(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run thirty isolated pytest conversion Agents concurrently"
    )
    parser.add_argument("--size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--operators", nargs="+")
    parser.add_argument("--flaggems-repo", type=Path, default=DEFAULT_FLAGGEMS_REPO)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--batches-dir", type=Path, default=DEFAULT_BATCHES)
    parser.add_argument("--model", default=os.environ.get("MODEL", "inherit"))
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.size < 1 or args.size > MAX_BATCH_SIZE:
        raise ValueError(f"batch size must be between 1 and {MAX_BATCH_SIZE}")

    repo = args.flaggems_repo.resolve()
    inventory = _read_json(args.manifest.resolve())
    state = _read_json(args.state.resolve())
    items = _select_items(
        inventory,
        state,
        args.size,
        args.operators,
        _pending_batch_operators(args.batches_dir.resolve()),
    )
    if not items:
        print("No pending operators selected")
        return

    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    batch_dir = args.batches_dir.resolve() / timestamp
    batch_dir.mkdir(parents=True, exist_ok=False)
    temporary_root = Path(tempfile.mkdtemp(prefix=f"pytest-converter-{timestamp}-"))
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    entries: list[dict[str, Any]] = []

    try:
        for ordinal, item in enumerate(items):
            safe_name = f"{ordinal:02d}-{_safe_name(item['source_operator'])}"
            agent_dir = batch_dir / "agents" / safe_name
            worktree = temporary_root / safe_name
            _run(
                ["git", "worktree", "add", "--detach", str(worktree), head],
                cwd=repo,
            )
            paths = _target_paths(item)
            _copy_targets(repo, worktree, paths)
            before_root = agent_dir / "before"
            _snapshot_targets(repo, before_root, paths)
            baseline_hashes = {
                path: _file_hash(repo / path) for path in paths
            }
            agent_state = agent_dir / "state.json"
            _write_json(
                agent_state,
                {
                    "schema_version": "kernelgen.pytest-conversion-state/v1",
                    "active_review": None,
                    "approved_file_hashes": baseline_hashes,
                    "history": [],
                },
            )
            entry = {
                "source_operator": item["source_operator"],
                "operator": item["operator"],
                "pytest_mark": item["pytest_mark"],
                "target_paths": paths,
                "baseline_hashes": baseline_hashes,
                "agent_dir": str(agent_dir),
                "worktree": str(worktree),
                "state": str(agent_state),
                "status": "running",
            }
            entries.append(entry)

        processes: list[tuple[dict[str, Any], subprocess.Popen[bytes], Any]] = []
        for entry in entries:
            agent_dir = Path(entry["agent_dir"])
            log_path = agent_dir / "agent.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_stream = log_path.open("wb")
            command = [
                sys.executable,
                str(RUN_ONE),
                "--operator",
                entry["source_operator"],
                "--flaggems-repo",
                entry["worktree"],
                "--manifest",
                str(args.manifest.resolve()),
                "--state",
                entry["state"],
                "--runs-dir",
                str(agent_dir / "runs"),
                "--model",
                args.model,
                "--timeout",
                str(args.timeout),
                "--no-tests",
            ]
            process = subprocess.Popen(
                command,
                cwd=KERNELGEN_ROOT,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            processes.append((entry, process, log_stream))
            print(
                f"started {entry['source_operator']} pid={process.pid}",
                flush=True,
            )

        pending = {process.pid for _, process, _ in processes}
        while pending:
            for entry, process, log_stream in processes:
                if process.pid not in pending or process.poll() is None:
                    continue
                pending.remove(process.pid)
                log_stream.close()
                entry["exit_code"] = process.returncode
                print(
                    f"finished {entry['source_operator']} exit={process.returncode}",
                    flush=True,
                )
            if pending:
                time.sleep(1)

        for entry, process, log_stream in processes:
            if not log_stream.closed:
                log_stream.close()
            agent_dir = Path(entry["agent_dir"])
            worktree = Path(entry["worktree"])
            after_root = agent_dir / "after"
            _snapshot_targets(worktree, after_root, entry["target_paths"])
            entry["after_hashes"] = {
                path: _file_hash(worktree / path)
                for path in entry["target_paths"]
            }
            patch = _make_patch(
                agent_dir / "before", after_root, entry["target_paths"]
            )
            patch_path = agent_dir / "changes.diff"
            patch_path.write_text(patch, encoding="utf-8")
            entry["patch"] = str(patch_path)
            agent_state = _read_json(Path(entry["state"]))
            active = agent_state.get("active_review") or {}
            entry["run_record"] = active.get("run_record")
            run_record = (
                _read_json(Path(entry["run_record"]))
                if entry["run_record"]
                else {}
            )
            agent_output = run_record.get("agent_output", {})
            entry["agent_status"] = agent_output.get("status")
            entry["remaining_issues"] = agent_output.get("remaining_issues", [])
            entry["command_audit_issues"] = run_record.get(
                "command_audit_issues", []
            )
            entry["out_of_scope_paths"] = run_record.get(
                "out_of_scope_paths", []
            )
            entry["changed_paths"] = run_record.get("changed_paths", [])
            if (
                entry.get("exit_code") == 0
                and _agent_result_is_reviewable(entry["agent_status"])
                and not entry["command_audit_issues"]
                and not entry["out_of_scope_paths"]
                and active.get("review_status") == "review_pending"
            ):
                entry["status"] = "ready_for_review"
            elif entry["agent_status"] == "blocked":
                entry["status"] = "protocol_gap"
            else:
                entry["status"] = "agent_failed"

        batch = {
            "schema_version": "kernelgen.pytest-conversion-batch/v1",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "flaggems_repo": str(repo),
            "flaggems_head": head,
            "temporary_root": str(temporary_root),
            "items": entries,
        }
        _write_json(batch_dir / "manifest.json", batch)
        print(f"Batch manifest: {batch_dir / 'manifest.json'}")
    finally:
        for entry in entries:
            worktree = Path(entry["worktree"])
            if worktree.exists():
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(worktree)],
                    cwd=repo,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    main()
