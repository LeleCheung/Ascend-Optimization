#!/usr/bin/env python3
"""Reopen an approved operator after a reviewed shared-file correction."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = KERNELGEN_ROOT / "kernel_todo_v2/pytest_conversion_state.json"
DEFAULT_INVENTORY = (
    KERNELGEN_ROOT / "kernel_todo_v2/pytest_conversion_inventory.json"
)
DEFAULT_BATCH_ROOT = (
    KERNELGEN_ROOT / "kernel_todo_v2/pytest_conversion_batches"
)
DEFAULT_FLAGGEMS_REPO = KERNELGEN_ROOT.parent / "FlagGems-master"


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


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument(
        "--flaggems-repo", type=Path, default=DEFAULT_FLAGGEMS_REPO
    )
    args = parser.parse_args()

    state_path = args.state.resolve()
    state = _read_json(state_path)
    if state.get("active_review"):
        raise RuntimeError("cannot reopen while another serial review is active")
    prior = next(
        (
            entry
            for entry in reversed(state.get("history", []))
            if entry["source_operator"] == args.operator
        ),
        None,
    )
    if prior is None or prior.get("decision") != "approved":
        raise RuntimeError("latest operator decision is not approved")

    inventory = _read_json(args.inventory.resolve())
    inventory_item = next(
        item
        for item in inventory["operators"]
        if item["source_operator"] == args.operator
    )
    target_paths = [
        *inventory_item["accuracy_files"],
        *inventory_item["benchmark_files"],
    ]
    repo = args.flaggems_repo.resolve()
    current_hashes = {path: _file_hash(repo / path) for path in target_paths}

    if args.batch:
        batch_path = args.batch.resolve()
        batch = _read_json(batch_path)
        matches = [
            item
            for item in batch["items"]
            if item["source_operator"] == args.operator
        ]
        if len(matches) != 1:
            raise RuntimeError(f"operator has {len(matches)} batch matches")
        item = matches[0]
        item["status"] = "reviewed_with_fixes"
        item["reviewed_hashes"] = current_hashes
        item["review_decision"] = "approved"
        item["review_notes"] = args.notes
        item["reviewed_at"] = _now()
    else:
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        batch_path = DEFAULT_BATCH_ROOT / f"{timestamp}-revalidation" / "manifest.json"
        item = {
            "source_operator": args.operator,
            "operator": inventory_item["operator"],
            "pytest_mark": inventory_item["pytest_mark"],
            "target_paths": target_paths,
            "baseline_hashes": prior["target_hashes"],
            "agent_dir": str(batch_path.parent / "manual-review"),
            "state": str(state_path),
            "status": "reviewed_with_fixes",
            "exit_code": 0,
            "after_hashes": current_hashes,
            "run_record": prior["run_record"],
            "agent_status": "manual_revalidation",
            "remaining_issues": [],
            "command_audit_issues": [],
            "out_of_scope_paths": [],
            "changed_paths": target_paths,
            "reviewed_hashes": current_hashes,
            "review_decision": "approved",
            "review_notes": args.notes,
            "reviewed_at": _now(),
        }
        batch = {
            "schema_version": "kernelgen.pytest-conversion-batch/v1",
            "created_at": _now(),
            "flaggems_repo": str(repo),
            "flaggems_head": None,
            "manual_revalidation": True,
            "items": [item],
        }

    reopened = {
        **prior,
        "decision": "needs_changes",
        "notes": args.notes,
        "reviewed_at": _now(),
        "revalidation_batch": str(batch_path),
    }
    state.setdefault("history", []).append(reopened)
    _write_json(batch_path, batch)
    _write_json(state_path, state)
    print(f"Prepared revalidation for {args.operator}: {batch_path}")


if __name__ == "__main__":
    main()
