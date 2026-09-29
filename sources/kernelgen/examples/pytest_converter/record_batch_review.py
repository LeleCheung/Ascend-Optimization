#!/usr/bin/env python3
"""Record primary review and snapshot the shared files selected for testing."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .run_one import _audit_reported_commands
except ImportError:
    from run_one import _audit_reported_commands


KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FLAGGEMS_REPO = KERNELGEN_ROOT.parent / "FlagGems-master"
DEFAULT_STATE = KERNELGEN_ROOT / "kernel_todo_v2" / "pytest_conversion_state.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record primary batch review")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--decision", choices=("approved", "rejected"), required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--flaggems-repo", type=Path, default=DEFAULT_FLAGGEMS_REPO)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()

    manifest_path = args.batch.resolve()
    batch = _read_json(manifest_path)
    matches = [
        item
        for item in batch["items"]
        if args.operator in (item["source_operator"], item["operator"])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"operator has {len(matches)} batch matches")
    item = matches[0]
    if item["status"] not in (
        "agent_failed",
        "ready_for_review",
        "reviewed",
        "reviewed_with_fixes",
        "rejected",
    ):
        raise RuntimeError(f"batch item is not reviewable: {item['status']}")

    repo = args.flaggems_repo.resolve()
    if args.decision == "approved" and item["status"] == "agent_failed":
        run_record = _read_json(Path(item["run_record"]))
        actual_commands = run_record.get("command_audit", {}).get(
            "actual_bash_commands", []
        )
        reported_commands = [
            command["command"]
            for command in run_record.get("agent_output", {}).get("commands", [])
        ]
        item["command_audit_issues"] = _audit_reported_commands(
            actual_commands, reported_commands
        )
        if (
            item.get("agent_status") not in ("converted", "already_compliant")
            or item["command_audit_issues"]
            or item.get("out_of_scope_paths")
        ):
            raise RuntimeError(
                "agent_failed item has unresolved conversion or audit issues"
            )
    current_hashes = {
        path: _file_hash(repo / path) for path in item["target_paths"]
    }
    reviewed_root = Path(item["agent_dir"]) / "reviewed"
    if args.decision == "approved":
        for relative_path in item["target_paths"]:
            source = repo / relative_path
            if not source.is_file():
                raise RuntimeError(f"reviewed target is missing: {relative_path}")
            destination = reviewed_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        item["reviewed_hashes"] = current_hashes
        item["status"] = (
            "reviewed"
            if current_hashes == item["after_hashes"]
            else "reviewed_with_fixes"
        )
    else:
        item["status"] = "rejected"
    item["review_decision"] = args.decision
    item["review_notes"] = args.notes
    item["reviewed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _write_json(manifest_path, batch)
    state_path = args.state.resolve()
    state = _read_json(state_path)
    active = state.get("active_review")
    if active and args.operator in (
        active["source_operator"],
        active["operator"],
    ):
        if args.decision != "approved":
            raise RuntimeError(
                "cannot reject a batch item while its serial review is active"
            )
        active["target_hashes"] = current_hashes
        active["batch_review_notes"] = args.notes
        active["review_status"] = "review_pending"
        _write_json(state_path, state)
    print(f"Recorded {args.decision} batch review for {item['source_operator']}")


if __name__ == "__main__":
    main()
