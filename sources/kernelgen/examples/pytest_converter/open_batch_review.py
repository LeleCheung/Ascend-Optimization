#!/usr/bin/env python3
"""Open the serial review gate for one already-applied batch result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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
    parser = argparse.ArgumentParser(
        description="Open review for one applied result from run_batch.py"
    )
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--flaggems-repo", type=Path, default=DEFAULT_FLAGGEMS_REPO)
    args = parser.parse_args()

    batch = _read_json(args.batch.resolve())
    matches = [
        item
        for item in batch["items"]
        if args.operator in (item["source_operator"], item["operator"])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"operator has {len(matches)} batch matches")
    item = matches[0]
    if item.get("review_decision") != "approved" or item["status"] not in (
        "reviewed",
        "reviewed_with_fixes",
    ):
        raise RuntimeError(f"batch item has not passed primary review: {item['status']}")

    state_path = args.state.resolve()
    state = _read_json(state_path)
    if state.get("active_review"):
        raise RuntimeError(
            "another operator is already waiting for serial device review"
        )
    repo = args.flaggems_repo.resolve()
    current_hashes = {
        path: _file_hash(repo / path) for path in item["target_paths"]
    }
    if current_hashes != item["reviewed_hashes"]:
        raise RuntimeError(
            "shared target files do not match the reviewed batch result; "
            f"current={current_hashes!r}, expected={item['reviewed_hashes']!r}"
        )
    state["active_review"] = {
        "source_operator": item["source_operator"],
        "operator": item["operator"],
        "review_status": "review_pending",
        "run_record": item["run_record"],
        "target_paths": item["target_paths"],
        "changed_paths": item["changed_paths"],
        "out_of_scope_paths": item["out_of_scope_paths"],
        "command_audit_issues": item["command_audit_issues"],
        "target_hashes": current_hashes,
        "scope_violation_baseline": {},
        "batch_manifest": str(args.batch.resolve()),
        "batch_review_notes": item["review_notes"],
    }
    _write_json(state_path, state)
    print(f"Opened serial review for {item['source_operator']}")


if __name__ == "__main__":
    main()
