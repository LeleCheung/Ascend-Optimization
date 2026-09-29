#!/usr/bin/env python3
"""Record the required human review of one pytest conversion attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = KERNELGEN_ROOT / "kernel_todo_v2" / "pytest_conversion_state.json"
DEFAULT_FLAGGEMS_REPO = KERNELGEN_ROOT.parent / "FlagGems-master"
DEFAULT_APPROVED_FILES = (
    KERNELGEN_ROOT / "kernel_todo_v2" / "pytest_conversion_approved_files"
)


def _file_hash(path: Path) -> str | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink():
        import os

        payload = f"symlink:{os.readlink(path)}".encode()
    elif path.is_file():
        payload = path.read_bytes()
    else:
        payload = b"directory"
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_approved_files(
    repo: Path, target_paths: list[str], destination: Path
) -> None:
    for relative_path in target_paths:
        source = repo / relative_path
        snapshot = destination / relative_path
        if not source.is_file():
            raise RuntimeError(
                f"approved target is not a regular file: {relative_path}"
            )
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        temporary = snapshot.with_suffix(snapshot.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open or close the serial review gate")
    parser.add_argument("--operator", required=True)
    parser.add_argument(
        "--decision", required=True, choices=("approved", "needs_changes")
    )
    parser.add_argument("--notes", required=True)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--flaggems-repo", type=Path, default=DEFAULT_FLAGGEMS_REPO)
    parser.add_argument(
        "--approved-files", type=Path, default=DEFAULT_APPROVED_FILES
    )
    args = parser.parse_args()

    state_path = args.state.resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    active = state.get("active_review")
    if not active:
        raise RuntimeError("there is no active conversion to review")
    if args.operator not in (active["source_operator"], active["operator"]):
        raise RuntimeError(
            f"active review is {active['source_operator']!r}, not {args.operator!r}"
        )

    repo = args.flaggems_repo.resolve()
    current_hashes = {
        path: _file_hash(repo / path) for path in active["target_paths"]
    }
    if current_hashes != active["target_hashes"]:
        raise RuntimeError(
            "target pytest files changed after the agent run; review the new diff "
            "and rerun the conversion before recording a decision"
        )
    if args.decision == "approved" and active.get("out_of_scope_paths"):
        raise RuntimeError("a run with out-of-scope changes cannot be approved")
    if args.decision == "approved" and active.get("command_audit_issues"):
        raise RuntimeError(
            "a run with command-audit issues cannot be approved: "
            + " | ".join(active["command_audit_issues"])
        )

    review = {
        **active,
        "decision": args.decision,
        "notes": args.notes,
        "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    state.setdefault("history", []).append(review)
    if args.decision == "approved":
        _save_approved_files(
            repo, active["target_paths"], args.approved_files.resolve()
        )
        state.setdefault("approved_file_hashes", {}).update(current_hashes)
        state["active_review"] = None
    else:
        active["review_status"] = "needs_changes"
        active["review_notes"] = args.notes
    _write_json(state_path, state)
    print(
        f"Recorded {args.decision} for {active['source_operator']}; "
        + (
            "the next operator may run"
            if args.decision == "approved"
            else "only this operator may be retried"
        )
    )


if __name__ == "__main__":
    main()
