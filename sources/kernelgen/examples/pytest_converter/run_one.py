#!/usr/bin/env python3
"""Run one pytest conversion and stop at the mandatory review gate.

The command never loops over the inventory.  A second operator cannot start
until ``record_review.py`` approves the active result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KERNELGEN_ROOT.parent))

from kernelgen.agents.pytest_converter import PytestConverterAgent
from kernelgen.framework.agent_roles import materialize_agent_role
from kernelgen.framework.runtime.claude import ClaudeRuntime


DEFAULT_FLAGGEMS_REPO = KERNELGEN_ROOT.parent / "FlagGems-master"
DEFAULT_MANIFEST = (
    KERNELGEN_ROOT / "kernel_todo_v2" / "pytest_conversion_inventory.json"
)
DEFAULT_STATE = KERNELGEN_ROOT / "kernel_todo_v2" / "pytest_conversion_state.json"
DEFAULT_RUNS = KERNELGEN_ROOT / "kernel_todo_v2" / "pytest_conversion_runs"
RUNTIME_ARTIFACT_PREFIXES = (".kernelgen/",)
RUNTIME_LOG = Path(".kernelgen/claude-runtime.log")


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


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "kernelgen.pytest-conversion-state/v1",
            "active_review": None,
            "approved_file_hashes": {},
            "history": [],
        }
    state = _read_json(path)
    if state.get("schema_version") != "kernelgen.pytest-conversion-state/v1":
        raise ValueError(f"unsupported conversion state: {path}")
    return state


def _find_operator(manifest: dict[str, Any], requested: str) -> dict[str, Any]:
    matches = [
        item
        for item in manifest["operators"]
        if requested in (item["source_operator"], item["operator"])
    ]
    if not matches:
        raise ValueError(f"operator {requested!r} is not in the conversion inventory")
    source_names = {item["source_operator"] for item in matches}
    if len(source_names) != 1:
        raise ValueError(
            f"operator {requested!r} is ambiguous; use one CSV spelling from "
            f"{sorted(source_names)}"
        )
    return matches[0]


def _file_hash(path: Path) -> str | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink():
        payload = f"symlink:{os.readlink(path)}".encode()
    elif path.is_file():
        payload = path.read_bytes()
    else:
        payload = b"directory"
    return hashlib.sha256(payload).hexdigest()


def _repo_snapshot(repo: Path) -> dict[str, str | None]:
    output = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    paths = [item.decode() for item in output.split(b"\0") if item]
    return {path: _file_hash(repo / path) for path in paths}


def _changed_paths(
    before: dict[str, str | None], after: dict[str, str | None]
) -> list[str]:
    return sorted(
        path
        for path in before.keys() | after.keys()
        if before.get(path) != after.get(path)
    )


def _read_appended_bytes(path: Path, offset: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(offset)
        return stream.read().decode("utf-8", errors="replace")


def _extract_bash_commands(log_text: str) -> list[str]:
    """Extract exact Bash tool commands from one Claude human-log segment."""
    lines = log_text.splitlines()
    commands: list[str] = []
    index = 0
    while index < len(lines):
        if not lines[index].startswith("[claude] 🔧 Bash:"):
            index += 1
            continue
        payload_start = index + 1
        while payload_start < len(lines) and lines[payload_start].strip() != "{":
            payload_start += 1
        parsed = None
        payload_end = payload_start
        while payload_end < len(lines):
            try:
                parsed = json.loads("\n".join(lines[payload_start : payload_end + 1]))
            except json.JSONDecodeError:
                payload_end += 1
                continue
            break
        if isinstance(parsed, dict) and isinstance(parsed.get("command"), str):
            commands.append(parsed["command"])
        index = max(index + 1, payload_end + 1)
    return commands


def _is_reportable_command(command: str) -> bool:
    stripped = command.strip()
    return (
        stripped == "pwd"
        or stripped.startswith("python ")
        or stripped.startswith("python3 ")
        or stripped.startswith("git ")
    )


def _audit_reported_commands(
    actual_bash_commands: list[str], reported_commands: list[str]
) -> list[str]:
    actual = [
        command
        for command in actual_bash_commands
        if _is_reportable_command(command)
    ]

    def collapse_repeated_validation(commands: list[str]) -> list[str]:
        collapsed: list[str] = []
        for command in commands:
            is_repeatable = (
                command == "pwd"
                or command.startswith("git diff --check ")
                or (command.startswith("python") and " -m py_compile " in command)
            )
            if is_repeatable and collapsed and collapsed[-1] == command:
                continue
            collapsed.append(command)
        return collapsed

    issues: list[str] = []
    if collapse_repeated_validation(actual) != collapse_repeated_validation(
        reported_commands
    ):
        issues.append(
            "reported commands do not exactly match the ordered pwd/python/git "
            f"commands from the runtime log; actual={actual!r}, "
            f"reported={reported_commands!r}"
        )
    forbidden = (" | ", " && ", " || ", "2>&1", " >> ", " > ", "; echo")
    invalid = [
        command for command in actual if any(token in command for token in forbidden)
    ]
    if invalid:
        issues.append(
            "validation commands contain forbidden shell composition: "
            + repr(invalid)
        )
    return issues


def _is_dirty(repo: Path, relative_path: str) -> bool:
    result = subprocess.run(
        [
            "git",
            "status",
            "--short",
            "--untracked-files=all",
            "--",
            relative_path,
        ],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return bool(result.stdout.strip())


def _check_target_baseline(
    repo: Path,
    target_paths: list[str],
    state: dict[str, Any],
    item: dict[str, Any],
) -> None:
    approved = state.get("approved_file_hashes", {})
    active = state.get("active_review")
    retry_hashes: dict[str, str | None] = {}
    if (
        active
        and active.get("source_operator") == item["source_operator"]
        and active.get("review_status") == "needs_changes"
    ):
        retry_hashes = active.get("target_hashes", {})

    unexpected = []
    for relative_path in target_paths:
        if not _is_dirty(repo, relative_path):
            continue
        current = _file_hash(repo / relative_path)
        expected = retry_hashes.get(relative_path, approved.get(relative_path))
        if expected is None or current != expected:
            unexpected.append(relative_path)
    if unexpected:
        raise RuntimeError(
            "target pytest files contain changes not approved by this workflow: "
            + ", ".join(unexpected)
        )


def _check_review_gate(
    repo: Path, state: dict[str, Any], item: dict[str, Any]
) -> None:
    active = state.get("active_review")
    if not active:
        return
    same_operator_retry = (
        active.get("source_operator") == item["source_operator"]
        and active.get("review_status") == "needs_changes"
    )
    if not same_operator_retry:
        raise RuntimeError(
            "a conversion is still waiting for review: "
            f"{active.get('source_operator')} ({active.get('review_status')})"
        )
    unresolved = []
    for path, expected in active.get("scope_violation_baseline", {}).items():
        if _file_hash(repo / path) != expected:
            unresolved.append(path)
    if unresolved:
        raise RuntimeError(
            "out-of-scope changes from the previous attempt must be restored "
            "before retry: "
            + ", ".join(unresolved)
        )


def _sync_native_agent(flaggems_repo: Path) -> Path:
    source = (
        KERNELGEN_ROOT / ".kernelgen" / "agents" / "kernel-pytest-converter.md"
    )
    materialize_agent_role(source, flaggems_repo)
    return flaggems_repo / ".claude" / "agents" / source.name


def _safe_name(value: str) -> str:
    return re_sub(r"[^A-Za-z0-9_.-]+", "_", value)


def re_sub(pattern: str, replacement: str, value: str) -> str:
    # Kept as a small seam for host tests without making regex global state.
    import re

    return re.sub(pattern, replacement, value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert exactly one FlagGems operator and wait for review"
    )
    parser.add_argument("--operator", required=True)
    parser.add_argument("--flaggems-repo", type=Path, default=DEFAULT_FLAGGEMS_REPO)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--model", default=os.environ.get("MODEL", "inherit"))
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-tests", action="store_true")
    args = parser.parse_args()

    repo = args.flaggems_repo.resolve()
    manifest = _read_json(args.manifest.resolve())
    item = _find_operator(manifest, args.operator)
    state = _load_state(args.state.resolve())
    _check_review_gate(repo, state, item)

    target_paths = list(
        dict.fromkeys([*item["accuracy_files"], *item["benchmark_files"]])
    )
    if not target_paths:
        raise RuntimeError(f"inventory has no pytest files for {args.operator!r}")
    _check_target_baseline(repo, target_paths, state, item)

    agent_definition = _sync_native_agent(repo)
    print(f"Synced native agent: {agent_definition}")
    print(
        f"Converting one operator: {item['source_operator']} -> {item['operator']}"
    )
    print("Review gate will remain closed after this invocation.")

    before = _repo_snapshot(repo)
    runtime = ClaudeRuntime(
        workspace=repo,
        model=args.model,
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        auth_token=(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        ),
        allowed_tools="Bash,Read,Write,Edit,Glob,Grep",
        timeout=args.timeout,
        idle_timeout=min(args.timeout / 2, 900),
        permission_mode="acceptEdits",
    )
    agent_input = {
        "source_operator": item["source_operator"],
        "operator": item["operator"],
        "pytest_mark": item["pytest_mark"],
        "accuracy_files": item["accuracy_files"],
        "benchmark_files": item["benchmark_files"],
        "shared_files": item.get("shared_files", {}),
        "review_feedback": (
            [state["active_review"]["review_notes"]]
            if state.get("active_review")
            and state["active_review"].get("review_status") == "needs_changes"
            else []
        ),
        "standard_path": "docs/content/zh-cn/testing/kernelgen-integration.md",
        "run_tests": not args.no_tests,
    }

    runtime_log = repo / RUNTIME_LOG
    runtime_log_offset = runtime_log.stat().st_size if runtime_log.exists() else 0
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    result = None
    agent_error = None
    try:
        result = PytestConverterAgent().run(agent_input, runtime)
    except Exception as exc:
        agent_error = f"{type(exc).__name__}: {exc}"
    after = _repo_snapshot(repo)
    runtime_log_segment = _read_appended_bytes(runtime_log, runtime_log_offset)
    actual_bash_commands = _extract_bash_commands(runtime_log_segment)
    reported_commands = (
        [command.command for command in result.commands] if result is not None else []
    )
    command_audit_issues = _audit_reported_commands(
        actual_bash_commands, reported_commands
    )
    changed = _changed_paths(before, after)
    changed_business_paths = [
        path
        for path in changed
        if not path.startswith(RUNTIME_ARTIFACT_PREFIXES)
    ]
    out_of_scope = [
        path for path in changed_business_paths if path not in target_paths
    ]

    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    record_path = args.runs_dir.resolve() / (
        f"{timestamp}-{_safe_name(item['source_operator'])}.json"
    )
    record = {
        "schema_version": "kernelgen.pytest-conversion-run/v1",
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "flaggems_repo": str(repo),
        "flaggems_head": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
        "input": agent_input,
        "agent_output": (
            result.model_dump(mode="json")
            if result is not None
            else {
                "source_operator": item["source_operator"],
                "operator": item["operator"],
                "status": "failed",
                "files_modified": [],
                "commands": [],
                "remaining_issues": [agent_error],
                "summary": "Claude invocation did not return a valid result",
            }
        ),
        "agent_error": agent_error,
        "command_audit": {
            "actual_bash_commands": actual_bash_commands,
            "issues": command_audit_issues,
        },
        "changed_paths": changed_business_paths,
        "out_of_scope_paths": out_of_scope,
        "command_audit_issues": command_audit_issues,
    }
    _write_json(record_path, record)

    state["active_review"] = {
        "source_operator": item["source_operator"],
        "operator": item["operator"],
        "review_status": "review_pending",
        "run_record": str(record_path),
        "target_paths": target_paths,
        "changed_paths": changed_business_paths,
        "out_of_scope_paths": out_of_scope,
        "target_hashes": {
            path: _file_hash(repo / path) for path in target_paths
        },
        "scope_violation_baseline": {
            path: before.get(path) for path in out_of_scope
        },
    }
    _write_json(args.state.resolve(), state)

    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Run record: {record_path}")
    print("REVIEW REQUIRED: inspect the scoped diff and record a decision.")
    if out_of_scope:
        print("SCOPE VIOLATION: " + ", ".join(out_of_scope), file=sys.stderr)
        raise SystemExit(2)
    if command_audit_issues:
        print(
            "COMMAND AUDIT FAILED: " + " | ".join(command_audit_issues),
            file=sys.stderr,
        )
        raise SystemExit(3)
    if agent_error:
        print(f"AGENT ERROR: {agent_error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
