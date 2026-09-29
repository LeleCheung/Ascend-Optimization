"""Prepare and deterministically audit one Native-to-FlagGems migration diff."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from kernelgen.agents.native_to_flaggems import (
    NativeToFlagGemsAgent,
    NativeToFlagGemsOutput,
)
from kernelgen.framework.agent_roles import materialize_agent_role
from kernelgen.framework.workflow import Workflow


AUDIT_SCHEMA = "kernelgen.native-to-flaggems-audit/v1"


class NativeToFlagGemsWorkflowInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    source_operator: str = ""
    vendor: str = Field(min_length=1)
    native_kernel_path: str
    native_definition_path: str
    flaggems_worktree: str
    accuracy_files: list[str] = Field(min_length=1)
    benchmark_files: list[str] = Field(min_length=1)
    standard_path: str = "docs/content/zh-cn/testing/kernelgen-integration.md"
    run_tests: bool = False


class NativeToFlagGemsWorkflowOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operator: str
    status: str
    ready_for_target_validation: bool = False
    changed_python_files: list[str] = Field(default_factory=list)
    audit_path: str
    agent_report: NativeToFlagGemsOutput | None = None
    reason: str = ""


def _run_git(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def _nul_paths(value: bytes) -> set[str]:
    return {
        item.decode("utf-8", errors="surrogateescape")
        for item in value.split(b"\0")
        if item
    }


def _dirty_paths(root: Path) -> set[str]:
    paths = set()
    paths.update(_nul_paths(_run_git(root, "diff", "--name-only", "-z")))
    paths.update(
        _nul_paths(_run_git(root, "diff", "--cached", "--name-only", "-z"))
    )
    paths.update(
        _nul_paths(
            _run_git(root, "ls-files", "--others", "--exclude-standard", "-z")
        )
    )
    return paths


def _staged_paths(root: Path) -> set[str]:
    return _nul_paths(_run_git(root, "diff", "--cached", "--name-only", "-z"))


def _file_state(root: Path, paths: set[str]) -> dict[str, tuple[bool, str]]:
    state = {}
    for relative in paths:
        path = root / relative
        if not path.is_file():
            state[relative] = (False, "")
            continue
        state[relative] = (True, hashlib.sha256(path.read_bytes()).hexdigest())
    return state


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normalize_reported_paths(paths: list[str]) -> set[str]:
    normalized: set[str] = set()
    for value in paths:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
            raise RuntimeError(f"Agent reported an unsafe changed path: {value!r}")
        normalized.add(path.as_posix())
    if len(normalized) != len(paths):
        raise RuntimeError("Agent reported duplicate changed paths")
    return normalized


def _allowed_python_path(
    path: str,
    *,
    accuracy_files: set[str],
    benchmark_files: set[str],
) -> bool:
    return (
        path.startswith("src/flag_gems/")
        or path in accuracy_files
        or path in benchmark_files
    )


def _validate_scoped_files(root: Path, paths: list[str], label: str) -> set[str]:
    normalized: set[str] = set()
    for value in paths:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".py"
            or str(path) in {"", "."}
        ):
            raise RuntimeError(f"invalid {label} path: {value!r}")
        relative = path.as_posix()
        if not (root / relative).is_file():
            raise RuntimeError(f"{label} file is missing: {relative}")
        normalized.add(relative)
    if len(normalized) != len(paths):
        raise RuntimeError(f"{label} paths contain duplicates")
    return normalized


def _syntax_check(root: Path, paths: set[str]) -> None:
    for relative in sorted(paths):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"migration deleted a Python file: {relative}")
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise RuntimeError(f"invalid Python after migration: {relative}: {exc}") from exc


class NativeToFlagGemsWorkflow(Workflow):
    name = "native_to_flaggems"
    InputModel = NativeToFlagGemsWorkflowInput
    OutputModel = NativeToFlagGemsWorkflowOutput

    def __init__(
        self,
        *,
        cwd: str = ".",
        runtime_factory: Callable[[str], Any] | None = None,
    ):
        self._cwd = Path(cwd)
        self._runtime_factory = runtime_factory

    @classmethod
    def bind(cls, path: str, runtime_factory) -> "NativeToFlagGemsWorkflow":
        return cls(cwd=path, runtime_factory=runtime_factory)

    def _execute(self, inp: NativeToFlagGemsWorkflowInput) -> dict[str, Any]:
        self._cwd.mkdir(parents=True, exist_ok=True)
        audit_path = self._cwd / "migration_audit.json"
        flaggems = Path(inp.flaggems_worktree).expanduser().resolve()
        kernel_path = Path(inp.native_kernel_path).expanduser().resolve()
        definition_path = Path(inp.native_definition_path).expanduser().resolve()
        base = {
            "schema_version": AUDIT_SCHEMA,
            "operator": inp.operator,
            "flaggems_worktree": str(flaggems),
            "native_kernel_path": str(kernel_path),
            "native_definition_path": str(definition_path),
        }

        try:
            repository_root = Path(
                _run_git(flaggems, "rev-parse", "--show-toplevel")
                .decode("utf-8")
                .strip()
            ).resolve()
            if repository_root != flaggems:
                raise RuntimeError(
                    f"flaggems_worktree must be the Git root: {repository_root}"
                )
            if not kernel_path.is_file() or not definition_path.is_file():
                raise RuntimeError("Native kernel or Definition file is missing")
            definition = json.loads(definition_path.read_text(encoding="utf-8"))
            if not isinstance(definition, dict):
                raise RuntimeError("Native Definition root must be an object")
            if definition.get("name") not in {None, inp.operator}:
                raise RuntimeError(
                    f"Native Definition name={definition.get('name')!r} "
                    f"does not match {inp.operator!r}"
                )
            accuracy_files = _validate_scoped_files(
                flaggems, inp.accuracy_files, "accuracy"
            )
            benchmark_files = _validate_scoped_files(
                flaggems, inp.benchmark_files, "benchmark"
            )
            standard = PurePosixPath(inp.standard_path)
            if (
                standard.is_absolute()
                or ".." in standard.parts
                or not (flaggems / standard).is_file()
            ):
                raise RuntimeError(
                    f"integration standard is missing or unsafe: {inp.standard_path!r}"
                )
            before_head = _run_git(flaggems, "rev-parse", "HEAD").decode().strip()
            before_branch = _run_git(
                flaggems, "symbolic-ref", "--short", "HEAD"
            ).decode().strip()
            before_dirty = _dirty_paths(flaggems)
            dirty_python = sorted(path for path in before_dirty if path.endswith(".py"))
            if dirty_python:
                raise RuntimeError(
                    "FlagGems worktree already has dirty Python files: "
                    f"{dirty_python}"
                )
            before_non_python_state = _file_state(flaggems, before_dirty)
            before_staged = _staged_paths(flaggems)
            if self._runtime_factory is None:
                raise RuntimeError("native_to_flaggems requires a runtime_factory")
            role = (
                Path(__file__).resolve().parents[1]
                / ".kernelgen"
                / "agents"
                / "kernel-native-to-flaggems.md"
            )
            materialize_agent_role(role, self._cwd)
            runtime = self._runtime_factory(str(self._cwd))
            add_directory = getattr(runtime, "add_directory", None)
            if callable(add_directory):
                add_directory(flaggems)

            report = NativeToFlagGemsAgent().run(
                {
                    "operator": inp.operator,
                    "source_operator": inp.source_operator,
                    "vendor": inp.vendor,
                    "flaggems_root": str(flaggems),
                    "native_kernel_code": kernel_path.read_text(encoding="utf-8"),
                    "native_definition": definition,
                    "accuracy_files": inp.accuracy_files,
                    "benchmark_files": inp.benchmark_files,
                    "standard_path": inp.standard_path,
                    "run_tests": inp.run_tests,
                },
                runtime,
            )
            if report.operator != inp.operator:
                raise RuntimeError(
                    f"Agent report operator={report.operator!r} does not match "
                    f"{inp.operator!r}"
                )

            after_head = _run_git(flaggems, "rev-parse", "HEAD").decode().strip()
            after_branch = _run_git(
                flaggems, "symbolic-ref", "--short", "HEAD"
            ).decode().strip()
            if (after_head, after_branch) != (before_head, before_branch):
                raise RuntimeError("Agent changed the FlagGems branch or HEAD")
            after_staged = _staged_paths(flaggems)
            if after_staged != before_staged:
                raise RuntimeError("Agent changed the FlagGems staging area")
            if _file_state(flaggems, before_dirty) != before_non_python_state:
                raise RuntimeError("Agent modified a pre-existing dirty non-Python file")
            after_dirty = _dirty_paths(flaggems)
            changed = after_dirty - before_dirty
            non_python = sorted(path for path in changed if not path.endswith(".py"))
            if non_python:
                raise RuntimeError(
                    f"migration created non-Python changes: {non_python}"
                )
            changed_python = {path for path in changed if path.endswith(".py")}
            out_of_scope = sorted(
                path
                for path in changed_python
                if not _allowed_python_path(
                    path,
                    accuracy_files=accuracy_files,
                    benchmark_files=benchmark_files,
                )
            )
            if out_of_scope:
                raise RuntimeError(f"migration changed out-of-scope files: {out_of_scope}")
            reported = _normalize_reported_paths(report.files_modified)
            if reported != changed_python:
                raise RuntimeError(
                    "Agent report differs from actual Python diff: "
                    f"reported={sorted(reported)}, actual={sorted(changed_python)}"
                )
            _syntax_check(flaggems, changed_python)
            source_changes = {
                path for path in changed_python if path.startswith("src/flag_gems/")
            }
            if report.status == "migrated" and not source_changes:
                raise RuntimeError("migrated report has no FlagGems source change")
            if report.status == "already_integrated" and changed_python:
                raise RuntimeError("already_integrated report unexpectedly changed files")
            if report.status in {"blocked", "failed"} and changed_python:
                raise RuntimeError(
                    f"{report.status} report left Python changes in the worktree"
                )
            ready = report.status in {"migrated", "already_integrated"}
            audit = {
                **base,
                "status": "READY_FOR_TARGET_VALIDATION" if ready else report.status.upper(),
                "ready_for_target_validation": ready,
                "changed_python_files": sorted(changed_python),
                "agent_report": report.model_dump(mode="json"),
                "reason": report.summary,
            }
        except Exception as exc:  # noqa: BLE001 - persist the deterministic gate
            report = locals().get("report")
            changed_after_failure: list[str] = []
            if "before_dirty" in locals():
                try:
                    changed_after_failure = sorted(
                        path
                        for path in (_dirty_paths(flaggems) - before_dirty)
                        if path.endswith(".py")
                    )
                except Exception:  # noqa: BLE001 - preserve the original reason
                    pass
            audit = {
                **base,
                "status": "AUDIT_FAILED",
                "ready_for_target_validation": False,
                "changed_python_files": changed_after_failure,
                "agent_report": (
                    report.model_dump(mode="json")
                    if isinstance(report, NativeToFlagGemsOutput)
                    else None
                ),
                "reason": f"{type(exc).__name__}: {exc}",
            }
        _atomic_json(audit_path, audit)
        return {
            "operator": inp.operator,
            "status": audit["status"],
            "ready_for_target_validation": audit["ready_for_target_validation"],
            "changed_python_files": audit["changed_python_files"],
            "audit_path": str(audit_path.resolve()),
            "agent_report": audit["agent_report"],
            "reason": audit["reason"],
        }


__all__ = [
    "AUDIT_SCHEMA",
    "NativeToFlagGemsWorkflow",
    "NativeToFlagGemsWorkflowInput",
    "NativeToFlagGemsWorkflowOutput",
]
