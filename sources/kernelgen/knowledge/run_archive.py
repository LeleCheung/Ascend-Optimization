"""Immutable, content-addressed archive for completed workspace states."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kernelgen.knowledge.models import (
    OperatorSignature,
    TargetContext,
)
from kernelgen.knowledge.contracts.runtime import WorkspaceKnowledgeState
from kernelgen.knowledge.layout import KnowledgeLayout, safe_name

LEDGER_FILENAME = ".ledger.json"


@dataclass(frozen=True)
class ArchivedRunSnapshot:
    root: Path
    ref: str
    snapshot_id: str
    manifest: dict


class KernelGenRunArchive:
    """Freeze ledger and referenced round artifacts without changing workspace."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def archive_workspace(
        self,
        workspace: Path,
        state: WorkspaceKnowledgeState,
        signature: OperatorSignature,
        target: TargetContext,
        *,
        run_id: str,
    ) -> ArchivedRunSnapshot:
        workspace = Path(workspace).resolve()
        ledger = workspace / LEDGER_FILENAME
        if not ledger.is_file():
            raise ValueError(f"workspace has no ledger: {workspace}")
        ledger_bytes = ledger.read_bytes()
        ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
        knowledge_logs = _knowledge_logs(workspace)
        if knowledge_logs:
            digest = hashlib.sha256(ledger_bytes)
            for name, data in sorted(knowledge_logs.items()):
                digest.update(b"\0")
                digest.update(name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(data)
            snapshot_id = digest.hexdigest()
        else:
            snapshot_id = ledger_sha
        run_key = safe_name(run_id)
        workspace_key = safe_name(state.workspace_id)
        snapshots = (
            self.root / run_key / workspace_key / "snapshots"
        )
        destination = snapshots / snapshot_id
        archive_ref = (
            f"archive://{run_key}/{workspace_key}/snapshots/{snapshot_id}"
        )
        if destination.is_dir():
            manifest = _read_manifest(destination)
            _verify_archive(
                destination,
                manifest,
                run_id=run_id,
                workspace_id=state.workspace_id,
                ledger_sha=ledger_sha,
            )
            return ArchivedRunSnapshot(
                root=destination,
                ref=archive_ref,
                snapshot_id=snapshot_id,
                manifest=manifest,
            )

        snapshots.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{snapshot_id}.", dir=str(snapshots))
        )
        try:
            (temporary / "ledger.json").write_bytes(ledger_bytes)
            context = temporary / "context"
            context.mkdir(parents=True, exist_ok=True)
            (context / "operator-signature.json").write_text(
                signature.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
            (context / "target-context.json").write_text(
                target.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
            if knowledge_logs:
                knowledge = temporary / "knowledge"
                knowledge.mkdir(parents=True, exist_ok=True)
                for name, data in sorted(knowledge_logs.items()):
                    (knowledge / name).write_bytes(data)
            payload = json.loads(ledger_bytes)
            artifact_paths = _artifact_paths(workspace, payload)
            for relative in artifact_paths:
                _copy_workspace_artifact(workspace, temporary, relative)
            manifest = {
                "schema_version": "1.0",
                "run_id": run_id,
                "workspace_id": state.workspace_id,
                "definition_id": signature.definition_id,
                "ledger_sha256": ledger_sha,
                "archive_ref": archive_ref,
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "files": _file_manifest(temporary),
            }
            (temporary / "manifest.json").write_text(
                json.dumps(
                    manifest,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            try:
                os.replace(temporary, destination)
            except FileExistsError:
                pass
            manifest = _read_manifest(destination)
            _verify_archive(
                destination,
                manifest,
                run_id=run_id,
                workspace_id=state.workspace_id,
                ledger_sha=ledger_sha,
            )
            return ArchivedRunSnapshot(
                root=destination,
                ref=archive_ref,
                snapshot_id=snapshot_id,
                manifest=manifest,
            )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)


def _knowledge_logs(workspace: Path) -> dict[str, bytes]:
    layout = KnowledgeLayout(workspace)
    logs = {}
    if layout.retrieval_log.is_file():
        logs["retrieval-log.jsonl"] = layout.retrieval_log.read_bytes()
    if layout.query_log.is_file():
        logs["query-log.legacy.jsonl"] = layout.query_log.read_bytes()
    return logs


def _artifact_paths(workspace: Path, ledger: dict) -> list[str]:
    paths = set()
    for round_record in ledger.get("rounds", []):
        if not isinstance(round_record, dict):
            continue
        solution = round_record.get("solution") or {}
        profile = round_record.get("profile") or {}
        for value in (
            solution.get("snapshot_path"),
            profile.get("analysis_path"),
        ):
            if isinstance(value, str) and value.strip():
                paths.add(value.strip())
    pending = list(paths)
    visited = set()
    while pending:
        relative = pending.pop()
        if relative in visited:
            continue
        visited.add(relative)
        path = workspace / relative
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for discovered in _json_artifact_paths(payload):
            candidate = workspace / discovered
            if candidate.exists() and discovered not in paths:
                paths.add(discovered)
                pending.append(discovered)
    return sorted(paths)


def _json_artifact_paths(value, key: str = "") -> list[str]:
    paths = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            paths.extend(_json_artifact_paths(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            paths.extend(_json_artifact_paths(child, key))
    elif (
        isinstance(value, str)
        and value.strip()
        and (key.endswith("_path") or key.endswith("_paths"))
    ):
        candidate = Path(value.strip())
        if not candidate.is_absolute() and ".." not in candidate.parts:
            paths.append(candidate.as_posix())
    return paths


def _copy_workspace_artifact(
    workspace: Path,
    archive_root: Path,
    relative: str,
) -> None:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"artifact path escapes workspace: {relative}")
    source = (workspace / candidate).resolve()
    try:
        source.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"artifact path escapes workspace: {relative}") from exc
    if not source.exists():
        return
    destination = archive_root / "artifacts" / candidate
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    elif source.is_file():
        shutil.copy2(source, destination, follow_symlinks=False)


def _file_manifest(root: Path) -> list[dict]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        records.append(
            {
                "path": relative,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return records


def _read_manifest(root: Path) -> dict:
    try:
        value = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid run archive at {root}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"invalid run archive manifest at {root}")
    return value


def _verify_archive(
    root: Path,
    manifest: dict,
    *,
    run_id: str,
    workspace_id: str,
    ledger_sha: str,
) -> None:
    expected = {
        "run_id": run_id,
        "workspace_id": workspace_id,
        "ledger_sha256": ledger_sha,
    }
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"run archive identity mismatch: expected={expected}, actual={actual}"
        )
    expected_files = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, dict) and item.get("path")
    }
    actual_paths = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative != "manifest.json":
            actual_paths.add(relative)
    expected_paths = set(expected_files)
    if expected_paths != actual_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise ValueError(
            "run archive file inventory mismatch: "
            f"missing_count={len(missing)}, "
            f"unexpected_count={len(unexpected)}, "
            f"missing_examples={missing[:8]}, "
            f"unexpected_examples={unexpected[:8]}"
        )
    for relative, expected_file in expected_files.items():
        data = (root / relative).read_bytes()
        actual_file = {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        expected_file_identity = {
            "size": expected_file.get("size"),
            "sha256": expected_file.get("sha256"),
        }
        if actual_file != expected_file_identity:
            raise ValueError(
                f"run archive checksum mismatch for {relative}: "
                f"expected={expected_file_identity}, actual={actual_file}"
            )
