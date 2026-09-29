"""Small, explicit Concept lifecycle operations."""

from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

from kernelgen.knowledge.catalog import (
    FilesystemCatalog,
    atomic_text,
    concept_ref,
)
from kernelgen.knowledge.context import normalize_device
from kernelgen.knowledge.index import SQLiteKnowledgeIndex
from kernelgen.knowledge.layout import CatalogLayout, safe_name
from kernelgen.knowledge.models import TargetContext, TargetScope
from kernelgen.knowledge.validation import canonical_concept_hash


_VERSION_FIELDS = (
    "language_version",
    "compiler_version",
    "runtime_version",
    "library_version",
    "driver_version",
)


def deprecate_concept(
    catalog_root: Path,
    concept_id: str,
    *,
    reason: str,
) -> dict:
    if not reason.strip():
        raise ValueError("deprecation reason is required")
    return _change_status(
        Path(catalog_root),
        concept_id,
        reason=reason,
        status="deprecated",
    )


def delete_concept(
    catalog_root: Path,
    concept_id: str,
    *,
    reason: str,
    apply: bool = False,
) -> dict:
    if not reason.strip():
        raise ValueError("deletion reason is required")
    root = Path(catalog_root)
    layout = CatalogLayout(root)
    layout.publish_lock.parent.mkdir(parents=True, exist_ok=True)
    with layout.publish_lock.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        catalog = FilesystemCatalog(root)
        concept = catalog.get_concept(concept_id)
        if concept.status != "deprecated":
            raise ValueError("hard delete requires a deprecated Concept")
        if concept.evidence:
            raise ValueError(
                "hard delete is limited to deprecated Concepts without evidence"
            )
        incoming = sorted(
            item.id
            for item in catalog.iter_concepts()
            if any(
                relation.target == concept_id
                for relation in item.relations
            )
        )
        if incoming:
            raise ValueError(
                "hard delete would leave dangling relations: "
                + ", ".join(incoming)
            )
        path = catalog.concept_path(concept)
        report = {
            "status": "planned",
            "concept_id": concept_id,
            "reason": reason,
            "path": str(path.relative_to(root)),
        }
        if not apply:
            return report
        backup = path.read_bytes()
        path.unlink()
        try:
            _rebuild_index(root)
        except BaseException:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(backup)
            _rebuild_index(root)
            raise
        report["status"] = "deleted"
        report["audit_ref"] = _write_audit(root, "delete", report)
        return report


def _change_status(
    root: Path,
    concept_id: str,
    *,
    reason: str,
    status: str,
) -> dict:
    layout = CatalogLayout(root)
    layout.publish_lock.parent.mkdir(parents=True, exist_ok=True)
    with layout.publish_lock.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        catalog = FilesystemCatalog(root)
        concept = catalog.get_concept(concept_id)
        if concept.status == status:
            return {
                "status": "noop",
                "concept_id": concept_id,
                "reason": reason,
            }
        path = catalog.concept_path(concept)
        backup = path.read_bytes()
        updated = concept.model_copy(
            update={
                "status": status,
                "managed": concept.managed.model_copy(
                    update={
                        "content_hash": "sha256:" + "0" * 64,
                        "updated_at": datetime.now(timezone.utc),
                    }
                ),
            }
        )
        updated = updated.model_copy(
            update={
                "managed": updated.managed.model_copy(
                    update={
                        "content_hash": canonical_concept_hash(updated)
                    }
                )
            }
        )
        try:
            catalog.write_concept(updated)
            _rebuild_index(root)
        except BaseException:
            path.write_bytes(backup)
            _rebuild_index(root)
            raise
        report = {
            "status": status,
            "concept_id": concept_id,
            "reason": reason,
            "path": str(path.relative_to(root)),
        }
        report["audit_ref"] = _write_audit(root, status, report)
        return report


def revalidation_candidates(
    catalog_root: Path,
    target: TargetContext,
) -> list[dict]:
    """Return Concepts whose recorded software differs on the same target."""

    candidates = []
    for concept in FilesystemCatalog(catalog_root).iter_concepts():
        expected = concept.scope.target
        if not _same_target_family(expected, target):
            continue
        if not _same_software(expected, target):
            continue
        changes = []
        for field in _VERSION_FIELDS:
            required = getattr(expected.software, field)
            observed = getattr(target.software, field)
            if required and observed and required != observed:
                changes.append(
                    {
                        "field": field,
                        "recorded": required,
                        "current": observed,
                    }
                )
        if changes:
            candidates.append(
                {
                    "concept_ref": concept_ref(concept),
                    "changes": changes,
                }
            )
    return sorted(candidates, key=lambda item: item["concept_ref"])


def rollback_publish_batch(
    catalog_root: Path,
    batch_id: str,
    *,
    apply: bool = False,
) -> dict:
    """Plan or apply rollback of an observation-only publish batch."""

    root = Path(catalog_root)
    layout = CatalogLayout(root)
    layout.publish_lock.parent.mkdir(parents=True, exist_ok=True)
    with layout.publish_lock.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _rollback_locked(root, batch_id, apply=apply)


def _rollback_locked(root: Path, batch_id: str, *, apply: bool) -> dict:
    layout = CatalogLayout(root)
    audit_path = layout.publish_audit / f"{safe_name(batch_id)}.json"
    if not audit_path.is_file():
        raise ValueError(f"publish audit not found: {batch_id}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("batch_id") != batch_id:
        raise ValueError(f"publish audit batch mismatch: {audit_path}")
    snapshot_before = str(audit.get("snapshot_before") or "")
    snapshot_after = str(audit.get("snapshot_after") or "")
    if not snapshot_before or not snapshot_after:
        raise ValueError("publish audit predates snapshot rollback support")

    catalog = FilesystemCatalog(root)
    current_snapshot = catalog.snapshot()
    if current_snapshot != snapshot_after:
        raise ValueError(
            "batch is not the current Catalog state: "
            f"expected {snapshot_after}, found {current_snapshot}"
        )
    result = audit.get("result") or {}
    concept_refs = sorted(
        {
            *result.get("created", []),
            *result.get("updated", []),
        }
    )
    if concept_refs:
        raise ValueError(
            "Concept rollback requires git revert because the Catalog stores "
            "one current file per Concept"
        )
    observation_ids = set(result.get("observations_added", []))
    observation_edits = _observation_edits(
        layout.observations,
        observation_ids,
    )
    report = {
        "status": "planned",
        "batch_id": batch_id,
        "snapshot_before": snapshot_before,
        "snapshot_after": snapshot_after,
        "concepts_changed": concept_refs,
        "observations_removed": sorted(observation_ids),
    }
    if not apply:
        return report

    backups = {
        path: path.read_bytes() if path.is_file() else None
        for path in observation_edits
    }
    try:
        for path, content in observation_edits.items():
            if content:
                atomic_text(path, content)
            else:
                path.unlink()
        _rebuild_index(root)
        restored_snapshot = catalog.snapshot()
        if restored_snapshot != snapshot_before:
            raise ValueError(
                "rollback did not restore audited snapshot: "
                f"expected {snapshot_before}, found {restored_snapshot}"
            )
    except BaseException:
        for path, content in backups.items():
            if content is None:
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        _rebuild_index(root)
        raise

    report["status"] = "rolled_back"
    rollback_path = root / "publish" / "rollback" / f"{safe_name(batch_id)}.json"
    atomic_text(
        rollback_path,
        json.dumps(
            {
                **report,
                "rolled_back_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
    )
    return report


def _same_target_family(
    expected: TargetScope,
    actual: TargetContext,
) -> bool:
    if expected.level == "portable":
        return True
    if expected.backend and expected.backend.casefold() != actual.backend.casefold():
        return False
    if expected.level in {"architecture", "exact"} and expected.architecture:
        if expected.architecture.casefold() != actual.architecture.casefold():
            return False
    if expected.level in {"device", "exact"} and expected.devices:
        actual_device = normalize_device(actual.device).casefold()
        accepted = {
            normalize_device(item).casefold()
            for item in expected.devices
        }
        if actual_device not in accepted:
            return False
    return True


def _same_software(
    expected: TargetScope,
    actual: TargetContext,
) -> bool:
    for field in ("language", "compiler", "runtime", "library"):
        required = getattr(expected.software, field)
        observed = getattr(actual.software, field)
        if required and (
            not observed or required.casefold() != observed.casefold()
        ):
            return False
    return True


def _observation_edits(
    observation_root: Path,
    observation_ids: set[str],
) -> dict[Path, str]:
    if not observation_ids:
        return {}
    found = set()
    edits = {}
    for path in sorted(observation_root.glob("*.jsonl")):
        kept = []
        changed = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            observation_id = str(payload.get("id") or "")
            if observation_id in observation_ids:
                found.add(observation_id)
                changed = True
            else:
                kept.append(line)
        if changed:
            edits[path] = "".join(item + "\n" for item in kept)
    missing = observation_ids - found
    if missing:
        raise ValueError(
            "audited observations are missing: " + ", ".join(sorted(missing))
        )
    return edits


def _rebuild_index(root: Path) -> None:
    catalog = FilesystemCatalog(root)
    layout = CatalogLayout(root)
    SQLiteKnowledgeIndex(layout.index).rebuild(
        catalog.iter_concepts(),
        catalog.snapshot(),
        observations=catalog.iter_observations(),
        retrievals=catalog.iter_retrieval_events(),
        usage_snapshot=catalog.usage_snapshot(),
    )


def _write_audit(root: Path, action: str, report: dict) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / "publish" / "maintenance" / f"{action}-{safe_name(report['concept_id'])}-{stamp}.json"
    atomic_text(
        path,
        json.dumps(
            {
                "schema_version": "1.0",
                "action": action,
                "changed_at": datetime.now(timezone.utc).isoformat(),
                **report,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
    )
    return str(path.relative_to(root))
