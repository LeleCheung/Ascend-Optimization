"""One-time migration from versioned Concept files to one current file."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import tempfile
from pathlib import Path

import yaml

from kernelgen.knowledge.catalog import (
    FilesystemCatalog,
)
from kernelgen.knowledge.index import SQLiteKnowledgeIndex
from kernelgen.knowledge.layout import CatalogLayout
from kernelgen.knowledge.validation import (
    canonical_concept_hash,
    parse_concept,
    validate_knowledge_base,
)


_VERSIONED_NAME = re.compile(r"@([1-9][0-9]*)\.md$")


def migrate_single_concept_files(kb: Path, *, apply: bool = False) -> dict:
    root = Path(kb)
    layout = CatalogLayout(root)
    layout.publish_lock.parent.mkdir(parents=True, exist_ok=True)
    with layout.publish_lock.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _migrate_locked(root, apply=apply)


def _migrate_locked(root: Path, *, apply: bool) -> dict:
    layout = CatalogLayout(root)
    catalog = FilesystemCatalog(root)
    source_paths = sorted(
        path
        for path in catalog.layout.concepts.glob("**/*.md")
        if path.name not in {"index.md", "log.md"}
    )
    grouped: dict[str, list[tuple[int, Path, object]]] = {}
    for path in source_paths:
        concept = parse_concept(path)
        grouped.setdefault(concept.id, []).append(
            (_legacy_revision(path), path, concept)
        )

    selected = []
    for concept_id, entries in sorted(grouped.items()):
        highest = max(item[0] for item in entries)
        choices = [item for item in entries if item[0] == highest]
        if len(choices) > 1:
            payloads = {
                item[2].model_dump_json(exclude={"managed"})
                for item in choices
            }
            if len(payloads) != 1:
                paths = ", ".join(str(item[1]) for item in choices)
                raise ValueError(
                    f"ambiguous latest Concept files for {concept_id}: {paths}"
                )
        chosen = sorted(choices, key=lambda item: str(item[1]))[-1]
        concept = chosen[2]
        concept = concept.model_copy(
            update={
                "managed": concept.managed.model_copy(
                    update={"content_hash": canonical_concept_hash(concept)}
                )
            }
        )
        selected.append((chosen[1], concept))

    targets = {catalog.concept_path(concept) for _, concept in selected}
    obsolete = sorted(set(source_paths) - targets)
    report = {
        "status": "planned",
        "concepts": len(selected),
        "source_files": len(source_paths),
        "target_files": len(targets),
        "obsolete_files": len(obsolete),
        "source_revisions_unchanged": True,
    }
    if not apply:
        return report

    affected = sorted(set(source_paths) | targets)
    backups = {
        path: path.read_bytes() if path.is_file() else None
        for path in affected
    }
    index_path = layout.index
    index_backup = (
        index_path.read_bytes() if index_path.is_file() else None
    )
    try:
        for _, concept in selected:
            catalog.write_concept(concept)
        for path in obsolete:
            path.unlink()
        validation = validate_knowledge_base(root)
        if not validation.valid:
            details = "; ".join(
                f"{item.path}: {item.message}"
                for item in validation.issues[:10]
            )
            raise ValueError(f"migrated knowledge base is invalid: {details}")
        snapshot = catalog.snapshot()
        SQLiteKnowledgeIndex(index_path).rebuild(
            catalog.iter_concepts(),
            snapshot,
            observations=catalog.iter_observations(),
            retrievals=catalog.iter_retrieval_events(),
            usage_snapshot=catalog.usage_snapshot(),
        )
    except BaseException:
        _restore_files(backups)
        _restore_files({index_path: index_backup})
        raise

    report.update(
        {
            "status": "migrated",
            "snapshot": snapshot,
            "valid": True,
        }
    )
    return report


def _legacy_revision(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        closing = lines.index("---", 1)
        frontmatter = yaml.safe_load("\n".join(lines[1:closing]))
    except (ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read Concept revision from {path}") from exc
    revision = (
        frontmatter.get("revision")
        if isinstance(frontmatter, dict)
        else None
    )
    if isinstance(revision, int) and revision >= 1:
        return revision
    match = _VERSIONED_NAME.search(path.name)
    return int(match.group(1)) if match else 0


def _restore_files(backups: dict[Path, bytes | None]) -> None:
    for path, content in backups.items():
        if content is None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        else:
            _atomic_bytes(path, content)


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", type=Path, default=Path("kb"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = migrate_single_concept_files(args.kb, apply=args.apply)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
