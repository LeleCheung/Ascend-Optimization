"""Filesystem-backed Concept catalog."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import yaml

from kernelgen.knowledge.layout import CatalogLayout
from kernelgen.knowledge.models import (
    Concept,
    KnowledgeUsageScope,
    ObservationRecord,
    OperatorSignature,
    SourcePackage,
    TargetContext,
)
from kernelgen.knowledge.validation import iter_jsonl, parse_concept


def concept_ref(concept: Concept) -> str:
    return concept.id


def compute_snapshot(
    concepts: Iterable[Concept],
    sources: Iterable[SourcePackage] = (),
    observations: Iterable[ObservationRecord] = (),
) -> str:
    identities = [
        f"concept:{concept_ref(item)}:{item.managed.content_hash}"
        for item in concepts
    ]
    identities.extend(
        f"source:{item.id}:{item.revision}:{item.checksum}"
        for item in sources
    )
    identities.extend(f"observation:{item.id}" for item in observations)
    identities.sort()
    payload = json.dumps(identities, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FilesystemCatalog:
    def __init__(self, root: Path):
        self.layout = CatalogLayout(Path(root))

    def iter_concepts(self) -> list[Concept]:
        concepts = []
        seen = set()
        if not self.layout.concepts.exists():
            return concepts
        for path in sorted(self.layout.concepts.glob("**/*.md")):
            if path.name in {"index.md", "log.md"}:
                continue
            concept = parse_concept(path)
            if concept.id in seen:
                raise ValueError(f"duplicate concept id: {concept.id}")
            seen.add(concept.id)
            concepts.append(concept)
        return sorted(concepts, key=lambda item: item.id)

    def snapshot(self) -> str:
        return compute_snapshot(
            self.iter_concepts(),
            self.iter_source_packages(),
            self.iter_observations(),
        )

    def iter_observations(self) -> list[ObservationRecord]:
        observations = []
        seen = set()
        root = self.layout.root / "observations"
        if not root.exists():
            return observations
        for path in sorted(root.glob("**/*.jsonl")):
            for _, raw in iter_jsonl(path):
                observation = ObservationRecord.model_validate(raw)
                if observation.id in seen:
                    raise ValueError(
                        f"duplicate observation id: {observation.id}"
                    )
                seen.add(observation.id)
                observations.append(observation)
        return observations

    def iter_retrieval_events(self) -> list[dict]:
        """Read successful detail fetches from each workspace's latest archive."""

        archive_root = self.layout.root.resolve().parent / "run-archive"
        latest = {}
        if not archive_root.exists():
            return []
        for manifest_path in sorted(
            archive_root.glob("*/*/snapshots/*/manifest.json")
        ):
            try:
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, TypeError):
                continue
            key = (
                str(manifest.get("run_id") or ""),
                str(manifest.get("workspace_id") or ""),
            )
            rank = str(manifest.get("archived_at") or "")
            if key not in latest or rank > latest[key][0]:
                latest[key] = (rank, manifest_path.parent)

        facts = {}
        for (run_id, workspace_id), (_, snapshot_root) in sorted(
            latest.items()
        ):
            log = snapshot_root / "knowledge" / "retrieval-log.jsonl"
            if not log.is_file():
                continue
            records = [raw for _, raw in iter_jsonl(log)]
            scopes = {}
            for raw in records:
                if raw.get("status") != "success":
                    continue
                scope = _retrieval_scope(raw)
                query_id = str(
                    raw.get("query_id") or raw.get("event_id") or ""
                )
                if scope is not None and query_id:
                    scopes[query_id] = scope
            for raw in records:
                if (
                    raw.get("status") != "success"
                    or raw.get("operation")
                    not in {"get_knowledge", "get_source"}
                ):
                    continue
                event_id = str(raw.get("event_id") or "")
                scope = scopes.get(str(raw.get("query_id") or ""))
                if not event_id or scope is None:
                    continue
                references = (
                    raw.get("returned_refs")
                    if raw.get("operation") == "get_knowledge"
                    else raw.get("returned_sources")
                ) or []
                for reference in references:
                    fact = {
                        "event_id": event_id,
                        "reference": str(reference),
                        "run_id": run_id,
                        "workspace_id": workspace_id,
                        "scope": scope.model_dump(mode="json"),
                    }
                    facts[(event_id, str(reference))] = fact
        return [facts[key] for key in sorted(facts)]

    def usage_snapshot(self) -> str:
        identities = [
            f"observation:{item.id}" for item in self.iter_observations()
        ]
        identities.extend(
            f"retrieval:{item['event_id']}:{item['reference']}"
            for item in self.iter_retrieval_events()
        )
        encoded = json.dumps(
            sorted(identities),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def iter_source_packages(self) -> list[SourcePackage]:
        packages = []
        root = self.layout.source_packages
        if not root.exists():
            return packages
        for path in sorted(root.glob("*.yaml")):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            packages.append(SourcePackage.model_validate(payload))
        return packages

    def get_concept(self, requested_ref: str) -> Concept:
        for concept in self.iter_concepts():
            if concept.id == requested_ref:
                return concept
        raise KeyError(f"concept ref not found: {requested_ref}")

    def find_by_claim(self, kind: str, claim_key: str) -> Concept | None:
        for concept in self.iter_concepts():
            if concept.kind == kind and concept.claim_key == claim_key:
                return concept
        return None

    def write_concept(self, concept: Concept) -> Path:
        destination = self.concept_path(concept)
        payload = _compact_optional(concept.model_dump(mode="json"))
        body = payload.pop("body")
        rendered = (
            "---\n"
            + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
            + "---\n\n"
            + body.rstrip()
            + "\n"
        )
        if (
            destination.is_file()
            and destination.read_text(encoding="utf-8") == rendered
        ):
            return destination
        atomic_text(destination, rendered)
        return destination

    def concept_path(self, concept: Concept) -> Path:
        slug = concept.id.rsplit(":", 1)[-1]
        return self.layout.concepts / f"{concept.kind}--{slug}.md"


def _compact_optional(value):
    """Omit only absent/empty optional metadata; preserve false and zero."""
    if isinstance(value, dict):
        compacted = {
            key: _compact_optional(item)
            for key, item in value.items()
        }
        return {
            key: item
            for key, item in compacted.items()
            if not _is_empty_optional(item)
        }
    if isinstance(value, list):
        return [_compact_optional(item) for item in value]
    return value


def _is_empty_optional(value) -> bool:
    return (
        value is None
        or (isinstance(value, str) and value == "")
        or (isinstance(value, (list, dict)) and not value)
    )


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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


def _retrieval_scope(raw: dict) -> KnowledgeUsageScope | None:
    request = raw.get("request")
    if not isinstance(request, dict):
        return None
    if raw.get("operation") == "query_sources":
        try:
            return KnowledgeUsageScope.model_validate(
                request.get("usage_scope")
            )
        except (TypeError, ValueError):
            return None
    if raw.get("operation") != "query_knowledge":
        return None
    try:
        return KnowledgeUsageScope.from_context(
            OperatorSignature.model_validate(
                request.get("operator_signature")
            ),
            TargetContext.model_validate(request.get("target_context")),
        )
    except (TypeError, ValueError):
        return None


def catalog_changed_paths(catalog_root: Path) -> set[str]:
    """Return tracked and untracked Catalog paths changed from HEAD."""

    root = Path(catalog_root)
    if not (root / ".git").is_dir():
        return set()
    try:
        resolved = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if Path(resolved).resolve() != root.resolve():
            return set()
        tracked = subprocess.run(
            ["git", "diff", "--name-only", "-z", "HEAD", "--"],
            cwd=str(root),
            capture_output=True,
            check=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=str(root),
            capture_output=True,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()
    return {
        os.fsdecode(item)
        for item in (tracked + untracked).split(b"\0")
        if item
    }


def _catalog_relative_path(root: Path, item: str | Path) -> str:
    path = Path(item)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"Catalog path is outside repository: {item}") from exc
    if not path.parts or ".." in path.parts:
        raise ValueError(f"invalid Catalog path: {item}")
    return path.as_posix()


def assert_catalog_paths_clean(
    catalog_root: Path,
    paths: Iterable[str | Path],
    *,
    changed_paths: Iterable[str | Path] | None = None,
) -> None:
    """Reject a transaction that would overwrite a pre-existing dirty path."""

    root = Path(catalog_root)
    dirty_items = (
        catalog_changed_paths(root)
        if changed_paths is None
        else changed_paths
    )
    dirty = {
        _catalog_relative_path(root, item)
        for item in dirty_items
    }
    requested = {
        _catalog_relative_path(root, item)
        for item in paths
    }
    conflicts = sorted(dirty.intersection(requested))
    if conflicts:
        raise RuntimeError(
            "Catalog transaction targets pre-existing dirty paths: "
            + ", ".join(conflicts)
        )


def commit_catalog(
    catalog_root: Path,
    message: str,
    *,
    paths: Iterable[str | Path] | None = None,
) -> bool:
    """Commit a standalone Catalog repository when it has changes."""

    root = Path(catalog_root)
    if not (root / ".git").is_dir():
        return False
    try:
        resolved = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if Path(resolved).resolve() != root.resolve():
            return False
        changed = catalog_changed_paths(root)
        requested = changed if paths is None else {
            _catalog_relative_path(root, item) for item in paths
        }
        selected = sorted(changed.intersection(requested))
        if not selected:
            return False
        subprocess.run(
            ["git", "add", "-A", "--", *selected],
            cwd=str(root),
            capture_output=True,
            check=True,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", *selected],
            cwd=str(root),
            capture_output=True,
        )
        if staged.returncode == 0:
            return False
        if staged.returncode != 1:
            raise subprocess.CalledProcessError(
                staged.returncode,
                staged.args,
                output=staged.stdout,
                stderr=staged.stderr,
            )
        environment = {
            **os.environ,
            "GIT_AUTHOR_NAME": "kb-publisher",
            "GIT_AUTHOR_EMAIL": "kb@publisher",
            "GIT_COMMITTER_NAME": "kb-publisher",
            "GIT_COMMITTER_EMAIL": "kb@publisher",
        }
        subprocess.run(
            ["git", "commit", "--only", "-m", message, "--", *selected],
            cwd=str(root),
            capture_output=True,
            env=environment,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
