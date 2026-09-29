"""Filesystem validation for KernelGen KB V1."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from pydantic import ValidationError

from kernelgen.knowledge.models import (
    Concept,
    GitSourceManifest,
    ObservationRecord,
    SourcePackage,
    StaticIngestionPlan,
)
from kernelgen.knowledge.publishing.validation import (
    candidate_round_references,
    normalize_candidate_vocabulary,
    validate_runtime_candidate,
    validate_static_candidate,
)
from kernelgen.knowledge.solutions import SolutionManifest
from kernelgen.knowledge.source_rules import source_path_matches
from kernelgen.knowledge.taxonomy import (
    OperatorTaxonomy,
    OperatorTaxonomyError,
)
from kernelgen.knowledge.vocabulary import Vocabulary


class KnowledgeValidationError(ValueError):
    """Raised when one knowledge artifact is invalid."""


@dataclass
class ValidationIssue:
    path: str
    message: str


@dataclass
class ValidationReport:
    concepts: int = 0
    observations: int = 0
    source_packages: int = 0
    source_entries: int = 0
    knowledge_documents: int = 0
    ingestion_plans: int = 0
    solutions: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "concepts": self.concepts,
            "observations": self.observations,
            "source_packages": self.source_packages,
            "source_entries": self.source_entries,
            "knowledge_documents": self.knowledge_documents,
            "ingestion_plans": self.ingestion_plans,
            "solutions": self.solutions,
            "issues": [
                {"path": issue.path, "message": issue.message}
                for issue in self.issues
            ],
        }


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise KnowledgeValidationError(f"unreadable YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise KnowledgeValidationError("YAML root must be a mapping")
    return value


def parse_concept(path: Path) -> Concept:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise KnowledgeValidationError(f"unreadable concept: {exc}") from exc
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise KnowledgeValidationError("concept must start with YAML frontmatter")
    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise KnowledgeValidationError("concept frontmatter is not closed") from exc
    try:
        frontmatter = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        raise KnowledgeValidationError(f"invalid frontmatter YAML: {exc}") from exc
    if not isinstance(frontmatter, dict):
        raise KnowledgeValidationError("concept frontmatter must be a mapping")
    body = "\n".join(lines[closing + 1 :]).strip()
    payload = {**frontmatter, "body": body}
    try:
        return Concept.model_validate(payload)
    except ValidationError as exc:
        raise KnowledgeValidationError(str(exc)) from exc


def canonical_concept_hash(concept: Concept) -> str:
    payload = concept.model_dump(mode="json")
    payload.pop("managed", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def iter_jsonl(path: Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise KnowledgeValidationError(f"unreadable JSONL: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise KnowledgeValidationError(
                f"invalid JSON on line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise KnowledgeValidationError(
                f"JSONL line {line_number} must be an object"
            )
        yield line_number, value


def validate_knowledge_base(kb_root: Path) -> ValidationReport:
    kb_root = Path(kb_root)
    report = ValidationReport()
    taxonomy = None
    try:
        taxonomy = OperatorTaxonomy.from_catalog(kb_root)
    except OperatorTaxonomyError as exc:
        report.issues.append(
            ValidationIssue(
                str(kb_root / "operator_taxonomy.yaml"),
                f"invalid operator taxonomy: {exc}",
            )
        )
    try:
        Vocabulary.from_catalog(kb_root)
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        report.issues.append(
            ValidationIssue(
                str(kb_root / "aliases.yaml"),
                f"invalid vocabulary: {exc}",
            )
        )
    concepts: Dict[str, tuple[Concept, Path]] = {}
    observation_ids: Dict[str, Path] = {}
    source_ids: Dict[str, tuple[SourcePackage, Path]] = {}

    for path in sorted((kb_root / "concepts").glob("**/*.md")):
        if path.name in {"index.md", "log.md"}:
            continue
        try:
            concept = parse_concept(path)
            if concept.id in concepts:
                other = concepts[concept.id][1]
                raise KnowledgeValidationError(
                    f"duplicate concept id {concept.id}; first defined in {other}"
                )
            expected_name = (
                f"{concept.kind}--{concept.id.rsplit(':', 1)[-1]}.md"
            )
            if path.name != expected_name:
                raise KnowledgeValidationError(
                    f"concept filename must be {expected_name}"
                )
            if taxonomy is not None:
                taxonomy.validate_operator_scope(
                    concept.scope.operator, label=concept.id
                )
            concepts[concept.id] = (concept, path)
            report.concepts += 1
        except (KnowledgeValidationError, OperatorTaxonomyError) as exc:
            report.issues.append(ValidationIssue(str(path), str(exc)))

    observation_root = kb_root / "observations"
    if observation_root.exists():
        for path in sorted(observation_root.glob("**/*.jsonl")):
            try:
                for line_number, raw in iter_jsonl(path):
                    try:
                        observation = ObservationRecord.model_validate(raw)
                    except ValidationError as exc:
                        raise KnowledgeValidationError(
                            f"line {line_number}: {exc}"
                        ) from exc
                    if observation.id in observation_ids:
                        raise KnowledgeValidationError(
                            f"duplicate observation id {observation.id}; "
                            "first defined in "
                            f"{observation_ids[observation.id]}"
                        )
                    observation_ids[observation.id] = path
                    report.observations += 1
            except KnowledgeValidationError as exc:
                report.issues.append(ValidationIssue(str(path), str(exc)))

    source_root = kb_root / "sources" / "packages"
    if source_root.exists():
        for path in sorted(source_root.glob("*.yaml")):
            try:
                source = SourcePackage.model_validate(_read_yaml(path))
                if source.id in source_ids:
                    raise KnowledgeValidationError(
                        f"duplicate source id; first defined in {source_ids[source.id][1]}"
                    )
                source_ids[source.id] = (source, path)
                _validate_source_content(kb_root, source)
                if source.search_mode == "searchable":
                    report.source_entries += _validate_source_manifest(
                        kb_root, source
                    )
                report.source_packages += 1
            except (KnowledgeValidationError, ValidationError) as exc:
                report.issues.append(ValidationIssue(str(path), str(exc)))

    ingestion_root = kb_root / "sources" / "ingestion"
    if ingestion_root.exists():
        for path in sorted(ingestion_root.glob("*.yaml")):
            try:
                plan = StaticIngestionPlan.model_validate(_read_yaml(path))
                if plan.source_package not in source_ids:
                    raise KnowledgeValidationError(
                        f"source package does not exist: {plan.source_package}"
                    )
                if taxonomy is not None:
                    for entry in plan.entries:
                        taxonomy.validate_operator_scope(
                            entry.scope.operator,
                            label=f"{path.name}:{entry.proposed_id}",
                        )
                source = source_ids[plan.source_package][0]
                if source.search_mode == "searchable":
                    report.knowledge_documents += _validate_plan_coverage(
                        kb_root, source, plan
                    )
                report.ingestion_plans += 1
            except (
                KnowledgeValidationError,
                OperatorTaxonomyError,
                ValidationError,
            ) as exc:
                report.issues.append(ValidationIssue(str(path), str(exc)))

    solution_root = kb_root / "solutions"
    if solution_root.exists():
        for path in sorted(solution_root.glob("**/manifest.json")):
            try:
                manifest = SolutionManifest.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
                code_path = path.with_name("best_kernel.py")
                code_path.read_text(encoding="utf-8")
                report.solutions += 1
            except (
                KnowledgeValidationError,
                OSError,
                UnicodeError,
                ValidationError,
            ) as exc:
                report.issues.append(ValidationIssue(str(path), str(exc)))

    concept_ids = {
        concept.id
        for concept, _ in concepts.values()
    }
    for concept, path in concepts.values():
        for relation in concept.relations:
            if relation.target not in concept_ids:
                report.issues.append(
                    ValidationIssue(
                        str(path),
                        f"relation target does not exist: {relation.target}",
                    )
                )
        for evidence in concept.evidence:
            if evidence.observation_ref not in observation_ids:
                report.issues.append(
                    ValidationIssue(
                        str(path),
                        "observation_ref does not exist: "
                        f"{evidence.observation_ref}",
                    )
                )
        for source in concept.sources:
            if source.resource.startswith("source:") and source.resource not in source_ids:
                report.issues.append(
                    ValidationIssue(
                        str(path),
                        f"source package does not exist: {source.resource}",
                    )
                )
                continue
            package_entry = source_ids.get(source.resource)
            if package_entry is None:
                continue
            package = package_entry[0]
            if package.allowed_backends:
                target = concept.scope.target
                if (
                    target.level == "portable"
                    or not target.backend
                    or target.backend not in package.allowed_backends
                ):
                    report.issues.append(
                        ValidationIssue(
                            str(path),
                            f"source {package.id} only permits backends "
                            f"{package.allowed_backends}, but Concept scope is "
                            f"level={target.level}, backend={target.backend!r}",
                        )
                    )

    return report


def _validate_source_content(kb_root: Path, source: SourcePackage) -> None:
    prefix = "kb://"
    if not source.content_root.startswith(prefix):
        return
    relative = Path(source.content_root.removeprefix(prefix))
    if relative.is_absolute() or ".." in relative.parts:
        raise KnowledgeValidationError(
            f"source content_root escapes KB: {source.content_root}"
        )
    content = (kb_root / relative).resolve()
    try:
        content.relative_to(kb_root.resolve())
    except ValueError as exc:
        raise KnowledgeValidationError(
            f"source content_root escapes KB: {source.content_root}"
        ) from exc
    if not content.exists():
        raise KnowledgeValidationError(
            f"source content is missing: {source.content_root}"
        )


def _resolve_kb_uri(kb_root: Path, value: str, field_name: str) -> Path:
    prefix = "kb://"
    if not value.startswith(prefix):
        raise KnowledgeValidationError(f"{field_name} must use kb://")
    relative = Path(value.removeprefix(prefix))
    if relative.is_absolute() or ".." in relative.parts:
        raise KnowledgeValidationError(f"{field_name} escapes KB: {value}")
    resolved = (kb_root / relative).resolve()
    try:
        resolved.relative_to(kb_root.resolve())
    except ValueError as exc:
        raise KnowledgeValidationError(f"{field_name} escapes KB: {value}") from exc
    return resolved


def _load_source_manifest(kb_root: Path, source: SourcePackage) -> GitSourceManifest:
    path = _resolve_kb_uri(kb_root, source.manifest_root, "manifest_root")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = GitSourceManifest.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
        raise KnowledgeValidationError(f"invalid Git source manifest: {exc}") from exc
    if manifest.source_package != source.id:
        raise KnowledgeValidationError(
            "Git source manifest source_package does not match package"
        )
    if manifest.revision != source.revision:
        raise KnowledgeValidationError(
            "Git source manifest revision does not match package"
        )
    return manifest


def _validate_source_manifest(kb_root: Path, source: SourcePackage) -> int:
    manifest = _load_source_manifest(kb_root, source)
    content = _resolve_kb_uri(kb_root, source.content_root, "content_root")
    expected = {
        entry.path: entry
        for entry in manifest.entries
        if entry.object_type == "blob"
    }
    for relative, entry in expected.items():
        path = content / relative
        if path.is_symlink():
            target = os.readlink(path)
            if Path(target).is_absolute():
                raise KnowledgeValidationError(
                    f"source symlink is absolute: {relative}"
                )
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(content.resolve())
            except ValueError as exc:
                raise KnowledgeValidationError(
                    f"source symlink escapes package: {relative}"
                ) from exc
    return len(manifest.entries)


def _validate_plan_coverage(
    kb_root: Path,
    source: SourcePackage,
    plan: StaticIngestionPlan,
) -> int:
    manifest = _load_source_manifest(kb_root, source)
    classifications = {
        entry.path: [
            rule.mode
            for rule in plan.rules
            if source_path_matches(entry.path, rule.include)
        ]
        for entry in manifest.entries
    }
    uncovered = [
        path
        for path, modes in classifications.items()
        if not modes
    ]
    if uncovered:
        raise KnowledgeValidationError(
            "ingestion plan leaves Git entries unclassified: "
            + ", ".join(uncovered[:10])
        )
    return sum(
        modes[-1] in {"indexed", "direct", "extract"}
        for modes in classifications.values()
    )


def source_content_checksum(content: Path) -> str:
    """Hash one source file or a deterministic directory tree.

    File hashing remains byte-for-byte compatible with V1 source packages.
    Directory hashing includes every relative path and file digest in sorted
    order. Relative symlinks are hashed by link target and must stay within the
    source root.
    """

    content = Path(content)
    if content.is_symlink():
        raise KnowledgeValidationError("source content root cannot be a symlink")
    if content.is_file():
        return "sha256:" + hashlib.sha256(content.read_bytes()).hexdigest()
    if not content.is_dir():
        raise KnowledgeValidationError(f"source content is not a file or directory: {content}")

    digest = hashlib.sha256()
    for path in sorted(content.rglob("*")):
        if path.is_symlink():
            relative = path.relative_to(content).as_posix()
            target = os.readlink(path)
            if Path(target).is_absolute():
                raise KnowledgeValidationError(
                    f"source symlink is absolute: {relative}"
                )
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(content.resolve())
            except ValueError as exc:
                raise KnowledgeValidationError(
                    f"source symlink escapes package: {relative}"
                ) from exc
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(b"symlink:")
            digest.update(target.encode("utf-8"))
            digest.update(b"\n")
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(content).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()
