"""Build deterministic source-backed Candidates from a reviewed ingestion plan."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.publishing.publisher import BatchPublisher
from kernelgen.knowledge.models import (
    CandidateConcept,
    SourcePackage,
    SourceReference,
    StaticIngestionPlan,
    StaticKnowledgeEntry,
)
from kernelgen.knowledge.contracts.runtime import PublishResult
from kernelgen.knowledge.layout import safe_name


_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


class StaticKnowledgeImporter:
    """Load one registered source package and its reviewed import plan."""

    def __init__(
        self,
        catalog_root: Path,
        plan: StaticIngestionPlan,
    ):
        self.catalog_root = Path(catalog_root)
        self.catalog = FilesystemCatalog(self.catalog_root)
        self.plan = plan
        self.package = self._find_package(plan.source_package)
        self.content_root = self._resolve_content_root(self.package)

    @classmethod
    def from_plan_file(
        cls,
        catalog_root: Path,
        plan_path: Path,
    ) -> "StaticKnowledgeImporter":
        payload = yaml.safe_load(Path(plan_path).read_text(encoding="utf-8"))
        return cls(
            catalog_root,
            StaticIngestionPlan.model_validate(payload),
        )

    def build_candidates(self) -> list[CandidateConcept]:
        return [
            self._build_candidate(entry)
            for entry in self.plan.entries
        ]

    def default_batch_id(self) -> str:
        return (
            f"static-{safe_name(self.package.id)}-"
            f"{safe_name(self.package.revision)}"
        )

    def publish(
        self,
        *,
        batch_id: str,
    ) -> PublishResult:
        candidates = self.build_candidates()
        return BatchPublisher(
            self.catalog_root,
            fact_reader=None,
        ).publish_static(
            candidates,
            batch_id=batch_id,
        )

    def _find_package(self, package_id: str) -> SourcePackage:
        for package in self.catalog.iter_source_packages():
            if package.id == package_id:
                return package
        raise ValueError(f"source package not found: {package_id}")

    def _resolve_content_root(self, package: SourcePackage) -> Path:
        prefix = "kb://"
        if not package.content_root.startswith(prefix):
            raise ValueError(
                "direct/extract ingestion requires a kb:// source snapshot"
            )
        relative = Path(package.content_root.removeprefix(prefix))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("source content_root escapes the KB")
        resolved = (self.catalog_root / relative).resolve()
        try:
            resolved.relative_to(self.catalog_root.resolve())
        except ValueError as exc:
            raise ValueError("source content_root escapes the KB") from exc
        return resolved

    def _build_candidate(
        self,
        entry: StaticKnowledgeEntry,
    ) -> CandidateConcept:
        source_path = self._resolve_source_file(entry.source_path)
        source_section = _read_markdown_section(source_path, entry.heading)
        if entry.mode == "direct":
            body = (
                f"# Claim\n\n{entry.summary}\n\n"
                "# Source material\n\n"
                f"{source_section.strip()}\n\n"
                "# Applicability\n\n"
                "The structured scope on this Concept is normative."
            )
        else:
            body = entry.body.strip()

        locator = entry.source_path
        if entry.heading:
            locator += f"#{entry.heading}"
        creator = (
            f"source-ingest:{self.package.id.removeprefix('source:')}:"
            f"{entry.mode}"
        )
        return CandidateConcept(
            candidate_id=(
                f"static:{safe_name(self.package.id)}:"
                f"{safe_name(self.package.revision)}:{safe_name(entry.claim_key)}"
            ),
            proposed_kind=entry.proposed_kind,
            proposed_id=entry.proposed_id,
            claim_key=entry.claim_key,
            title=entry.title,
            summary=entry.summary,
            domains=entry.domains,
            scope=entry.scope,
            retrieval=entry.retrieval,
            body=body,
            relations=entry.relations,
            source_refs=[
                SourceReference(
                    resource=self.package.id,
                    revision=self.package.revision,
                    locator=locator,
                    title=entry.title,
                )
            ],
            created_by=creator,
        )

    def _resolve_source_file(self, source_path: str) -> Path:
        path = (self.content_root / source_path).resolve()
        try:
            path.relative_to(self.content_root.resolve())
        except ValueError as exc:
            raise ValueError(f"source path escapes package: {source_path}") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source file is missing or unsafe: {source_path}")
        return path


def _read_markdown_section(path: Path, heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    if not heading:
        return text.strip()

    lines = text.splitlines()
    headings = _markdown_headings(lines)
    matches = [
        (index, level)
        for index, level, title in headings
        if title == heading
    ]
    if not matches:
        raise ValueError(f"heading not found in {path}: {heading}")
    if len(matches) > 1:
        raise ValueError(f"heading is ambiguous in {path}: {heading}")

    start, level = matches[0]
    end = len(lines)
    for index, candidate_level, _ in headings:
        if index > start and candidate_level <= level:
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _markdown_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return ATX headings while ignoring heading-like lines in code fences."""

    headings = []
    fence_character = ""
    fence_length = 0
    for index, line in enumerate(lines):
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if not fence_character:
                fence_character = marker[0]
                fence_length = len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character = ""
                fence_length = 0
            continue
        if fence_character:
            continue
        match = _HEADING.match(line)
        if match:
            headings.append(
                (index, len(match.group(1)), match.group(2).strip())
            )
    return headings
