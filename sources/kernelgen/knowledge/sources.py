"""Bounded lexical search over immutable local SourcePackage snapshots."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.fts import chinese_sequences, english_tokens
from kernelgen.knowledge.layout import CatalogLayout
from kernelgen.knowledge.ranking import (
    usage_reasons,
    usage_score,
)
from kernelgen.knowledge.records import retrieval_record
from kernelgen.knowledge.contracts.runtime import (
    SourceDocument,
    SourceSearchHit,
    SourceSearchResult,
)
from kernelgen.knowledge.models import (
    GitSourceManifest,
    KnowledgeUsageScope,
    SourcePackage,
)
from kernelgen.knowledge.source_index import (
    MAX_FILE_BYTES,
    TEXT_SUFFIXES,
    SQLiteSourceIndex,
    SourceIndexMatch,
)
from kernelgen.knowledge.vocabulary import Vocabulary


_SNIPPET_CONTEXT_LINES = 3
_PROMOTED_CONCEPT_WEIGHT = 10.0
_TARGET_BACKEND_MATCH_WEIGHT = 8.0
_CROSS_BACKEND_PENALTY = 4.0
_PATH_TOKEN_WEIGHT = 2.0
_PATH_TOKEN_BOOST_CAP = 6.0
_PATH_TOKEN_EXCLUSIONS = {
    "ascend",
    "documentation",
    "kernel",
    "language",
    "python",
    "source",
    "triton",
}
_LINE_LOCATOR = re.compile(r"^(?P<path>.+):L(?P<start>[1-9][0-9]*)-L(?P<end>[1-9][0-9]*)$")


class SourceSearcher:
    """Search indexed source material without converting it into a Concept."""

    def __init__(
        self,
        catalog_root: Path,
        retrieval_audit=None,
        *,
        usage_index=None,
        usage_scope: KnowledgeUsageScope | None = None,
        source_index_path: Path | None = None,
        allow_index_rebuild: bool = True,
    ):
        self.catalog_root = Path(catalog_root).resolve()
        self.catalog = FilesystemCatalog(self.catalog_root)
        self.retrieval_audit = retrieval_audit
        self.usage_index = usage_index
        self.usage_scope = usage_scope
        self.allow_index_rebuild = allow_index_rebuild
        self.vocabulary = Vocabulary.from_catalog(self.catalog_root)
        self.source_index = SQLiteSourceIndex(
            source_index_path or CatalogLayout(self.catalog_root).source_index,
            self.catalog_root,
            read_only=not allow_index_rebuild,
        )

    def search(
        self,
        query: str,
        *,
        package_ids: list[str] | None = None,
        path_globs: list[str] | None = None,
        include_source_only: bool = False,
        max_results: int = 12,
        origin: str = "application",
    ) -> SourceSearchResult:
        query_id = f"source-query:{uuid4().hex}"
        request = {
            "query": query,
            "package_ids": sorted(package_ids or []),
            "path_globs": path_globs or ["*"],
            "include_source_only": include_source_only,
            "max_results": max_results,
        }
        if self.usage_scope is not None:
            request["usage_scope"] = self.usage_scope.model_dump(mode="json")
        try:
            expanded_terms = self.vocabulary.expand(query)
            request["expanded_terms"] = expanded_terms
            result = self._search(
                query,
                query_id=query_id,
                expanded_terms=expanded_terms,
                package_ids=package_ids,
                path_globs=path_globs,
                include_source_only=include_source_only,
                max_results=max_results,
            )
        except Exception as exc:
            self._record(
                operation="query_sources",
                origin=origin,
                status="error",
                event_id=query_id,
                query_id=query_id,
                request=request,
                error=str(exc),
            )
            raise
        self._record(
            operation="query_sources",
            origin=origin,
            status="success",
            event_id=query_id,
            query_id=result.query_id,
            snapshot=result.snapshot,
            request=request,
            returned_sources=[
                _source_ref(
                    hit.source_package,
                    hit.revision,
                    hit.locator,
                )
                for hit in result.hits
            ],
            returned_refs=sorted(
                {
                    concept_ref
                    for hit in result.hits
                    for concept_ref in hit.promoted_concept_refs
                }
            ),
        )
        return result

    def _search(
        self,
        query: str,
        *,
        query_id: str,
        expanded_terms: list[str],
        package_ids: list[str] | None,
        path_globs: list[str] | None,
        include_source_only: bool,
        max_results: int,
    ) -> SourceSearchResult:
        query = query.strip()
        if not query:
            raise ValueError("source query cannot be empty")
        if not 1 <= max_results <= 50:
            raise ValueError("max_results must be between 1 and 50")
        lexical_text = " ".join(expanded_terms)
        tokens = _tokens(lexical_text)
        if not (
            english_tokens(lexical_text)
            or chinese_sequences(lexical_text)
        ):
            raise ValueError("source query has no searchable terms")
        selected = set(package_ids or [])
        globs = path_globs or ["*"]
        packages = list(self.catalog.iter_source_packages())
        unknown = sorted(selected - {package.id for package in packages})
        if unknown:
            raise ValueError(
                "unknown source package id(s): "
                + ", ".join(unknown)
                + "; use canonical source:<id> values returned by the catalog "
                + "or omit package_ids"
            )
        snapshot = self.catalog.snapshot()
        concepts = self.catalog.iter_concepts()
        usage_snapshot = self.catalog.usage_snapshot()
        if (
            self.usage_index is not None
            and (
                self.usage_index.snapshot() != snapshot
                or self.usage_index.usage_snapshot() != usage_snapshot
            )
        ):
            if not self.allow_index_rebuild:
                raise RuntimeError(
                    "knowledge index is missing or stale in read_only_v1; "
                    "prebuild the Catalog .derived index or provide an "
                    "external derived_root"
                )
            self.usage_index.rebuild(
                concepts,
                snapshot,
                observations=self.catalog.iter_observations(),
                retrievals=self.catalog.iter_retrieval_events(),
                usage_snapshot=usage_snapshot,
            )
        packages_by_id = {package.id: package for package in packages}
        if self.usage_scope is not None and selected:
            blocked = sorted(
                package_id
                for package_id in selected
                if not _target_backend_relevance(
                    packages_by_id[package_id],
                    self.usage_scope,
                )[0]
            )
            if blocked:
                raise ValueError(
                    "source package is not allowed for target backend "
                    f"{self.usage_scope.target_backend}: "
                    + ", ".join(blocked)
                )
        for package in packages:
            if (
                selected
                and package.id in selected
                and package.search_mode == "static_only"
            ):
                raise ValueError(
                    "source package is static_only and cannot be searched: "
                    + package.id
                )
        scored: list[SourceSearchHit] = []
        indexed = self.source_index.search(
            expanded_terms,
            package_ids=selected,
            path_globs=globs,
            include_source_only=include_source_only,
            limit=max(max_results * 20, 200),
        )
        for match in indexed:
            package = packages_by_id[match.source_package]
            eligible, target_adjustment, target_reason = (
                _target_backend_relevance(package, self.usage_scope)
            )
            if not eligible:
                continue
            path = self._kb_path(package.content_root) / match.path
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            hit = _indexed_hit(match, text, tokens)
            if target_adjustment:
                hit = hit.model_copy(
                    update={
                        "score": max(0.001, hit.score + target_adjustment),
                        "recommendation_reasons": [
                            *hit.recommendation_reasons,
                            target_reason,
                        ],
                    }
                )
            if self.usage_index is not None and self.usage_scope is not None:
                summary = self.usage_index.usage(
                    _source_ref(
                        hit.source_package,
                        hit.revision,
                        hit.locator,
                    ),
                    self.usage_scope,
                )
                hit = hit.model_copy(
                    update={
                        "score": max(
                            0.001,
                            hit.score + usage_score(summary),
                        ),
                        "usage_summary": summary,
                        "recommendation_reasons": [
                            *hit.recommendation_reasons,
                            *usage_reasons(
                                summary,
                                scope_label="exact evaluation scope",
                            ),
                        ],
                    }
                )
            promoted = _promoted_concepts(hit, concepts)
            if promoted:
                hit = hit.model_copy(
                    update={
                        "score": hit.score + _PROMOTED_CONCEPT_WEIGHT,
                        "promoted_concept_refs": promoted,
                        "recommendation_reasons": [
                            *hit.recommendation_reasons,
                            "actionable promoted concepts: " + str(len(promoted)),
                        ],
                    }
                )
            scored.append(hit)

        hits = sorted(
            scored,
            key=lambda item: (-item.score, item.source_package, item.path),
        )[:max_results]
        return SourceSearchResult(
            query_id=query_id,
            snapshot=snapshot,
            query=query,
            expanded_terms=expanded_terms,
            hits=hits,
        )

    def read(
        self,
        *,
        query_id: str,
        source_package: str,
        path: str,
        line_start: int = 1,
        line_end: int | None = None,
        origin: str = "application",
    ) -> SourceDocument:
        request = {
            "source_package": source_package,
            "path": path,
            "line_start": line_start,
            "line_end": line_end,
        }
        try:
            document = self._read(
                query_id=query_id,
                source_package=source_package,
                path=path,
                line_start=line_start,
                line_end=line_end,
            )
        except Exception as exc:
            self._record(
                operation="get_source",
                origin=origin,
                status="error",
                query_id=query_id,
                request=request,
                error=str(exc),
            )
            raise
        self._record(
            operation="get_source",
            origin=origin,
            status="success",
            query_id=query_id,
            request=request,
            returned_sources=[
                _source_ref(
                    document.source_package,
                    document.revision,
                    document.locator,
                )
            ],
        )
        return document

    def _read(
        self,
        *,
        query_id: str,
        source_package: str,
        path: str,
        line_start: int,
        line_end: int | None,
    ) -> SourceDocument:
        if query_id and not query_id.startswith("source-query:"):
            raise ValueError(
                "get_source query_id must start with source-query:"
            )
        if not source_package.startswith("source:"):
            raise ValueError(
                "source_package must use canonical source:<id> form"
            )
        snapshot = self.catalog.snapshot()
        source_ref = f"{source_package}::{path}"
        if line_start < 1:
            raise ValueError("line_start must be at least 1")
        if line_end is not None and line_end < line_start:
            raise ValueError("line_end must be greater than or equal to line_start")
        requested_end = line_end or (line_start + 199)
        if requested_end - line_start + 1 > 400:
            raise ValueError("get_source is limited to 400 lines per call")
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"source path escapes package: {path}")

        package = next(
            (
                item
                for item in self.catalog.iter_source_packages()
                if item.id == source_package
            ),
            None,
        )
        if package is None:
            raise ValueError(f"unknown source package: {source_package}")
        eligible, _, _ = _target_backend_relevance(package, self.usage_scope)
        if not eligible:
            assert self.usage_scope is not None
            raise ValueError(
                "source package is not allowed for target backend "
                f"{self.usage_scope.target_backend}: {source_package}"
            )
        if package.search_mode == "static_only":
            raise ValueError(
                "source package is static_only and cannot be read: "
                + source_package
            )
        if (
            not package.content_root.startswith("kb://")
            or not package.manifest_root
            or not package.manifest_root.startswith("kb://")
        ):
            raise ValueError("source package is not materialized in this catalog")
        if package.usage_policy == "federated_only":
            raise ValueError("federated-only source content cannot be read locally")
        manifest = GitSourceManifest.model_validate_json(
            self._kb_path(package.manifest_root).read_text(encoding="utf-8")
        )
        entry = next(
            (item for item in manifest.entries if item.path == path),
            None,
        )
        if entry is None or entry.object_type != "blob" or entry.mode == "120000":
            raise ValueError(f"source file is not readable: {source_ref}")
        if (entry.size or 0) > MAX_FILE_BYTES:
            raise ValueError("source file exceeds the 1 MB read limit")
        if Path(path).suffix.lower() not in TEXT_SUFFIXES:
            raise ValueError("source file is not a supported text format")

        source_path = (self._kb_path(package.content_root) / relative).resolve()
        source_path.relative_to(self._kb_path(package.content_root))
        data = source_path.read_bytes()
        text = data.decode("utf-8")
        lines = text.splitlines()
        if not lines:
            raise ValueError(f"source file is empty: {source_ref}")
        if line_start > len(lines):
            raise ValueError(
                f"line_start exceeds source length ({len(lines)}): {line_start}"
            )
        actual_end = min(requested_end, len(lines))
        content = "\n".join(lines[line_start - 1 : actual_end])
        return SourceDocument(
            query_id=query_id,
            snapshot=snapshot,
            source_package=package.id,
            revision=package.revision,
            path=path,
            locator=f"{path}:L{line_start}-L{actual_end}",
            usage_policy=package.usage_policy,
            usage_notes=package.usage_notes,
            line_start=line_start,
            line_end=actual_end,
            total_lines=len(lines),
            truncated=line_start > 1 or actual_end < len(lines),
            content=content,
        )

    def _record(self, **fields) -> None:
        if self.retrieval_audit is not None:
            self.retrieval_audit.append(retrieval_record(**fields))

    def _kb_path(self, value: str) -> Path:
        relative = Path(value.removeprefix("kb://"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"source path escapes catalog: {value}")
        resolved = (self.catalog_root / relative).resolve()
        resolved.relative_to(self.catalog_root)
        return resolved


def _target_backend_relevance(
    package: SourcePackage,
    usage_scope: KnowledgeUsageScope | None,
) -> tuple[bool, float, str]:
    """Apply SourcePackage backend policy without hiding portable sources."""

    if usage_scope is None or not package.allowed_backends:
        return True, 0.0, ""
    target_backend = str(usage_scope.target_backend).strip().lower()
    allowed = {
        str(item).strip().lower()
        for item in package.allowed_backends
        if str(item).strip()
    }
    if target_backend in allowed:
        return (
            True,
            _TARGET_BACKEND_MATCH_WEIGHT,
            f"target backend match: {target_backend}",
        )
    if package.usage_policy in {"restricted", "federated_only"}:
        return False, 0.0, ""
    return (
        True,
        -_CROSS_BACKEND_PENALTY,
        "cross-backend analogy; source scope: " + ", ".join(sorted(allowed)),
    )


def _source_ref(source_package: str, revision: str, locator: str) -> str:
    return f"{source_package}@{revision}::{locator}"


def _promoted_concepts(hit: SourceSearchHit, concepts) -> list[str]:
    return sorted(
        {
            concept.id
            for concept in concepts
            for source in concept.sources
            if source.resource == hit.source_package
            and source.revision == hit.revision
            and _same_fragment(
                source.locator,
                hit.path,
                hit.line_start,
                hit.line_end,
            )
        }
    )


def _same_fragment(
    locator: str,
    path: str,
    line_start: int,
    line_end: int,
) -> bool:
    matched = _LINE_LOCATOR.fullmatch(locator)
    if matched:
        return (
            matched.group("path") == path
            and int(matched.group("start")) <= line_end
            and line_start <= int(matched.group("end"))
        )
    return locator == path


def _tokens(query: str) -> list[str]:
    normalized = "".join(
        character.lower() if character.isalnum() or character in {"_", "-"} else " "
        for character in query
    )
    return sorted({token for token in normalized.split() if len(token) >= 2})


def _path_token_boost(path: str, tokens: list[str]) -> float:
    lowered_path = path.casefold()
    matched = {
        token.casefold()
        for token in tokens
        if len(token) >= 5
        and token.casefold() not in _PATH_TOKEN_EXCLUSIONS
        and token.casefold() in lowered_path
    }
    return min(_PATH_TOKEN_BOOST_CAP, _PATH_TOKEN_WEIGHT * len(matched))


def _indexed_hit(
    match: SourceIndexMatch,
    text: str,
    tokens: list[str],
) -> SourceSearchHit:
    lines = text.splitlines()
    lowered = text.casefold()
    matched = [token for token in tokens if token.casefold() in lowered]
    if matched:
        best_index = 0
        best_count = -1
        for index, line in enumerate(lines):
            line_lower = line.casefold()
            count = sum(token.casefold() in line_lower for token in matched)
            if count > best_count:
                best_index = index
                best_count = count
    else:
        best_index = max(0, match.line_start - 1)
    start = max(0, best_index - _SNIPPET_CONTEXT_LINES)
    end = min(len(lines), best_index + _SNIPPET_CONTEXT_LINES + 1)
    snippet = "\n".join(lines[start:end]).strip()
    if not snippet:
        snippet = match.chunk_text[:4000]
        start = match.line_start - 1
        end = match.line_end
    return SourceSearchHit(
        source_package=match.source_package,
        revision=match.revision,
        path=match.path,
        locator=f"{match.path}:L{start + 1}-L{end}",
        classification=match.classification,
        usage_policy=match.usage_policy,
        usage_notes=match.usage_notes,
        line_start=start + 1,
        line_end=end,
        score=match.score + _path_token_boost(match.path, tokens),
        snippet=snippet[:4000],
    )
