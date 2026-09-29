"""Versioned FTS5 projection over bounded SourcePackage line chunks."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.fts import (
    chinese_sequences,
    english_tokens,
    fts_or_query,
    weighted_rrf,
)
from kernelgen.knowledge.models import GitSourceManifest, StaticIngestionPlan
from kernelgen.knowledge.source_rules import source_path_matches


SOURCE_INDEX_SCHEMA_VERSION = "1"
TEXT_SUFFIXES = {
    "",
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".h",
    ".hpp",
    ".htm",
    ".html",
    ".json",
    ".md",
    ".mlir",
    ".py",
    ".rst",
    ".sh",
    ".td",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
MAX_FILE_BYTES = 1_000_000
_CHUNK_LINES = 80
_CHUNK_OVERLAP_LINES = 16
_SEARCH_BATCH = 1024
_CHANNEL_WEIGHTS = {"english": 1.5, "chinese": 1.5}
_RRF_SCALE = 1_000.0


@dataclass(frozen=True)
class SourceIndexMatch:
    source_package: str
    revision: str
    path: str
    classification: str
    usage_policy: str
    usage_notes: str
    content_identity: str
    entry_order: int
    line_start: int
    line_end: int
    chunk_text: str
    score: float


class SQLiteSourceIndex:
    def __init__(
        self,
        path: Path,
        catalog_root: Path,
        *,
        read_only: bool = False,
    ):
        self.path = Path(path)
        self.catalog_root = Path(catalog_root).resolve()
        self.read_only = read_only

    def _connect(self):
        if self.read_only:
            return sqlite3.connect(
                self.path.resolve().as_uri() + "?mode=ro",
                uri=True,
            )
        return sqlite3.connect(self.path)

    def ensure_current(self) -> None:
        expected = self.expected_key()
        if self.current_key() != expected:
            if self.read_only:
                raise RuntimeError(
                    "source index is missing or stale in read_only_v1; "
                    "prebuild the Catalog .derived index or provide an "
                    "external derived_root"
                )
            self.rebuild(expected_key=expected)

    def expected_key(self) -> str:
        catalog = FilesystemCatalog(self.catalog_root)
        plans = _plans(self.catalog_root)
        payload = {
            "packages": [
                item.model_dump(mode="json")
                for item in catalog.iter_source_packages()
            ],
            "plans": [
                plans[key].model_dump(mode="json") for key in sorted(plans)
            ],
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def current_key(self) -> str:
        if not self.path.is_file():
            return ""
        try:
            with self._connect() as connection:
                version = connection.execute(
                    "SELECT value FROM metadata WHERE key='schema_version'"
                ).fetchone()
                if version is None or version[0] != SOURCE_INDEX_SCHEMA_VERSION:
                    return ""
                tables = connection.execute(
                    """
                    SELECT COUNT(*) FROM sqlite_master
                    WHERE type = 'table'
                      AND name IN ('chunks', 'source_fts_en', 'source_fts_zh')
                    """
                ).fetchone()
                if tables is None or tables[0] != 3:
                    return ""
                key = connection.execute(
                    "SELECT value FROM metadata WHERE key='source_key'"
                ).fetchone()
        except sqlite3.DatabaseError:
            return ""
        return str(key[0]) if key else ""

    def rebuild(self, *, expected_key: str | None = None) -> dict[str, int]:
        if self.read_only:
            raise RuntimeError(
                "source index is read-only; provide an external derived_root "
                "to permit rebuilding"
            )
        expected_key = expected_key or self.expected_key()
        catalog = FilesystemCatalog(self.catalog_root)
        plans = _plans(self.catalog_root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=".sources.",
            suffix=".db",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        files_indexed = 0
        chunks_indexed = 0
        try:
            with sqlite3.connect(temporary) as connection:
                connection.executescript(
                    """
                    CREATE TABLE metadata(
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE chunks(
                        id INTEGER PRIMARY KEY,
                        source_package TEXT NOT NULL,
                        revision TEXT NOT NULL,
                        path TEXT NOT NULL,
                        classification TEXT NOT NULL,
                        usage_policy TEXT NOT NULL,
                        usage_notes TEXT NOT NULL,
                        content_identity TEXT NOT NULL,
                        entry_order INTEGER NOT NULL,
                        line_start INTEGER NOT NULL,
                        line_end INTEGER NOT NULL,
                        chunk_text TEXT NOT NULL
                    );
                    CREATE INDEX chunks_filter
                    ON chunks(source_package, classification, path);
                    CREATE VIRTUAL TABLE source_fts_en USING fts5(
                        chunk_id UNINDEXED,
                        path,
                        text,
                        tokenize = 'porter unicode61'
                    );
                    CREATE VIRTUAL TABLE source_fts_zh USING fts5(
                        chunk_id UNINDEXED,
                        path,
                        text,
                        tokenize = 'trigram'
                    );
                    """
                )
                connection.executemany(
                    "INSERT INTO metadata VALUES(?, ?)",
                    [
                        ("schema_version", SOURCE_INDEX_SCHEMA_VERSION),
                        ("source_key", expected_key),
                    ],
                )
                chunk_id = 0
                for package in catalog.iter_source_packages():
                    if package.search_mode == "static_only":
                        continue
                    if package.usage_policy == "federated_only":
                        continue
                    if (
                        not package.content_root.startswith("kb://")
                        or not package.manifest_root
                        or not package.manifest_root.startswith("kb://")
                    ):
                        continue
                    content_root = _kb_path(
                        self.catalog_root,
                        package.content_root,
                    )
                    manifest = GitSourceManifest.model_validate_json(
                        _kb_path(
                            self.catalog_root,
                            package.manifest_root,
                        ).read_text(encoding="utf-8")
                    )
                    plan = plans.get(package.id)
                    for entry_order, entry in enumerate(manifest.entries):
                        classification = _classification(plan, entry.path)
                        if (
                            entry.object_type != "blob"
                            or entry.mode == "120000"
                            or (entry.size or 0) > MAX_FILE_BYTES
                            or Path(entry.path).suffix.lower() not in TEXT_SUFFIXES
                            or classification == "federated"
                        ):
                            continue
                        source_path = (content_root / entry.path).resolve()
                        source_path.relative_to(content_root)
                        try:
                            text = source_path.read_text(encoding="utf-8")
                        except (OSError, UnicodeError):
                            continue
                        lines = text.splitlines()
                        if not lines:
                            continue
                        files_indexed += 1
                        for line_start, line_end, chunk_text in _chunks(lines):
                            chunk_id += 1
                            chunks_indexed += 1
                            connection.execute(
                                """
                                INSERT INTO chunks VALUES(
                                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                                )
                                """,
                                (
                                    chunk_id,
                                    package.id,
                                    package.revision,
                                    entry.path,
                                    classification,
                                    package.usage_policy,
                                    package.usage_notes,
                                    entry.sha256,
                                    entry_order,
                                    line_start,
                                    line_end,
                                    chunk_text,
                                ),
                            )
                            connection.execute(
                                "INSERT INTO source_fts_en VALUES(?, ?, ?)",
                                (chunk_id, entry.path, chunk_text),
                            )
                            connection.execute(
                                "INSERT INTO source_fts_zh VALUES(?, ?, ?)",
                                (chunk_id, entry.path, chunk_text),
                            )
                integrity = connection.execute(
                    "PRAGMA integrity_check"
                ).fetchone()
                if not integrity or integrity[0] != "ok":
                    raise RuntimeError(
                        f"SQLite source index integrity check failed: {integrity}"
                    )
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return {"files": files_indexed, "chunks": chunks_indexed}

    def search(
        self,
        expanded_terms: Sequence[str],
        *,
        package_ids: set[str],
        path_globs: Sequence[str],
        include_source_only: bool,
        limit: int,
    ) -> list[SourceIndexMatch]:
        self.ensure_current()
        lexical_text = "\n".join(
            str(item).strip() for item in expanded_terms if str(item).strip()
        )
        channel_queries = {
            "english": fts_or_query(english_tokens(lexical_text)),
            "chinese": fts_or_query(chinese_sequences(lexical_text)),
        }
        rows: dict[int, tuple] = {}
        rankings: dict[str, list[str]] = {}
        with self._connect() as connection:
            for channel, fts_query in channel_queries.items():
                if not fts_query:
                    continue
                table = f"source_fts_{'en' if channel == 'english' else 'zh'}"
                ranking, channel_rows = _channel_rows(
                    connection,
                    table=table,
                    fts_query=fts_query,
                    package_ids=package_ids,
                    include_source_only=include_source_only,
                    path_globs=path_globs,
                    limit=max(limit * 32, 512),
                )
                rankings[channel] = ranking
                rows.update(channel_rows)
        scores = weighted_rrf(
            rankings,
            weights=_CHANNEL_WEIGHTS,
            scale=_RRF_SCALE,
        )
        by_file: dict[tuple[str, str], SourceIndexMatch] = {}
        for reference in sorted(
            scores,
            key=lambda item: (-scores[item], int(item)),
        ):
            row = rows[int(reference)]
            item = SourceIndexMatch(
                source_package=row[1],
                revision=row[2],
                path=row[3],
                classification=row[4],
                usage_policy=row[5],
                usage_notes=row[6],
                content_identity=row[7],
                entry_order=int(row[8]),
                line_start=int(row[9]),
                line_end=int(row[10]),
                chunk_text=row[11],
                score=max(0.001, scores[reference]),
            )
            key = (item.source_package, item.path)
            prior = by_file.get(key)
            if prior is None or (
                item.score,
                -item.line_start,
            ) > (
                prior.score,
                -prior.line_start,
            ):
                by_file[key] = item

        candidates = list(by_file.values())
        if not include_source_only:
            by_content: dict[tuple[str, str], SourceIndexMatch] = {}
            for item in candidates:
                key = (item.source_package, item.content_identity)
                prior = by_content.get(key)
                if prior is None or (item.entry_order, item.path) < (
                    prior.entry_order,
                    prior.path,
                ):
                    by_content[key] = item
            candidates = list(by_content.values())
        return sorted(
            candidates,
            key=lambda item: (-item.score, item.source_package, item.path),
        )[:limit]


def _channel_rows(
    connection: sqlite3.Connection,
    *,
    table: str,
    fts_query: str,
    package_ids: set[str],
    include_source_only: bool,
    path_globs: Sequence[str],
    limit: int,
) -> tuple[list[str], dict[int, tuple]]:
    filters = [f"{table} MATCH ?"]
    parameters: list[object] = [fts_query]
    if package_ids:
        placeholders = ",".join("?" for _ in sorted(package_ids))
        filters.append(f"c.source_package IN ({placeholders})")
        parameters.extend(sorted(package_ids))
    if not include_source_only:
        filters.append("c.classification != 'source_only'")
    query = f"""
        SELECT c.*
        FROM {table}
        JOIN chunks AS c ON c.id = {table}.chunk_id
        WHERE {' AND '.join(filters)}
        ORDER BY bm25({table}, 0.0, 6.0, 1.0), c.id
        LIMIT ? OFFSET ?
    """
    ranking: list[str] = []
    rows: dict[int, tuple] = {}
    offset = 0
    while len(ranking) < limit:
        batch = connection.execute(
            query,
            (*parameters, _SEARCH_BATCH, offset),
        ).fetchall()
        if not batch:
            break
        for row in batch:
            if not any(
                source_path_matches(str(row[3]), glob) for glob in path_globs
            ):
                continue
            chunk_id = int(row[0])
            rows[chunk_id] = row
            ranking.append(str(chunk_id))
            if len(ranking) >= limit:
                break
        offset += len(batch)
    return ranking, rows


def _chunks(lines: list[str]):
    step = _CHUNK_LINES - _CHUNK_OVERLAP_LINES
    for start in range(0, len(lines), step):
        end = min(len(lines), start + _CHUNK_LINES)
        yield start + 1, end, "\n".join(lines[start:end])
        if end == len(lines):
            break


def _plans(catalog_root: Path) -> dict[str, StaticIngestionPlan]:
    root = Path(catalog_root) / "sources" / "ingestion"
    plans = {}
    for path in sorted(root.glob("*.yaml")) if root.exists() else []:
        plan = StaticIngestionPlan.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        plans[plan.source_package] = plan
    return plans


def _classification(plan: StaticIngestionPlan | None, path: str) -> str:
    if plan is None:
        return "source_only"
    matches = [
        rule.mode
        for rule in plan.rules
        if source_path_matches(path, rule.include)
    ]
    return matches[-1] if matches else "source_only"


def _kb_path(catalog_root: Path, value: str) -> Path:
    relative = Path(value.removeprefix("kb://"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"source path escapes catalog: {value}")
    resolved = (Path(catalog_root) / relative).resolve()
    resolved.relative_to(Path(catalog_root).resolve())
    return resolved
