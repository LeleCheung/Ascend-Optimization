"""Disposable SQLite projection for deterministic Concept recall."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from kernelgen.knowledge.models import (
    Concept,
    KnowledgeUsageScope,
    KnowledgeUsageSummary,
    ObservationRecord,
)
from kernelgen.knowledge.contracts.runtime import (
    SearchHit,
    SearchRoute,
)
from kernelgen.knowledge.context import normalize_device
from kernelgen.knowledge.fts import (
    chinese_sequences,
    english_tokens,
    fts_or_query,
    weighted_rrf,
)


_INDEX_SCHEMA_VERSION = "2"
_CHANNEL_WEIGHTS = {
    "structured": 1.25,
    "english": 1.5,
    "chinese": 1.5,
}
_RRF_SCALE = 400.0


class SQLiteKnowledgeIndex:
    def __init__(self, path: Path, *, read_only: bool = False):
        self.path = Path(path)
        self.read_only = read_only

    def _connect(self):
        if self.read_only:
            return sqlite3.connect(
                self.path.resolve().as_uri() + "?mode=ro",
                uri=True,
            )
        return sqlite3.connect(self.path)

    def rebuild(
        self,
        concepts: Iterable[Concept],
        snapshot: str,
        observations: Iterable[ObservationRecord] = (),
        retrievals: Iterable[dict] = (),
        usage_snapshot: str = "",
    ) -> None:
        if self.read_only:
            raise RuntimeError(
                "knowledge index is read-only; provide an external "
                "derived_root to permit rebuilding"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=".knowledge.",
            suffix=".db",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with sqlite3.connect(temporary) as connection:
                connection.executescript(
                    """
                    CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE concepts(
                        id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        evidence_state TEXT NOT NULL,
                        title TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        body TEXT NOT NULL
                    );
                    CREATE VIRTUAL TABLE concept_fts_en USING fts5(
                        concept_id UNINDEXED,
                        title,
                        summary,
                        body,
                        terms,
                        tokenize = 'porter unicode61'
                    );
                    CREATE VIRTUAL TABLE concept_fts_zh USING fts5(
                        concept_id UNINDEXED,
                        title,
                        summary,
                        body,
                        terms,
                        tokenize = 'trigram'
                    );
                    CREATE TABLE terms(
                        concept_id TEXT NOT NULL,
                        route TEXT NOT NULL,
                        term TEXT NOT NULL
                    );
                    CREATE INDEX terms_lookup ON terms(route, term);
                    CREATE TABLE applications(
                        reference TEXT NOT NULL,
                        observation_ref TEXT NOT NULL,
                        workspace_id TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        disposition TEXT NOT NULL,
                        usage_mode TEXT NOT NULL,
                        effect TEXT NOT NULL,
                        agent_assessment TEXT NOT NULL,
                        evaluated_at TEXT NOT NULL,
                        PRIMARY KEY(reference, observation_ref)
                    );
                    CREATE INDEX applications_lookup
                    ON applications(reference, scope);
                    CREATE TABLE retrievals(
                        reference TEXT NOT NULL,
                        event_id TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        PRIMARY KEY(reference, event_id)
                    );
                    CREATE INDEX retrievals_lookup
                    ON retrievals(reference, scope);
                    """
                )
                connection.execute(
                    """
                    INSERT INTO metadata(key, value)
                    VALUES('schema_version', ?)
                    """,
                    (_INDEX_SCHEMA_VERSION,),
                )
                connection.execute(
                    "INSERT INTO metadata(key, value) VALUES('snapshot', ?)",
                    (snapshot,),
                )
                connection.execute(
                    """
                    INSERT INTO metadata(key, value)
                    VALUES('usage_snapshot', ?)
                    """,
                    (usage_snapshot,),
                )
                for concept in concepts:
                    search_fields = _concept_search_fields(concept)
                    connection.execute(
                        "INSERT INTO concepts VALUES(?, ?, ?, ?, ?, ?, ?)",
                        (
                            concept.id,
                            concept.kind,
                            concept.status,
                            concept.evidence_state,
                            concept.title,
                            concept.summary,
                            concept.body,
                        ),
                    )
                    connection.execute(
                        "INSERT INTO concept_fts_en VALUES(?, ?, ?, ?, ?)",
                        (concept.id, *search_fields),
                    )
                    connection.execute(
                        "INSERT INTO concept_fts_zh VALUES(?, ?, ?, ?, ?)",
                        (concept.id, *search_fields),
                    )
                    connection.executemany(
                        "INSERT INTO terms VALUES(?, ?, ?)",
                        [
                            (concept.id, route, term.lower())
                            for route, term in _concept_terms(concept)
                            if term
                        ],
                    )
                for observation in observations:
                    if observation.evaluation_scope is None:
                        continue
                    scope = _scope_key(observation.evaluation_scope)
                    evaluated_at = (
                        observation.evaluated_at or observation.recorded_at
                    ).isoformat()
                    effect = observation.knowledge_effect()
                    for application in observation.applications:
                        reference = _application_ref(application)
                        connection.execute(
                            """
                            INSERT OR IGNORE INTO applications
                            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                reference,
                                observation.id,
                                observation.workspace_id,
                                scope,
                                application.disposition,
                                observation.usage_mode,
                                effect,
                                (
                                    application.agent_assessment.assessment
                                    if application.agent_assessment is not None
                                    else "unassessed"
                                ),
                                evaluated_at,
                            ),
                        )
                for retrieval in retrievals:
                    try:
                        reference = str(retrieval["reference"])
                        event_id = str(retrieval["event_id"])
                        scope = _scope_key(
                            KnowledgeUsageScope.model_validate(
                                retrieval["scope"]
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                    connection.execute(
                        "INSERT OR IGNORE INTO retrievals VALUES(?, ?, ?)",
                        (reference, event_id, scope),
                    )
                result = connection.execute("PRAGMA integrity_check").fetchone()
                if not result or result[0] != "ok":
                    raise RuntimeError(f"SQLite integrity check failed: {result}")
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def search(
        self,
        routes: Sequence[SearchRoute],
        limit: int,
        *,
        lexical_queries: Sequence[str] = (),
    ) -> list[SearchHit]:
        if not self.path.is_file():
            return []
        structured_scores: dict[str, float] = defaultdict(float)
        matches: dict[str, set[str]] = defaultdict(set)
        rankings: dict[str, list[str]] = {}
        with self._connect() as connection:
            for route in routes:
                rows = connection.execute(
                    """
                    SELECT concept_id
                    FROM terms
                    WHERE route = ? AND term = ?
                    """,
                    (route.route, route.term.lower()),
                ).fetchall()
                for (concept_id,) in rows:
                    structured_scores[concept_id] += route.weight
                    matches[concept_id].add(f"{route.route}:{route.term}")
            rankings["structured"] = sorted(
                structured_scores,
                key=lambda concept_id: (
                    -structured_scores[concept_id],
                    concept_id,
                ),
            )[:limit]

            lexical_text = "\n".join(
                str(item).strip() for item in lexical_queries if str(item).strip()
            )
            english_query = fts_or_query(english_tokens(lexical_text))
            if english_query:
                rankings["english"] = [
                    row[0]
                    for row in connection.execute(
                        """
                        SELECT concept_id
                        FROM concept_fts_en
                        WHERE concept_fts_en MATCH ?
                        ORDER BY bm25(
                            concept_fts_en,
                            0.0, 8.0, 4.0, 1.0, 5.0
                        ), concept_id
                        LIMIT ?
                        """,
                        (english_query, limit),
                    ).fetchall()
                ]
                for concept_id in rankings["english"]:
                    matches[concept_id].add("lexical:english_bm25")

            chinese_query = fts_or_query(chinese_sequences(lexical_text))
            if chinese_query:
                rankings["chinese"] = [
                    row[0]
                    for row in connection.execute(
                        """
                        SELECT concept_id
                        FROM concept_fts_zh
                        WHERE concept_fts_zh MATCH ?
                        ORDER BY bm25(
                            concept_fts_zh,
                            0.0, 8.0, 4.0, 1.0, 5.0
                        ), concept_id
                        LIMIT ?
                        """,
                        (chinese_query, limit),
                    ).fetchall()
                ]
                for concept_id in rankings["chinese"]:
                    matches[concept_id].add("lexical:chinese_trigram_bm25")

        scores = weighted_rrf(
            rankings,
            weights=_CHANNEL_WEIGHTS,
            scale=_RRF_SCALE,
        )
        ordered = sorted(
            scores,
            key=lambda concept_id: (-scores[concept_id], concept_id),
        )[:limit]
        return [
            SearchHit(
                concept_id=concept_id,
                score=scores[concept_id],
                matched_on=sorted(matches[concept_id]),
            )
            for concept_id in ordered
        ]

    def usage(
        self,
        reference: str,
        scope: KnowledgeUsageScope,
    ) -> KnowledgeUsageSummary:
        if not self.path.is_file():
            return KnowledgeUsageSummary()
        with self._connect() as connection:
            retrieved_count = connection.execute(
                """
                SELECT COUNT(*) FROM retrievals
                WHERE reference = ? AND scope = ?
                """,
                (reference, _scope_key(scope)),
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT observation_ref, disposition, usage_mode, effect,
                       agent_assessment, evaluated_at
                FROM applications
                WHERE reference = ? AND scope = ?
                ORDER BY evaluated_at, observation_ref
                """,
                (reference, _scope_key(scope)),
            ).fetchall()
        applied = [
            row for row in rows if row[1] in {"adopted", "adapted"}
        ]
        positive = {"correctness_recovered", "performance_improved"}
        regressions = {"correctness_regressed", "performance_regressed"}
        return KnowledgeUsageSummary(
            retrieved_count=int(retrieved_count),
            considered_count=len(rows),
            applied_count=len(applied),
            evaluated_count=len(applied),
            single_success_count=sum(
                row[2] == "single" and row[3] in positive
                for row in applied
            ),
            combined_success_count=sum(
                row[2] == "combined" and row[3] in positive
                for row in applied
            ),
            correctness_recovery_count=sum(
                row[3] == "correctness_recovered" for row in applied
            ),
            regression_count=sum(
                row[3] in regressions for row in applied
            ),
            agent_confirmed_count=sum(
                row[4] == "confirmed" for row in applied
            ),
            agent_partially_confirmed_count=sum(
                row[4] == "partially_confirmed" for row in applied
            ),
            agent_not_confirmed_count=sum(
                row[4] == "not_confirmed" for row in applied
            ),
            agent_inconclusive_count=sum(
                row[4] == "inconclusive" for row in applied
            ),
            last_used_at=(
                max(row[5] for row in applied) if applied else None
            ),
            observation_refs=sorted({row[0] for row in rows}),
        )

    def usage_snapshot(self) -> str:
        if not self.path.is_file():
            return ""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='usage_snapshot'"
            ).fetchone()
        return str(row[0]) if row else ""

    def snapshot(self) -> str:
        if not self.path.is_file():
            return ""
        try:
            with self._connect() as connection:
                version = connection.execute(
                    "SELECT value FROM metadata WHERE key='schema_version'"
                ).fetchone()
                if version is None or version[0] != _INDEX_SCHEMA_VERSION:
                    return ""
                schema = connection.execute(
                    """
                    SELECT COUNT(*) FROM sqlite_master
                    WHERE type = 'table'
                      AND name IN (
                          'applications',
                          'retrievals',
                          'concept_fts_en',
                          'concept_fts_zh'
                      )
                    """
                ).fetchone()
                if schema is None or schema[0] != 4:
                    return ""
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key='snapshot'"
                ).fetchone()
        except sqlite3.DatabaseError:
            return ""
        return str(row[0]) if row else ""


def _concept_terms(concept: Concept) -> list[tuple[str, str]]:
    terms = [
        ("kind", concept.kind),
        *[("domain", item) for item in concept.domains],
        *[("phase", item) for item in concept.retrieval.phases],
        *[("task", item) for item in concept.retrieval.tasks],
        *[("symptom", item) for item in concept.retrieval.symptoms],
        *[("technique", item) for item in concept.retrieval.techniques],
        *[("keyword", item) for item in concept.retrieval.keywords],
        *[("keyword", item) for item in concept.domains],
        *[("keyword", item) for item in concept.retrieval.techniques],
        *[("definition_id", item) for item in concept.scope.operator.definition_ids],
        *[("op_type", item) for item in concept.scope.operator.op_types],
        *[("motif", item) for item in concept.scope.operator.motifs],
        *[("dataflow", item) for item in concept.scope.operator.dataflow],
        *[("dtype", item) for item in concept.scope.operator.dtypes],
        *[("layout", item) for item in concept.scope.operator.layouts],
    ]
    target = concept.scope.target
    terms.extend(
        [
            ("target_level", target.level),
            ("backend", target.backend),
            ("architecture", target.architecture),
            *[("device", normalize_device(item)) for item in target.devices],
            *[("capability", item) for item in target.capabilities],
        ]
    )
    for token in _lexical_tokens(
        " ".join((concept.title, concept.summary, concept.body))
    ):
        terms.append(("keyword", token))
    return sorted(
        {
            (route, str(term).lower())
            for route, term in terms
            if str(term)
        }
    )


def _concept_search_fields(concept: Concept) -> tuple[str, str, str, str]:
    terms = " ".join(term for _, term in _concept_terms(concept))
    return concept.title, concept.summary, concept.body, terms


def _scope_key(scope: KnowledgeUsageScope) -> str:
    return json.dumps(
        scope.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _application_ref(application) -> str:
    if application.concept_ref:
        return application.concept_ref
    source = application.source_ref
    return f"{source.resource}@{source.revision}::{source.locator}"


def _lexical_tokens(text: str) -> list[str]:
    normalized = "".join(
        character.lower() if character.isalnum() or character in {"_", "-"} else " "
        for character in text
    )
    return sorted({token for token in normalized.split() if len(token) >= 3})
