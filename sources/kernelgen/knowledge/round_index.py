"""Rebuildable SQLite/FTS projection over immutable archived rounds."""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import tempfile
from pathlib import Path

from kernelgen.knowledge.contracts.runtime import RoundSearchHit


EMPTY_ROUND_SNAPSHOT = "sha256:" + hashlib.sha256(b"[]").hexdigest()


class SQLiteRoundIndex:
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

    def rebuild(self, archive_root: Path) -> None:
        if self.read_only:
            raise RuntimeError(
                "round index is read-only; provide an external derived_root "
                "to permit rebuilding"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=".rounds.",
            suffix=".db",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            rows = _latest_round_rows(Path(archive_root))
            snapshot = _round_snapshot(rows)
            with sqlite3.connect(temporary) as connection:
                connection.executescript(
                    """
                    CREATE TABLE metadata(
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE rounds(
                        run_id TEXT NOT NULL,
                        workspace_id TEXT NOT NULL,
                        round_num INTEGER NOT NULL,
                        archive_ref TEXT NOT NULL,
                        snapshot_id TEXT NOT NULL,
                        definition_id TEXT NOT NULL,
                        definition_name TEXT NOT NULL,
                        target_backend TEXT NOT NULL,
                        target_architecture TEXT NOT NULL,
                        target_device TEXT NOT NULL,
                        operator_motifs TEXT NOT NULL,
                        operator_dtypes TEXT NOT NULL,
                        plan_kind TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        code_changes TEXT NOT NULL,
                        hypothesis TEXT NOT NULL,
                        key_params TEXT NOT NULL,
                        status TEXT NOT NULL,
                        geo_mean REAL,
                        performance_baseline_round_num INTEGER,
                        geo_mean_delta_pct REAL,
                        solution_sha256 TEXT NOT NULL,
                        profile_status TEXT NOT NULL,
                        profile_summary TEXT NOT NULL,
                        conclusion TEXT NOT NULL,
                        PRIMARY KEY(run_id, workspace_id, round_num)
                    );
                    CREATE VIRTUAL TABLE rounds_fts USING fts5(
                        run_id UNINDEXED,
                        workspace_id UNINDEXED,
                        round_num UNINDEXED,
                        text
                    );
                    CREATE INDEX rounds_scope ON rounds(
                        definition_id,
                        target_backend,
                        target_architecture,
                        target_device,
                        status
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO metadata VALUES('snapshot', ?)",
                    (snapshot,),
                )
                for row in rows:
                    values = tuple(row[key] for key in _COLUMNS)
                    connection.execute(
                        f"INSERT INTO rounds({','.join(_COLUMNS)}) "
                        f"VALUES({','.join('?' for _ in _COLUMNS)})",
                        values,
                    )
                    connection.execute(
                        "INSERT INTO rounds_fts VALUES(?, ?, ?, ?)",
                        (
                            row["run_id"],
                            row["workspace_id"],
                            row["round_num"],
                            row["search_text"],
                        ),
                    )
                result = connection.execute(
                    "PRAGMA integrity_check"
                ).fetchone()
                if not result or result[0] != "ok":
                    raise RuntimeError(
                        f"round index integrity check failed: {result}"
                    )
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def snapshot(self) -> str:
        if not self.path.is_file():
            return EMPTY_ROUND_SNAPSHOT
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'snapshot'"
                ).fetchone()
        except sqlite3.Error:
            return EMPTY_ROUND_SNAPSHOT
        return str(row[0]) if row else EMPTY_ROUND_SNAPSHOT

    def search(
        self,
        query: str,
        *,
        definition_id: str = "",
        target_backend: str = "",
        target_architecture: str = "",
        target_device: str = "",
        limit: int = 12,
    ) -> list[RoundSearchHit]:
        if not self.path.is_file():
            return []
        clauses = []
        parameters: list[object] = []
        if definition_id:
            clauses.append("r.definition_id = ?")
            parameters.append(definition_id)
        for column, value in (
            ("target_backend", target_backend),
            ("target_architecture", target_architecture),
            ("target_device", target_device),
        ):
            if value:
                clauses.append(f"r.{column} = ?")
                parameters.append(value)
        where = " AND ".join(clauses) or "1 = 1"
        lexical = _fts_query(query)
        with self._connect() as connection:
            connection.row_factory = sqlite3.Row
            if lexical:
                rows = connection.execute(
                    f"""
                    SELECT r.*, bm25(rounds_fts) AS lexical_score
                    FROM rounds_fts
                    JOIN rounds r USING(run_id, workspace_id, round_num)
                    WHERE rounds_fts MATCH ? AND {where}
                    ORDER BY lexical_score ASC, r.run_id, r.workspace_id,
                             r.round_num DESC
                    LIMIT ?
                    """,
                    [lexical, *parameters, limit],
                ).fetchall()
            else:
                rows = connection.execute(
                    f"""
                    SELECT r.*, 0.0 AS lexical_score
                    FROM rounds r
                    WHERE {where}
                    ORDER BY r.run_id, r.workspace_id, r.round_num DESC
                    LIMIT ?
                    """,
                    [*parameters, limit],
                ).fetchall()
        return [_to_hit(row) for row in rows]


_COLUMNS = (
    "run_id",
    "workspace_id",
    "round_num",
    "archive_ref",
    "snapshot_id",
    "definition_id",
    "definition_name",
    "target_backend",
    "target_architecture",
    "target_device",
    "operator_motifs",
    "operator_dtypes",
    "plan_kind",
    "strategy",
    "code_changes",
    "hypothesis",
    "key_params",
    "status",
    "geo_mean",
    "performance_baseline_round_num",
    "geo_mean_delta_pct",
    "solution_sha256",
    "profile_status",
    "profile_summary",
    "conclusion",
)


def _latest_round_rows(archive_root: Path) -> list[dict]:
    selected: dict[tuple[str, str, int], tuple[str, dict]] = {}
    if not archive_root.exists():
        return []
    for manifest_path in sorted(archive_root.glob("*/*/snapshots/*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            snapshot_root = manifest_path.parent
            ledger = json.loads(
                (snapshot_root / "ledger.json").read_text(encoding="utf-8")
            )
            signature = json.loads(
                (snapshot_root / "context" / "operator-signature.json").read_text(
                    encoding="utf-8"
                )
            )
            target = json.loads(
                (snapshot_root / "context" / "target-context.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError, TypeError):
            continue
        for raw_round in ledger.get("rounds", []):
            if not isinstance(raw_round, dict):
                continue
            round_num = raw_round.get("round_num")
            if not isinstance(round_num, int) or round_num <= 0:
                continue
            row = _round_row(
                manifest,
                ledger,
                signature,
                target,
                raw_round,
            )
            key = (row["run_id"], row["workspace_id"], round_num)
            rank = str(manifest.get("archived_at") or "")
            if key not in selected or rank > selected[key][0]:
                selected[key] = (rank, row)
    return [selected[key][1] for key in sorted(selected)]


def _round_snapshot(rows: list[dict]) -> str:
    identities = [
        {
            "run_id": row["run_id"],
            "workspace_id": row["workspace_id"],
            "round_num": row["round_num"],
            "snapshot_id": row["snapshot_id"],
            "solution_sha256": row["solution_sha256"],
        }
        for row in rows
    ]
    encoded = json.dumps(
        identities,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _round_row(
    manifest: dict,
    ledger: dict,
    signature: dict,
    target: dict,
    raw_round: dict,
) -> dict:
    plan = raw_round.get("plan") or {}
    evaluation = raw_round.get("evaluation") or {}
    comparison = evaluation.get("comparison") or {}
    profile = raw_round.get("profile") or {}
    conclusion = raw_round.get("conclusion") or {}
    key_params = plan.get("key_params") or {}
    conclusion_text = " ".join(
        str(conclusion.get(key) or "")
        for key in (
            "root_cause",
            "perf_gap_analysis",
            "debug_lesson",
            "strategy_evolution",
            "next_suggestion",
        )
    ).strip()
    row = {
        "run_id": str(manifest.get("run_id") or ""),
        "workspace_id": str(manifest.get("workspace_id") or ""),
        "round_num": int(raw_round["round_num"]),
        "archive_ref": str(manifest.get("archive_ref") or ""),
        "snapshot_id": str(manifest.get("snapshot_id") or manifest.get("ledger_sha256") or ""),
        "definition_id": str(signature.get("definition_id") or ""),
        "definition_name": str(
            signature.get("definition_name")
            or ledger.get("definition_name")
            or ""
        ),
        "target_backend": str(target.get("backend") or ""),
        "target_architecture": str(target.get("architecture") or ""),
        "target_device": str(
            target.get("device") or ledger.get("target_hardware") or ""
        ),
        "operator_motifs": json.dumps(
            signature.get("motifs") or [], sort_keys=True
        ),
        "operator_dtypes": json.dumps(
            signature.get("dtypes") or [], sort_keys=True
        ),
        "plan_kind": str(plan.get("kind") or ""),
        "strategy": str(plan.get("strategy") or ""),
        "code_changes": str(plan.get("code_changes") or ""),
        "hypothesis": str(plan.get("hypothesis") or ""),
        "key_params": json.dumps(
            key_params, sort_keys=True, ensure_ascii=False
        ),
        "status": str(evaluation.get("status") or ""),
        "geo_mean": evaluation.get("geo_mean"),
        "performance_baseline_round_num": comparison.get(
            "performance_baseline_round_num",
            comparison.get("baseline_round_num"),
        ),
        "geo_mean_delta_pct": comparison.get("geo_mean_delta_pct"),
        "solution_sha256": str(
            (raw_round.get("solution") or {}).get("sha256") or ""
        ),
        "profile_status": str(profile.get("status") or ""),
        "profile_summary": str(profile.get("summary") or ""),
        "conclusion": conclusion_text,
    }
    row["search_text"] = " ".join(
        [
            row["definition_name"],
            row["operator_motifs"],
            row["operator_dtypes"],
            row["plan_kind"],
            row["strategy"],
            row["code_changes"],
            row["hypothesis"],
            row["key_params"],
            row["status"],
            row["profile_status"],
            row["profile_summary"],
            row["conclusion"],
        ]
    )
    return row


def _fts_query(query: str) -> str:
    tokens = [
        token
        for token in "".join(
            character.lower() if character.isalnum() or character in "_-" else " "
            for character in query
        ).split()
        if len(token) >= 2
    ]
    return " OR ".join(f'"{token}"' for token in sorted(set(tokens)))


def _to_hit(row: sqlite3.Row) -> RoundSearchHit:
    return RoundSearchHit(
        run_id=row["run_id"],
        workspace_id=row["workspace_id"],
        round_num=row["round_num"],
        archive_ref=row["archive_ref"],
        definition_id=row["definition_id"],
        target_backend=row["target_backend"],
        target_architecture=row["target_architecture"],
        target_device=row["target_device"],
        status=row["status"],
        geo_mean=row["geo_mean"],
        performance_baseline_round_num=(
            row["performance_baseline_round_num"]
        ),
        geo_mean_delta_pct=row["geo_mean_delta_pct"],
        strategy=row["strategy"],
        code_changes=row["code_changes"],
        key_params=json.loads(row["key_params"]),
        profile_summary=row["profile_summary"],
        conclusion=row["conclusion"],
        score=-float(row["lexical_score"]),
    )
