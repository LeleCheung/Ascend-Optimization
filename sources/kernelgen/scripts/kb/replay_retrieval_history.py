"""Replay successful historical Concept and Source queries without mutating runs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.index import SQLiteKnowledgeIndex
from kernelgen.knowledge.models import QueryContext
from kernelgen.knowledge.query import QueryKnowledge
from kernelgen.knowledge.records import FileRetrievalAudit
from kernelgen.knowledge.scope import match_scope
from kernelgen.knowledge.sources import SourceSearcher


_COHORT_SCHEMA = "1.0"
_REPORT_SCHEMA = "1.0"
_TOP_K = 4
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_LINE_RANGE_RE = re.compile(r":L[1-9][0-9]*-L[1-9][0-9]*$")


def _language(text: str) -> str:
    has_cjk = bool(_CJK_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "chinese"
    if has_latin:
        return "english"
    return "other"


def _source_key(reference: str) -> str:
    if "::" not in reference:
        return reference
    package_revision, locator = reference.split("::", 1)
    package = package_revision.split("@", 1)[0]
    path = _LINE_RANGE_RE.sub("", locator)
    return f"{package}::{path}"


def _historical_direct(event: dict[str, Any]) -> list[str]:
    levels = event.get("result_levels") or {}
    returned = list(event.get("returned_refs") or [])
    return [
        reference
        for reference in returned
        if not levels or levels.get(reference) == "direct"
    ]


def _discover_workspace_paths(runs_root: Path) -> list[str]:
    return [
        str(path.parent.relative_to(runs_root))
        for path in sorted(
            runs_root.glob("**/.kernelgen/knowledge/publish-result.json")
        )
    ]


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _COHORT_SCHEMA:
        raise ValueError("cohort manifest schema_version must be '1.0'")
    if not isinstance(payload.get("workspaces"), list):
        raise ValueError("cohort manifest workspaces must be a list")
    if not isinstance(payload.get("query_ids"), list):
        raise ValueError("cohort manifest query_ids must be a list")
    return payload


def _read_events(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"{path}:{line_number}: event must be an object")
        rows.append((line_number, event))
    return rows


def _collect_cohort(
    runs_root: Path,
    *,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    workspace_paths = (
        list(manifest["workspaces"])
        if manifest is not None
        else _discover_workspace_paths(runs_root)
    )
    allowed_query_ids = (
        set(manifest["query_ids"]) if manifest is not None else None
    )
    cases: list[dict[str, Any]] = []
    missing_logs: list[str] = []
    seen_query_ids: set[str] = set()

    for workspace_path in workspace_paths:
        knowledge_dir = runs_root / workspace_path
        log_path = knowledge_dir / "retrieval-log.jsonl"
        if not log_path.is_file():
            missing_logs.append(workspace_path)
            continue
        events = _read_events(log_path)
        detail_query_ids: set[str] = set()
        detail_refs: dict[str, list[str]] = {}
        for _, event in events:
            if event.get("status") != "success" or event.get("origin") != "mcp":
                continue
            operation = event.get("operation")
            query_id = str(event.get("query_id") or "")
            if operation == "get_knowledge":
                detail_query_ids.add(query_id)
                detail_refs.setdefault(query_id, []).extend(
                    event.get("returned_refs") or []
                )
            elif operation == "get_source":
                detail_query_ids.add(query_id)
                detail_refs.setdefault(query_id, []).extend(
                    _source_key(item)
                    for item in event.get("returned_sources") or []
                )

        for line_number, event in events:
            if event.get("status") != "success" or event.get("origin") != "mcp":
                continue
            operation = event.get("operation")
            if operation not in {"query_knowledge", "query_sources"}:
                continue
            query_id = str(event.get("query_id") or "")
            if allowed_query_ids is not None and query_id not in allowed_query_ids:
                continue
            if not query_id:
                raise ValueError(f"{log_path}:{line_number}: query_id is empty")
            if query_id in seen_query_ids:
                raise ValueError(f"duplicate historical query_id: {query_id}")
            seen_query_ids.add(query_id)
            request = event.get("request") or {}
            if operation == "query_knowledge":
                historical = _historical_direct(event)
                text = str(request.get("question") or "")
                definition_id = str(
                    (request.get("operator_signature") or {}).get("definition_id")
                    or "<missing>"
                )
            else:
                historical = [
                    _source_key(item)
                    for item in event.get("returned_sources") or []
                ]
                text = str(request.get("query") or "")
                definition_id = str(
                    (request.get("usage_scope") or {}).get("definition_id")
                    or "<missing>"
                )
            cases.append(
                {
                    "query_id": query_id,
                    "workspace": workspace_path,
                    "line_number": line_number,
                    "operation": operation,
                    "created_at": event.get("created_at") or "",
                    "definition_id": definition_id,
                    "language": _language(text),
                    "request": request,
                    "historical_top4": historical[:_TOP_K],
                    "historical_detail_read": query_id in detail_query_ids,
                    "historical_detail_refs": sorted(
                        set(detail_refs.get(query_id) or [])
                    ),
                }
            )

    if allowed_query_ids is not None:
        missing_query_ids = sorted(allowed_query_ids - seen_query_ids)
        if missing_query_ids:
            raise ValueError(
                "cohort manifest query_ids missing from logs: "
                + ", ".join(missing_query_ids[:8])
            )
    cases.sort(key=lambda row: (row["workspace"], row["line_number"]))
    return {
        "workspace_paths": workspace_paths,
        "missing_logs": missing_logs,
        "cases": cases,
    }


def _cohort_counts(cohort: dict[str, Any]) -> dict[str, int | float]:
    cases = cohort["cases"]
    concept_count = sum(row["operation"] == "query_knowledge" for row in cases)
    source_count = sum(row["operation"] == "query_sources" for row in cases)
    detail_count = sum(row["historical_detail_read"] for row in cases)
    query_count = len(cases)
    return {
        "workspace_count": len(cohort["workspace_paths"]),
        "workspace_logs_present": len(cohort["workspace_paths"])
        - len(cohort["missing_logs"]),
        "missing_log_count": len(cohort["missing_logs"]),
        "query_count": query_count,
        "concept_query_count": concept_count,
        "source_query_count": source_count,
        "detail_read_query_count": detail_count,
        "detail_read_rate": detail_count / query_count if query_count else 0.0,
    }


def _manifest_payload(cohort: dict[str, Any], cohort_id: str) -> dict[str, Any]:
    return {
        "schema_version": _COHORT_SCHEMA,
        "cohort_id": cohort_id,
        "selector": "workspace contains .kernelgen/knowledge/publish-result.json",
        "counts": _cohort_counts(cohort),
        "workspaces": cohort["workspace_paths"],
        "query_ids": [row["query_id"] for row in cohort["cases"]],
    }


def _validate_manifest_counts(
    manifest: dict[str, Any],
    cohort: dict[str, Any],
) -> None:
    expected = manifest.get("counts") or {}
    actual = _cohort_counts(cohort)
    for key in (
        "workspace_count",
        "workspace_logs_present",
        "missing_log_count",
        "query_count",
        "concept_query_count",
        "source_query_count",
        "detail_read_query_count",
    ):
        if key in expected and expected[key] != actual[key]:
            raise ValueError(
                f"cohort count mismatch for {key}: "
                f"expected {expected[key]}, got {actual[key]}"
            )


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / len(left_set | right_set)


def _rank(reference: str, returned: list[str]) -> int | None:
    try:
        return returned.index(reference) + 1
    except ValueError:
        return None


def _replay(
    kb: Path,
    cohort: dict[str, Any],
    temporary: Path,
) -> list[dict[str, Any]]:
    catalog = FilesystemCatalog(kb)
    concepts = {concept.id: concept for concept in catalog.iter_concepts()}
    audit = FileRetrievalAudit(temporary / "retrieval.jsonl")
    concept_service = QueryKnowledge(
        catalog,
        SQLiteKnowledgeIndex(temporary / "knowledge.db"),
        audit,
    )
    source_service = SourceSearcher(kb, audit)
    existing_source_index = getattr(source_service, "source_index", None)
    if existing_source_index is not None:
        replay_source_index = type(existing_source_index)(
            temporary / "sources.db",
            kb,
        )
        if (
            existing_source_index.current_key()
            == existing_source_index.expected_key()
        ):
            shutil.copy2(existing_source_index.path, replay_source_index.path)
        source_service.source_index = replay_source_index

    rows = []
    for case in cohort["cases"]:
        row = dict(case)
        row.update(
            {
                "replay_top4": [],
                "replay_analogies_top4": [],
                "replay_conflicts_top4": [],
                "runtime_incompatible_direct": [],
                "error": "",
            }
        )
        try:
            if case["operation"] == "query_knowledge":
                context = QueryContext.model_validate(case["request"])
                bundle = concept_service.execute(context, origin="application")
                direct = [item.concept_ref for item in bundle.direct]
                row["replay_top4"] = direct[:_TOP_K]
                row["replay_analogies_top4"] = [
                    item.concept_ref for item in bundle.analogies[:_TOP_K]
                ]
                row["replay_conflicts_top4"] = [
                    item.concept_ref for item in bundle.conflicts[:_TOP_K]
                ]
                row["runtime_incompatible_direct"] = [
                    reference
                    for reference in row["replay_top4"]
                    if reference in concepts
                    and match_scope(concepts[reference].scope, context).level
                    == "incompatible"
                ]
            else:
                request = case["request"]
                result = source_service.search(
                    str(request.get("query") or ""),
                    package_ids=request.get("package_ids") or None,
                    path_globs=request.get("path_globs") or None,
                    include_source_only=bool(
                        request.get("include_source_only", False)
                    ),
                    max_results=int(request.get("max_results") or 12),
                    origin="application",
                )
                row["replay_top4"] = [
                    f"{hit.source_package}::{hit.path}"
                    for hit in result.hits[:_TOP_K]
                ]
        except Exception as exc:  # Keep replay evidence for all other queries.
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _operation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if not row["error"]]
    detail_rows = [row for row in successful if row["historical_detail_read"]]
    detail_ref_total = sum(len(row["historical_detail_refs"]) for row in detail_rows)
    detail_ref_hits = sum(
        sum(
            reference in row["replay_top4"]
            for reference in row["historical_detail_refs"]
        )
        for row in detail_rows
    )
    detail_query_hits = sum(
        bool(set(row["historical_detail_refs"]) & set(row["replay_top4"]))
        for row in detail_rows
    )
    return {
        "query_count": len(rows),
        "successful_replays": len(successful),
        "replay_errors": len(rows) - len(successful),
        "empty_top4": sum(not row["replay_top4"] for row in successful),
        "historical_top1_retained": sum(
            bool(row["historical_top4"])
            and bool(row["replay_top4"])
            and row["historical_top4"][0] == row["replay_top4"][0]
            for row in successful
        ),
        "exact_top4_matches": sum(
            row["historical_top4"] == row["replay_top4"]
            for row in successful
        ),
        "mean_top4_jaccard": (
            sum(
                _jaccard(row["historical_top4"], row["replay_top4"])
                for row in successful
            )
            / len(successful)
            if successful
            else 0.0
        ),
        "detail_read_queries": len(detail_rows),
        "detail_query_top4_hits": detail_query_hits,
        "detail_query_top4_hit_rate": (
            detail_query_hits / len(detail_rows) if detail_rows else 1.0
        ),
        "detail_refs": detail_ref_total,
        "detail_ref_top4_hits": detail_ref_hits,
        "detail_ref_top4_recall": (
            detail_ref_hits / detail_ref_total if detail_ref_total else 1.0
        ),
        "runtime_incompatible_direct": sum(
            len(row["runtime_incompatible_direct"]) for row in successful
        ),
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "languages": dict(sorted(Counter(row["language"] for row in rows).items())),
        "definitions": dict(
            sorted(Counter(row["definition_id"] for row in rows).items())
        ),
        "phases": dict(
            sorted(
                Counter(
                    str(row["request"].get("phase") or "<source>")
                    for row in rows
                ).items()
            )
        ),
        "tasks": dict(
            sorted(
                Counter(
                    str(row["request"].get("task") or "<source>")
                    for row in rows
                ).items()
            )
        ),
    }


def _strict_incompatible(
    row: dict[str, Any],
    concepts: dict[str, Any],
) -> list[str]:
    if row["operation"] != "query_knowledge":
        return []
    context = QueryContext.model_validate(row["request"])
    return [
        reference
        for reference in row["replay_top4"]
        if reference in concepts
        and match_scope(concepts[reference].scope, context).level == "incompatible"
    ]


def _compare(
    candidate_rows: list[dict[str, Any]],
    baseline_report: dict[str, Any],
    catalog: FilesystemCatalog,
) -> dict[str, Any]:
    baseline_rows = {
        row["query_id"]: row for row in baseline_report.get("cases") or []
    }
    candidate_by_id = {row["query_id"]: row for row in candidate_rows}
    if set(baseline_rows) != set(candidate_by_id):
        raise ValueError("candidate and baseline reports use different query cohorts")
    concepts = {concept.id: concept for concept in catalog.iter_concepts()}
    summary: dict[str, Any] = {
        "baseline_label": baseline_report.get("label") or "baseline",
        "candidate_label": "",
        "operations": {},
        "detail_retention_changes": [],
        "strict_scope_leakage_cases": [],
    }
    for operation in ("query_knowledge", "query_sources"):
        pairs = [
            (baseline_rows[query_id], candidate_by_id[query_id])
            for query_id in sorted(candidate_by_id)
            if candidate_by_id[query_id]["operation"] == operation
        ]
        detail_gains = detail_losses = 0
        rank_improvements = rank_regressions = rank_ties = 0
        strict_baseline = strict_candidate = 0
        strict_cases = []
        changed_by_language: Counter[str] = Counter()
        changed_by_definition: Counter[str] = Counter()
        for baseline, candidate in pairs:
            baseline_hit = bool(
                set(candidate["historical_detail_refs"])
                & set(baseline["replay_top4"])
            )
            candidate_hit = bool(
                set(candidate["historical_detail_refs"])
                & set(candidate["replay_top4"])
            )
            if candidate["historical_detail_read"] and baseline_hit != candidate_hit:
                if candidate_hit:
                    detail_gains += 1
                else:
                    detail_losses += 1
                summary["detail_retention_changes"].append(
                    {
                        "query_id": candidate["query_id"],
                        "operation": operation,
                        "definition_id": candidate["definition_id"],
                        "language": candidate["language"],
                        "change": "gain" if candidate_hit else "loss",
                        "historical_detail_refs": candidate["historical_detail_refs"],
                        "baseline_top4": baseline["replay_top4"],
                        "candidate_top4": candidate["replay_top4"],
                    }
                )
            for reference in candidate["historical_detail_refs"]:
                baseline_rank = _rank(reference, baseline["replay_top4"]) or (_TOP_K + 1)
                candidate_rank = _rank(reference, candidate["replay_top4"]) or (_TOP_K + 1)
                if candidate_rank < baseline_rank:
                    rank_improvements += 1
                elif candidate_rank > baseline_rank:
                    rank_regressions += 1
                else:
                    rank_ties += 1
            if baseline["replay_top4"] != candidate["replay_top4"]:
                changed_by_language[candidate["language"]] += 1
                changed_by_definition[candidate["definition_id"]] += 1
            baseline_incompatible = _strict_incompatible(baseline, concepts)
            candidate_incompatible = _strict_incompatible(candidate, concepts)
            strict_baseline += len(baseline_incompatible)
            strict_candidate += len(candidate_incompatible)
            if baseline_incompatible or candidate_incompatible:
                strict_cases.append(
                    {
                        "query_id": candidate["query_id"],
                        "definition_id": candidate["definition_id"],
                        "baseline_incompatible": baseline_incompatible,
                        "candidate_incompatible": candidate_incompatible,
                    }
                )
        summary["strict_scope_leakage_cases"].extend(strict_cases)
        summary["operations"][operation] = {
            "query_count": len(pairs),
            "top1_changes": sum(
                (baseline["replay_top4"][:1] != candidate["replay_top4"][:1])
                for baseline, candidate in pairs
            ),
            "top4_changes": sum(
                baseline["replay_top4"] != candidate["replay_top4"]
                for baseline, candidate in pairs
            ),
            "mean_candidate_baseline_top4_jaccard": (
                sum(
                    _jaccard(baseline["replay_top4"], candidate["replay_top4"])
                    for baseline, candidate in pairs
                )
                / len(pairs)
                if pairs
                else 0.0
            ),
            "detail_query_retention_gains": detail_gains,
            "detail_query_retention_losses": detail_losses,
            "detail_ref_rank_improvements": rank_improvements,
            "detail_ref_rank_regressions": rank_regressions,
            "detail_ref_rank_ties": rank_ties,
            "strict_incompatible_baseline": strict_baseline,
            "strict_incompatible_candidate": strict_candidate,
            "changed_by_language": dict(sorted(changed_by_language.items())),
            "changed_by_definition": dict(sorted(changed_by_definition.items())),
        }
    return summary


def _markdown(report: dict[str, Any]) -> str:
    cohort = report["cohort"]
    lines = [
        f"# Historical retrieval replay: {report['label']}",
        "",
        f"- Cohort: `{cohort['cohort_id']}`",
        f"- Published workspaces: {cohort['counts']['workspace_count']} ({cohort['counts']['workspace_logs_present']} with retrieval logs, {cohort['counts']['missing_log_count']} without)",
        f"- Successful queries: {cohort['counts']['query_count']} ({cohort['counts']['concept_query_count']} Concept + {cohort['counts']['source_query_count']} Source)",
        f"- Queries followed by a successful detail read: {cohort['counts']['detail_read_query_count']} ({cohort['counts']['detail_read_rate']:.1%})",
        "",
        "## Interpretation",
        "",
        "Historical detail reads are a change-audit proxy, not relevance labels: they reflect the legacy ranker, legacy catalog snapshots, and what an Agent chose to inspect. Candidate acceptance is determined by required golden cases and strict Scope checks; detail-retention gains and losses identify cases for semantic review.",
        "",
        "## Replay metrics",
        "",
        "| Operation | Queries | Errors | Empty Top-4 | Historical Top-1 retained | Exact Top-4 | Mean Jaccard | Detail-query hit rate | Detail-ref recall | Incompatible direct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for operation in ("query_knowledge", "query_sources"):
        metrics = report["metrics"][operation]
        lines.append(
            f"| `{operation}` | {metrics['query_count']} | {metrics['replay_errors']} | {metrics['empty_top4']} | {metrics['historical_top1_retained']} | {metrics['exact_top4_matches']} | {metrics['mean_top4_jaccard']:.1%} | {metrics['detail_query_top4_hit_rate']:.1%} | {metrics['detail_ref_top4_recall']:.1%} | {metrics['runtime_incompatible_direct']} |"
        )
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            "### Query language",
            "",
            "| Language | Queries |",
            "|---|---:|",
        ]
    )
    for key, value in report["coverage"]["languages"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "### Definition",
            "",
            "| Definition | Queries |",
            "|---|---:|",
        ]
    )
    for key, value in report["coverage"]["definitions"].items():
        lines.append(f"| `{key}` | {value} |")

    comparison = report.get("comparison")
    if comparison:
        lines.extend(
            [
                "",
                f"## Comparison to {comparison['baseline_label']}",
                "",
                "| Operation | Top-1 changes | Top-4 changes | Mean overlap | Detail gains | Detail losses | Detail-rank better | Detail-rank worse | Strict incompatible baseline | Strict incompatible candidate |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for operation in ("query_knowledge", "query_sources"):
            metrics = comparison["operations"][operation]
            lines.append(
                f"| `{operation}` | {metrics['top1_changes']} | {metrics['top4_changes']} | {metrics['mean_candidate_baseline_top4_jaccard']:.1%} | {metrics['detail_query_retention_gains']} | {metrics['detail_query_retention_losses']} | {metrics['detail_ref_rank_improvements']} | {metrics['detail_ref_rank_regressions']} | {metrics['strict_incompatible_baseline']} | {metrics['strict_incompatible_candidate']} |"
            )
        if comparison["detail_retention_changes"]:
            lines.extend(
                [
                    "",
                    "### Detail-read retention changes",
                    "",
                    "| Query | Operation | Definition | Change |",
                    "|---|---|---|---|",
                ]
            )
            for row in comparison["detail_retention_changes"]:
                lines.append(
                    f"| `{row['query_id']}` | `{row['operation']}` | `{row['definition_id']}` | {row['change']} |"
                )
    errors = [row for row in report["cases"] if row["error"]]
    if errors:
        lines.extend(["", "## Replay errors", ""])
        for row in errors:
            lines.append(f"- `{row['query_id']}`: {row['error']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--cohort-id", default="published-workspaces-v1")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    runs_root = args.runs_root.resolve()
    manifest = _load_manifest(args.manifest) if args.manifest else None
    cohort = _collect_cohort(runs_root, manifest=manifest)
    if manifest is not None:
        _validate_manifest_counts(manifest, cohort)
    if args.write_manifest:
        if manifest is not None:
            raise ValueError("--manifest and --write-manifest cannot be combined")
        manifest = _manifest_payload(cohort, args.cohort_id)
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.manifest_only:
        if manifest is None:
            manifest = _manifest_payload(cohort, args.cohort_id)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    kb = args.kb.resolve()
    with tempfile.TemporaryDirectory(prefix="kernelgen-history-replay-") as temp_name:
        rows = _replay(kb, cohort, Path(temp_name))
    catalog = FilesystemCatalog(kb)
    cohort_manifest = manifest or _manifest_payload(cohort, args.cohort_id)
    report = {
        "schema_version": _REPORT_SCHEMA,
        "label": args.label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kb": str(kb),
        "catalog_snapshot": catalog.snapshot(),
        "runs_root": str(runs_root),
        "cohort": {
            "cohort_id": cohort_manifest["cohort_id"],
            "manifest": str(args.manifest.resolve()) if args.manifest else "",
            "counts": _cohort_counts(cohort),
            "missing_logs": cohort["missing_logs"],
        },
        "methodology": {
            "historical_detail_retention": "diagnostic proxy, not a relevance label",
            "acceptance_evidence": "required golden cases plus strict Scope checks",
        },
        "metrics": {
            operation: _operation_metrics(
                [row for row in rows if row["operation"] == operation]
            )
            for operation in ("query_knowledge", "query_sources")
        },
        "coverage": _coverage(rows),
        "cases": rows,
    }
    if args.compare_to:
        baseline_report = json.loads(args.compare_to.read_text(encoding="utf-8"))
        report["comparison"] = _compare(rows, baseline_report, catalog)
        report["comparison"]["candidate_label"] = args.label

    rendered_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    rendered_markdown = _markdown(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered_json, encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(rendered_markdown, encoding="utf-8")
    print(rendered_json, end="")
    return 1 if any(row["error"] for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
