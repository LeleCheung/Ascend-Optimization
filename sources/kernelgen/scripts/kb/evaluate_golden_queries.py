"""Evaluate operator-specific Concept and Source retrieval golden cases."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from kernelgen.knowledge.catalog import FilesystemCatalog
from kernelgen.knowledge.index import SQLiteKnowledgeIndex
from kernelgen.knowledge.models import OperatorSignature, QueryContext, TargetContext
from kernelgen.knowledge.query import QueryKnowledge
from kernelgen.knowledge.records import FileRetrievalAudit
from kernelgen.knowledge.scope import match_scope
from kernelgen.knowledge.sources import SourceSearcher


REQUIRED_SCENARIOS = {
    "initial",
    "constraint",
    "post_error",
    "post_profile",
    "plateau",
}


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("golden fixture root must be a mapping")
    if payload.get("schema_version") != "1.0":
        raise ValueError("golden fixture schema_version must be '1.0'")
    operators = payload.get("operators")
    cases = payload.get("cases")
    if not isinstance(operators, dict) or len(operators) != 10:
        raise ValueError("golden fixture must define exactly 10 operators")
    if not isinstance(cases, list) or len(cases) != 50:
        raise ValueError("golden fixture must define exactly 50 Concept cases")
    grouped: dict[str, set[str]] = defaultdict(set)
    seen_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("each golden case must be a mapping")
        case_id = str(case.get("id") or "")
        definition_id = str(case.get("definition_id") or "")
        scenario = str(case.get("scenario") or "")
        if not case_id or case_id in seen_ids:
            raise ValueError(f"golden case id is empty or duplicated: {case_id!r}")
        if definition_id not in operators:
            raise ValueError(f"{case_id}: unknown definition_id {definition_id!r}")
        if scenario not in REQUIRED_SCENARIOS:
            raise ValueError(f"{case_id}: invalid scenario {scenario!r}")
        if not case.get("expected_any"):
            raise ValueError(f"{case_id}: expected_any must not be empty")
        seen_ids.add(case_id)
        grouped[definition_id].add(scenario)
    for definition_id in operators:
        if grouped[definition_id] != REQUIRED_SCENARIOS:
            raise ValueError(
                f"{definition_id}: scenarios differ from {sorted(REQUIRED_SCENARIOS)}"
            )
    return payload


def _signature(definition_id: str, fixture: dict[str, Any]) -> OperatorSignature:
    spec = fixture["operators"][definition_id]
    return OperatorSignature(
        definition_id=definition_id,
        definition_name=definition_id,
        op_type=spec["family"],
        motifs=spec["motifs"],
        dataflow=spec["dataflow"],
        dtypes=spec["dtypes"],
        layouts=spec["layouts"],
        workload_features=spec.get("workload_features") or {},
    )


def _rank(returned: list[str], expected: list[str]) -> int | None:
    positions = [returned.index(item) + 1 for item in expected if item in returned]
    return min(positions) if positions else None


def _evaluate_concepts(
    kb: Path,
    fixture: dict[str, Any],
    temporary: Path,
) -> tuple[list[dict[str, Any]], int]:
    catalog = FilesystemCatalog(kb)
    concepts = {concept.id: concept for concept in catalog.iter_concepts()}
    service = QueryKnowledge(
        catalog,
        SQLiteKnowledgeIndex(temporary / "knowledge.db"),
        FileRetrievalAudit(temporary / "retrieval.jsonl"),
    )
    target = TargetContext.model_validate(fixture["target"])
    rows: list[dict[str, Any]] = []
    incompatible_direct = 0
    max_results = int(fixture.get("max_results") or 4)
    for case in fixture["cases"]:
        context = QueryContext(
            phase=case["phase"],
            task=case["task"],
            round_num=(1 if case["phase"] != "initial" else None),
            question=case["question"],
            operator_signature=_signature(case["definition_id"], fixture),
            target_context=target,
            findings=case.get("findings") or [],
            errors=case.get("errors") or [],
            max_results=max_results,
        )
        bundle = service.execute(context)
        direct = [item.concept_ref for item in bundle.direct[:max_results]]
        analogies = [item.concept_ref for item in bundle.analogies[:max_results]]
        conflicts = [item.concept_ref for item in bundle.conflicts[:max_results]]
        incompatible_refs = []
        for reference in direct:
            concept = concepts.get(reference)
            if concept is not None and match_scope(concept.scope, context).level == "incompatible":
                incompatible_refs.append(reference)
        incompatible_direct += len(incompatible_refs)
        expected = list(case["expected_any"])
        rank = _rank(direct, expected)
        forbidden = sorted(set(direct) & set(case.get("forbidden_direct") or []))
        expected_first = case.get("expected_first")
        first_ok = expected_first is None or (
            bool(direct) and direct[0] == expected_first
        )
        rows.append(
            {
                "id": case["id"],
                "definition_id": case["definition_id"],
                "scenario": case["scenario"],
                "phase": case["phase"],
                "task": case["task"],
                "question": case["question"],
                "expected_any": expected,
                "expected_first": expected_first,
                "rank": rank,
                "hit_at_4": rank is not None and rank <= 4,
                "first_ok": first_ok,
                "forbidden_direct": forbidden,
                "incompatible_direct": incompatible_refs,
                "direct": direct,
                "analogies": analogies,
                "conflicts": conflicts,
            }
        )
    return rows, incompatible_direct


def _evaluate_sources(kb: Path, fixture: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    searcher = SourceSearcher(kb)
    default_limit = int(fixture.get("max_results") or 4)
    for case in fixture.get("source_cases") or []:
        limit = int(case.get("max_results") or default_limit)
        result = searcher.search(
            case["query"],
            package_ids=case.get("package_ids"),
            path_globs=case.get("path_globs"),
            include_source_only=bool(case.get("include_source_only", False)),
            max_results=limit,
        )
        returned = [item.path for item in result.hits[:limit]]
        expected = list(case["expected_any_paths"])
        rank = _rank(returned, expected)
        expected_first = case.get("expected_first_path")
        first_ok = expected_first is None or (
            bool(returned) and returned[0] == expected_first
        )
        forbidden = sorted(
            set(returned) & set(case.get("forbidden_paths") or [])
        )
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expected_any_paths": expected,
                "expected_first_path": expected_first,
                "rank": rank,
                "hit_at_4": rank is not None and rank <= 4,
                "required_hit": bool(case.get("required_hit", False)),
                "first_ok": first_ok,
                "forbidden_paths": forbidden,
                "returned_paths": returned,
            }
        )
    return rows


def _metrics(
    fixture: dict[str, Any],
    concepts: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    incompatible_direct: int,
) -> dict[str, Any]:
    concept_hits = sum(row["hit_at_4"] for row in concepts)
    source_hits = sum(row["hit_at_4"] for row in sources)
    concept_recall = concept_hits / len(concepts)
    source_recall = source_hits / len(sources) if sources else 1.0
    rank_failures = sum(not row["first_ok"] for row in [*concepts, *sources])
    required_hit_failures = sum(
        row["required_hit"] and not row["hit_at_4"] for row in sources
    )
    forbidden = sum(
        bool(row["forbidden_direct"]) for row in concepts
    ) + sum(bool(row["forbidden_paths"]) for row in sources)
    threshold = float(fixture.get("minimum_recall_at_4") or 0.9)
    passed = (
        concept_recall >= threshold
        and source_recall >= threshold
        and incompatible_direct == 0
        and forbidden == 0
        and rank_failures == 0
        and required_hit_failures == 0
    )
    return {
        "concept_cases": len(concepts),
        "concept_hits_at_4": concept_hits,
        "concept_recall_at_4": concept_recall,
        "source_cases": len(sources),
        "source_hits_at_4": source_hits,
        "source_recall_at_4": source_recall,
        "incompatible_direct": incompatible_direct,
        "forbidden_violations": forbidden,
        "first_rank_failures": rank_failures,
        "required_hit_failures": required_hit_failures,
        "minimum_recall_at_4": threshold,
        "passed": passed,
    }


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        f"# Golden retrieval evaluation: {report['label']}",
        "",
        f"- Concept Recall@4: {metrics['concept_recall_at_4']:.1%} ({metrics['concept_hits_at_4']}/{metrics['concept_cases']})",
        f"- Source Recall@4: {metrics['source_recall_at_4']:.1%} ({metrics['source_hits_at_4']}/{metrics['source_cases']})",
        f"- Incompatible direct results: {metrics['incompatible_direct']}",
        f"- Forbidden-result violations: {metrics['forbidden_violations']}",
        f"- Required first-rank failures: {metrics['first_rank_failures']}",
        f"- Required Source Hit@4 failures: {metrics['required_hit_failures']}",
        f"- Acceptance: {'PASS' if metrics['passed'] else 'FAIL'}",
        "",
        "## Concept cases",
        "",
        "| Case | Definition | Scenario | Hit@4 | Rank | First | Returned direct refs |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in report["concept_cases"]:
        returned = "<br>".join(row["direct"]) or "-"
        lines.append(
            f"| `{row['id']}` | `{row['definition_id']}` | {row['scenario']} | "
            f"{'yes' if row['hit_at_4'] else 'no'} | {row['rank'] or '-'} | "
            f"{'yes' if row['first_ok'] else 'no'} | {returned} |"
        )
    lines.extend(
        [
            "",
            "## Source cases",
            "",
            "| Case | Required | Hit@4 | Rank | First | Returned paths |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["source_cases"]:
        returned = "<br>".join(row["returned_paths"]) or "-"
        lines.append(
            f"| `{row['id']}` | {'yes' if row['required_hit'] else 'no'} | "
            f"{'yes' if row['hit_at_4'] else 'no'} | "
            f"{row['rank'] or '-'} | {'yes' if row['first_ok'] else 'no'} | "
            f"{returned} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    kb = args.kb.resolve()
    fixture = _load_fixture(args.fixture)
    with tempfile.TemporaryDirectory(prefix="kernelgen-golden-") as temp_name:
        concepts, incompatible_direct = _evaluate_concepts(
            kb,
            fixture,
            Path(temp_name),
        )
    sources = _evaluate_sources(kb, fixture)
    catalog = FilesystemCatalog(kb)
    report = {
        "schema_version": "1.0",
        "label": args.label,
        "fixture": str(args.fixture.resolve()),
        "kb": str(kb),
        "catalog_snapshot": catalog.snapshot(),
        "metrics": _metrics(
            fixture,
            concepts,
            sources,
            incompatible_direct,
        ),
        "concept_cases": concepts,
        "source_cases": sources,
    }
    rendered_json = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    rendered_markdown = _markdown(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered_json, encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(rendered_markdown, encoding="utf-8")
    print(rendered_json, end="")
    if args.enforce and not report["metrics"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
