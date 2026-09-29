#!/usr/bin/env python3
"""Run one VCD catalog oracle through a live KGS as the candidate solution."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY))

from kernelgen_server import (
    BoundEvaluateRequest,
    Catalog,
    EvaluationSettings,
    EvaluatorBinding,
    Implementation,
    InspectRequest,
    SourceFile,
)
from kernelgen_server.client import evaluate, inspect, preflight


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="path to the source-specific VCD native catalog",
    )
    parser.add_argument("--operator", required=True)
    parser.add_argument("--warmup-ms", type=int, default=5)
    parser.add_argument("--benchmark-ms", type=int, default=5)
    parser.add_argument(
        "--execution-timeout",
        type=int,
        default=1200,
        help="server-side timeout in seconds for each preflight/evaluate request",
    )
    preflight = parser.add_mutually_exclusive_group()
    preflight.add_argument(
        "--skip-preflight",
        dest="skip_preflight",
        action="store_true",
        help="run evaluate directly; evaluate still includes correctness before timing",
    )
    preflight.add_argument(
        "--with-preflight",
        dest="skip_preflight",
        action="store_false",
        help="run the optional correctness-only preflight before evaluate",
    )
    parser.set_defaults(skip_preflight=False)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _candidate_sources(operator_root: Path) -> list[SourceFile]:
    oracle_path = operator_root / "oracle.py"
    oracle = oracle_path.read_text(encoding="utf-8")
    functions = {
        node.name
        for node in ast.parse(oracle, filename=str(oracle_path)).body
        if isinstance(node, ast.FunctionDef)
    }
    if "run" not in functions:
        raise ValueError(f"reference-as-solution requires run(): {oracle_path}")
    sources = [SourceFile(path="solution.py", content=oracle)]
    assets_root = operator_root / "assets"
    if assets_root.is_dir():
        for path in sorted(
            item
            for item in assets_root.rglob("*")
            if item.is_file()
            and "__pycache__" not in item.parts
            and item.suffix not in {".pyc", ".pyo"}
        ):
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"reference-as-solution only supports text assets: {path}"
                ) from exc
            relative = path.relative_to(operator_root).as_posix()
            sources.append(SourceFile(path=relative, content=content))
    return sources


def main() -> int:
    args = _arguments()
    catalog = Catalog(args.catalog)
    operator = catalog.load(args.operator)
    relative = operator.relative
    operator_root = args.catalog.resolve() / "ops" / relative
    catalog_name = catalog.manifest.get("name")
    if not isinstance(catalog_name, str) or not catalog_name:
        raise ValueError("catalog manifest must declare a non-empty name")
    binding = EvaluatorBinding(
        catalog_name=catalog_name,
        definition=args.operator,
    )
    implementation = Implementation(
        name=f"{args.operator}-reference-as-solution",
        definition=operator.definition.name,
        language="python",
        entrypoint="solution.py::run",
        sources=_candidate_sources(operator_root),
    )
    request = BoundEvaluateRequest(
        binding=binding,
        implementation=implementation,
        settings=EvaluationSettings(
            warmup_ms=args.warmup_ms,
            benchmark_ms=args.benchmark_ms,
            timeout_seconds=args.execution_timeout,
        ),
    )
    manifest = inspect(InspectRequest(binding=binding), args.server)
    print(
        f"inspect: operator={manifest.case_list.operator} "
        f"signature={manifest.candidate_contract.signature} "
        f"timing_cases={len(manifest.case_list.cases)}",
        flush=True,
    )
    if not args.skip_preflight:
        flight = preflight(request, args.server, timeout=args.timeout)
        print(
            f"preflight: status={flight.get('status')} stage={flight.get('stage')} "
            f"cases={flight.get('num_cases')}",
            flush=True,
        )
        if flight.get("status") != "PASSED":
            if flight.get("log"):
                print(flight["log"], file=sys.stderr)
            return 2
    result = evaluate(request, args.server, timeout=args.timeout)
    payload = result.model_dump(mode="json")
    phase_counts: dict[str, int] = {}
    for row in result.per_workload:
        key = f"{row.phase}:{row.status.value}"
        phase_counts[key] = phase_counts.get(key, 0) + 1
    print(
        "evaluate: "
        + json.dumps(
            {
                "status": result.status.value,
                "num_workloads": result.num_workloads,
                "num_passed": result.num_passed,
                "timing_skipped": result.timing_skipped,
                "geo_mean": result.geo_mean,
                "phase_counts": phase_counts,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0 if result.status.value == "PASSED" else 3


if __name__ == "__main__":
    sys.exit(main())
