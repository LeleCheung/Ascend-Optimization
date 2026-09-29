#!/usr/bin/env python3
"""Extract a list of FlagGems operators into one v5 catalog.

LLM extraction runs concurrently, while catalog persistence is serialized in
the parent process so concurrent workers cannot overwrite ``manifest.json``.
Reference validation is intentionally a separate step and should use
``kernelgen_server/tests/live_server_validation.py`` on the target device.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from kernelgen.agents.extractor.flaggems import FlagGemsExtractorAgent
from kernelgen.examples.flaggems_extract.run_example import prepare_native_agent
from kernelgen.framework.runtime.claude import ClaudeRuntime
from kernelgen.workflows.flaggems_extract import FlagGemsExtractWorkflow


_SAFE_OPERATOR = re.compile(r"^[A-Za-z0-9_.-]+$")


def read_operators(path: Path) -> list[str]:
    """Read either a plain one-name-per-line file or a Markdown table."""
    operators: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value.startswith("|") and value.count("|") >= 2:
            value = value.split("|", 2)[1].strip()
        if (
            not value
            or value in {"算子名称", "operator", "Operator"}
            or set(value) <= {"-", ":", " "}
        ):
            continue
        if not _SAFE_OPERATOR.fullmatch(value):
            raise ValueError(f"invalid operator name in {path}: {value!r}")
        if value not in seen:
            seen.add(value)
            operators.append(value)
    return operators


def _runtime(workspace: Path, model: str, timeout: int) -> ClaudeRuntime:
    prepare_native_agent(workspace)
    return ClaudeRuntime(
        workspace=workspace,
        model=model,
        allowed_tools="Bash,Read,Grep,Glob",
        permission_mode="acceptEdits",
        timeout=timeout,
        idle_timeout=max(120, timeout // 2),
    )


def _extract_one(
    operator: str,
    *,
    flaggems_repo: str,
    workspace_root: Path,
    model: str,
    timeout: int,
    case_list_path: str | None,
) -> tuple[Any, dict[str, Any]]:
    started = time.monotonic()
    workspace = workspace_root / operator
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        output = FlagGemsExtractorAgent().run(
            {
                "operator": operator,
                "flaggems_repo": flaggems_repo,
                "case_list_path": case_list_path,
            },
            _runtime(workspace, model, timeout),
        )
        return output, {
            "operator": operator,
            "status": "EXTRACTED" if output.results else "NO_DEFINITION",
            "definitions": [
                result.definition.name for result in output.results
            ],
            "seconds": round(time.monotonic() - started, 1),
        }
    except Exception as exc:  # noqa: BLE001 - report each independent operator
        return None, {
            "operator": operator,
            "status": "EXTRACT_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.monotonic() - started, 1),
        }


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-file", type=Path, required=True)
    parser.add_argument("--flaggems-repo", required=True)
    parser.add_argument("--catalog-root", required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("MODEL", "inherit"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--case-list-path",
        help="authoritative target flaggems.benchmark-case-list/v2 report",
    )
    args = parser.parse_args()
    if args.max_workers < 1 or args.timeout < 1:
        parser.error("max-workers and timeout must be positive")

    operators = read_operators(args.operator_file)
    args.workspace_root.mkdir(parents=True, exist_ok=True)
    print(
        f"extracting {len(operators)} operators with "
        f"{args.max_workers} workers -> {args.catalog_root}",
        flush=True,
    )

    rows_by_operator: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                _extract_one,
                operator,
                flaggems_repo=args.flaggems_repo,
                workspace_root=args.workspace_root,
                model=args.model,
                timeout=args.timeout,
                case_list_path=args.case_list_path,
            ): operator
            for operator in operators
        }
        for future in as_completed(futures):
            operator = futures[future]
            output, row = future.result()
            if output is not None and output.results:
                try:
                    FlagGemsExtractWorkflow._dump_catalog(
                        output.results,
                        args.catalog_root,
                    )
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "PERSIST_FAILED"
                    row["error"] = f"{type(exc).__name__}: {exc}"
            rows_by_operator[operator] = row
            ordered = [
                rows_by_operator[name]
                for name in operators
                if name in rows_by_operator
            ]
            _write_report(args.report, ordered)
            print(
                f"[{len(ordered)}/{len(operators)}] "
                f"{operator}: {row['status']} "
                f"{row.get('definitions', row.get('error', ''))}",
                flush=True,
            )

    rows = [rows_by_operator[name] for name in operators]
    failures = [
        row for row in rows if row["status"] != "EXTRACTED"
    ]
    print(
        f"completed: {len(rows) - len(failures)}/{len(rows)} extracted",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
