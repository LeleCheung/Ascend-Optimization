#!/usr/bin/env python3
"""Extract multiple FlagGems operators into one Native v6.2 Catalog."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from kernelgen.agents.extractor.flaggems.v62_agent import (
    FlagGemsV62ExtractorInput,
    load_flaggems_timing_cases,
    persist_flaggems_v62_extraction,
)
from kernelgen.framework.agent_roles import materialize_agent_role
from kernelgen.framework.runtime.claude import ClaudeRuntime
from kernelgen.workflows.catalog_extract import (
    CatalogExtractOutput,
    CatalogExtractWorkflow,
)


KERNELGEN_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "kernelgen.flaggems-v62-batch-extract/v1"
_SAFE_OPERATOR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PACKAGE_FILES = {
    "definition.json",
    "oracle.py",
    "correctness.jsonl",
    "timing.jsonl",
}


def read_operators(path: Path) -> list[str]:
    """Read unique operator names from a plain list or Markdown table."""

    operators: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value.startswith("|") and value.count("|") >= 2:
            value = value.split("|", 2)[1].strip().strip("`")
        if (
            not value
            or value in {"算子", "算子名称", "operator", "Operator"}
            or set(value) <= {"-", ":", " "}
        ):
            continue
        if not _SAFE_OPERATOR.fullmatch(value):
            raise ValueError(f"invalid operator name in {path}: {value!r}")
        if value not in seen:
            seen.add(value)
            operators.append(value)
    if not operators:
        raise ValueError(f"operator file is empty: {path}")
    return operators


def _runtime(workspace: Path, model: str, timeout: int) -> ClaudeRuntime:
    role = KERNELGEN_ROOT / ".kernelgen/agents/kernel-flaggems-v62-extractor.md"
    materialize_agent_role(role, workspace)
    return ClaudeRuntime(
        workspace=workspace,
        model=model,
        allowed_tools="Read",
        permission_mode="acceptEdits",
        timeout=timeout,
        idle_timeout=max(120, timeout // 2),
    )


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _existing_package_status(catalog_root: Path, operator: str) -> str | None:
    operator_root = catalog_root / "ops" / operator
    if not operator_root.exists():
        return None
    if not operator_root.is_dir():
        return "target operator path is not a directory"
    actual = {path.name for path in operator_root.iterdir() if path.is_file()}
    missing = sorted(_PACKAGE_FILES - actual)
    empty = sorted(
        name
        for name in _PACKAGE_FILES & actual
        if (operator_root / name).stat().st_size == 0
    )
    if missing or empty:
        return f"incomplete existing package: missing={missing}, empty={empty}"
    return "complete"


def _extract_one(
    operator: str,
    *,
    flaggems_repo: Path,
    case_list_path: Path,
    workspace_root: Path,
    runtime_factory: Callable[[Path], Any],
) -> tuple[CatalogExtractOutput | None, dict[str, Any]]:
    started = time.monotonic()
    workspace = workspace_root / operator
    try:
        workflow = CatalogExtractWorkflow(
            cwd=str(workspace),
            runtime_factory=lambda _: runtime_factory(workspace),
        )
        output = workflow.run(
            {
                "operator": operator,
                "flaggems_repo": str(flaggems_repo),
                "case_list_path": str(case_list_path),
            }
        )
        return output, {
            "operator": operator,
            "status": "EXTRACTED",
            "seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001 - one operator must not abort the batch
        return None, {
            "operator": operator,
            "status": "EXTRACT_FAILED",
            "reason": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.monotonic() - started, 3),
        }


def _persist_one(
    output: CatalogExtractOutput,
    *,
    flaggems_repo: Path,
    case_list_path: Path,
    catalog_root: Path,
    adapter_catalog_root: Path | None,
    workspace_root: Path,
) -> dict[str, Any]:
    operator = output.operator
    coverage_path = workspace_root / operator / "accuracy_coverage.json"
    _atomic_json(coverage_path, output.accuracy_coverage)
    inp = FlagGemsV62ExtractorInput(
        operator=operator,
        flaggems_repo=str(flaggems_repo),
        case_list_path=str(case_list_path),
    )
    result = persist_flaggems_v62_extraction(
        inp,
        output.extraction,
        catalog_root,
        adapter_catalog_root=adapter_catalog_root,
    )
    return {
        "operator_root": str(result.operator_root),
        "adapter_definition": (
            str(result.adapter_definition_path)
            if result.adapter_definition_path is not None
            else None
        ),
        "correctness_workloads": result.num_correctness_workloads,
        "timing_workloads": result.num_timing_workloads,
        "accuracy_coverage": str(coverage_path),
    }


def run_batch(
    args: argparse.Namespace,
    *,
    extract_one: Callable[..., tuple[Any, dict[str, Any]]] = _extract_one,
    persist_one: Callable[..., dict[str, Any]] = _persist_one,
) -> tuple[int, dict[str, Any]]:
    operators = read_operators(args.operator_file.resolve())
    flaggems_repo = args.flaggems_repo.resolve()
    case_list_path = args.case_list_path.resolve()
    catalog_root = args.catalog_root.resolve()
    workspace_root = args.workspace_root.resolve()
    report_path = args.report.resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    adapter_catalog_root = (
        args.adapter_catalog_root.resolve()
        if args.adapter_catalog_root is not None
        else catalog_root.parent / "flaggems-adapter-definitions"
    )
    if not (adapter_catalog_root / "manifest.json").is_file():
        adapter_catalog_root = None

    results: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for operator in operators:
        existing = _existing_package_status(catalog_root, operator)
        if existing == "complete":
            results[operator] = {
                "operator": operator,
                "status": "ALREADY_PRESENT",
                "operator_root": str(catalog_root / "ops" / operator),
            }
            continue
        if existing is not None:
            results[operator] = {
                "operator": operator,
                "status": "TARGET_CONFLICT",
                "reason": existing,
            }
            continue
        try:
            load_flaggems_timing_cases(case_list_path, operator)
        except Exception as exc:  # noqa: BLE001 - report protocol gaps per operator
            results[operator] = {
                "operator": operator,
                "status": "CASE_LIST_INVALID",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            continue
        pending.append(operator)
        results[operator] = {
            "operator": operator,
            "status": "PENDING",
        }

    def report() -> dict[str, Any]:
        ordered = [results[operator] for operator in operators]
        counts: dict[str, int] = {}
        for row in ordered:
            status = str(row["status"])
            counts[status] = counts.get(status, 0) + 1
        return {
            "schema_version": SCHEMA_VERSION,
            "catalog_root": str(catalog_root),
            "adapter_catalog_root": (
                str(adapter_catalog_root) if adapter_catalog_root is not None else None
            ),
            "operator_count": len(operators),
            "counts": counts,
            "results": ordered,
        }

    _atomic_json(report_path, report())
    runtime_factory = lambda workspace: _runtime(
        workspace, args.model, args.timeout
    )
    completed = len(operators) - len(pending)
    with ThreadPoolExecutor(max_workers=min(args.max_workers, max(1, len(pending)))) as pool:
        futures = {
            pool.submit(
                extract_one,
                operator,
                flaggems_repo=flaggems_repo,
                case_list_path=case_list_path,
                workspace_root=workspace_root,
                runtime_factory=runtime_factory,
            ): operator
            for operator in pending
        }
        for future in as_completed(futures):
            operator = futures[future]
            try:
                output, row = future.result()
            except Exception as exc:  # noqa: BLE001 - injected workers may fail too
                output = None
                row = {
                    "operator": operator,
                    "status": "EXTRACT_FAILED",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            if output is not None:
                try:
                    row.update(
                        persist_one(
                            output,
                            flaggems_repo=flaggems_repo,
                            case_list_path=case_list_path,
                            catalog_root=catalog_root,
                            adapter_catalog_root=adapter_catalog_root,
                            workspace_root=workspace_root,
                        )
                    )
                    row["status"] = "PERSISTED"
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "PERSIST_FAILED"
                    row["reason"] = f"{type(exc).__name__}: {exc}"
            results[operator] = row
            completed += 1
            _atomic_json(report_path, report())
            print(
                f"[{completed}/{len(operators)}] {operator}: {row['status']}",
                flush=True,
            )

    final = report()
    _atomic_json(report_path, final)
    successful = {"PERSISTED", "ALREADY_PRESENT"}
    failures = [row for row in final["results"] if row["status"] not in successful]
    return (1 if failures else 0), final


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-file", type=Path, required=True)
    parser.add_argument("--flaggems-repo", type=Path, required=True)
    parser.add_argument("--case-list-path", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--adapter-catalog-root", type=Path)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("MODEL", "inherit"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if args.timeout < 1 or args.max_workers < 1:
        parser.error("timeout and max-workers must be positive")
    return args


def main() -> int:
    exit_code, report = run_batch(_parse_args())
    counts: dict[str, int] = {}
    for row in report["results"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"counts": counts}, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
