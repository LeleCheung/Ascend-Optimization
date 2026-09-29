#!/usr/bin/env python3
"""Update one Kernel Todo V2 results page from terminal optimizer outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
from typing import Any

from kernelgen.tools.kernel_todo_v2_pipeline import (
    StageResult,
    load_pipeline_results,
    refresh_markdown_summary,
    write_pipeline_results,
    write_second_stage_markdown,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ROW_OPERATOR = re.compile(r"`([^`]+)`")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", required=True)
    parser.add_argument("--workspace", type=pathlib.Path, required=True)
    parser.add_argument("--results-md", type=pathlib.Path)
    parser.add_argument("--pipeline-json", type=pathlib.Path)
    parser.add_argument("--kernelgen-results-md", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _rows(lines: list[str]) -> dict[str, tuple[int, list[str]]]:
    result: dict[str, tuple[int, list[str]]] = {}
    for index, line in enumerate(lines):
        if not line.startswith("| `"):
            continue
        columns = [item.strip() for item in line.strip().strip("|").split("|")]
        match = ROW_OPERATOR.search(columns[0])
        if match and len(columns) >= 9:
            result[match.group(1)] = (index, columns)
    return result


def _definition_to_source(chip: str, source_names: set[str]) -> dict[str, str]:
    inventory = _read_json(
        REPO_ROOT / "kernel_todo_v2" / "pytest_conversion_inventory.json"
    )["operators"]
    candidates: dict[str, list[str]] = {}
    for entry in inventory:
        source = entry["source_operator"]
        if source not in source_names or (
            chip not in entry["chips"] and chip != "nvidia"
        ):
            continue
        candidates.setdefault(entry["operator"], []).append(source)
    mapping = {name: name for name in source_names}
    for definition, sources in candidates.items():
        if len(sources) == 1:
            mapping.setdefault(definition, sources[0])
    return mapping


def _relative(from_dir: pathlib.Path, target: pathlib.Path) -> str:
    return pathlib.Path(
        os.path.relpath(target.resolve(), from_dir.resolve())
    ).as_posix()


def _hack_text(ledger: dict[str, Any]) -> str:
    best_round = ledger.get("best_round")
    rounds = ledger.get("rounds") or []
    selected = next(
        (item for item in rounds if item.get("round_num") == best_round),
        rounds[-1] if rounds else {},
    )
    evaluation = selected.get("evaluation") or {}
    if not evaluation.get("is_hack"):
        return "未发现"
    reason = str(evaluation.get("hack_reason") or "Server 标记为 hack").replace("|", "\\|")
    return f"是：{reason}"


def _terminal_output(path: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    output = _read_json(path)
    ledger_path = path.parent / ".ledger.json"
    if not ledger_path.is_file():
        raise RuntimeError(f"terminal output has no ledger: {path}")
    ledger = _read_json(ledger_path)
    if output.get("definition_name") != ledger.get("definition_name"):
        raise RuntimeError(f"definition mismatch between output and ledger: {path}")
    return output, ledger


def main() -> int:
    args = _parse_args()
    workspace = args.workspace.resolve()
    results_path = (
        args.results_md.resolve()
        if args.results_md
        else REPO_ROOT / "kernel_todo_v2" / args.chip / "results.md"
    )
    pipeline_path = (
        args.pipeline_json.resolve()
        if args.pipeline_json
        else results_path.parent / "pipeline_results.json"
    )
    kernelgen_results_path = (
        args.kernelgen_results_md.resolve()
        if args.kernelgen_results_md
        else results_path.parent / "kernelgen_results.md"
    )
    lines = results_path.read_text(encoding="utf-8").splitlines()
    rows = _rows(lines)
    pipeline = load_pipeline_results(pipeline_path, chip=args.chip)
    definition_to_source = _definition_to_source(args.chip, set(rows))
    updated: list[str] = []
    unresolved: list[str] = []

    for output_path in sorted(
        workspace.glob("definitions/*/optimize_definition_output.json")
    ):
        output, ledger = _terminal_output(output_path)
        definition = str(output.get("definition_name") or output_path.parent.name)
        source = definition_to_source.get(definition)
        if source is None or source not in rows:
            unresolved.append(definition)
            continue
        index, columns = rows[source]
        passed = output.get("status") == "PASSED"
        speed = output.get("best_geo_mean")
        if passed and (not isinstance(speed, (int, float)) or not math.isfinite(speed)):
            raise RuntimeError(
                f"PASSED output has no finite best_geo_mean: {output_path}"
            )
        below_threshold = passed and float(speed) < pipeline.threshold
        columns[3] = (
            "待二阶段优化"
            if below_threshold
            else "成功"
            if passed
            else "失败"
        )
        columns[4] = f"{float(speed):.3f}x" if passed else "—"
        columns[5] = _hack_text(ledger)
        ledger_path = output_path.parent / ".ledger.json"
        evidence = (
            f"[ledger]({_relative(results_path.parent, ledger_path)}) / "
            f"[output]({_relative(results_path.parent, output_path)})"
        )
        if passed:
            columns[6] = f"—（{evidence}）"
            columns[7] = (
                "已达到 0.8x；保留 best code 并进行提交前复验。"
                if not below_threshold
                else "正确性和计时完整但未达到 0.8x；进入 Native KernelGen 二阶段优化。"
            )
        else:
            columns[6] = f"优化终态未获得全量正确候选。（{evidence}）"
            columns[7] = "根据 ledger 的最后失败阶段继续排查。"
        code_path = output_path.parent / ".best_kernel.py"
        columns[8] = (
            f"[best code]({_relative(results_path.parent, code_path)})"
            if passed and code_path.is_file()
            else f"[output]({_relative(results_path.parent, output_path)})"
        )
        evidence_paths = [
            _relative(results_path.parent, ledger_path),
            _relative(results_path.parent, output_path),
        ]
        if passed and code_path.is_file():
            evidence_paths.append(_relative(results_path.parent, code_path))
        pipeline.upsert_simple_opt(
            source_operator=source,
            definition_name=definition,
            result=StageResult(
                attempt_id=_relative(results_path.parent, output_path),
                verdict="passed" if passed else "failed",
                geo_mean=float(speed) if passed else None,
                evidence=evidence_paths,
                reason=(
                    ""
                    if passed
                    else "优化终态未获得全量正确候选。"
                ),
            ),
        )
        lines[index] = "| " + " | ".join(columns) + " |"
        rows[source] = (index, columns)
        updated.append(source)

    summary = refresh_markdown_summary(lines, pipeline)

    report = {
        "updated": updated,
        "unresolved": unresolved,
        "summary": summary.model_dump(),
        "pipeline_json": str(pipeline_path),
        "kernelgen_results_md": str(kernelgen_results_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if unresolved:
        raise RuntimeError(f"ambiguous or missing results rows: {unresolved}")
    if not args.dry_run:
        write_pipeline_results(pipeline_path, pipeline)
        write_second_stage_markdown(kernelgen_results_path, pipeline)
        temporary = results_path.with_suffix(results_path.suffix + ".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
