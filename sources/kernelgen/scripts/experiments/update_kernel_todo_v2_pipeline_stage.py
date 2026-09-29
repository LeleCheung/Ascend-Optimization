#!/usr/bin/env python3
"""Record one Native KernelGen stage result and refresh Kernel Todo V2 views."""

from __future__ import annotations

import argparse
import json
import pathlib
import re

from kernelgen.tools.kernel_todo_v2_pipeline import (
    PipelineStatus,
    StageName,
    StageResult,
    load_pipeline_results,
    refresh_markdown_summary,
    write_pipeline_results,
    write_second_stage_markdown,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ROW_OPERATOR = re.compile(r"`([^`]+)`")
SECOND_STAGE_NAMES = (
    StageName.NATIVE_BASELINE,
    StageName.KERNELGEN_NATIVE,
    StageName.FINAL_GEMS,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[stage.value for stage in SECOND_STAGE_NAMES],
    )
    parser.add_argument(
        "--attempt-id",
        required=True,
        help="stable identifier for this append-only stage attempt",
    )
    parser.add_argument(
        "--verdict",
        required=True,
        choices=["passed", "failed", "blocked"],
    )
    parser.add_argument("--geo-mean", type=float)
    parser.add_argument("--evidence", action="append", required=True)
    parser.add_argument("--reason", default="")
    parser.add_argument("--code-path")
    parser.add_argument("--results-md", type=pathlib.Path)
    parser.add_argument("--pipeline-json", type=pathlib.Path)
    parser.add_argument("--kernelgen-results-md", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _atomic_markdown(path: pathlib.Path, lines: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _direction(stage: StageName, status: PipelineStatus, result: StageResult) -> str:
    if status == PipelineStatus.BLOCKED:
        return "排除阻塞后复用原二阶段 workspace 续跑。"
    if status == PipelineStatus.FAILED:
        return "根据结构化阶段证据定位失败；代码或工具发生明确变化后再重试。"
    if stage == StageName.NATIVE_BASELINE:
        return "Native baseline 与 profiling 门禁已完成；进入 KernelGen workflow。"
    if stage == StageName.KERNELGEN_NATIVE:
        return "KernelGen Native 结果已完成；迁回 FlagGems 做最终验收。"
    if result.geo_mean is not None and status == PipelineStatus.SECOND_STAGE_QUALIFIED:
        return "最终 Gems 已达到 0.8x；归档代码和全部阶段证据。"
    return "最终 Gems 正确且可计时，但二阶段仍未达到 0.8x；保留四阶段结果。"


def _sync_result_row(
    lines: list[str],
    *,
    source_operator: str,
    stage: StageName,
    result: StageResult,
    status: PipelineStatus,
    code_path: str | None,
) -> None:
    status_text = {
        PipelineStatus.FIRST_STAGE_QUALIFIED: "成功",
        PipelineStatus.PENDING_SECOND_STAGE: "待二阶段优化",
        PipelineStatus.SECOND_STAGE_RUNNING: "二阶段处理中",
        PipelineStatus.SECOND_STAGE_QUALIFIED: "成功",
        PipelineStatus.FAILED: "失败",
        PipelineStatus.BLOCKED: "阻塞",
    }[status]
    for index, line in enumerate(lines):
        if not line.startswith("| `"):
            continue
        columns = [item.strip() for item in line.strip().strip("|").split("|")]
        match = ROW_OPERATOR.search(columns[0])
        if not match or match.group(1) != source_operator or len(columns) < 9:
            continue
        columns[3] = status_text
        if stage == StageName.FINAL_GEMS:
            columns[4] = (
                f"{result.geo_mean:.3f}x"
                if result.verdict.value == "passed" and result.geo_mean is not None
                else "—"
            )
        columns[6] = result.reason or "—"
        columns[7] = _direction(stage, status, result)
        if code_path:
            columns[8] = code_path
        lines[index] = "| " + " | ".join(columns) + " |"
        return
    raise RuntimeError(f"results row not found: {source_operator}")


def main() -> int:
    args = _parse_args()
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
    stage = StageName(args.stage)
    result = StageResult(
        attempt_id=args.attempt_id,
        verdict=args.verdict,
        geo_mean=args.geo_mean,
        evidence=args.evidence,
        reason=args.reason,
    )
    pipeline = load_pipeline_results(pipeline_path, chip=args.chip)
    changed = pipeline.update_stage(
        source_operator=args.operator,
        name=stage,
        result=result,
    )
    operator_result = pipeline.operators[args.operator]
    status = operator_result.status(pipeline.threshold)
    lines = results_path.read_text(encoding="utf-8").splitlines()
    _sync_result_row(
        lines,
        source_operator=args.operator,
        stage=stage,
        result=result,
        status=status,
        code_path=args.code_path,
    )
    summary = refresh_markdown_summary(lines, pipeline)
    report = {
        "operator": args.operator,
        "stage": stage.value,
        "changed": changed,
        "status": status.value,
        "summary": summary.model_dump(),
        "pipeline_json": str(pipeline_path),
        "kernelgen_results_md": str(kernelgen_results_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    write_pipeline_results(pipeline_path, pipeline)
    write_second_stage_markdown(kernelgen_results_path, pipeline)
    _atomic_markdown(results_path, lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
