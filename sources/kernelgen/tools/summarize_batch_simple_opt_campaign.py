"""Summarize one BatchSimpleOpt campaign spread across multiple batch dirs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from kernelgen.workflows.optimization.single_coder import (
    KERNEL_OPTIMIZATION_OUTPUT_FILENAME,
)


@dataclass(frozen=True)
class CampaignResult:
    name: str
    state: str
    rounds: int
    best_geo_mean: Optional[float]
    best_round: Optional[int]
    candidate_latency_ms: Optional[float]
    reference_latency_ms: Optional[float]
    timing_workload_count: int
    last_eval_status: str
    authoritative: bool
    workspace: Path
    source: Path
    batch_dir: Path
    batch_exit_code: Optional[int]


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _read_exit_code(batch_dir: Path) -> Optional[int]:
    try:
        return int((batch_dir / "exit").read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _definition_names(batch_dir: Path) -> list[str]:
    try:
        requested = [
            line.strip()
            for line in (batch_dir / "definitions.txt").read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]
    except OSError:
        requested = []
    definitions_dir = batch_dir / "workspace" / "definitions"
    discovered = (
        [
            path.name
            for path in sorted(definitions_dir.iterdir())
            if path.is_dir()
        ]
        if definitions_dir.is_dir()
        else []
    )
    return list(dict.fromkeys([*requested, *discovered]))


def load_batch(batch_dir: Path) -> list[CampaignResult]:
    batch_dir = batch_dir.expanduser().resolve()
    exit_code = _read_exit_code(batch_dir)
    results = []
    for name in _definition_names(batch_dir):
        workspace = batch_dir / "workspace" / "definitions" / name
        output_path = workspace / KERNEL_OPTIMIZATION_OUTPUT_FILENAME
        ledger_path = workspace / ".ledger.json"
        output = _read_json(output_path)
        ledger = _read_json(ledger_path)
        ledger_rounds = ledger.get("rounds", []) if ledger else []
        if not isinstance(ledger_rounds, list):
            ledger_rounds = []

        if output:
            state = str(output.get("status", "DONE"))
            rounds = int(output.get("rounds", len(ledger_rounds)))
            best_geo_mean = output.get("best_geo_mean")
            authoritative = True
            source = output_path
        else:
            state = "RUNNING" if exit_code is None else "INTERRUPTED"
            rounds = len(ledger_rounds)
            best_geo_mean = ledger.get("best_geo_mean") if ledger else None
            authoritative = False
            source = ledger_path

        last_evaluation = (
            ledger_rounds[-1].get("evaluation") if ledger_rounds else None
        )
        last_eval_status = (
            str(last_evaluation.get("status", "-"))
            if isinstance(last_evaluation, dict)
            else "-"
        )
        best_round = ledger.get("best_round") if ledger else None
        best_record = next(
            (
                record
                for record in ledger_rounds
                if record.get("round_num") == best_round
            ),
            None,
        )
        best_evaluation = (
            best_record.get("evaluation")
            if isinstance(best_record, dict)
            else None
        )
        workloads = (
            best_evaluation.get("workloads", [])
            if isinstance(best_evaluation, dict)
            else []
        )
        timing_workloads = [
            workload
            for workload in workloads
            if isinstance(workload, dict)
            and workload.get("phase") == "timing"
            and workload.get("latency_ms") is not None
        ]
        if len(timing_workloads) == 1:
            candidate_latency_ms = timing_workloads[0].get("latency_ms")
            reference_latency_ms = timing_workloads[0].get(
                "reference_latency_ms",
            )
        else:
            candidate_latency_ms = (
                best_evaluation.get("latency_ms")
                if isinstance(best_evaluation, dict)
                else None
            )
            reference_latency_ms = None
        results.append(
            CampaignResult(
                name=name,
                state=state,
                rounds=rounds,
                best_geo_mean=best_geo_mean,
                best_round=best_round,
                candidate_latency_ms=candidate_latency_ms,
                reference_latency_ms=reference_latency_ms,
                timing_workload_count=len(timing_workloads),
                last_eval_status=last_eval_status,
                authoritative=authoritative,
                workspace=workspace,
                source=source,
                batch_dir=batch_dir,
                batch_exit_code=exit_code,
            )
        )
    return results


def collect_campaign(batch_dirs: Iterable[Path]) -> list[CampaignResult]:
    results = []
    seen = set()
    for batch_dir in batch_dirs:
        for result in load_batch(batch_dir):
            if result.name in seen:
                raise ValueError(
                    f"duplicate definition across batch dirs: {result.name}",
                )
            seen.add(result.name)
            results.append(result)
    return results


def _status_label(result: CampaignResult) -> str:
    labels = {
        "PASSED": "完成",
        "FAILED": "失败",
        "RUNNING": "运行中",
        "INTERRUPTED": "中止",
    }
    label = labels.get(result.state, result.state)
    return label if result.authoritative else f"{label}（暂定）"


def _speedup_label(result: CampaignResult) -> str:
    if result.best_geo_mean is None:
        return "—"
    suffix = "" if result.authoritative else "（暂定）"
    return f"{result.best_geo_mean:.6f}x{suffix}"


def _decision_label(result: CampaignResult) -> str:
    if result.best_geo_mean is None:
        return "无有效最佳值"
    if result.state == "FAILED":
        return "失败"
    prefix = "" if result.authoritative else "暂定；"
    if result.best_geo_mean > 1.0:
        return f"{prefix}快于 reference"
    if result.best_geo_mean >= 0.8:
        return f"{prefix}达到 0.8 合格线，但未快于 reference"
    return f"{prefix}未达到 0.8 合格线"


def _latency_label(value: Optional[float], authoritative: bool) -> str:
    if value is None:
        return "—"
    suffix = "" if authoritative else "（暂定）"
    return f"{value:.6f}{suffix}"


def summary_counts(results: list[CampaignResult]) -> dict[str, int]:
    final_passed = [
        result
        for result in results
        if result.authoritative and result.state == "PASSED"
    ]
    return {
        "total": len(results),
        "finalized": sum(result.authoritative for result in results),
        "passed": len(final_passed),
        "accelerated": sum(
            result.best_geo_mean is not None
            and result.best_geo_mean > 1.0
            for result in final_passed
        ),
        "qualified": sum(
            result.best_geo_mean is not None
            and result.best_geo_mean >= 0.8
            for result in final_passed
        ),
        "running": sum(result.state == "RUNNING" for result in results),
        "interrupted": sum(
            result.state == "INTERRUPTED" for result in results
        ),
    }


def render_markdown(
    results: list[CampaignResult],
    generated_at: str,
    title: str,
) -> str:
    counts = summary_counts(results)
    lines = [
        f"# {title}",
        "",
        f"> 自动汇总时间：{generated_at}。加速比和状态直接读取各 workspace "
        "的最终输出或 ledger。",
        "",
        "## 汇总结论",
        "",
        f"- 共 {counts['total']} 个算子；{counts['finalized']} 个已有权威最终输出，"
        f"{counts['running']} 个运行中，{counts['interrupted']} 个中止。",
        f"- 已完成结果中，{counts['accelerated']} 个 `best_geo_mean > 1`，"
        f"{counts['qualified']} 个达到项目的 `best_geo_mean >= 0.8` 合格线。",
        "- `PASSED` 只表示数值正确且计时有效；是否实际加速以 "
        "`best_geo_mean > 1` 为准。",
        "- “暂定”值来自 ledger，未写入最终 "
        "`optimize_definition_output.json`，不计入完成结果统计。",
        "",
        "## 逐算子结果",
        "",
        "| 算子 | 状态 | 轮数 | 最佳轮次 | Reference latency (ms) | "
        "Candidate latency (ms) | 加速比 | 判定 | Workspace |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        best_round = (
            str(result.best_round) if result.best_round is not None else "—"
        )
        workspace = str(result.workspace)
        lines.append(
            f"| `{result.name}` | {_status_label(result)} | "
            f"{result.rounds} | {best_round} | "
            f"{_latency_label(result.reference_latency_ms, result.authoritative)} | "
            f"{_latency_label(result.candidate_latency_ms, result.authoritative)} | "
            f"{_speedup_label(result)} | "
            f"{_decision_label(result)} | "
            f"[workspace]({workspace}) |"
        )

    lines.extend([
        "",
        "## 实验口径",
        "",
        "- Catalog：`akg-bench-lite-v5`；每个算子分别运行 correctness 和 timing "
        "workload。",
        "- Correctness：`torch.allclose` 等价口径，`rtol=0.01`、"
        "`atol=0.01`，要求全部元素匹配。",
        "- 目标硬件：`Ascend910B4-1`；KernelGen Server API `v5.1`；"
        "计时使用严格 `torch_npu.profiler`，没有 wall-time fallback。",
        "- Coder 模型：`deepseek-v4-flash[1m]`。",
        "- 加速比：`best_geo_mean`，即候选实现相对 reference 的 workload "
        "加速比几何平均。",
        "- Latency：取最佳轮次 timing workload 的严格 profiler 结果；本批每个"
        "算子只有一个 timing workload，因此 "
        "`加速比 = reference latency / candidate latency`。",
        "",
        "## 结果来源",
        "",
    ])
    for result in results:
        source_kind = (
            "最终输出" if result.authoritative else "ledger（暂定）"
        )
        lines.append(
            f"- `{result.name}`：[{source_kind}]({result.source})；"
            f"[batch]({result.batch_dir})"
        )
    return "\n".join(lines) + "\n"


def report_json(
    results: list[CampaignResult],
    generated_at: str,
    title: str,
) -> dict[str, Any]:
    return {
        "title": title,
        "generated_at": generated_at,
        "summary": summary_counts(results),
        "results": [
            {
                **asdict(result),
                "workspace": str(result.workspace),
                "source": str(result.source),
                "batch_dir": str(result.batch_dir),
            }
            for result in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-dir",
        action="append",
        required=True,
        type=Path,
        help="BatchSimpleOpt batch directory; repeat in report order",
    )
    parser.add_argument(
        "--title",
        default="BatchSimpleOpt campaign results",
    )
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    try:
        results = collect_campaign(args.batch_dir)
    except ValueError as exc:
        parser.error(str(exc))
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    markdown = render_markdown(results, generated_at, args.title)
    print(markdown, end="")
    if args.markdown_out:
        args.markdown_out.write_text(markdown, encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                report_json(results, generated_at, args.title),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
