#!/usr/bin/env python3
"""Bootstrap per-chip Kernel Todo V2 result ledgers from verified evidence.

The script imports only same-chip V1 FlagGems Adapter results, optionally adds
per-chip V2 baseline and inspect summaries, copies the selected V1 candidate
files, and renders one complete ``results.md`` per chip.  It does not execute
candidate code or infer target-device support from another chip.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CHIP_LABELS = {
    "haiguang": "海光",
    "huawei": "华为",
    "kunlunxin": "昆仑芯",
    "moer": "摩尔",
    "muxi": "沐曦",
    "nvidia": "英伟达",
    "pingtouge": "平头哥",
    "suiyuan": "燧原",
    "tianshu": "天数",
}
PPU_BASELINE_FAILURES = {
    "linalg_ldl_factor_ex": (
        "PyTorch reference 调用的 cusolverDnSsytrf_bufferSize 不受 PPU HGGC 支持，进程退出。",
        "等待厂商补齐 LDL factorization 能力后重跑完整 core baseline。",
    ),
    "linalg_ldl_factor": (
        "PyTorch reference 调用的 cusolverDnSsytrf_bufferSize 不受 PPU HGGC 支持，进程退出。",
        "等待厂商补齐 LDL factorization 能力后重跑完整 core baseline。",
    ),
    "linalg_ldl_solve": (
        "构造 LDL 输入时触发 PPU HGGC 不支持的 factorization 路径。",
        "等待厂商补齐 LDL factorization 能力后重跑完整 core baseline。",
    ),
    "rnn_relu": (
        "PyTorch reference 构造 RNN 时返回 ACDNN_STATUS_NOT_SUPPORTED，目标运行时不支持 RNN mode 0。",
        "等待厂商补齐对应 RNN mode，或由验收方确认该算子不属于当前硬件能力范围。",
    ),
}


class BootstrapError(RuntimeError):
    """Raised when source evidence is inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise BootstrapError(f"{path}: expected a JSON object")
    return value


def _compact(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _markdown_cell(value: Any) -> str:
    return _compact(value).replace("|", "\\|") or "—"


def _low_or_failure_reason(result: dict[str, Any]) -> str:
    analysis = " ".join(str(result.get("analysis") or "").split())
    speedup = result.get("speedup")
    if speedup is None:
        return _compact(analysis.split("Coder 轮次归因：", 1)[0])
    marker = "可能原因（基于现有 timing，尚未 profile 验证）："
    if marker in analysis:
        return _compact(analysis.split(marker, 1)[1].split("Coder 轮次归因：", 1)[0])
    return _compact(
        f"候选已通过正确性，但几何平均加速比仅 {float(speedup):.3f}x；现有记录没有经过 profile 验证的单一根因。"
    )


def _next_step(result: dict[str, Any]) -> str:
    value = " ".join(str(result.get("next_step") or "").split())
    if value.startswith("改进策略："):
        value = value[len("改进策略：") :].split("Coder 建议：", 1)[0]
    elif value.startswith("Coder 建议："):
        value = value[len("Coder 建议：") :]
    return _compact(value) or "在 V2 pytest 下复验后，根据失败 workload 或 profile 结果继续处理。"


def _hack_summary(result: dict[str, Any]) -> str:
    review = result.get("review") or {}
    status = str(review.get("status") or "未检查")
    detail = " ".join(str(review.get("detail") or "").split())
    if status == "未发现明显风险":
        return "未发现（V1 静态审查，Server is_hack=false）"
    return _compact(f"{status}：{detail}" if detail else status)


def _parse_chip_path(value: str) -> tuple[str, Path]:
    chip, separator, raw_path = value.partition("=")
    if not separator or chip not in CHIP_LABELS or not raw_path:
        raise argparse.ArgumentTypeError("expected <chip>=<json-path>")
    return chip, Path(raw_path)


def _inventory_by_chip(root: Path) -> dict[str, list[dict[str, Any]]]:
    payload = _load_json(root / "kernel_todo_v2/pytest_conversion_inventory.json")
    result = {chip: [] for chip in CHIP_LABELS}
    for entry in payload.get("operators", []):
        for chip in entry.get("chips", []):
            if chip in result:
                result[chip].append(entry)
    for chip, entries in result.items():
        source = root / f"kernel_todo_v2/{chip}/failed_ops.txt"
        expected = [
            line.strip() for line in source.read_text().splitlines() if line.strip()
        ]
        actual = [str(entry["source_operator"]) for entry in entries]
        if actual != expected:
            raise BootstrapError(f"{chip}: inventory order differs from failed_ops.txt")
    return result


def _v1_results(root: Path, chip: str) -> dict[str, dict[str, Any]]:
    source = root / f"kernel_todo_v1/{chip}-flaggems-adapter/result_sources.json"
    if not source.is_file():
        return {}
    payload = _load_json(source)
    devices = payload.get("devices") or {}
    if len(devices) != 1:
        raise BootstrapError(f"{source}: expected exactly one device")
    rows = next(iter(devices.values())).get("results") or []
    return {str(row["operator"]): row for row in rows}


def _baseline_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _load_json(path)
    return {str(row["operator"]): row for row in payload.get("rows", [])}


def _inspect_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = _load_json(path)
    return {str(row["source_operator"]): row for row in payload.get("results", [])}


def _copy_v1_code(root: Path, chip: str, operator: str, target: Path) -> None:
    source = root / f"kernel_todo_v1/{chip}-flaggems-adapter/codes/{operator}.py"
    if not source.is_file():
        raise BootstrapError(f"missing V1 candidate: {source}")
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise BootstrapError(f"refusing to overwrite non-matching candidate: {target}")
    shutil.copyfile(source, target)


def _render_chip(
    root: Path,
    chip: str,
    entries: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    inspect: dict[str, dict[str, Any]],
    snapshot_date: str,
) -> tuple[int, int, int, int, int, int]:
    device_dir = root / f"kernel_todo_v2/{chip}"
    code_dir = device_dir / "codes"
    code_dir.mkdir(parents=True, exist_ok=True)
    old_results = _v1_results(root, chip)
    rows: list[dict[str, Any]] = []
    expected_codes: set[str] = set()

    for entry in entries:
        source_operator = str(entry["source_operator"])
        operator = str(entry["operator"])
        old = old_results.get(operator)
        base = baseline.get(source_operator)
        inspected = inspect.get(source_operator)
        reference_passed = bool(base and base.get("reference_verified"))
        timing_passed = bool(base and base.get("gems_timing_valid"))
        if old is not None:
            reference_passed = True
            timing_passed = True

        optimization_status = "未跑"
        reference_status = "通过" if reference_passed else "未验证"
        timing_status = "通过" if timing_passed else "未验证"
        speedup: float | None = None
        hack = "未检查"
        reason = "—"
        direction = "先在目标芯片跑完整 core baseline；Reference 与 Gems 可计时均通过后启动 BatchSimpleOpt。"
        code_path: str | None = None

        if old is not None:
            speedup_value = old.get("speedup")
            speedup = (
                float(speedup_value)
                if isinstance(speedup_value, (int, float))
                else None
            )
            optimization_status = (
                "成功"
                if speedup is not None and speedup >= 0.8
                else "待二阶段优化"
                if speedup is not None
                else "失败"
            )
            hack = _hack_summary(old)
            reason = (
                "—"
                if speedup is not None and speedup >= 0.8
                else _low_or_failure_reason(old)
            )
            direction = (
                "在 V2 pytest 下复验历史候选；通过后可直接复用。"
                if speedup is not None and speedup >= 0.8
                else _next_step(old).rstrip("。；")
                + "；修改后必须重新执行 V2 全量 correctness 与 timing。"
            )
            code_path = f"codes/{operator}.py"
            expected_codes.add(f"{operator}.py")
            _copy_v1_code(root, chip, operator, code_dir / f"{operator}.py")
        elif base is not None and not reference_passed:
            if chip == "pingtouge" and source_operator == "addmv_":
                timing_status = "无法计时"
                reason = "core benchmark 与独立复验均达到单阶段 850 秒上限，没有生成完整 benchmark JSON。"
                direction = "先用单 case 定位超时位置；修复计时链或确认合理上限后重跑完整 core baseline。"
            else:
                reference_status = "未通过"
                reason, direction = PPU_BASELINE_FAILURES.get(
                    source_operator,
                    (
                        "目标芯片 baseline 没有产生有效的 reference timing。",
                        "定位 reference 失败原因并重跑完整 core baseline。",
                    ),
                )
        elif base is not None and reference_passed and timing_passed:
            if inspected and inspected.get("status") == "failed":
                reason = "V2 baseline 已通过，但 /inspect 无法从旧式 benchmark 枚举稳定 Workload。"
                direction = "先把 benchmark 迁移到两阶段 Case API，使 /inspect 可枚举并重放 Workload。"
            else:
                direction = "V2 baseline 与 /inspect 已通过；待启动 BatchSimpleOpt。"

        rows.append(
            {
                "source_operator": source_operator,
                "operator": operator,
                "optimization_status": optimization_status,
                "reference_status": reference_status,
                "timing_status": timing_status,
                "speedup": speedup,
                "reference_passed": reference_passed,
                "timing_passed": timing_passed,
                "hack": hack,
                "reason": reason,
                "direction": direction,
                "code_path": code_path,
            }
        )

    actual_codes = {path.name for path in code_dir.glob("*.py")}
    unexpected_codes = actual_codes - expected_codes
    if unexpected_codes:
        raise BootstrapError(
            f"{chip}: refusing to remove unexpected V2 candidates: "
            f"{sorted(unexpected_codes)}"
        )
    keep = code_dir / ".gitkeep"
    if expected_codes and keep.exists():
        keep.unlink()
    elif not expected_codes:
        keep.touch(exist_ok=True)

    total = len(rows)
    reference_count = sum(row["reference_passed"] for row in rows)
    timing_count = sum(
        row["reference_passed"] and row["timing_passed"] for row in rows
    )
    first_stage_qualified = sum(
        row["optimization_status"] == "成功"
        and row["speedup"] is not None
        and row["speedup"] >= 0.8
        for row in rows
    )
    pending_second_stage = sum(
        row["optimization_status"] == "待二阶段优化" for row in rows
    )
    failed = sum(row["optimization_status"] == "失败" for row in rows)

    lines = [
        f"# {CHIP_LABELS[chip]} Kernel Todo V2 结果",
        "",
        (
            f"本页是 {snapshot_date} 的保守快照。V1 同芯片、同 canonical operator "
            "的历史结果已迁入并复制候选代码，但在 V2 pytest 下复验前不视为 V2 新实验"
            "终态；跨芯片结果不复用。`reference通过数` 统计 Reference 独立通过的算子，"
            "`Gems 可计时数` 只统计 Reference 与 Gems 可计时同时通过的算子；两项都只"
            "统计已有目标芯片证据的算子，未知项保持“未跑”。"
        ),
        "",
        "## 汇总",
        "",
        "| 总算子数 | reference通过数 | Gems 可计时数 | 第一阶段达标数 | "
        "待二阶段优化数 | 第二阶段达标数 | 最终失败数 | 阻塞数 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {total} | {reference_count} | {timing_count} | "
        f"{first_stage_qualified} | {pending_second_stage} | 0 | {failed} | 0 |",
        "",
        (
            "字段口径：逐算子 `Reference` 和 `Gems 可计时` 两列分别记录 baseline 是否可"
            "执行、是否取得完整有效计时；汇总表采用漏斗口径，`Gems 可计时数` 要求两列"
            "同时通过。`优化状态=成功` 表示候选已达到 0.8；全量正确且可计时但低于 "
            "0.8 的候选写为`待二阶段优化`；`失败`表示已运行优化但没有全量正确候选；"
            "`未跑`表示尚无 SimpleOpt 结果。"
        ),
        "",
        "## 逐算子结果",
        "",
        "| 算子名 | Reference | Gems 可计时 | 优化状态 | 加速比 | "
        "Hack 情况 | 失败/低加速比/阻塞原因 | 后续方向 | code_path |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        source_operator = str(row["source_operator"])
        operator = str(row["operator"])
        name = (
            f"`{source_operator}`"
            if source_operator == operator
            else f"`{source_operator}`（`{operator}`）"
        )
        speed = f"{row['speedup']:.3f}x" if row["speedup"] is not None else "—"
        code = f"[{row['code_path']}]({row['code_path']})" if row["code_path"] else "—"
        lines.append(
            f"| {name} | {row['reference_status']} | {row['timing_status']} | "
            f"{row['optimization_status']} | {speed} | "
            f"{_markdown_cell(row['hack'])} | {_markdown_cell(row['reason'])} | "
            f"{_markdown_cell(row['direction'])} | {code} |"
        )
    (device_dir / "results.md").write_text("\n".join(lines) + "\n")
    return (
        total,
        reference_count,
        timing_count,
        first_stage_qualified,
        pending_second_stage,
        failed,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--snapshot-date", default=dt.date.today().isoformat())
    parser.add_argument(
        "--baseline-summary",
        action="append",
        default=[],
        type=_parse_chip_path,
        metavar="CHIP=PATH",
    )
    parser.add_argument(
        "--inspect-summary",
        action="append",
        default=[],
        type=_parse_chip_path,
        metavar="CHIP=PATH",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    inventory = _inventory_by_chip(root)
    baseline_paths = dict(args.baseline_summary)
    inspect_paths = dict(args.inspect_summary)
    for chip in CHIP_LABELS:
        stats = _render_chip(
            root,
            chip,
            inventory[chip],
            _baseline_rows(baseline_paths.get(chip)),
            _inspect_rows(inspect_paths.get(chip)),
            args.snapshot_date,
        )
        print(
            f"{chip}: total={stats[0]} reference={stats[1]} timing={stats[2]} "
            f"first_stage_qualified={stats[3]} "
            f"pending_second_stage={stats[4]} failed={stats[5]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
