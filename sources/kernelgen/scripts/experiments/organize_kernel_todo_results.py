#!/usr/bin/env python3
"""Collect, render, and verify organized kernel_todo experiment results.

This tool never imports candidate code and never runs an evaluation.  It only
reads BatchSimpleOpt JSON/ledger artifacts and copies the selected code text.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "kernel_todo_v1/result_sources.json"
BAD_WORKLOAD_STATUSES = {
    "RUNTIME_ERROR",
    "TIMEOUT",
    "SUSPECTED_DEVICE_ERROR",
    "SERVER_STATUS_FAILED",
}
INFRA_MARKERS = (
    "no available gateway",
    "server unreachable",
    "authentication",
    "api not support",
    "api 不支持",
    "suspected_device_error",
    "runtime_error",
    "server_status_failed",
    "agent output parse",
)
HOST_SYNC_PATTERN = re.compile(r"\.(?:cpu|numpy|item|tolist)\s*\(")
TORCH_COMPUTE_PATTERN = re.compile(
    r"\btorch\.(?:ops|linalg|nn\.functional|sort|topk|nonzero|matmul|mm|bmm|addmm)\b"
)
FIXED_INPUT_PATTERN = re.compile(
    r"(?:\.shape\s*\[[^\]]+\]|\.numel\s*\(\s*\)|tuple\s*\([^\n]*\.shape\s*\))"
    r"\s*(?:==|!=|in)\s*(?:\d|\(|\[|\{)"
)


class ResultError(RuntimeError):
    """Raised when an artifact cannot be accepted as a completed result."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ResultError(f"{path}: expected a JSON object")
    return value


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ResultError(f"artifact is outside repository: {path}") from exc


def _operator_name(definition_name: str) -> str:
    return re.sub(r"^flaggems_", "", definition_name).lstrip("_")


def _selected_round(ledger: dict[str, Any], code_kind: str) -> dict[str, Any]:
    rounds = ledger.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ResultError(f"{ledger.get('definition_name')}: ledger has no rounds")
    if code_kind == "last":
        return rounds[-1]
    if code_kind != "best":
        raise ResultError(f"unsupported code_kind: {code_kind}")
    best_round = ledger.get("best_round")
    selected = next(
        (item for item in rounds if item.get("round_num") == best_round), None
    )
    if selected is None:
        raise ResultError(
            f"{ledger.get('definition_name')}: missing best round {best_round}"
        )
    return selected


def _selected_code(ledger: dict[str, Any], code_kind: str) -> str:
    if code_kind == "best":
        code = ledger.get("best_code")
    else:
        code = _selected_round(ledger, code_kind).get("solution", {}).get("code")
    if not isinstance(code, str) or not code.strip():
        raise ResultError(
            f"{ledger.get('definition_name')}: empty {code_kind} candidate code"
        )
    return code.rstrip() + "\n"


def _compact_text(value: Any, limit: int = 420) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _fallback_next_step(operator: str, speed: float | None) -> str:
    if speed is None:
        return "先修复最后一轮未通过的 correctness workload，再讨论性能。"
    if speed >= 0.8:
        return "扩展 shape/dtype workload 复核泛化性，并复测低于几何平均值的 timing workload。"
    if re.search(r"(?:addmm|linear|conv|attention|einsum|linalg)", operator):
        return "对照厂商库基线检查 tile、layout、tensor-core 路径和 shape/dtype 分桶。"
    if re.search(r"(?:hist|nonzero|index|scatter|gather|reduce|pool|median|norm)", operator):
        return "检查归约/原子冲突、访存合并、中间张量和 launch 配置。"
    if re.search(r"(?:broadcast|conj|unsqueeze|unbind|narrow|permute|split)", operator):
        return "先确认 Reference 是否为 view；若语义成本不对称，不应继续比较物化 kernel。"
    return "优先减少 launch 和临时张量开销，再检查向量宽度、访存合并与参数分桶。"


def _low_speed_guidance(
    operator: str, timing: list[dict[str, Any]]
) -> tuple[str, str]:
    """Infer a conservative cause and action from operator class and timing data."""

    worst = min(timing, key=lambda item: float(item["speedup"])) if timing else None
    worst_speed = float(worst["speedup"]) if worst is not None else None
    reference_us = (
        float(worst["reference_latency_ms"]) * 1000
        if worst is not None
        and isinstance(worst.get("reference_latency_ms"), (int, float))
        else None
    )
    spread = (
        max(float(item["speedup"]) for item in timing)
        / max(min(float(item["speedup"]) for item in timing), 1e-12)
        if timing
        else 1.0
    )
    shape_note = (
        "；不同 timing workload 落差较大，也说明单一配置的 shape 泛化不足"
        if spread >= 3.0
        else ""
    )

    if operator in {"broadcast_tensors", "broadcast_to", "conj"}:
        cause = (
            "Reference 很可能只创建 view/stride 或 lazy-conjugate 元数据，而候选启动 "
            "kernel 并物化输出，双方实际工作量不对称"
        )
        strategy = (
            "先核对 alias、stride 和 lazy-conjugate 语义；能返回 view 时直接实现元数据"
            "路径，否则应让 benchmark 两侧都物化后再比较，不再单纯调 Triton kernel"
        )
    elif re.search(r"(?:linalg_eig|linalg_cholesky)", operator):
        cause = (
            "候选的分解算法、分块或同步方式尚未接近厂商 linalg 库，复杂 shape 上的"
            "串行步骤和中间结果开销被放大"
        )
        strategy = (
            "先 profile 定位 panel factorization、更新阶段和同步热点，再按矩阵规模"
            "分别选择 block/tile；若无法接近库算法，应把该实现标为实验性而非继续微调"
        )
    elif re.search(r"(?:addmm|linear|conv|attention)", operator):
        cause = (
            "Reference 使用高度优化的厂商 GEMM/卷积/Attention 路径，当前候选的 tile、"
            "layout、流水深度或并行分解没有覆盖其优势"
        )
        strategy = (
            "按退化最严重的 shape/dtype 单独 profile 和分桶，系统搜索 tile、num_warps、"
            "num_stages、layout 与 tensor-core 路径，并避免额外转置或中间张量"
        )
    elif operator == "histc":
        cause = "直方图更新存在严重原子冲突或串行循环，候选耗时远高于 Reference"
        strategy = (
            "改用分块私有直方图后再归并，降低全局原子竞争；按 bins 和输入规模选择"
            "不同 block，并 profile 确认不是 host sync 或超长循环"
        )
    elif re.search(r"(?:col2im|fractional_max_pool|upsample)", operator):
        cause = (
            "索引映射、重复计算或 scatter/gather 访存主导，当前实现没有复用数据，"
            "且可能产生原子竞争或过多中间步骤"
        )
        strategy = (
            "按输出 tile 重排工作，复用坐标计算并合并访存；对重叠写入采用分段归约，"
            "同时针对边界与常见 shape 建立独立配置"
        )
    elif re.search(r"(?:gcd|lcm)", operator):
        cause = "逐元素整数迭代存在分支发散，最慢元素决定 warp 执行时间"
        strategy = (
            "减少 Euclid 迭代中的除法和分支，按 dtype/数值范围选择算法，并检查"
            "in-place 写回与向量化是否引入额外 kernel 或同步"
        )
    elif operator == "narrow_copy":
        cause = "这是小规模纯拷贝路径，Reference 已接近低延迟复制，候选主要受 launch 和低占用影响"
        strategy = "按连续性和拷贝长度分桶，合并索引计算，针对小张量减少 program 数并提高向量宽度"
    elif reference_us is not None and reference_us <= 5.0:
        cause = (
            f"最弱 workload 的 Reference 仅约 {reference_us:.3f} μs，候选主要受 "
            "kernel launch、Python 调度和低占用影响"
        )
        strategy = (
            "合并 kernel/中间张量并缩短发射路径；小 shape 使用更少 program 和更宽向量，"
            "必要时单独建立小张量 fast path"
        )
    else:
        cause = (
            "当前 elementwise/数据移动实现仍受数学指令吞吐、访存效率、launch 或"
            "in-place 写回开销限制"
        )
        strategy = (
            "对最弱 workload 做 profile，分别检查指令、带宽和 launch 占比；再调整"
            "向量宽度、program 数、num_warps，并按 shape/dtype 建立参数分桶"
        )

    if worst_speed is not None:
        cause += f"；最弱 workload 只有 {worst_speed:.3f}x{shape_note}"
    return cause + "。", strategy + "。"


def _initial_review(
    operator: str,
    ledger: dict[str, Any],
    output: dict[str, Any],
    code_kind: str,
    speed: float | None,
) -> tuple[dict[str, str], str, str, str]:
    """Build a reproducible, conservative first-pass review from saved artifacts."""

    selected = _selected_round(ledger, code_kind)
    evaluation = selected.get("evaluation", {})
    code = _selected_code(ledger, code_kind)
    findings: list[str] = []
    severe = False

    if evaluation.get("is_hack"):
        reason = _compact_text(evaluation.get("hack_reason"), 180)
        findings.append(f"Server 标记 is_hack=true{f'（{reason}）' if reason else ''}")
        severe = True
    if "gen_inputs" in code or re.search(r"(?:corr|time)-\d+", code):
        findings.append("代码疑似引用 workload/输入生成细节")
        severe = True
    if TORCH_COMPUTE_PATTERN.search(code):
        findings.append("发现 Torch 计算 API，需排除框架 fallback")
        severe = True
    if HOST_SYNC_PATTERN.search(code):
        findings.append("发现 host sync/host materialization")
        severe = True
    if FIXED_INPUT_PATTERN.search(code):
        findings.append("发现固定 shape/numel 分支，需扩展 workload 复核")

    if severe:
        review_status = "需人工复核"
    elif findings:
        review_status = "泛化风险"
    else:
        review_status = "未发现明显风险"
        findings.append("未发现 workload 标识、Torch 目标计算、host sync 或固定 shape 等式")
    hack_state = "true" if evaluation.get("is_hack") else "false"
    review = {
        "status": review_status,
        "detail": f"Server is_hack={hack_state}；" + "；".join(findings) + "。",
    }

    conclusion = selected.get("conclusion") or {}
    evaluated_at = str(evaluation.get("evaluated_at") or "")
    tested_at = evaluated_at[:10] if re.match(r"\d{4}-\d{2}-\d{2}", evaluated_at) else ""
    timing: list[dict[str, Any]] = []
    if output.get("status") == "FAILED":
        correctness = [
            item
            for item in evaluation.get("workloads", [])
            if isinstance(item, dict) and item.get("phase") == "correctness"
        ]
        failed = [item for item in correctness if item.get("status") != "PASSED"]
        failed_text = ", ".join(
            f"{item.get('uuid')}={item.get('status')}"
            f"（abs={item.get('abs_err')}, rel={item.get('rel_err')}）"
            for item in failed[:3]
        ) or "ledger 未列出具体失败 workload"
        analysis = (
            f"最后一轮 correctness 通过 "
            f"{len(correctness) - len(failed)}/{len(correctness)}；{failed_text}。"
        )
    else:
        timing = [
            item
            for item in evaluation.get("workloads", [])
            if isinstance(item, dict)
            and item.get("phase") == "timing"
            and isinstance(item.get("speedup"), (int, float))
        ]
        worst = min(timing, key=lambda item: float(item["speedup"])) if timing else None
        analysis = (
            f"best round {selected.get('round_num')} 全部通过，几何平均 "
            f"{float(speed):.3f}x。"
        )
        if worst is not None:
            candidate_us = (
                float(worst["latency_ms"]) * 1000
                if isinstance(worst.get("latency_ms"), (int, float))
                else None
            )
            reference_us = (
                float(worst["reference_latency_ms"]) * 1000
                if isinstance(worst.get("reference_latency_ms"), (int, float))
                else None
            )
            latency_text = (
                f"，candidate/reference={candidate_us:.3f}/{reference_us:.3f} μs"
                if candidate_us is not None and reference_us is not None
                else ""
            )
            analysis += (
                f" 最低 timing 为 {worst.get('uuid')} "
                f"{float(worst['speedup']):.3f}x{latency_text}。"
            )

    low_speed_strategy = ""
    if speed is not None and speed < 0.8:
        possible_cause, low_speed_strategy = _low_speed_guidance(operator, timing)
        analysis += f" 可能原因（基于现有 timing，尚未 profile 验证）：{possible_cause}"

    coder_analysis = _compact_text(
        conclusion.get("perf_gap_analysis") or conclusion.get("root_cause"), 360
    )
    if coder_analysis:
        analysis += f" Coder 轮次归因：{coder_analysis}"
    coder_next_step = _compact_text(conclusion.get("next_suggestion"), 320)
    if low_speed_strategy:
        next_step = f"改进策略：{low_speed_strategy}"
        if coder_next_step:
            next_step += f" Coder 建议：{coder_next_step}"
    elif coder_next_step:
        next_step = f"Coder 建议：{coder_next_step}"
    else:
        next_step = _fallback_next_step(operator, speed)
    return review, _compact_text(analysis, 850), next_step, tested_at


def _normal_completion(
    output: dict[str, Any], ledger: dict[str, Any]
) -> tuple[bool, str]:
    status = output.get("status")
    output_rounds = output.get("rounds")
    rounds = ledger.get("rounds")
    if not isinstance(output_rounds, int) or output_rounds <= 0:
        return False, f"workflow rounds={output_rounds!r}"
    if not isinstance(rounds, list) or len(rounds) != output_rounds:
        return False, "output/ledger round count mismatch"

    summary = str(output.get("summary", "")).lower()
    marker = next((item for item in INFRA_MARKERS if item in summary), None)
    if marker is not None:
        return False, f"infrastructure/API marker: {marker}"

    last_evaluation = rounds[-1].get("evaluation", {})
    workload_statuses = {
        str(item.get("status"))
        for item in last_evaluation.get("workloads", [])
        if isinstance(item, dict)
    }
    bad = sorted(workload_statuses & BAD_WORKLOAD_STATUSES)
    if bad:
        return False, f"last evaluation contains {','.join(bad)}"

    if status == "PASSED":
        if not output.get("best_code") or not ledger.get("best_code"):
            return False, "PASSED result has no best code"
        if not isinstance(ledger.get("best_geo_mean"), (int, float)):
            return False, "PASSED result has no numeric best_geo_mean"
        return True, "completed with a valid best candidate"

    if status == "FAILED":
        if last_evaluation.get("status") != "PARTIAL_PASS":
            return False, f"FAILED final evaluation={last_evaluation.get('status')!r}"
        try:
            _selected_code(ledger, "last")
        except ResultError as exc:
            return False, str(exc)
        return True, "completed with no fully correct candidate"

    return False, f"unsupported workflow status={status!r}"


def _validate_entry(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    ledger_path = _repo_path(entry["ledger_path"])
    output_path = _repo_path(entry["output_path"])
    if not ledger_path.is_file():
        raise ResultError(f"missing ledger: {ledger_path}")
    if not output_path.is_file():
        raise ResultError(f"missing workflow output: {output_path}")

    ledger = _read_json(ledger_path)
    output = _read_json(output_path)
    normal, reason = _normal_completion(output, ledger)
    if not normal:
        raise ResultError(f"{entry['operator']}: not a normal completion: {reason}")

    actual_operator = _operator_name(str(ledger.get("definition_name", "")))
    if actual_operator.rstrip("_") != str(entry["operator"]).rstrip("_"):
        raise ResultError(
            f"operator mismatch: manifest={entry['operator']} ledger={actual_operator}"
        )
    if len(ledger["rounds"]) != entry["rounds"]:
        raise ResultError(
            f"{entry['operator']}: rounds mismatch "
            f"{entry['rounds']} != {len(ledger['rounds'])}"
        )

    code_kind = entry["code_kind"]
    if output["status"] == "PASSED" and code_kind != "best":
        raise ResultError(f"{entry['operator']}: PASSED result must use best code")
    if output["status"] == "FAILED" and code_kind != "last":
        raise ResultError(f"{entry['operator']}: FAILED result must use last code")

    actual_speed = ledger.get("best_geo_mean")
    expected_speed = entry.get("speedup")
    if expected_speed is None:
        if output["status"] != "FAILED":
            raise ResultError(f"{entry['operator']}: missing speedup for PASSED result")
    elif not isinstance(actual_speed, (int, float)) or round(
        float(actual_speed), 3
    ) != round(float(expected_speed), 3):
        raise ResultError(
            f"{entry['operator']}: speed mismatch {expected_speed} != {actual_speed}"
        )
    return ledger, output, _selected_code(ledger, code_kind)


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _relative_link(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target, from_dir)).as_posix()


def _render_device(
    device: str, config: dict[str, Any], *, write: bool
) -> tuple[int, int, int]:
    device_dir = REPO_ROOT / "kernel_todo" / device
    code_dir = device_dir / "codes"
    if write:
        code_dir.mkdir(parents=True, exist_ok=True)

    result_lines = [
        f"# {config['display_name']} 已完成优化结果",
        "",
        "<!-- 由 scripts/experiments/organize_kernel_todo_results.py 根据 result_sources.json 生成。 -->",
        "",
        "这里只收录工作流正常结束的 BatchSimpleOpt 任务。API 不支持、网关、鉴权、",
        "设备错误或 Agent 解析错误导致的中断任务不进入本目录。",
        "",
        f"- 计时：{config['timing']}",
        "- 达标线：最佳几何平均加速比不低于 `0.8x`。",
        "- `best` 是全部 workload 正确的最佳候选；`last` 是正常跑满但没有全量正确",
        "  候选时的最后一轮代码。",
        "- 原始 `runs/` 是本地实验产物，链接用于定位证据，不保证随 Git 分发。",
        "",
        "| 算子 | 结果 | 加速比 | 轮数 | 模型 / 日期 | 代码 | 原始记录 |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    analysis_lines = [
        f"# {config['display_name']} best/last code 静态初析",
        "",
        "<!-- 由 scripts/experiments/organize_kernel_todo_results.py 根据 result_sources.json 生成。 -->",
        "",
        "这是基于 ledger 和代码文本的初步审查，没有重新运行候选。`Server is_hack=false`",
        "只表示当时策略未命中；本页额外标出 workload 硬编码、Torch 计算参与、host",
        "sync 和泛化风险。低于 `0.8x` 的条目根据 timing 和算子结构给出可能原因与",
        "改进策略，但没有 profile 证据时只视为待验证假设。后续采纳代码前仍应扩展",
        "workload 复核。",
        "",
        "| 算子 | hack / 泛化审查 | 初步性能或失败原因 | 后续方向 |",
        "| --- | --- | --- | --- |",
    ]

    qualified = below = no_best = 0
    expected_codes: set[str] = set()
    seen_operators: set[str] = set()
    for entry in config["results"]:
        operator = entry["operator"]
        if operator in seen_operators:
            raise ResultError(f"{device}: duplicate operator in manifest: {operator}")
        seen_operators.add(operator)
        ledger, output, code = _validate_entry(entry)
        code_path = code_dir / f"{operator}.py"
        expected_codes.add(code_path.name)
        if write:
            code_path.write_text(code)
        elif not code_path.is_file() or code_path.read_text() != code:
            raise ResultError(f"{device}/{operator}: organized code differs from ledger")

        speed = entry.get("speedup")
        if speed is None:
            no_best += 1
            speed_text = "—"
            code_label = "last（未通过）"
        else:
            speed = float(speed)
            speed_text = f"{speed:.3f}x"
            code_label = "best"
            if speed >= 0.8:
                qualified += 1
            else:
                below += 1

        ledger_path = _repo_path(entry["ledger_path"])
        output_path = _repo_path(entry["output_path"])
        result_lines.append(
            f"| `{operator}` | {_markdown_escape(entry['result'])} | {speed_text} | "
            f"{entry['rounds']} | `{_markdown_escape(entry['model'])}` / "
            f"{entry['tested_at']} | "
            f"[{code_label}]({_relative_link(device_dir, code_path)}) | "
            f"[ledger]({_relative_link(device_dir, ledger_path)}) / "
            f"[output]({_relative_link(device_dir, output_path)}) |"
        )
        review = entry["review"]
        analysis_lines.append(
            f"| [`{operator}`]({_relative_link(device_dir, code_path)}) | "
            f"**{_markdown_escape(review['status'])}**："
            f"{_markdown_escape(review['detail'])} | "
            f"{_markdown_escape(entry['analysis'])} | "
            f"{_markdown_escape(entry['next_step'])} |"
        )

    total = len(config["results"])
    result_lines[7:7] = [
        f"本目录共整理 **{total}** 个正常完成任务：达标 **{qualified}**、未达标",
        f"**{below}**、无全量正确候选 **{no_best}**。",
        "",
    ]
    overview = "\n".join(
        [
            f"# {config['display_name']}",
            "",
            f"正常完成任务：**{total}**；达到 `0.8x`：**{qualified}**；低于 `0.8x`：",
            f"**{below}**；无全量正确候选：**{no_best}**。",
            "",
            "- [Reference / Profiler 验证状态](reference_status.md)",
            "- [正常完成的优化结果与原始记录](results.md)",
            "- [best/last code 静态初析](analysis.md)",
            "- [`codes/`](codes/)：逐算子 best code；无正确 best 时保存最后候选。",
            "",
        ]
    )
    if write:
        (device_dir / "results.md").write_text("\n".join(result_lines) + "\n")
        (device_dir / "analysis.md").write_text("\n".join(analysis_lines) + "\n")
        (device_dir / "README.md").write_text(overview)
        for stale_code in code_dir.glob("*.py"):
            if stale_code.name not in expected_codes:
                stale_code.unlink()
    else:
        for filename in ("results.md", "analysis.md", "README.md", "reference_status.md"):
            if not (device_dir / filename).is_file():
                raise ResultError(f"{device}: missing {filename}")

    actual_codes = {path.name for path in code_dir.glob("*.py")}
    if actual_codes != expected_codes:
        raise ResultError(
            f"{device}: code inventory mismatch; "
            f"missing={sorted(expected_codes - actual_codes)}, "
            f"extra={sorted(actual_codes - expected_codes)}"
        )
    return qualified, below, no_best


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if manifest.get("schema_version") != "1.0":
        raise ResultError(f"{path}: unsupported schema_version")
    devices = manifest.get("devices")
    if not isinstance(devices, dict) or not devices:
        raise ResultError(f"{path}: devices must be a non-empty object")
    return manifest


def command_render(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    totals = [0, 0, 0]
    for device, config in manifest["devices"].items():
        stats = _render_device(device, config, write=True)
        totals = [left + right for left, right in zip(totals, stats)]
    print(
        f"rendered {sum(totals)} results: qualified={totals[0]} "
        f"below={totals[1]} no_best={totals[2]}"
    )
    return 0


def command_verify(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest)
    totals = [0, 0, 0]
    for device, config in manifest["devices"].items():
        stats = _render_device(device, config, write=False)
        totals = [left + right for left, right in zip(totals, stats)]
    print(
        f"verified {sum(totals)} results: qualified={totals[0]} "
        f"below={totals[1]} no_best={totals[2]}"
    )
    return 0


def _scan_definition(
    ledger_path: Path, model: str, tested_at: str
) -> tuple[dict[str, Any] | None, str]:
    output_path = ledger_path.parent / "optimize_definition_output.json"
    if not output_path.is_file():
        return None, "missing optimize_definition_output.json"
    ledger = _read_json(ledger_path)
    output = _read_json(output_path)
    normal, reason = _normal_completion(output, ledger)
    if not normal:
        return None, reason

    workflow_status = output["status"]
    speed = float(ledger["best_geo_mean"]) if workflow_status == "PASSED" else None
    result = (
        "达标"
        if speed is not None and speed >= 0.8
        else "未达标"
        if speed is not None
        else "未通过（无全量正确候选）"
    )
    operator = _operator_name(str(ledger["definition_name"]))
    code_kind = "best" if workflow_status == "PASSED" else "last"
    review, analysis, next_step, evaluated_at = _initial_review(
        operator, ledger, output, code_kind, speed
    )
    entry = {
        "operator": operator,
        "result": result,
        "speedup": speed,
        "rounds": len(ledger["rounds"]),
        "model": model,
        "tested_at": evaluated_at if tested_at == "auto" and evaluated_at else tested_at,
        "code_kind": code_kind,
        "ledger_path": _relative_to_repo(ledger_path),
        "output_path": _relative_to_repo(output_path),
        "review": review,
        "analysis": analysis,
        "next_step": next_step,
    }
    return entry, reason


def command_scan(args: argparse.Namespace) -> int:
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    by_operator: dict[str, list[str]] = {}
    roots = [
        run_root if run_root.is_absolute() else REPO_ROOT / run_root
        for run_root in args.run_root
    ]
    for root in roots:
        if not root.is_dir():
            raise ResultError(f"run root does not exist: {root}")
        definition_dirs = {
            path.parent for path in root.rglob(".ledger.json")
        } | {
            path.parent for path in root.rglob("optimize_definition_output.json")
        }
        for definition_dir in sorted(definition_dirs):
            ledger_path = definition_dir / ".ledger.json"
            if not ledger_path.is_file():
                output_path = definition_dir / "optimize_definition_output.json"
                detail = "missing .ledger.json"
                if output_path.is_file():
                    output = _read_json(output_path)
                    detail += (
                        f"; status={output.get('status')!r}, "
                        f"rounds={output.get('rounds')!r}, "
                        f"summary={str(output.get('summary', ''))[:200]!r}"
                    )
                excluded.append(
                    {
                        "artifact_path": _relative_to_repo(definition_dir),
                        "reason": detail,
                    }
                )
                continue
            try:
                entry, reason = _scan_definition(
                    ledger_path, args.model, args.tested_at
                )
            except (KeyError, TypeError, ValueError, ResultError) as exc:
                entry, reason = None, str(exc)
            if entry is None:
                excluded.append(
                    {
                        "artifact_path": _relative_to_repo(ledger_path),
                        "reason": reason,
                    }
                )
                continue
            included.append(entry)
            by_operator.setdefault(entry["operator"], []).append(entry["ledger_path"])

    duplicates = {
        operator: paths for operator, paths in by_operator.items() if len(paths) > 1
    }
    payload = {
        "schema_version": "1.0",
        "device": args.device,
        "run_roots": [_relative_to_repo(path) for path in roots],
        "included": included,
        "excluded": excluded,
        "duplicates": duplicates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        f"scan included={len(included)} excluded={len(excluded)} "
        f"duplicate_operators={len(duplicates)} -> {args.output}"
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(func=None)
    subparsers = parser.add_subparsers(dest="command")

    for name, func in (("render", command_render), ("verify", command_verify)):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--manifest", type=Path, default=DEFAULT_MANIFEST
        )
        command.set_defaults(func=func)

    scan = subparsers.add_parser("scan")
    scan.add_argument("--device", required=True)
    scan.add_argument("--run-root", type=Path, action="append", required=True)
    scan.add_argument("--model", required=True)
    scan.add_argument("--tested-at", required=True)
    scan.add_argument("--output", type=Path, required=True)
    scan.set_defaults(func=command_scan)

    args = parser.parse_args()
    if args.func is None:
        parser.print_help()
        parser.exit(2)
    return args


def main() -> int:
    try:
        args = _parse_args()
        return int(args.func(args))
    except ResultError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
