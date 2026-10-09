#!/usr/bin/env python3
"""生成 fixed launch-plan 候选的中文重复评测与历史候选对照。"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def timing(result: dict) -> dict[str, dict]:
    return {row["uuid"]: row for row in result["per_workload"] if row["phase"] == "timing"}


def verified(result: dict) -> dict[str, dict]:
    return {row["uuid"]: row for row in result["comparisons"]}


def speedup(row: dict) -> float:
    after = row["after"]
    return after["reference_latency_ms"] / after["latency_ms"]


def render(fixed: Path, prior: Path, candidate: Path) -> str:
    source_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    runs = [read(fixed / name) for name in ("eval.json", "eval-02.json", "eval-03.json")]
    for run in runs:
        if (run.get("status") != "PASSED" or run.get("num_passed") != 33
                or run.get("num_workloads") != 33 or run.get("is_hack")
                or run.get("candidate_sha256") != source_hash):
            raise ValueError("重复评测没有通过 33/33，或候选源码哈希不一致")
    new_cases = [timing(run) for run in runs]
    if len(new_cases[0]) != 15 or any(set(rows) != set(new_cases[0]) for rows in new_cases):
        raise ValueError("三次运行的 15 个 timing UUID 不一致")
    if any(row["status"] != "PASSED" for rows in new_cases for row in rows.values()):
        raise ValueError("存在未通过的 timing case")
    old = {
        "无 profiler A/B": read(prior / "ab-no-profile-final-verification.json"),
        "有 profiler A/B": read(prior / "ab-profile-final-verification.json"),
        "早期 launcher 缓存": read(prior / "profiled-final-verification.json"),
    }
    old_cases = {name: verified(result) for name, result in old.items()}
    if any(result["status"] != "PASSED" or set(old_cases[name]) != set(new_cases[0])
           for name, result in old.items()):
        raise ValueError("历史独立复验的 15 个 case 或状态不匹配")
    geo = [run["geo_mean"] for run in runs]
    lines = ["# narrow_copy launch-plan 修复候选", "",
             f"候选源码 SHA-256：`{source_hash}`。三次 KGS `/evaluate` 都通过 18/18 correctness、"
             "15/15 timing，`is_hack=false`。", "",
             "| 次数 | 设备 | geo mean vs PyTorch |",
             "| --- | --- | ---: |"]
    for index, run in enumerate(runs, 1):
        lines.append(f"| {index} | {run['device']} | {run['geo_mean']:.4f}× |")
    lines.extend(["", f"三次 geo mean 的算术平均为 **{statistics.mean(geo):.4f}×**，"
                  f"范围 **{min(geo):.4f}–{max(geo):.4f}×**，样本标准差 "
                  f"**{statistics.stdev(geo):.4f}**。这是三次不同设备上的独立运行，"
                  "不是同卡重复的置信区间。", "",
                  "| dtype | shape / dim / start / length | PyTorch 中位 us | 候选中位 us | "
                  "候选 speedup 中位 | 三次范围 | 无 profiler A/B | 有 profiler A/B | 早期 launcher 缓存 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for uuid in sorted(new_cases[0]):
        rows = [case[uuid] for case in new_cases]
        axes = rows[0]["axes"]
        if any(row["axes"] != axes for row in rows[1:]):
            raise ValueError(f"workload axes 不一致: {uuid}")
        for name, cases in old_cases.items():
            if cases[uuid]["axes"] != axes:
                raise ValueError(f"{name} 的 axes 不一致: {uuid}")
        detail = axes["shape_detail"]
        shape = " × ".join(map(str, detail[0]))
        params = "/".join(map(str, detail[1:]))
        ratios = [row["speedup"] for row in rows]
        baseline = statistics.median(row["reference_latency_ms"] * 1000 for row in rows)
        candidate_us = statistics.median(row["latency_ms"] * 1000 for row in rows)
        historical = " | ".join(f"{speedup(old_cases[name][uuid]):.3f}×" for name in old)
        lines.append(f"| {axes['dtype']} | {shape} / {params} | {baseline:.2f} | "
                     f"{candidate_us:.2f} | {statistics.median(ratios):.3f}× | "
                     f"{min(ratios):.3f}–{max(ratios):.3f}× | {historical} |")
    lines.extend(["", "## 解释边界", "",
                  "历史三列来自其他时间的独立复验，reference 在每次运行重新测量；"
                  "它们可用于识别值得复查的 case，不是严格同卡、同时段的因果消融。",
                  "修复候选低于 PyTorch；早期 launcher 缓存候选的独立复验 geo mean "
                  f"为 {old['早期 launcher 缓存']['geo_mean']:.4f}×，优于本次候选，"
                  "不能把本次修复称为当前最优。", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixed_dir", type=Path)
    parser.add_argument("prior_dir", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(render(args.fixed_dir, args.prior_dir, args.candidate), encoding="utf-8")


if __name__ == "__main__":
    main()
