#!/usr/bin/env python3
"""校验配对 KG 运行并生成 narrow_copy 的中文逐 case A/B 报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verified_cases(verification: dict) -> dict[str, dict]:
    return {item["uuid"]: item for item in verification["comparisons"]}


def config(request: dict) -> dict:
    opt = request["workflow_input"]["optimization"]
    keys = ("definition_name", "catalog_name", "eval_server_url", "implementation_language",
            "warmup_ms", "benchmark_ms", "num_trials", "eval_timeout_seconds",
            "early_stop_rounds", "min_rounds", "max_round", "max_coder_sessions",
            "seed_code_path", "reference_code_path", "reference_code_prompt_path")
    return {"mode": request["mode"], "runtime": request["runtime"], "model": request["model"],
            **{key: opt.get(key) for key in keys}}


def compare(a: dict, b: dict, ar: dict, br: dict, av: dict, bv: dict) -> str:
    if not ar["workflow_input"]["optimization"]["profile_enabled"] is False:
        raise ValueError("A 组必须关闭 profiling")
    if not br["workflow_input"]["optimization"]["profile_enabled"] is True:
        raise ValueError("B 组必须开启 profiling")
    if config(ar) != config(br):
        raise ValueError(f"两组 KG 参数不一致: {config(ar)} != {config(br)}")
    if len(a["rounds"]) != 2 or len(b["rounds"]) != 2:
        raise ValueError("两组都必须包含两轮")
    if av.get("status") != "PASSED" or bv.get("status") != "PASSED":
        raise ValueError("两组最终独立复验都必须通过")
    for ledger, verification in ((a, av), (b, bv)):
        selected = ledger["rounds"][verification["round_num"] - 1]
        if selected["solution"]["sha256"] != verification["solution_sha256"]:
            raise ValueError("最终独立复验的源码与 ledger 选中轮次不一致")
    lines = ["# narrow_copy 配对 A/B", "", "A：`--no-profile`；B：`--profile`。加速比 = PyTorch/reference walltime ÷ 候选 walltime。", ""]
    lines.append("两组使用同一 seed 和提示文件；第一轮候选源码哈希分别为 "
                 f"`{a['rounds'][0]['solution']['sha256']}` 和 "
                 f"`{b['rounds'][0]['solution']['sha256']}`。候选可能因模型搜索不同而分叉。")
    lines.append("")
    a_case = verified_cases(av)
    b_case = verified_cases(bv)
    if set(a_case) != set(b_case) or len(a_case) != 15:
        raise ValueError("两组独立复验的 timing case 不是同一组 15 项")
    for label, ledger, verification in (("A", a, av), ("B", b, bv)):
        r1, r2 = ledger["rounds"]
        rounds = "; ".join(f"R{r['round_num']} {r['evaluation']['status']} / "
                             f"{r['evaluation']['geo_mean']:.4f}x" if r['evaluation']['geo_mean'] is not None
                             else f"R{r['round_num']} {r['evaluation']['status']} / 无有效几何平均值"
                             for r in (r1, r2))
        lines.append(f"- {label}：{rounds}；最终选 R{verification['round_num']}，独立复验 "
                     f"{verification['geo_mean']:.4f}x / {verification['status']}；"
                     f"R1 profile={r1['profile']['status']}。")
    failed = [row for row in b["rounds"][1]["evaluation"]["workloads"]
              if row["phase"] == "correctness" and row["status"] != "PASSED"]
    if failed:
        lines.append(f"B 组 R2 有 {len(failed)} 个 correctness workload 未通过，因此其 timing 数字只作诊断，"
                     "不作为有效优化候选；以下表格改用两组最终通过独立复验的候选。")
    lines.extend(["", "## 最终独立复验逐 case", "", "| dtype | shape / dim / start / length | A PyTorch us | A 候选 us | A 加速比 | B PyTorch us | B 候选 us | B 加速比 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for key in sorted(a_case):
        x, y = a_case[key], b_case[key]
        if x.get("axes") != y.get("axes"):
            raise ValueError(f"shape/dtype 不一致: {key}")
        axes = x.get("axes", {})
        detail = axes.get("shape_detail", [])
        shape = " × ".join(map(str, detail[0])) if detail else "unknown"
        args = "/".join(map(str, detail[1:])) if detail else "unknown"
        xa, ya = x["after"], y["after"]
        lines.append(f"| {axes.get('dtype')} | {shape} / {args} | {xa['reference_latency_ms']*1000:.2f} | {xa['latency_ms']*1000:.2f} | {xa['reference_latency_ms']/xa['latency_ms']:.3f}x | {ya['reference_latency_ms']*1000:.2f} | {ya['latency_ms']*1000:.2f} | {ya['reference_latency_ms']/ya['latency_ms']:.3f}x |")
    valid_profile = b["rounds"][0]["profile"]["status"] == "completed"
    lines.extend(["", "## 因果边界", "",
                  f"B 组第一轮 profile 状态：`{b['rounds'][0]['profile']['status']}`。" +
                  ("本次日志顺序证据见同目录 README；第二轮候选未通过正确性，不能主张有效性能收益。" if valid_profile else "未在第一轮形成完成的 profile 分析，不能称为由 KGS profiling 反馈驱动的第二轮。"),
                  "两组搜索具有模型随机性，PyTorch reference 也逐组重测；本表仅报告本次配对结果，不声称平均因果收益。", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("a_ledger", "b_ledger", "a_request", "b_request", "a_final", "b_final", "output"):
        parser.add_argument(name, type=Path)
    p = parser.parse_args()
    result = compare(*(load(getattr(p, name)) for name in
                       ("a_ledger", "b_ledger", "a_request", "b_request", "a_final", "b_final")))
    p.output.parent.mkdir(parents=True, exist_ok=True)
    p.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
