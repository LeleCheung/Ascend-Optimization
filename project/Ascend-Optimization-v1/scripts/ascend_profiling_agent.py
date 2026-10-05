#!/usr/bin/env python3
"""把 KGS Ascend profiling JSON 压缩成可注入 KernelGen/Claude 的中文诊断提示。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("instruction", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8-sig"))
    instruction = json.loads(args.instruction.read_text(encoding="utf-8-sig"))
    top = (metrics.get("metrics", {}).get("ops") or [{}])[0]
    summary = instruction.get("summary", {})
    text = f'''# Ascend profiling agent 诊断输入

目标设备：{instruction.get("hardware", {}).get("target", {}).get("device", "Ascend910B")}
Profiler：{instruction.get("profiler", "msprof")}；本报告仅用于诊断，不能替代正式 walltime。

主 kernel：{top.get("op_name", "unknown")}
平均设备时间：{top.get("avg_duration_us", "unknown")} us
MTE2 占比：{top.get("aiv_mte2_ratio", "unknown")}
MTE3 占比：{top.get("aiv_mte3_ratio", "unknown")}
Vector 占比：{top.get("aiv_vec_ratio", "unknown")}
Scalar 占比：{top.get("aiv_scalar_ratio", "unknown")}
Cube 利用率：{top.get("cube_utilization_pct", "unknown")}
Simulator 指令数：{summary.get("simulator_instruction_count", "unknown")}
源码映射指令数：{summary.get("simulator_mapped_instruction_count", "unknown")}

诊断结论：这是以 MTE2/MTE3 数据搬运为主的 narrow_copy 路径，Vector/Cube 计算不是主要限制。优先减少 program 数量、循环控制、同步和边界处理；连续 row-major 拷贝应保持直接搬运，不要引入复杂索引或额外计算。

实验约束：保持 narrow_copy 的 ABI、完整 workload 和 PyTorch baseline；任何候选必须通过全部 correctness，并用 KGS walltime 重新比较逐 case 延迟。不要把 msprof 或 simulator 的采集时间当作加速比。
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
