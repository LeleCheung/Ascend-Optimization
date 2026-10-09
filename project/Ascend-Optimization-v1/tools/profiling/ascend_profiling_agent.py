#!/usr/bin/env python3
"""将单个 case 的 KGS Eval、msprof 和 simulator 证据整理为中文诊断。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def read_json(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError(f"需要 JSON 对象: {path}")
    return result


def analyze(eval_path: Path, metrics_path: Path, instruction_path: Path, source_path: Path,
            request_paths: tuple[Path, ...] = ()) -> dict:
    evaluated, metrics, instruction = map(read_json, (eval_path, metrics_path, instruction_path))
    case_id = metrics.get("workload_name")
    if not case_id or instruction.get("workload_name") != case_id:
        raise ValueError("metrics 与 instruction 的 case_id 不一致")
    if any(item.get("status") != "completed" or item.get("profiler") != "msprof" for item in (metrics, instruction)):
        raise ValueError("需要两份已完成的 msprof 采集")
    if metrics.get("options", {}).get("level") != "metrics" or instruction.get("options", {}).get("level") != "instruction":
        raise ValueError("需要 metrics 和 instruction 两个采集级别")
    rows = evaluated.get("per_workload", evaluated.get("workloads", []))
    matches = [r for r in rows if r.get("uuid") == case_id and r.get("phase") == "timing"]
    if len(matches) != 1 or matches[0].get("status") != "PASSED":
        raise ValueError("正式评测中应恰有一个通过的同名 timing case")
    row = matches[0]
    ops = [op for op in metrics.get("metrics", {}).get("ops", []) if "narrow_copy" in op.get("op_name", "").lower()]
    if not ops:
        raise ValueError("metrics 没有 narrow_copy 候选 kernel")
    op = max(ops, key=lambda value: value.get("avg_duration_us") or 0)
    summary = instruction.get("summary", {})
    source_text = source_path.read_text(encoding="utf-8-sig")
    for request_path in request_paths:
        request = read_json(request_path)
        if request.get("case_id") != case_id:
            raise ValueError(f"profile 请求体 case_id 不匹配: {request_path}")
        sources = request.get("implementation", {}).get("sources", [])
        if len(sources) != 1 or sources[0].get("content", "").rstrip("\r\n") != source_text.rstrip("\r\n"):
            raise ValueError(f"profile 请求体候选源码不匹配: {request_path}")
    max_line = len(source_text.splitlines())
    mapped = sorted({int(n) for n in re.findall(r"narrow_copy\.py:(\d+)", summary.get("text", "")) if int(n) <= max_line})
    candidate_ms, pytorch_ms = row.get("latency_ms"), row.get("reference_latency_ms")
    if not all(isinstance(value, (float, int)) and value > 0 for value in (candidate_ms, pytorch_ms)):
        raise ValueError("正式评测缺少正数 walltime")
    return {
        "schema_version": "1.0", "case_id": case_id,
        "device": metrics.get("hardware", {}).get("target", {}).get("device"),
        "axes": row.get("axes", {}),
        "walltime": {"candidate_us": candidate_ms * 1000, "pytorch_us": pytorch_ms * 1000,
                     "speedup": row.get("speedup"), "source": str(eval_path)},
        "hardware_metrics": {"kernel": op.get("op_name"), "device_duration_us": op.get("avg_duration_us"),
                             "mte2_ratio": op.get("aiv_mte2_ratio"), "mte3_ratio": op.get("aiv_mte3_ratio"),
                             "vector_ratio": op.get("aiv_vec_ratio"), "scalar_ratio": op.get("aiv_scalar_ratio"),
                             "source": str(metrics_path), "scope": "msprof 真机采集，独立于正式 walltime"},
        "simulator": {"instruction_count": summary.get("simulator_instruction_count"),
                      "mapped_instruction_count": summary.get("simulator_mapped_instruction_count"),
                      "mapped_source_lines": mapped, "source": str(instruction_path),
                      "scope": "AI Core simulator，非真机延迟"},
        "roofline": {"status": "unavailable", "reason": "缺少同一 kernel/range 的可信 FLOPs、实际搬运 Bytes 及 910B 实测 roof；纯拷贝不以计算 Roofline 判定上限"},
        "source": {"path": str(source_path), "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()},
        "verified_profile_requests": [str(path) for path in request_paths],
        "limitations": ["Eval walltime 与 msprof/simulator 来自不同采集，不得相减解释为精确 host 时间",
                        "instruction 采集可能包含输入生成 kernel；只使用映射到候选源码的行定位",
                        "单个 case 的证据不能推广为完整 15 case 的硬件结论"],
    }


def render(data: dict) -> str:
    w, h, s = data["walltime"], data["hardware_metrics"], data["simulator"]
    lines = ["# narrow_copy 单 case profiling 诊断", "",
             f"- case：`{data['case_id']}`；设备：`{data['device']}`",
             f"- 正式 KGS walltime：候选 {w['candidate_us']:.2f} us，PyTorch {w['pytorch_us']:.2f} us，加速比 {w['speedup']:.3f}x。",
             f"- 真机 msprof：`{h['kernel']}` 设备时间 {h['device_duration_us']} us；MTE2 {h['mte2_ratio']}，MTE3 {h['mte3_ratio']}，Vector {h['vector_ratio']}，Scalar {h['scalar_ratio']}。",
             f"- AI Core simulator：指令 {s['instruction_count']} 条，源码映射 {s['mapped_instruction_count']} 条；候选源码行 {s['mapped_source_lines']}。",
             "", "## 推断与下一步", "",
             "候选是以数据搬运为主的拷贝 kernel，设备指令中的 MTE2/MTE3 值得核查；正式 walltime 与设备时间的差距提示应单独测量 host 启动、输出分配和同步。先为最差和最大 shape 分别采证，再只选择一个有证据的修改，并以完整 correctness 和 15 case walltime 复验。",
             "", f"Roofline：**{data['roofline']['status']}**。{data['roofline']['reason']}。",
             "", "## 证据边界", "", *[f"- {item}" for item in data["limitations"]], "",
             f"来源：`{w['source']}`、`{h['source']}`、`{s['source']}`；候选 SHA-256：`{data['source']['sha256']}`。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation", "metrics", "instruction", "source", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--profile-request", type=Path, action="append", default=[])
    args = parser.parse_args()
    data = analyze(args.evaluation, args.metrics, args.instruction, args.source,
                   tuple(args.profile_request))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
