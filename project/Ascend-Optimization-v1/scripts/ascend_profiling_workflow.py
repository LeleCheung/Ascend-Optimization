#!/usr/bin/env python3
"""校验 KGS 原始证据并生成逐 case 的 Ascend 中文体检报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"需要 JSON 对象: {path}")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_in(request: dict) -> str:
    sources = request.get("implementation", {}).get("sources", [])
    if len(sources) != 1 or not isinstance(sources[0].get("content"), str):
        raise ValueError("请求体必须包含一份候选源码")
    return sources[0]["content"]


def check_artifacts(response: dict, directory: Path) -> list[dict]:
    manifest = read_json(directory / "manifest.json")
    if manifest.get("profile_id") != response.get("profile_id"):
        raise ValueError("artifact manifest 与 profile_id 不匹配")
    expected = {item["id"]: item for item in response.get("artifacts", [])}
    checked = []
    for item in manifest.get("artifacts", []):
        remote = expected.get(item.get("id"))
        if remote is None or item.get("kind") != remote.get("kind"):
            raise ValueError(f"artifact 未在 profile 响应中登记: {item.get('id')}")
        path = directory / f"{item['id']}-{Path(remote['filename']).name}"
        if not path.is_file() or path.stat().st_size != item.get("size_bytes"):
            raise ValueError(f"artifact 缺失或大小不匹配: {path}")
        digest = sha256(path)
        if digest != item.get("sha256") or path.stat().st_size != remote.get("size_bytes"):
            raise ValueError(f"artifact SHA 或响应大小不匹配: {path}")
        checked.append({"id": item["id"], "kind": item["kind"], "path": str(path),
                        "size_bytes": path.stat().st_size, "sha256": digest})
    if not checked:
        raise ValueError("profile 没有已校验的原始 artifact")
    return checked


def select_kernel(profile: dict, operator: str) -> dict | None:
    ops = [op for op in profile.get("metrics", {}).get("ops", [])
           if operator.lower() in op.get("op_name", "").lower()]
    if not ops:
        return None
    return max(ops, key=lambda op: op.get("avg_duration_us") or 0)


def mapped_lines(profile: dict, source_name: str, line_count: int) -> list[int]:
    text = profile.get("summary", {}).get("text", "")
    pattern = re.escape(source_name) + r":(\d+)"
    return sorted({int(line) for line in re.findall(pattern, text)
                   if 1 <= int(line) <= line_count})


def diagnose(case: dict) -> dict:
    wall = case["walltime"]
    metrics = case.get("metrics")
    if metrics is None:
        return {"bound": "inconclusive", "confidence": "low",
                "inference": "缺少同一候选的真机 kernel 指标，不能分类设备瓶颈。",
                "next_experiment": "先采集该 case 的 msprof metrics，并保留原始 artifact。",
                "falsifier": "指标采集仍无法识别候选 kernel。"}
    device_us = metrics.get("device_duration_us")
    mte2 = metrics.get("aiv_mte2_ratio")
    mte3 = metrics.get("aiv_mte3_ratio")
    if isinstance(device_us, (int, float)) and device_us < 10 and wall["candidate_us"] > 30:
        return {"bound": "host_or_launch_hypothesis", "confidence": "low",
                "inference": "独立 msprof 采集中的设备 kernel 很短，而正式 walltime 较长；host 分配、调度或同步值得单独测量。两次计时不能相减为 host 开销。",
                "next_experiment": "同卡固定输入，分别测量输出分配、已编译 launcher 调用和同步，并只改变一项 host 路径操作后完整复验。",
                "falsifier": "分项 host 测量没有可重复差距，或完整评测未改善。"}
    if isinstance(mte2, (int, float)) and mte2 >= 0.7 and isinstance(mte3, (int, float)) and mte3 >= 0.4:
        return {"bound": "device_copy_pipeline_hypothesis", "confidence": "medium",
                "inference": "该 kernel 的 MTE2/MTE3 活跃比例较高，值得测试搬运分块和并行度；比例不等于 HBM 带宽利用率或已达峰值。",
                "next_experiment": "仅调整持续拷贝 kernel 的每 program 分块数，比较同卡设备指标与完整 15 case walltime。",
                "falsifier": "MTE2/MTE3 指标和完整评测没有稳定改善，或任一正确性用例失败。"}
    return {"bound": "inconclusive", "confidence": "low",
            "inference": "现有指标不足以区分设备计算、搬运或 host 路径。",
            "next_experiment": "补充针对该 case 的流水线、访存或独立 host 分项采集。",
            "falsifier": "补充采集仍无法稳定复现同一受限类型。"}


def analyze(operator: str, source_path: Path, eval_path: Path, eval_request_path: Path,
            inspect_path: Path, profiles: list[tuple[Path, Path, Path]],
            ir_paths: tuple[Path, ...] = ()) -> dict:
    source = source_path.read_text(encoding="utf-8-sig")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    evaluated = read_json(eval_path)
    eval_request = read_json(eval_request_path)
    inspect = read_json(inspect_path)
    binding = eval_request.get("binding", {})
    if binding.get("definition") != operator or source_in(eval_request) != source:
        raise ValueError("评测请求的算子或候选源码不匹配")
    if evaluated.get("candidate_sha256") != source_hash or evaluated.get("status") != "PASSED":
        raise ValueError("正式评测未通过或候选源码 SHA 不匹配")
    fingerprint = inspect.get("benchmark_fingerprint")
    if not fingerprint:
        raise ValueError("inspect 缺少 benchmark_fingerprint")
    timing = {row["uuid"]: row for row in evaluated.get("per_workload", [])
              if row.get("phase") == "timing" and row.get("status") == "PASSED"}
    if not timing or len(timing) != sum(row.get("phase") == "timing"
                                      for row in evaluated.get("per_workload", [])):
        raise ValueError("正式评测存在未通过的 timing case")
    cases = {}
    source_name = ""
    for response_path, request_path, artifact_dir in profiles:
        response, request = read_json(response_path), read_json(request_path)
        case_id = request.get("case_id")
        level = request.get("options", {}).get("level")
        if (case_id not in timing or case_id != response.get("workload_name")
                or level not in {"metrics", "instruction"}
                or response.get("options", {}).get("level") != level
                or response.get("status") != "completed"
                or response.get("profiler") != "msprof"):
            raise ValueError(f"profile 状态、级别或 case 身份不匹配: {response_path}")
        if (request.get("binding") != binding or request.get("benchmark_fingerprint") != fingerprint
                or source_in(request) != source):
            raise ValueError(f"profile Definition、指纹或候选源码不匹配: {request_path}")
        source_name = Path(request["implementation"]["sources"][0]["path"]).name
        artifacts = check_artifacts(response, artifact_dir)
        entry = cases.setdefault(case_id, {"axes": timing[case_id].get("axes", {}), "profiles": {}})
        if level in entry["profiles"]:
            raise ValueError(f"重复的 case/level: {case_id} {level}")
        op = select_kernel(response, operator)
        entry["profiles"][level] = {
            "profile_id": response["profile_id"], "device": response.get("device"),
            "software": response.get("hardware", {}).get("software"),
            "response": str(response_path), "request": str(request_path),
            "artifacts": artifacts, "kernel": op,
            "mapped_source_lines": mapped_lines(response, source_name, len(source.splitlines()))
            if level == "instruction" else [],
            "instruction_count": response.get("summary", {}).get("simulator_instruction_count")
            if level == "instruction" else None,
        }
    result_cases = []
    for case_id in sorted(timing):
        row = timing[case_id]
        if not all(isinstance(row.get(key), (int, float)) and row[key] > 0
                   for key in ("latency_ms", "reference_latency_ms")):
            raise ValueError(f"缺少正式 walltime: {case_id}")
        entry = cases.get(case_id, {"axes": row.get("axes", {}), "profiles": {}})
        metric_profile = entry["profiles"].get("metrics")
        op = metric_profile["kernel"] if metric_profile else None
        wall = {"candidate_us": row["latency_ms"] * 1000,
                "pytorch_us": row["reference_latency_ms"] * 1000,
                "speedup": row["speedup"], "source": str(eval_path),
                "scope": "KGS 独立正式评测 walltime"}
        record = {"case_id": case_id, "axes": entry["axes"], "walltime": wall,
                  "metrics": ({"kernel": op.get("op_name"),
                               "device_duration_us": op.get("avg_duration_us"),
                               "aiv_mte2_ratio": op.get("aiv_mte2_ratio"),
                               "aiv_mte3_ratio": op.get("aiv_mte3_ratio"),
                               "aiv_scalar_ratio": op.get("aiv_scalar_ratio"),
                               "aiv_vec_ratio": op.get("aiv_vec_ratio"),
                               "scope": "独立真机 msprof 采集"} if op else None),
                  "profiles": entry["profiles"]}
        record["diagnosis"] = diagnose(record)
        result_cases.append(record)
    covered = sum(bool(case["profiles"]) for case in result_cases)
    return {"schema_version": "1.0", "operator": operator,
            "candidate": {"path": str(source_path), "sha256": source_hash},
            "evaluation": {"path": str(eval_path), "request": str(eval_request_path),
                           "inspect": str(inspect_path), "benchmark_fingerprint": fingerprint,
                           "device": evaluated.get("device"), "num_workloads": evaluated.get("num_workloads"),
                           "num_passed": evaluated.get("num_passed"), "geo_mean": evaluated.get("geo_mean")},
            "coverage": {"timing_cases": len(result_cases), "profiled_cases": covered},
            "cases": result_cases,
            "ir": [{"path": str(path), "sha256": sha256(path), "status": "unmapped",
                    "reason": "尚无经验证的 PC 到该 IR 的对应关系"} for path in ir_paths],
            "roofline": {"status": "unavailable", "reason": "未验证同一 kernel/range 的 FLOPs、实际搬运 Bytes、执行时间和硬件 roof；拷贝算子不能用计算 Roofline 推断上限"},
            "limitations": ["profile 与正式 Eval 是独立采集，device 时间不得从 walltime 中相减",
                            "profile evaluation_id 不是正式 Eval ID；通过 Definition 指纹、case 与完整候选源码绑定",
                            "未采集的 case 只保留 walltime，不推广其他 case 的瓶颈结论"]}


def render(report: dict) -> str:
    ev, cov = report["evaluation"], report["coverage"]
    lines = [f"# {report['operator']} 独立 Profiling 体检", "",
             f"候选 SHA-256：`{report['candidate']['sha256']}`；KGS 正式评测 "
             f"{ev['num_passed']}/{ev['num_workloads']}，geo mean {ev['geo_mean']:.4f}×；"
             f"{cov['profiled_cases']}/{cov['timing_cases']} 个 timing case 有独立采集。", "",
             "| case | PyTorch us | 候选 us | 加速比 | 真机 kernel us | 诊断 | 置信度 |",
             "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for case in report["cases"]:
        w, m, d = case["walltime"], case["metrics"], case["diagnosis"]
        short = case["case_id"].split("::")[-2:]
        device = f"{m['device_duration_us']:.2f}" if m and isinstance(m["device_duration_us"], (int, float)) else "—"
        lines.append(f"| {'/'.join(short)} | {w['pytorch_us']:.2f} | {w['candidate_us']:.2f} | "
                     f"{w['speedup']:.3f}× | {device} | {d['bound']} | {d['confidence']} |")
    lines.extend(["", "## 逐 case 判断", ""])
    for case in report["cases"]:
        if not case["profiles"]:
            continue
        diagnosis = case["diagnosis"]
        lines.extend([f"### {case['case_id']}", "", diagnosis["inference"], "",
                      f"下一项实验：{diagnosis['next_experiment']}", "",
                      f"反证条件：{diagnosis['falsifier']}", ""])
        for level, profile in case["profiles"].items():
            lines.append(f"- `{level}`：`{profile['response']}`；profile ID `{profile['profile_id']}`；"
                         f"原始 artifact {len(profile['artifacts'])} 个，SHA 已校验。")
            if level == "instruction":
                lines.append(f"  模拟器指令 {profile.get('instruction_count', '—')} 条；"
                             f"映射源码行 {profile.get('mapped_source_lines', [])}。")
        lines.append("")
    lines.extend(["## 证据边界", "", f"Roofline：**{report['roofline']['status']}**。{report['roofline']['reason']}",
                  *[f"- {item}" for item in report["limitations"]], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator")
    parser.add_argument("source", type=Path)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("evaluation_request", type=Path)
    parser.add_argument("inspect", type=Path)
    parser.add_argument("output", type=Path, help="中文 Markdown 输出路径；同名 JSON 同时生成")
    parser.add_argument("--profile", type=Path, nargs=3, action="append", default=[],
                        metavar=("RESPONSE", "REQUEST", "ARTIFACT_DIR"))
    parser.add_argument("--ir", type=Path, action="append", default=[])
    args = parser.parse_args()
    report = analyze(args.operator, args.source, args.evaluation, args.evaluation_request,
                     args.inspect, args.profile, tuple(args.ir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
