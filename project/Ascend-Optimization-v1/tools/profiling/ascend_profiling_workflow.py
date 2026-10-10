#!/usr/bin/env python3
"""校验 KGS 原始证据并生成逐 case 的 Ascend 中文体检报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def select_kernel(profile: dict, operator: str, kernel_prefix: str | None = None) -> dict | None:
    entries = profile.get("metrics", {}).get("ops", [])
    if kernel_prefix is not None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", kernel_prefix):
            raise ValueError("kernel 前缀须为实际内核符号")
        # 优化版的符号可能不含算子名；按明确符号及编译后缀边界匹配。
        pattern = re.compile(re.escape(kernel_prefix) + r"(?:_|$)")
        ops = [op for op in entries if pattern.match(op.get("op_name", ""))]
        if len({op.get("op_name") for op in ops}) > 1:
            raise ValueError("kernel 前缀匹配多个符号，须指定更完整的符号")
    else:
        ops = [op for op in entries if operator.lower() in op.get("op_name", "").lower()]
    if not ops:
        return None
    return max(ops, key=lambda op: op.get("avg_duration_us") or 0)


def mapped_lines(profile: dict, source_name: str, line_count: int) -> list[int]:
    text = profile.get("summary", {}).get("text", "")
    pattern = re.escape(source_name) + r":(\d+)"
    return sorted({int(line) for line in re.findall(pattern, text)
                   if 1 <= int(line) <= line_count})


def workload_model(operator: str, axes: dict) -> dict:
    """只计算语义上的工作量；逻辑字节不替代硬件搬运计数。"""
    dtype = str(axes.get("dtype", "")).removeprefix("torch.")
    element_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}.get(dtype)
    detail = axes.get("shape_detail")
    unavailable = {"status": "unavailable", "reason": "缺少可识别的 dtype/shape_detail"}
    if element_bytes is None or not isinstance(detail, list) or not detail:
        return unavailable

    def shape(value):
        if not isinstance(value, list) or not value or any(type(n) is not int or n <= 0 for n in value):
            raise ValueError("形状须为正整数列表")
        return value

    try:
        dims = shape(detail[0])
        inputs = math.prod(dims)
        if operator == "amin":
            if len(detail) == 1:
                reduction, outputs, axis = inputs, 1, None
            else:
                axis = detail[1]
                if type(axis) is not int or not -len(dims) <= axis < len(dims):
                    return unavailable
                axis %= len(dims)
                reduction, outputs = dims[axis], inputs // dims[axis]
            data = {"semantic": "最小值归约", "compute_path": "AIV Vector",
                    "input_elements": inputs, "output_elements": outputs,
                    "reduction_axis": axis, "reduction_length": reduction,
                    "minimum_comparisons": outputs * (reduction - 1),
                    "flops": None, "logical_bytes": (inputs + outputs) * element_bytes,
                    "note": "比较次数不是 GEMM FLOPs；单次读输入、写输出的逻辑模型不含转置和多阶段临时张量。"}
        elif operator == "matmul_bias_activation":
            rhs = shape(detail[1])
            if len(dims) != 2 or len(rhs) != 2 or dims[1] != rhs[0]:
                return unavailable
            m, k = dims
            n = rhs[1]
            bias = shape(detail[2])
            data = {"semantic": "矩阵乘、bias 与 ReLU 融合", "compute_path": "AIC Cube + AIV Vector",
                    "m": m, "n": n, "k": k, "flops": 2 * m * n * k,
                    "logical_bytes": (m * k + k * n + math.prod(bias) + m * n) * element_bytes,
                    "note": "FLOPs 仅计矩阵乘，FMA 计两次；逻辑字节按输入各读一次与输出写一次，不含 tile 重读。"}
        elif operator == "narrow_copy":
            axis, start, length = detail[1:4]
            if (type(axis) is not int or not -len(dims) <= axis < len(dims)
                    or type(start) is not int or type(length) is not int or length < 0):
                return unavailable
            axis %= len(dims)
            if start < 0:
                start %= dims[axis]
            if not 0 <= start <= dims[axis] or length > dims[axis] - start:
                return unavailable
            outputs = inputs // dims[axis] * length
            data = {"semantic": "切片并复制到独立输出", "compute_path": "AIV MTE2/MTE3 或 D2D",
                    "input_elements": inputs, "output_elements": outputs,
                    "flops": 0, "logical_bytes": 2 * outputs * element_bytes,
                    "note": "只计所选元素的一读一写；非连续输入的预处理和实际物理搬运另测。"}
        else:
            return {"status": "unavailable", "reason": "尚无该算子的工作量模型"}
    except (IndexError, ValueError, TypeError):
        return unavailable
    return {"status": "algorithm_estimate", "dtype": dtype, "element_bytes": element_bytes,
            "byte_source": "logical_minimum", "physical_bytes": None,
            "roofline_usable": False, **data}


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
    if case.get("timing_scope", "walltime") == "walltime" and isinstance(device_us, (int, float)) and device_us < 10 and wall["candidate_us"] > 30:
        return {"bound": "host_or_launch_hypothesis", "confidence": "low",
                "inference": "独立 msprof 采集中的设备 kernel 很短，而正式 walltime 较长；host 分配、调度或同步值得单独测量。两次计时不能相减为 host 开销。",
                "next_experiment": "同卡固定输入，分别测量输出分配、已编译 launcher 调用和同步，并只改变一项 host 路径操作后完整复验。",
                "falsifier": "分项 host 测量没有可重复差距，或完整评测未改善。"}
    aic_mte2 = metrics.get("aic_mte2_ratio")
    aic_scalar = metrics.get("aic_scalar_ratio")
    if case.get("operator") == "amin":
        vec = metrics.get("aiv_vec_ratio")
        scalar = metrics.get("aiv_scalar_ratio")
        if isinstance(mte2, (int, float)) and mte2 >= 0.7:
            return {"bound": "reduction_data_pipeline", "confidence": "medium",
                    "inference": f"最小值归约的 AIV MTE2 活跃比例为 {mte2:.1%}。优先检查连续列搬运粒度、归约分块及中间类型；该比例不等于 HBM 带宽利用率。",
                    "next_experiment": "比较连续宽列分块和 bf16 minimum 后显式恢复类型，与 PyTorch 逐值比较后完整复验；记录设备延迟和 MTE2/Vector 变化。",
                    "falsifier": "搬运粒度或类型修改没有改善设备延迟，或任一正确性用例失败。"}
        if isinstance(vec, (int, float)) and vec >= 0.7:
            return {"bound": "reduction_vector_pipeline", "confidence": "medium",
                    "inference": f"最小值归约的 Vector 活跃比例为 {vec:.1%}。优先检查类型转换与归约指令，不能将它解释为 Cube GEMM 算力已达峰值。",
                    "next_experiment": "固定布局与分块，比较 fp32 累积和可编译的输入精度 min，保存 IR/指标并完整复验。",
                    "falsifier": "向量流水线变化不伴随延迟改善，或类型变化破坏 min 的结果。"}
        if isinstance(scalar, (int, float)) and scalar >= 0.5:
            return {"bound": "reduction_control_pipeline", "confidence": "medium",
                    "inference": f"最小值归约的 Scalar 活跃比例为 {scalar:.1%}，应检查归约循环、地址计算和 program 覆盖。",
                    "next_experiment": "增大单次归约片，减少循环次数，并测量 UB 容量与完整性能。",
                    "falsifier": "循环减少却使搬运或 UB 压力增加，整体延迟没有改善。"}
    if case.get("operator") == "matmul_bias_activation" and isinstance(aic_mte2, (int, float)) and aic_mte2 >= 0.7:
        return {"bound": "gemm_data_pipeline", "confidence": "medium",
                "inference": f"矩阵乘的 AIC MTE2 活跃比例为 {aic_mte2:.1%}，Scalar 为 {aic_scalar}。优先增加 K 分块、减少循环和地址计算，再重叠搬运与计算。流水线比例不等于 HBM 带宽利用率，各比例不能相加。",
                "next_experiment": "增大 K 分块至 128/256，按 shape 调整 M/N tile，测试 multibuffer/unit_flag；完整正确性与所有 timing case 复验。",
                "falsifier": "MTE2 和总延迟不改善，或新增分块导致精度失败。"}
    if case.get("operator") == "matmul_bias_activation" and isinstance(aic_scalar, (int, float)) and aic_scalar >= 0.5:
        return {"bound": "gemm_control_and_parallelism", "confidence": "medium",
                "inference": "小矩阵的 Scalar 活跃比例超过一半，原 tile 的 program 数较少；循环控制和并行覆盖应优先优化。",
                "next_experiment": "用较小 M/N tile 覆盖更多核，同时增大 K 分块以减少循环；完整复验。",
                "falsifier": "更高 program 数反而增加搬运，或总延迟没有改善。"}
    if case.get("operator") == "narrow_copy" and isinstance(mte2, (int, float)) and mte2 >= 0.7 and isinstance(mte3, (int, float)) and mte3 >= 0.4:
        return {"bound": "device_copy_pipeline_hypothesis", "confidence": "medium",
                "inference": "该 kernel 的 MTE2/MTE3 活跃比例较高，值得测试搬运分块和并行度；比例不等于 HBM 带宽利用率或已达峰值。",
                "next_experiment": "仅调整持续拷贝 kernel 的每 program 分块数，比较同卡设备指标与完整性能评测；计时范围沿用该合同。",
                "falsifier": "MTE2/MTE3 指标和完整评测没有稳定改善，或任一正确性用例失败。"}
    return {"bound": "inconclusive", "confidence": "low",
            "inference": "现有指标不足以区分设备计算、搬运或 host 路径。",
            "next_experiment": "补充针对该 case 的流水线、访存或独立 host 分项采集。",
            "falsifier": "补充采集仍无法稳定复现同一受限类型。"}


def analyze(operator: str, source_path: Path, eval_path: Path, eval_request_path: Path,
            inspect_path: Path, profiles: list[tuple[Path, Path, Path]],
            ir_paths: tuple[Path, ...] = (), *, timing_scope: str = "walltime",
            evaluation_provenance: Path | None = None,
            kernel_prefix: str | None = None) -> dict:
    if timing_scope not in {"walltime", "device_kernel"}:
        raise ValueError("计时范围必须为 walltime 或 device_kernel")
    source = source_path.read_text(encoding="utf-8-sig")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    evaluated = read_json(eval_path)
    eval_request = read_json(eval_request_path)
    inspect = read_json(inspect_path)
    binding = eval_request.get("binding", {})
    if binding.get("definition") != operator or source_in(eval_request) != source:
        raise ValueError("评测请求的算子或候选源码不匹配")
    evaluated_hash = evaluated.get("candidate_sha256")
    if evaluated_hash is None and evaluation_provenance is not None:
        provenance = read_json(evaluation_provenance)
        if provenance.get("binding") != binding:
            raise ValueError("评测 provenance 的 binding 不匹配")
        evaluated_hash = provenance.get("source_sha256")
    if evaluated_hash != source_hash or evaluated.get("status") != "PASSED":
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
        op = select_kernel(response, operator, kernel_prefix)
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
            raise ValueError(f"缺少正式计时: {case_id}")
        entry = cases.get(case_id, {"axes": row.get("axes", {}), "profiles": {}})
        metric_profile = entry["profiles"].get("metrics")
        op = metric_profile["kernel"] if metric_profile else None
        wall = {"candidate_us": row["latency_ms"] * 1000,
                "pytorch_us": row["reference_latency_ms"] * 1000,
                "speedup": row["speedup"], "source": str(eval_path),
                "scope": "完整调用 walltime" if timing_scope == "walltime" else "FlagGems Ascend 设备 kernel 计时"}
        record = {"case_id": case_id, "operator": operator, "timing_scope": timing_scope,
                  "axes": entry["axes"], "workload_model": workload_model(operator, entry["axes"]), "walltime": wall,
                  "metrics": ({"kernel": op.get("op_name"),
                               "device_duration_us": op.get("avg_duration_us"),
                               "aiv_mte2_ratio": op.get("aiv_mte2_ratio"),
                               "aiv_mte3_ratio": op.get("aiv_mte3_ratio"),
                               "aiv_scalar_ratio": op.get("aiv_scalar_ratio"),
                               "aiv_vec_ratio": op.get("aiv_vec_ratio"),
                               "aic_mte2_ratio": op.get("aic_mte2_ratio"),
                               "aic_mte1_ratio": op.get("aic_mte1_ratio"),
                               "aic_scalar_ratio": op.get("aic_scalar_ratio"),
                               "aic_mac_ratio": op.get("aic_mac_ratio"),
                               "cube_utilization_pct": op.get("cube_utilization_pct"),
                               "scope": "独立真机 msprof 采集"} if op else None),
                  "profiles": entry["profiles"]}
        record["diagnosis"] = diagnose(record)
        result_cases.append(record)
    covered = sum(bool(case["profiles"]) for case in result_cases)
    return {"schema_version": "1.3", "operator": operator, "timing_scope": timing_scope,
            "kernel_prefix": kernel_prefix,
            "candidate": {"path": str(source_path), "sha256": source_hash},
            "evaluation": {"path": str(eval_path), "request": str(eval_request_path),
                           "inspect": str(inspect_path), "benchmark_fingerprint": fingerprint,
                           "device": evaluated.get("device"), "num_workloads": evaluated.get("num_workloads"),
                           "num_passed": evaluated.get("num_passed"), "geo_mean": evaluated.get("geo_mean")},
            "coverage": {"timing_cases": len(result_cases), "profiled_cases": covered},
            "cases": result_cases,
            "ir": [{"path": str(path), "sha256": sha256(path), "status": "unmapped",
                    "reason": "尚无经验证的 PC 到该 IR 的对应关系"} for path in ir_paths],
            "roofline": {"status": "unavailable", "reason": "未验证同一 kernel/range 的 FLOPs、实际搬运 Bytes、执行时间和硬件 roof"},
            "limitations": ["profile 与正式 Eval 是独立采集，不能相减推断精确 host 开销",
                            "profile evaluation_id 不是正式 Eval ID；通过 Definition 指纹、case 与完整候选源码绑定",
                            "未采集的 case 只保留正式计时，不推广其他 case 的瓶颈结论"]}


def render(report: dict) -> str:
    ev, cov = report["evaluation"], report["coverage"]
    lines = [f"# {report['operator']} 独立 Profiling 体检", "",
             f"候选 SHA-256：`{report['candidate']['sha256']}`；KGS 正式评测 "
             f"{ev['num_passed']}/{ev['num_workloads']}，geo mean {ev['geo_mean']:.4f}×；"
             f"{cov['profiled_cases']}/{cov['timing_cases']} 个 timing case 有独立采集。", "",
             f"计时范围：`{report['timing_scope']}`。", "",
             "| case | PyTorch us | 候选 us | 加速比 | 真机 kernel us | 诊断 | 置信度 |",
             "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for case in report["cases"]:
        w, m, d = case["walltime"], case["metrics"], case["diagnosis"]
        short = case["case_id"].split("::")[-2:]
        device = f"{m['device_duration_us']:.2f}" if m and isinstance(m["device_duration_us"], (int, float)) else "—"
        lines.append(f"| {'/'.join(short)} | {w['pytorch_us']:.2f} | {w['candidate_us']:.2f} | "
                     f"{w['speedup']:.3f}× | {device} | {d['bound']} | {d['confidence']} |")
    lines.extend(["", "## 语义与逻辑工作量", "",
                  "以下由合同中的形状与 dtype 计算，不是硬件计数；逻辑字节不能代替物理 HBM/GM 搬运量或作为数值 Roofline 点。", "",
                  "| case | 语义 | 逻辑字节 | GEMM FLOPs / min 比较次数 |",
                  "| --- | --- | ---: | ---: |"])
    for case in report["cases"]:
        model = case["workload_model"]
        if model.get("status") != "algorithm_estimate":
            continue
        work = (f"min 比较 {model['minimum_comparisons']}" if model.get("flops") is None
                else str(model["flops"]))
        lines.append(f"| {'/'.join(case['case_id'].split('::')[-2:])} | {model['semantic']} | {model['logical_bytes']} | {work} |")
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
    parser.add_argument("--timing-scope", choices=["walltime", "device_kernel"], default="walltime")
    parser.add_argument("--evaluation-provenance", type=Path)
    parser.add_argument("--kernel-prefix", help="优化内核实际符号，如 mba_pipeline_kernel；匹配不唯一时拒绝分析")
    args = parser.parse_args()
    report = analyze(args.operator, args.source, args.evaluation, args.evaluation_request,
                     args.inspect, args.profile, tuple(args.ir), timing_scope=args.timing_scope,
                     evaluation_provenance=args.evaluation_provenance, kernel_prefix=args.kernel_prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")


if __name__ == "__main__":
    main()
