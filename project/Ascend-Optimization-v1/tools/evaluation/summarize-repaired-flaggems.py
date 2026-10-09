#!/usr/bin/env python3
"""核验修复基线与候选的合同、源码和正确性，计算直接性能提升。"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


OPERATORS = ("_prelu_kernel_backward", "batch_norm_backward", "smooth_l1_loss_backward")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def geo(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("延迟或加速比必须为有限正数")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    measurements = args.report / "measurements"
    experiment = read(measurements / "experiment.json")
    if experiment["repeats"] != 3:
        raise ValueError("最终报告要求完整三轮复测")
    repaired_sources = {row["operator"]: row for row in read(args.report / "repairs/repairs.json")}
    rows, cases = [], []
    devices = set()
    for operator in OPERATORS:
        folder = measurements / operator
        inspected = read(folder / "inspect.json")
        provenance = read(folder / "provenance.json")
        assert inspected["benchmark_fingerprint"] == provenance["benchmark_fingerprint"]
        protocol = read(args.report / "protocol.json")[operator]
        repeat_rows = []
        for repeat in range(1, experiment["repeats"] + 1):
            labels = ("repaired-flaggems", "candidate")
            paths = [folder / f"{label}-{repeat}.result.json" for label in labels]
            if not all(path.exists() for path in paths):
                raise ValueError(f"缺少完整三轮结果：{operator} 第 {repeat} 轮")
            results = [read(path) for path in paths]
            requests = [read(folder / f"{label}-{repeat}.request.json") for label in labels]
            assert requests[0]["binding"] == requests[1]["binding"] == provenance["binding"]
            assert requests[0]["settings"] == requests[1]["settings"] == provenance["settings"]
            for label, request, filename in zip(labels, requests, ("repaired.py", "candidate.py")):
                source = request["implementation"]["sources"][0]["content"].encode()
                assert hashlib.sha256(source).hexdigest() == provenance["sources_sha256"][label]
                assert source == (folder / filename).read_bytes()
                assert source == (args.report / "inputs" / operator / filename).read_bytes()
            expected = protocol["workloads"]
            assert {c["case_id"] for c in inspected["case_list"]["cases"]} == {
                u for u, w in expected.items() if w["phase"] == "timing"}
            all_cases = []
            for result in results:
                if result.get("status") != "PASSED":
                    raise ValueError(f"{operator} 第 {repeat} 轮未通过完整正确性，不能汇报性能")
                current = {w["uuid"]: w for w in result["per_workload"]}
                assert set(current) == set(expected)
                assert len(current) == len(result["per_workload"])
                assert all(w["status"] == "PASSED" and w["phase"] == expected[u]["phase"]
                           and w["axes"] == expected[u]["axes"] for u, w in current.items())
                all_cases.append(current)
            assert results[0]["device"] == results[1]["device"], operator
            assert results[0]["num_passed"] == results[0]["num_workloads"] == len(expected)
            assert results[1]["num_passed"] == results[1]["num_workloads"] == len(expected)
            devices.add(results[0]["device"])
            repaired_source = (folder / "repaired.py").read_bytes()
            assert repaired_source == (args.report / "repairs" / operator / "repaired.py").read_bytes()
            assert hashlib.sha256(repaired_source).hexdigest() == repaired_sources[operator]["repaired_sha256"]
            timings = [{u: w for u, w in current.items() if w["phase"] == "timing"} for current in all_cases]
            ratios, baseline_speedups, candidate_speedups, drifts = [], [], [], []
            for uuid, baseline in timings[0].items():
                candidate = timings[1][uuid]
                assert baseline["axes"] == candidate["axes"]
                ratio = baseline["latency_ms"] / candidate["latency_ms"]
                ratios.append(ratio)
                baseline_speedups.append(baseline["reference_latency_ms"] / baseline["latency_ms"])
                candidate_speedups.append(candidate["reference_latency_ms"] / candidate["latency_ms"])
                drifts.append(candidate["reference_latency_ms"] / baseline["reference_latency_ms"])
                cases.append({"operator": operator, "repeat": repeat, "uuid": uuid,
                              "baseline_ms": baseline["latency_ms"], "candidate_ms": candidate["latency_ms"],
                              "speedup_vs_repaired_flaggems": ratio,
                              "pytorch_baseline_ms": baseline["reference_latency_ms"],
                              "pytorch_candidate_ms": candidate["reference_latency_ms"]})
            repeat_rows.append({"repeat": repeat, "speedup_vs_repaired_flaggems": geo(ratios),
                                "baseline_vs_pytorch": geo(baseline_speedups), "candidate_vs_pytorch": geo(candidate_speedups),
                                "pytorch_drift": geo(drifts), "wins": sum(v > 1 for v in ratios),
                                "timing_cases": len(ratios),
                                "correctness_cases": sum(w["phase"] == "correctness" for w in all_cases[0].values()),
                                "logical_device": results[0]["device"]})
        ratios = [r["speedup_vs_repaired_flaggems"] for r in repeat_rows]
        rows.append({"operator": operator, "median_speedup_vs_repaired_flaggems": statistics.median(ratios),
                     "min_speedup": min(ratios), "max_speedup": max(ratios),
                     "median_baseline_vs_pytorch": statistics.median(r["baseline_vs_pytorch"] for r in repeat_rows),
                     "median_candidate_vs_pytorch": statistics.median(r["candidate_vs_pytorch"] for r in repeat_rows),
                     "repeats": repeat_rows})
    assert len(devices) == 1, "全部算子必须使用同一设备"
    (args.report / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.report / "per-case.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
