#!/usr/bin/env python3
"""核验同合同 FlagGems 对照，按逐项延迟计算候选相对原版的提升。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


OPERATORS = ["mse_loss_backward", "_prelu_kernel_backward", "t_copy",
             "batch_norm_backward", "smooth_l1_loss_backward"]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def geometric(values: list[float]) -> float | None:
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    measurements = args.report / "measurements"
    summary, cases = [], []
    for operator in OPERATORS:
        folder = measurements / operator
        retry = measurements / "prelu-retry" / operator
        if operator == "_prelu_kernel_backward" and retry.exists():
            folder = retry
        original, candidate = [read(folder / f"{v}-1.result.json")
                               for v in ("flaggems", "candidate")]
        requests = [read(folder / f"{v}-1.request.json") for v in ("flaggems", "candidate")]
        assert requests[0]["binding"] == requests[1]["binding"], operator
        assert requests[0]["settings"] == requests[1]["settings"], operator
        provenance = read(folder / "provenance.json")
        inspect = read(folder / "inspect.json")
        original_source = (args.report / "source-audit" / operator / "flaggems-original-source.py").read_text(encoding="utf-8")
        assert requests[0]["implementation"]["sources"][0]["content"].startswith(original_source), operator
        candidate_source = requests[1]["implementation"]["sources"][0]["content"]
        assert hashlib.sha256(candidate_source.encode()).hexdigest() == provenance["candidate_sha256"], operator
        archived_candidate = (args.report / "source-audit" / operator / "candidate.py").read_bytes()
        assert hashlib.sha256(archived_candidate).hexdigest() == provenance["candidate_sha256"], operator
        original_bytes = (args.report / "source-audit" / operator / "flaggems-original-source.py").read_bytes()
        assert hashlib.sha256(original_bytes).hexdigest() == provenance["flaggems_file_sha256"], operator
        phases = [Counter(w.get("phase") for w in result.get("per_workload", []))
                  for result in (original, candidate)]
        phase_statuses = [Counter(f"{w.get('phase')}:{w.get('status')}"
                                  for w in result.get("per_workload", []))
                          for result in (original, candidate)]
        timing = [{w["uuid"]: w for w in result.get("per_workload", []) if w.get("phase") == "timing"}
                  for result in (original, candidate)]
        same_cases = set(timing[0]) == set(timing[1]) and bool(timing[0])
        valid = original.get("status") == candidate.get("status") == "PASSED" and same_cases
        ratios, native_ratios = [], []
        if valid:
            for uuid in timing[0]:
                a, b = timing[0][uuid], timing[1][uuid]
                assert a["status"] == b["status"] == "PASSED", uuid
                assert a.get("axes") == b.get("axes"), uuid
                assert a["latency_ms"] > 0 and b["latency_ms"] > 0, uuid
                ratio = a["latency_ms"] / b["latency_ms"]
                ratios.append(ratio)
                native_ratios.append(a["reference_latency_ms"] / b["reference_latency_ms"])
                cases.append({"operator": operator, "uuid": uuid,
                              "axes": json.dumps(b.get("axes"), ensure_ascii=False, sort_keys=True),
                              "flaggems_us": a["latency_ms"] * 1000,
                              "candidate_us": b["latency_ms"] * 1000,
                              "flaggems_pytorch_us": a["reference_latency_ms"] * 1000,
                              "candidate_pytorch_us": b["reference_latency_ms"] * 1000,
                              "candidate_over_flaggems": ratio})
        summary.append({"operator": operator,
                        "timing_scope": "Ascend FlagGems kernel 模式，设备侧耗时",
                        "flaggems_matches_locked_upstream": provenance["flaggems_file_sha256"] == provenance["upstream_file_sha256"],
                        "flaggems_status": original.get("status"),
                        "candidate_status": candidate.get("status"),
                        "flaggems_vs_pytorch": original.get("geo_mean") if original.get("status") == "PASSED" else None,
                        "candidate_vs_pytorch": candidate.get("geo_mean") if candidate.get("status") == "PASSED" else None,
                        "candidate_vs_flaggems": geometric(ratios),
                        "pytorch_reference_drift_ratio": geometric(native_ratios),
                        "timing_wins": sum(r > 1 for r in ratios), "timing_cases_compared": len(ratios),
                        "same_timing_case_ids": same_cases,
                        "flaggems_phase_counts": dict(phases[0]), "candidate_phase_counts": dict(phases[1]),
                        "flaggems_phase_status_counts": dict(phase_statuses[0]),
                        "candidate_phase_status_counts": dict(phase_statuses[1]),
                        "flaggems_failed_workloads": [
                            {k: w.get(k) for k in ("uuid", "phase", "status", "log")}
                            for w in original.get("per_workload", []) if w.get("status") != "PASSED"],
                        "flaggems_passed": original.get("num_passed"),
                        "candidate_passed": candidate.get("num_passed"),
                        "flaggems_device": original.get("device"), "candidate_device": candidate.get("device"),
                        "benchmark_fingerprint": inspect.get("benchmark_fingerprint"),
                        "settings": requests[0]["settings"], "provenance": folder.relative_to(args.report).as_posix(),
                        "flaggems_log": original.get("log", "")})
    (args.report / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.report / "per-case.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]) if cases else ["operator"])
        writer.writeheader()
        writer.writerows(cases)
    for row in summary:
        print(row["operator"], row["flaggems_status"], row["candidate_status"],
              row["flaggems_vs_pytorch"], row["candidate_vs_pytorch"], row["candidate_vs_flaggems"])


if __name__ == "__main__":
    main()
