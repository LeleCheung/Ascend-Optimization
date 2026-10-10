#!/usr/bin/env python3
"""从已结束的 KG 工作区导出通过完整评测的最佳轮次，供独立复验。"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("operator")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    progress = read(args.workspace / ".kernelgen/run-progress.json")
    if progress.get("state") not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
        parser.error("KG 工作区尚未结束，不能导出最终候选")
    candidates = []
    for directory in sorted(args.workspace.glob("stages/optimize/work/.kernelgen/evals/round-*")):
        result_path = directory / "result.json"
        if not result_path.is_file():
            continue
        result = read(result_path)
        score = result.get("geo_mean")
        if (result.get("status") != "PASSED" or result.get("is_hack")
                or not isinstance(score, (int, float)) or score <= 0
                or not result.get("num_workloads")
                or result.get("num_passed") != result.get("num_workloads")):
            continue
        identity = read(directory / "identity.json")
        solution = read(directory / "solution.json")
        if identity.get("definition_name") != args.operator or solution.get("definition") != args.operator:
            raise ValueError("轮次算子身份不匹配")
        sources = solution.get("sources", [])
        if len(sources) != 1 or solution.get("entrypoint") != "main.py::run":
            raise ValueError("导出工具只支持单文件 main.py::run 候选")
        source = (directory / "main.py").read_text(encoding="utf-8")
        if sources[0].get("path") != "main.py" or sources[0].get("content") != source:
            raise ValueError("源码快照与提交给 KGS 的 solution 不一致")
        candidates.append((score, directory, source, identity, result))
    if not candidates:
        parser.error("没有通过完整评测的候选，保留失败工作区进行排障")
    score, directory, source, identity, result = max(candidates, key=lambda item: item[0])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "main.py").write_text(source, encoding="utf-8", newline="\n")
    for name in ("result.json", "identity.json", "solution.json", "definition.json", "workloads.json"):
        path = directory / name
        if path.is_file():
            (args.output / name).write_bytes(path.read_bytes())
    provenance = {"operator": args.operator, "workspace": str(args.workspace),
                  "workspace_state": progress["state"], "round": directory.name,
                  "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "benchmark_fingerprint": identity.get("benchmark_fingerprint"),
                  "geo_mean": score,
                  "next_step": "导出的是 KG 轮次结果；仍须使用 evaluate-operator.py 独立完整复验"}
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
