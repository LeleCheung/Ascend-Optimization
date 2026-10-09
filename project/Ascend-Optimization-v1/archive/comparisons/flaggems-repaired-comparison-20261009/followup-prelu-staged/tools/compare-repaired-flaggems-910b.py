#!/usr/bin/env python3
"""同合同、同设备复测最小修复版 FlagGems 与既有候选，逐轮归档原始结果。"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("comparison_http", HERE / "compare-flaggems-910b.py")
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:19656")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--operators", nargs="+", default=list(client.RUNS)[1:2] + list(client.RUNS)[3:])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = client.http(args.server, "/status", method="GET", timeout=60)
    client.save(args.output / "server-status.json", status)
    if len(status["devices"]) != 1:
        raise ValueError("本轮专用 KGS 应只暴露一张 NPU，确保所有变体使用相同设备")
    client.save(args.output / "experiment.json", {
        "server": args.server, "physical_device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), "repeats": args.repeats,
        "variants": ["repaired-flaggems", "candidate"],
        "baseline": "历史 FlagGems 源码的最小本机正确性修复，非未修改的上游原版",
        "timing": "FlagGems adapter 设备侧 kernel 计时；按逐 case 直接延迟计算提升",
        "ordering": "奇数轮先基线，偶数轮先候选",
    })
    summary = []
    for operator in args.operators:
        source_dir = args.inputs / operator
        output = args.output / operator
        output.mkdir()
        blueprint = json.loads((source_dir / "blueprint.json").read_text(encoding="utf-8"))
        binding = blueprint["binding"]
        bundle_path = source_dir / "operator-bundle.tar"
        if "bundle_id" in binding:
            data = bundle_path.read_bytes()
            response = client.http(args.server, "/operator-bundles/" + hashlib.sha256(data).hexdigest(), data, "PUT")
            client.save(output / "bundle-response.json", response)
            binding = {"bundle_id": response["bundle_id"], "definition": operator}
        inspected = client.http(args.server, "/inspect", {"api_version": "v6.2", "binding": binding}, timeout=300)
        client.save(output / "inspect.json", inspected)
        expected = json.loads((source_dir / "historical-inspect.json").read_text(encoding="utf-8"))
        if inspected["benchmark_fingerprint"] != expected["benchmark_fingerprint"] or inspected["case_list"] != expected["case_list"]:
            raise ValueError(f"{operator} 的测试合同与历史候选不一致")
        sources = {}
        for label, filename in (("repaired-flaggems", "repaired.py"), ("candidate", "candidate.py")):
            data = (source_dir / filename).read_bytes()
            (output / filename).write_bytes(data)
            sources[label] = data.decode("utf-8")
        client.save(output / "provenance.json", {
            "binding": binding, "settings": blueprint["settings"],
            "benchmark_fingerprint": inspected["benchmark_fingerprint"],
            "sources_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()},
        })
        for repeat in range(1, args.repeats + 1):
            labels = list(sources) if repeat % 2 else list(reversed(sources))
            for label in labels:
                name = f"{label}-{repeat}"
                request = {
                    "api_version": "v6.2", "binding": binding,
                    "implementation": {"name": operator + "-" + name, "definition": operator,
                        "language": "triton", "entrypoint": "main.py::run",
                        "sources": [{"path": "main.py", "content": sources[label]}]},
                    "settings": blueprint["settings"],
                }
                client.save(output / (name + ".request.json"), request)
                print("START", operator, name, flush=True)
                result = client.http(args.server, "/evaluate", request,
                                     timeout=blueprint["settings"]["timeout_seconds"] + 120)
                client.save(output / (name + ".result.json"), result)
                row = {"operator": operator, "variant": label, "repeat": repeat,
                       **{key: result.get(key) for key in ("status", "geo_mean", "device", "num_passed", "num_workloads")}}
                summary.append(row)
                client.save(args.output / "summary.json", summary)
                print("DONE", json.dumps(row), flush=True)
            if any(row["status"] != "PASSED" for row in summary[-2:]):
                print("STOP_OPERATOR_CORRECTNESS", operator, flush=True)
                break
    print("COMPLETE", str(args.output), flush=True)


if __name__ == "__main__":
    main()
