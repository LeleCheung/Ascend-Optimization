#!/usr/bin/env python3
"""对已通过完整评测的固定 case 采集 Ascend 指标/指令并校验原始产物。"""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from evaluate_operator_import import call, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--inspect", type=Path, required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--server", default="http://127.0.0.1:19655")
    p.add_argument("--level", choices=["metrics", "instruction"], default="metrics")
    p.add_argument("--aic-metrics")
    p.add_argument("--op-metrics")
    p.add_argument("--timeout", type=int, default=900)
    a = p.parse_args()
    if a.output.exists():
        p.error("输出已存在，请使用新批次以保留原始证据")
    ev = json.loads(a.evaluation.read_text())
    req = json.loads(a.request.read_text())
    inspect = json.loads(a.inspect.read_text())
    rows = [x for x in ev.get("per_workload", []) if x.get("phase") == "timing"]
    if ev.get("status") != "PASSED" or not any(x["uuid"] == a.case_id for x in rows):
        p.error("须使用通过完整测试的候选及其 timing case")
    a.output.mkdir(parents=True)
    save(a.output / "inspect.json", inspect)
    payload = {"api_version": "v6.2", "binding": req["binding"],
               "implementation": req["implementation"],
               "benchmark_fingerprint": inspect["benchmark_fingerprint"],
               "case_id": a.case_id, "expected_backend": "npu",
               "options": {"level": a.level, "warmup": 1, "iterations": 2,
                           "timeout_sec": a.timeout}}
    options = {}
    if a.aic_metrics:
        options["aic_metrics"] = a.aic_metrics
    if a.op_metrics:
        if a.level != "instruction":
            p.error("op-metrics 仅用于 instruction")
        options["op_metrics"] = a.op_metrics
    if options:
        payload["options"]["backend_options"] = {"npu": options}
    save(a.output / "request.json", payload)
    print("PROFILE", a.case_id, a.level, flush=True)
    response = call(a.server, "/profile", payload, a.timeout + 120)
    save(a.output / "response.json", response)
    artifacts = a.output / "artifacts"
    artifacts.mkdir()
    downloaded = []
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for item in response.get("artifacts", []):
        url = item.get("download_url", "")
        if not url.startswith("/profile_artifacts/"):
            raise ValueError("unexpected artifact URL")
        path = artifacts / (item["id"] + "-" + Path(item["filename"]).name)
        with opener.open(a.server + url, timeout=120) as stream:
            data = stream.read()
        if len(data) != item["size_bytes"]:
            raise ValueError("artifact size mismatch")
        path.write_bytes(data)
        downloaded.append({"id": item["id"], "kind": item["kind"], "path": str(path),
                           "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    save(artifacts / "manifest.json", {"profile_id": response.get("profile_id"),
                                      "artifacts": downloaded})
    print("DONE", response.get("status"), response.get("error"), flush=True)
    if response.get("status") != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
