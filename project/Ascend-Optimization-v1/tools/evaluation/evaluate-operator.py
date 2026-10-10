#!/usr/bin/env python3
"""固定源码，通过 KGS 原生 FlagGems 合同评测单个算子并保存原始证据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def save(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def call(server: str, route: str, payload: dict | None = None, timeout: int = 1800) -> dict:
    # 仅此工具绕过系统代理，避免本机 loopback 请求进入代理。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        server + route, data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with opener.open(request, timeout=timeout) as response:
        result = json.load(response)
    if result.get("operation_id") and "per_workload" not in result:
        operation_id = result["operation_id"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(5)
            response = call(server, "/operations/" + operation_id, timeout=60)
            if response.get("result") is not None:
                return response["result"]
            if response.get("state", response.get("status", "")).lower() in {"failed", "cancelled"}:
                return response
        raise TimeoutError(operation_id)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--operator", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--server", default="http://127.0.0.1:19655")
    p.add_argument("--catalog", default="flaggems-adapter-definitions")
    p.add_argument("--label", required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--timing-scope", choices=("device_kernel", "device_task", "walltime"), default="device_kernel",
                   help="记录 benchmark 已固定的实际计时口径，不修改 KGS 或 benchmark")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    source = a.source.read_text(encoding="utf-8")
    binding = {"catalog_name": a.catalog, "definition": a.operator}
    save(a.output / "server-status.json", call(a.server, "/status", timeout=60))
    inspect_path = a.output / "inspect.json"
    if not inspect_path.exists():
        print("INSPECT", a.operator, flush=True)
        save(inspect_path, call(a.server, "/inspect", {"api_version": "v6.2", "binding": binding}))
    source_path = a.output / (a.label + ".py")
    if source_path.exists() and source_path.read_text(encoding="utf-8") != source:
        raise RuntimeError("输出目录已有不同的候选源码，请使用新批次")
    source_path.write_text(source, encoding="utf-8")
    save(a.output / (a.label + ".provenance.json"), {
        "operator": a.operator, "source_path": str(a.source.resolve()),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "server": a.server, "binding": binding,
        "timing_scope": a.timing_scope,
    })
    for index in range(1, a.repeats + 1):
        name = f"{a.label}-{index}"
        result_path = a.output / (name + ".result.json")
        if result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if previous.get("status") != "PASSED":
                raise SystemExit("已存在未通过结果；保留该证据，复测请使用新目录")
            print("EXISTS PASSED", result_path, flush=True)
            continue
        request = {
            "api_version": "v6.2", "binding": binding,
            "implementation": {"name": a.operator + "-" + name, "definition": a.operator,
                "language": "triton", "entrypoint": "main.py::run",
                "sources": [{"path": "main.py", "content": source}]},
            "settings": {"warmup_ms": 1000, "benchmark_ms": 100,
                "num_trials": 3, "timeout_seconds": a.timeout},
        }
        save(a.output / (name + ".request.json"), request)
        print("START", name, flush=True)
        try:
            result = call(a.server, "/evaluate", request, a.timeout + 120)
        except urllib.error.HTTPError as e:
            result = {"status": "HTTP_ERROR", "http_status": e.code,
                      "detail": e.read().decode()}
        save(result_path, result)
        print("DONE", json.dumps({k: result.get(k) for k in
            ("status", "geo_mean", "min_speedup", "device", "num_passed", "num_workloads")}), flush=True)
        if result.get("status") != "PASSED":
            raise SystemExit("评测未通过；查看原始结果，勿将失败数据计入加速比")


if __name__ == "__main__":
    main()
