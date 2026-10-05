#!/usr/bin/env python3
"""在专用 KGS 上评测一份固定的 narrow_copy Triton 候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def post_json(server: str, route: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        server + route,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:19652")
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"输出已存在: {args.output}")
    source = args.candidate.read_text(encoding="utf-8")
    binding = {"catalog_name": "flaggems-adapter-definitions", "definition": "narrow_copy"}
    inspect = post_json(args.server, "/inspect", {"api_version": "v6.2", "binding": binding}, 60)
    if inspect.get("adapter_kind") not in (None, "flaggems"):
        raise ValueError(f"意外的 adapter: {inspect.get('adapter_kind')}")

    request = {
        "api_version": "v6.2",
        "binding": binding,
        "implementation": {
            "name": "narrow_copy_launch_plan_fixed",
            "definition": "narrow_copy",
            "language": "triton",
            "entrypoint": "main.py::run",
            "sources": [{"path": "main.py", "content": source}],
        },
        "settings": {
            "warmup_ms": 1000,
            "benchmark_ms": 100,
            "num_trials": 3,
            "timeout_seconds": 1500,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output.with_suffix(".inspect.json"), inspect)
    write_json(args.output.with_suffix(".request.json"), request)
    response = post_json(args.server, "/evaluate", request, 1560)
    response["candidate_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    write_json(args.output, response)
    print(json.dumps({key: response.get(key) for key in
                      ("status", "geo_mean", "num_workloads", "num_passed", "log")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
