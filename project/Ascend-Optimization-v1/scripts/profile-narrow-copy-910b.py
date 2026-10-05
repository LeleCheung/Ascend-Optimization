#!/usr/bin/env python3
"""在 910B 容器内对固定 narrow_copy 候选发起 KGS 精确 case profiling。"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path


def request_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("case_id", help="正式评测 JSON 中的 timing uuid")
    parser.add_argument("output", type=Path)
    parser.add_argument("--level", choices=("metrics", "instruction"), default="metrics")
    parser.add_argument("--server", default="http://127.0.0.1:19652")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    binding = {"catalog_name": "flaggems-adapter-definitions", "definition": "narrow_copy"}
    inspect = request_json(
        args.server + "/inspect", {"api_version": "v6.2", "binding": binding}, 60
    )
    fingerprint = inspect["benchmark_fingerprint"]
    source = args.candidate.read_text(encoding="utf-8")
    implementation = {
        "name": "narrow_copy_profile_probe",
        "definition": "narrow_copy",
        "language": "triton",
        "entrypoint": "narrow_copy.py::run",
        "sources": [{"path": "narrow_copy.py", "content": source}],
    }
    payload = {
        "api_version": "v6.2",
        "binding": binding,
        "implementation": implementation,
        "benchmark_fingerprint": fingerprint,
        "case_id": args.case_id,
        "expected_backend": "npu",
        "options": {
            "level": args.level,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "timeout_sec": args.timeout,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / (args.output.stem + "-request.json")).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        result = request_json(args.server + "/profile", payload, args.timeout + 30)
    except urllib.error.HTTPError as error:
        result = {"http_status": error.code, "error": error.read().decode("utf-8", "replace")}
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result.get(key) for key in ("status", "error", "summary", "warnings")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
