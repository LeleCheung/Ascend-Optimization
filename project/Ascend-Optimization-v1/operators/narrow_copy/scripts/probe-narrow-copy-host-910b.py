#!/usr/bin/env python3
"""分别测量 narrow_copy 的 host 调用与同步完成耗时。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def candidate_run(path: Path):
    spec = importlib.util.spec_from_file_location("narrow_copy_host_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载候选: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def sample(function, source, dim, start, length) -> tuple[float, float]:
    torch.npu.synchronize()
    begin = time.perf_counter_ns()
    output = function(source, dim, start, length)
    enqueued = time.perf_counter_ns()
    torch.npu.synchronize()
    finished = time.perf_counter_ns()
    del output
    return (enqueued - begin) / 1000, (finished - begin) / 1000


def describe(samples: list[tuple[float, float]]) -> dict:
    enqueue, wall = zip(*samples)
    return {"enqueue_us_median": statistics.median(enqueue),
            "enqueue_us_min": min(enqueue), "enqueue_us_max": max(enqueue),
            "wall_us_median": statistics.median(wall),
            "wall_us_min": min(wall), "wall_us_max": max(wall)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    if args.repeats < 3 or args.output.exists():
        parser.error("repeats 至少为 3 且输出路径必须不存在")
    torch.npu.set_device(args.device)
    run = candidate_run(args.candidate)
    cases = (
        ("small-f32", (64, 64), torch.float32, 0, 16, 32),
        ("mid-bf16", (4096, 4096), torch.bfloat16, 0, 1024, 2048),
        ("large-f16", (1024, 65536), torch.float16, 0, 256, 512),
    )
    results = []
    for label, shape, dtype, dim, start, length in cases:
        source = torch.ones(shape, device=args.device, dtype=dtype)
        samples = {"candidate": [], "pytorch": []}
        for iteration in range(args.repeats + 5):
            order = (("candidate", run), ("pytorch", torch.narrow_copy))
            if iteration % 2:
                order = tuple(reversed(order))
            for name, function in order:
                value = sample(function, source, dim, start, length)
                if iteration >= 5:
                    samples[name].append(value)
        results.append({"case": label, "shape": shape, "dtype": str(dtype),
                        "dim": dim, "start": start, "length": length,
                        "candidate": describe(samples["candidate"]),
                        "pytorch": describe(samples["pytorch"])})
    result = {"measurement_scope": "单次 Python 调用返回和后续 NPU 同步，非 KGS 正式计时",
              "device": args.device,
              "repeats": args.repeats, "cases": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
