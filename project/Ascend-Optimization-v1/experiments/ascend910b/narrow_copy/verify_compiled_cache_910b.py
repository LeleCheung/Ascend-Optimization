#!/usr/bin/env python3
"""验证候选确实命中编译后 launcher，而不只是普通 JIT 路径。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401


class CountingCache(dict):
    hits = 0

    def get(self, key, default=None):
        value = super().get(key, default)
        if value is not None:
            self.hits += 1
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="npu:4")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"输出已存在: {args.output}")
    torch.npu.set_device(args.device)
    spec = importlib.util.spec_from_file_location("narrow_copy_compiled_probe", args.candidate)
    if spec is None or spec.loader is None:
        raise ValueError("无法加载候选")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._COMPILED = CountingCache()
    source = torch.arange(4096, device=args.device, dtype=torch.float32).reshape(64, 64)
    expected = torch.narrow_copy(source, 0, 16, 32)
    counts = []
    for _ in range(6):
        actual = module.run(source, 0, 16, 32)
        torch.npu.synchronize()
        if not torch.equal(actual, expected):
            raise ValueError("编译缓存验证期间结果错误")
        counts.append({"compiled": len(module._COMPILED), "hits": module._COMPILED.hits,
                       "disabled": len(module._NO_COMPILED)})
    result = {"device": args.device, "counts_after_each_call": counts,
              "cache_hit_verified": counts[-1]["hits"] > 0,
              "disabled_signatures": counts[-1]["disabled"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["cache_hit_verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
