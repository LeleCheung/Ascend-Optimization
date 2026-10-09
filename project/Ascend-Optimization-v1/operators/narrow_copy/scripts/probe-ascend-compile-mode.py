#!/usr/bin/env python3
"""验证 FlagTree Ascend 后端公开的 compile_mode=simt_only 入口。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def copy_kernel(out, inp, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(out + offsets, tl.load(inp + offsets, mask=mask), mask=mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    inp = torch.arange(128, device=args.device, dtype=torch.float32)
    out = torch.empty_like(inp)
    result = {"device": args.device, "kernel": "copy_kernel", "compile_mode": "simt_only"}
    begin = time.perf_counter_ns()
    try:
        # compile_mode 是后端 launch option；若版本不接受它，这里应明确失败。
        copy_kernel[(1,)](out, inp, 128, BLOCK=128, compile_mode="simt_only")
        torch.npu.synchronize()
        result.update({"status": "PASSED", "cold_wall_us": (time.perf_counter_ns() - begin) / 1000,
                       "equal": bool(torch.equal(out, inp))})
    except Exception as error:
        result.update({"status": "FAILED", "error_type": type(error).__name__,
                       "error": str(error)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
