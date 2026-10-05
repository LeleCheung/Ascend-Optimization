#!/usr/bin/env python3
"""在 910B 上对 narrow_copy 候选做固定 workload 之外的语义复验。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("narrow_copy_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载候选: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def check(run, label: str, source: torch.Tensor, dim: int, start: int, length: int) -> dict:
    expected = torch.narrow_copy(source, dim, start, length)
    actual = run(source, dim, start, length)
    torch.npu.synchronize()
    matched = (actual.shape == expected.shape and actual.dtype == expected.dtype
               and actual.device == expected.device and torch.equal(actual, expected))
    no_alias = actual.numel() == 0 or actual.data_ptr() != source.data_ptr()
    return {"case": label, "status": "PASSED" if matched and no_alias else "FAILED",
            "shape": list(source.shape), "stride": list(source.stride()),
            "dim": dim, "start": start, "length": length,
            "matched": matched, "no_alias": no_alias}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"输出已存在: {args.output}")
    run = load_candidate(args.candidate)
    cases = []
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        base = torch.arange(120, dtype=torch.float32).reshape(4, 5, 6).to("npu", dtype=dtype)
        suffix = str(dtype).split(".")[-1]
        cases.extend([
            check(run, suffix + "/runs", base, 1, 1, 3),
            check(run, suffix + "/negative-start", base, 1, -2, 2),
            check(run, suffix + "/negative-dim", base, -1, 1, 3),
            check(run, suffix + "/transpose", base.transpose(0, 1), 1, 1, 2),
            check(run, suffix + "/strided", base[:, ::2, :], 1, 1, 2),
            check(run, suffix + "/zero-length", base, 1, 2, 0),
            check(run, suffix + "/full-length", base, 1, 0, 5),
            check(run, suffix + "/cache-second-input", base + 100, 1, 1, 3),
            check(run, suffix + "/empty-axis", base[:, :0, :], 1, 0, 0),
        ])
    result = {"candidate": str(args.candidate), "num_cases": len(cases),
              "num_passed": sum(item["status"] == "PASSED" for item in cases),
              "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("num_cases", "num_passed")}, ensure_ascii=False))
    if result["num_passed"] != result["num_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
