#!/usr/bin/env python3
"""在 910B 上分解 narrow_copy 的分配、launcher、编译和同步开销。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401


def load(path: Path):
    spec = importlib.util.spec_from_file_location("decompose_candidate", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载候选: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host_call(function, *args):
    torch.npu.synchronize()
    begin = time.perf_counter_ns()
    value = function(*args)
    enqueued = time.perf_counter_ns()
    torch.npu.synchronize()
    finished = time.perf_counter_ns()
    return value, (enqueued - begin) / 1000, (finished - begin) / 1000


def median(values):
    return {"median_us": statistics.median(values), "min_us": min(values),
            "max_us": max(values)}


def measure_case(module, source, shape, dtype, dim, start, length, repeats):
    # 先建立 plan 和编译缓存；cold_run 单独保留，不与 warm 样本混合。
    torch.npu.synchronize()
    cold_begin = time.perf_counter_ns()
    cold_output = module.run(source, dim, start, length)
    torch.npu.synchronize()
    cold_us = (time.perf_counter_ns() - cold_begin) / 1000
    del cold_output

    warm = {"run": [], "empty": [], "launch": [], "pytorch": []}
    # 对连续 dim=0 case，使用候选自己的 plan 直接调用已解析 launcher，
    # 只测已分配 output 后的 launcher 路径；这不是改变候选 ABI 的优化。
    plan = module._make_plan(dim, start, length, tuple(source.shape),
                             tuple(source.stride()), source.ndim)
    launcher, args, block, even = plan[1], plan[2], plan[3], plan[4]
    output_shape = plan[0]
    if launcher is None or launcher is module._GENERIC:
        direct_available = False
    else:
        direct_available = True
    # 编译缓存中的对象是已经绑定 grid/编译选项的低层 launcher。若后端
    # 暴露该对象，优先测它；否则退回 JIT wrapper，并在结果中注明。
    cached_launcher = None
    # _COMPILED 的 key 含 shape/dtype、device 和输入/输出地址低位；不能
    # 随意取第一个 specialization，否则大 case 会误用小 case 的 runner。
    try:
        probe_output = source.new_empty(output_shape)
        cache_key = ((dim, start, length, tuple(source.shape), tuple(source.stride()), source.dtype),
                     source.device.index, source.data_ptr() & 15, probe_output.data_ptr() & 15)
        cached_launcher = getattr(module, "_COMPILED", {}).get(cache_key)
        del probe_output
    except Exception:
        cached_launcher = None
    cached_launcher_error = None
    direct_kind = "compiled_cache" if cached_launcher is not None else "jit_wrapper"

    for index in range(repeats + 5):
        # 交替顺序降低同一进程中两条路径的时段偏差。
        for kind in (("run", module.run), ("pytorch", torch.narrow_copy)):
            if index % 2:
                kind = ("pytorch", torch.narrow_copy) if kind[0] == "run" else ("run", module.run)
            name, function = kind
            value, enqueue, wall = host_call(function, source, dim, start, length)
            if index >= 5:
                warm[name].append((enqueue, wall))
            del value
            if index >= 5 and name == "run":
                torch.npu.synchronize()

        if index >= 5:
            _, enqueue, wall = host_call(source.new_empty, output_shape)
            warm["empty"].append((enqueue, wall))
            if direct_available:
                output = source.new_empty(output_shape)
                torch.npu.synchronize()
                begin = time.perf_counter_ns()
                direct = cached_launcher or launcher
                try:
                    if cached_launcher is not None:
                        # run() 自身也会把 BLOCK/EVEN 作为 constexpr 参数传给
                        # compiled runner；缺少这两个参数会制造假的 ABI 错误。
                        direct(output, source, *args, block, even)
                    elif len(args) == 2:
                        direct(output, source, args[0], args[1], block, even)
                    elif len(args) == 3:
                        direct(output, source, args[0], args[1], args[2], block, even)
                    else:
                        direct(output, source, args[0], args[1], args[2], args[3], block, even)
                except TypeError as error:
                    # FlagTree 的 compiled runner 不是稳定的 Python ABI；记录
                    # 失败签名并继续测 JIT wrapper，避免把内部对象当成公开入口。
                    cached_launcher_error = f"{type(error).__name__}: {error}"
                    cached_launcher = None
                    direct_kind = "jit_wrapper_after_compiled_abi_error"
                    continue
                enqueued = time.perf_counter_ns()
                torch.npu.synchronize()
                finished = time.perf_counter_ns()
                warm["launch"].append(((enqueued - begin) / 1000, (finished - begin) / 1000))
                del output

    def summarize(samples):
        return {"enqueue": median([item[0] for item in samples]),
                "wall": median([item[1] for item in samples])}

    return {"shape": list(shape), "dtype": str(dtype), "dim": dim,
            "start": start, "length": length, "cold_run_wall_us": cold_us,
            "direct_launcher_available": direct_available,
            "direct_launcher_kind": direct_kind,
            "compiled_launcher_error": cached_launcher_error,
            "warm": {name: summarize(values) for name, values in warm.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 3:
        parser.error("输出必须不存在，repeats 至少为 3")
    torch.npu.set_device(args.device)
    module = load(args.candidate)
    cases = (("small-f32", (64, 64), torch.float32, 0, 16, 32),
             ("large-f16", (1024, 65536), torch.float16, 0, 256, 512))
    results = []
    for label, shape, dtype, dim, start, length in cases:
        source = torch.ones(shape, device=args.device, dtype=dtype)
        value = measure_case(module, source, shape, dtype, dim, start, length, args.repeats)
        value["case"] = label
        results.append(value)
    result = {"schema_version": "1.0", "candidate": str(args.candidate),
              "device": args.device, "repeats": args.repeats,
              "scope": "同一进程的 host perf_counter 与显式 NPU synchronize；不是 KGS walltime",
              "cases": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
