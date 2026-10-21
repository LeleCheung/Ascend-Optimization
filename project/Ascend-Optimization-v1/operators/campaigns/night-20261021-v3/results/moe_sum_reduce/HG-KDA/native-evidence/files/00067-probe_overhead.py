"""Diagnose host-dispatch vs device-time split for moe_sum_reduce candidate."""
import importlib.util
import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

OUT = {}


def load_cand():
    spec = importlib.util.spec_from_file_location("cand_main", "tmp/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@triton.jit
def _trivial_kernel(ptr):
    tl.store(ptr + tl.program_id(0), 1.0)


@triton.jit
def _trivial_kernel_many(ptr, a, b, c, d, e, f, g, h, BLOCK: tl.constexpr):
    tl.store(ptr + tl.program_id(0), a + b + c + d + e + f + g + h)


def bench(fn, n, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
        torch.npu.synchronize()
    t_sync = (time.perf_counter() - t0) / n * 1e6
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    t_nosync = (time.perf_counter() - t0) / n * 1e6
    # device-side event timing (dispatch pipelined)
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.npu.synchronize()
    t_dev = s.elapsed_time(e) / n * 1e3
    return {"sync_us": round(t_sync, 2), "nosync_us": round(t_nosync, 2), "event_us": round(t_dev, 2)}


def main():
    props = torch.npu.get_device_properties(0)
    info = {"name": props.name}
    for k in ("num_aicore", "num_vectorcore", "total_memory"):
        if hasattr(props, k):
            info[k] = getattr(props, k)
    OUT["device"] = info

    cand = load_cand()
    ref = lambda x, s: x.float().sum(dim=1).mul(s).to(x.dtype)

    shapes = [(1, 8, 7168), (8, 8, 7168), (64, 8, 7168), (512, 8, 7168), (4096, 8, 7168)]
    tmp = torch.empty(4, dtype=torch.float32, device="npu")
    OUT["trivial_1prog"] = bench(lambda: _trivial_kernel[(1,)](tmp), 200)
    OUT["trivial_1prog_12args"] = bench(
        lambda: _trivial_kernel_many[(1,)](tmp, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, BLOCK=8), 200
    )
    OUT["trivial_1024prog"] = bench(lambda: _trivial_kernel[(1024,)](tmp), 200)
    OUT["torch_empty"] = bench(lambda: torch.empty((4096, 7168), dtype=torch.bfloat16, device="npu"), 200)

    for shape in shapes:
        x = torch.randn(shape, dtype=torch.float32, device="npu").to(torch.bfloat16)
        n = 50 if shape[0] >= 512 else 200
        OUT[f"cand_{shape[0]}"] = bench(lambda: cand.run(x, 2.5), n)
        OUT[f"ref_{shape[0]}"] = bench(lambda: ref(x, 2.5), n)
        del x

    path = os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "overhead_probe.json")
    with open(path, "w") as f:
        json.dump(OUT, f, indent=2)
    print(json.dumps(OUT, indent=2))


if __name__ == "__main__":
    main()
