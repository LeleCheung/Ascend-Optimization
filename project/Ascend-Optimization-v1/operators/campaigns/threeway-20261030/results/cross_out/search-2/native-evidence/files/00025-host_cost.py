"""Price host dispatch versus device kernel for cross_out, and tune BLOCK.

Levels measured on the same tensors:
  bare   : only the raw launcher call with everything precomputed
  lean   : minimal wrapper (memoized plan, cached stream)
  run    : the submitted candidate's own run() entry point
Also sweeps BLOCK for the 1M-vector fp16 and fp32 cases.
Diagnostic evidence only.
"""
import importlib.util
import json
import os
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl


def load_candidate():
    spec = importlib.util.spec_from_file_location("cand_main", "tmp/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    m = load_candidate()
    dev = "npu:0"
    kern = m._cross_flat_kernel
    results = {}

    # ---- host dispatch levels on small and mid sizes ---------------------- #
    for name, nvec, iters in (("small_1024", 1024, 200), ("mid_65536", 65536, 200)):
        n = nvec * 3
        a = torch.randn(n, dtype=torch.float16, device=dev)
        b = torch.randn(n, dtype=torch.float16, device=dev)
        out = torch.empty(n, dtype=torch.float16, device=dev)
        av, bv, ov = a.view(nvec, 3), b.view(nvec, 3), out.view(nvec, 3)
        row = {}

        row["run"] = bench(lambda: m.run(av, bv, -1, out=ov), iters)

        block, step = 1024, 1023
        grid = (n + step - 1) // step
        compiled = kern.warmup(a, b, out, n, BLOCK=block, STEP=step, grid=(1,))
        launcher = compiled.run
        lf, fc, pm = launcher.launch, compiled.function, compiled.packed_metadata
        stream = m._current_stream(a.device.index)
        pa, pb, po = a.data_ptr(), b.data_ptr(), out.data_ptr()

        def bare():
            lf(grid, 1, 1, stream, fc, pm, None, None, None, pa, pb, po, n, block, step)

        def bare_streamq():
            lf(grid, 1, 1, m._current_stream(a.device.index), fc, pm, None, None, None,
               pa, pb, po, n, block, step)

        row["bare"] = bench(bare, iters)
        row["bare_plus_streamquery"] = bench(bare_streamq, iters)
        row["run_minus_bare_ms"] = row["run"] - row["bare"]
        results[name] = row
        print(name, json.dumps(row), flush=True)

    # ---- BLOCK sweep on the large cases ---------------------------------- #
    for name, dt, nvec, blocks, iters in (
        ("fp16_1M", torch.float16, 1048576, (1024, 2048, 4096), 30),
        ("fp32_1M", torch.float32, 1048576, (512, 1024, 2048), 30),
    ):
        n = nvec * 3
        torch.manual_seed(0)
        a = torch.randn(n, dtype=dt, device=dev)
        b = torch.randn(n, dtype=dt, device=dev)
        out = torch.empty(n, dtype=dt, device=dev)
        ref = torch.cross(a.view(nvec, 3), b.view(nvec, 3), dim=-1).reshape(-1)
        row = {}
        for blk in blocks:
            step = (blk // 3) * 3
            g = ((n + step - 1) // step,)
            entry = {}
            fn = (lambda blk=blk, step=step, g=g:
                  kern[g](a, b, out, n, BLOCK=blk, STEP=step))
            try:
                out.zero_()
                torch.npu.synchronize()
                fn()
                torch.npu.synchronize()
                entry["max_abs_err"] = (out.float() - ref.float()).abs().max().item()
                entry["ms"] = bench(fn, iters)
            except Exception:
                entry["error"] = traceback.format_exc().splitlines()[-1]
            row["b%d" % blk] = entry
        row["torch_cross_out"] = {"ms": bench(
            lambda: torch.cross(a.view(nvec, 3), b.view(nvec, 3), dim=-1, out=out.view(nvec, 3)),
            iters)}
        results[name] = row
        print(name, json.dumps(row), flush=True)

    path = os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "host_cost.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE", path)


if __name__ == "__main__":
    main()
