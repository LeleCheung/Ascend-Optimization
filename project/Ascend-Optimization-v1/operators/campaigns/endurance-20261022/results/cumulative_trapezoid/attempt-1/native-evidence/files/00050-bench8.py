"""Find the cheapest correct Triton launch path on this backend.

Compares: standard JIT dispatch, CompiledKernel.__getitem__ runner, and a raw
ck.run(...) call, for a 2-pointer kernel and an 8-parameter kernel.  Also times
output-allocation variants.
"""
import json
import os
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
res = {"cases": [], "errors": [], "info": {}}


@triton.jit
def k2(a_ptr, b_ptr):
    i = tl.program_id(0) + tl.arange(0, 64)
    tl.store(b_ptr + i, tl.load(a_ptr + i))


@triton.jit
def k8(a_ptr, b_ptr, NOUT: tl.constexpr, ROW_IN: tl.constexpr, SCAN_IN: tl.constexpr,
       BR: tl.constexpr, L: tl.constexpr, SCALE: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BR + tl.arange(0, BR)
    c = tl.arange(0, L)
    m = (c < NOUT)[None, :]
    x = tl.load(a_ptr + r[:, None] * ROW_IN + c[None, :] * SCAN_IN, mask=m, other=0.0).to(tl.float32)
    x = tl.cumsum(x * SCALE, axis=1)
    tl.store(b_ptr + r[:, None] * NOUT + c[None, :], x, mask=m)


def wall(fn, iters=100, warm=20):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    torch.npu.set_device(0)
    from triton.runtime import driver
    a = torch.randn(128, 4096, dtype=torch.float16).npu()
    b = torch.empty(128, 4095, dtype=torch.float16, device="npu:0")

    res["info"]["torch_empty"] = wall(lambda: torch.empty((128, 4095), dtype=torch.float16, device="npu:0"))
    res["info"]["torch_new_empty"] = wall(lambda: a.new_empty((128, 4095)))
    res["info"]["torch_empty_like"] = wall(lambda: torch.empty_like(b))

    # --- standard dispatch (reference point)
    res["info"]["k2_standard"] = wall(lambda: k2[(8,)](a, b))
    res["info"]["k8_standard"] = wall(
        lambda: k8[(4,)](a, b, 4095, 4096, 1, 32, 128, 0.5))

    # --- warmup shapes: positional vs keyword constexprs
    for mode in ("kw", "pos"):
        try:
            if mode == "kw":
                ck = k8.warmup(a, b, NOUT=4095, ROW_IN=4096, SCAN_IN=1, BR=32, L=128,
                               SCALE=0.5, grid=(4,))
            else:
                ck = k8.warmup(a, b, 4095, 4096, 1, 32, 128, 0.5, grid=(4,))
            info = {"mode": mode, "md": str(ck.metadata)[:400]}
            try:
                runner = ck[(4, 1, 1)]
                runner(a, b)
                torch.npu.synchronize()
                info["runner_ok"] = True
                info["runner_ms"] = wall(lambda: runner(a, b))
            except Exception as e:
                info["runner_err"] = repr(e)[:200]
                runner = None
            # raw ck.run with explicit stream
            try:
                stream = driver.active.get_current_stream(0)
                md = ck.launch_metadata((4, 1, 1), stream, a, b, 4095, 4096, 1, 32, 128, 0.5)
                info["md_sig_ok"] = True
            except Exception as e:
                info["md_sig_err"] = repr(e)[:200]
                md = None
            if md is not None:
                try:
                    ck.run(4, 1, 1, stream, ck.function, ck.packed_metadata, md, None, None,
                           a, b, 4095, 4096, 1, 32, 128, 0.5, md, md, md)
                    torch.npu.synchronize()
                    info["raw_ok"] = True
                except Exception as e:
                    info["raw_err"] = repr(e)[:200]
            res["info"][f"k8_{mode}"] = info
        except Exception as e:
            res["errors"].append({"what": f"k8_{mode}", "err": repr(e)[-300:]})

    # --- k2 pre-bound runner (2 pointers only)
    try:
        ck2 = k2.warmup(a, b, grid=(8,))
        info = {"md": str(ck2.metadata)[:400]}
        runner2 = ck2[(8, 1, 1)]
        runner2(a, b)
        torch.npu.synchronize()
        info["runner_ms"] = wall(lambda: runner2(a, b))
        # try dropping the metadata/stream lookups: call ck.run directly
        stream = driver.active.get_current_stream(0)
        try:
            ck2.run(8, 1, 1, stream, ck2.function, ck2.packed_metadata, None, None, None, a, b)
            torch.npu.synchronize()
            info["raw_direct_ms"] = wall(
                lambda: ck2.run(8, 1, 1, stream, ck2.function, ck2.packed_metadata, None, None, None, a, b))
        except Exception as e:
            info["raw_direct_err"] = repr(e)[:300]
        res["info"]["k2_prebuilt"] = info
    except Exception as e:
        res["errors"].append({"what": "k2_prebuilt", "err": repr(e)[-300:]})

    # --- stream read cost
    res["info"]["get_current_stream"] = wall(lambda: driver.active.get_current_stream(0))
    res["info"]["npustream"] = wall(lambda: torch.npu.current_stream())

    print(json.dumps(res, indent=1)[:9000])
    with open(os.path.join(ART, "bench8.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
