"""Measure host launch overhead vs device time, and dump failing-kernel errors."""
import inspect
import json
import os
import time

import torch
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
res = {"wall": {}, "dev": {}, "errors": [], "info": {}}


@triton.jit
def k_nop(x_ptr):
    tl.store(x_ptr + tl.program_id(0), 0.0)


@triton.jit
def k_nop4(a_ptr, b_ptr, n, s, BR: tl.constexpr, BLOCK: tl.constexpr):
    tl.store(a_ptr + tl.program_id(0), 0.0)


def wall(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def dev(fn, iters=50, warm=10):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    e0 = torch.npu.Event(enable_timing=True)
    e1 = torch.npu.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.npu.synchronize()
    return e0.elapsed_time(e1) / iters


def main():
    torch.npu.set_device(0)
    res["info"]["triton_version"] = triton.__version__
    z = torch.zeros(8192, dtype=torch.float32, device="npu:0")
    y = torch.randn(128, 1024, dtype=torch.float16, device="npu:0")

    res["wall"]["nop_grid1"] = wall(lambda: k_nop[(1,)](z))
    res["wall"]["nop_grid64"] = wall(lambda: k_nop[(64,)](z))
    res["wall"]["nop4_grid64"] = wall(lambda: k_nop4[(64,)](z, z, 1024, 1, BR=1, BLOCK=1024))
    res["dev"]["nop_grid64"] = dev(lambda: k_nop[(64,)](z))
    res["wall"]["torch_mul"] = wall(lambda: y.mul(1.0))
    res["wall"]["torch_mul_"] = wall(lambda: y.mul_(1.0))
    res["wall"]["torch_cumsum"] = wall(lambda: torch.cumsum(y, dim=-1))
    res["wall"]["torch_cumtrapz"] = wall(lambda: torch.cumulative_trapezoid(y, dx=1, dim=-1))
    res["dev"]["torch_cumsum"] = dev(lambda: torch.cumsum(y, dim=-1))

    # ---- introspection of the launch path
    from triton.runtime.jit import JITFunction
    try:
        src = inspect.getsource(JITFunction.run)
        res["info"]["JITFunction.run"] = src
    except Exception as e:
        res["info"]["JITFunction.run"] = repr(e)
    try:
        ck = k_nop.warmup(z, grid=(64,))
        res["info"]["compiled_kernel_type"] = type(ck).__name__
        res["info"]["ck_attrs"] = [a for a in dir(ck) if not a.startswith("__")]
        res["info"]["ck_run_isdoc"] = (ck.run.__doc__ or "")[:500]
        res["info"]["ck_run_sig"] = str(inspect.signature(ck.run)) if hasattr(ck.run, "__call__") else "?"
        try:
            res["info"]["ck_getitem_src"] = inspect.getsource(type(ck).__getitem__)
        except Exception as e:
            res["info"]["ck_getitem_src"] = repr(e)
    except Exception as e:
        res["errors"].append({"what": "warmup", "err": repr(e)[:600]})

    # ---- fast-launch candidates
    try:
        ck = k_nop.warmup(z, grid=(64,))
        runner = ck[(64, 1, 1)]
        runner(z)
        torch.npu.synchronize()
        res["wall"]["fast_getitem_nop"] = wall(lambda: runner(z))
        res["dev"]["fast_getitem_nop"] = dev(lambda: runner(z))
    except Exception as e:
        res["errors"].append({"what": "fast_getitem_nop", "err": repr(e)[:800]})

    try:
        ck = k_nop4.warmup(z, z, 1024, 1, grid=(64,), BR=1, BLOCK=1024)
        runner = ck[(64, 1, 1)]
        runner(z, z, 1024, 1)
        torch.npu.synchronize()
        res["wall"]["fast_getitem_nop4"] = wall(lambda: runner(z, z, 1024, 1))
        res["dev"]["fast_getitem_nop4"] = dev(lambda: runner(z, z, 1024, 1))
    except Exception as e:
        res["errors"].append({"what": "fast_getitem_nop4", "err": repr(e)[:800]})

    # dimension-0 scan support probe (full error text)
    INNER = 128
    N = 1024
    yy = torch.randn(N, INNER, dtype=torch.float16, device="npu:0")
    out = torch.empty(N - 1, INNER, dtype=torch.float16, device="npu:0")

    @triton.jit
    def k_ax0(y_ptr, out_ptr, N, INNER, BI: tl.constexpr, L: tl.constexpr):
        i = tl.program_id(0) * BI + tl.arange(0, BI)
        k = tl.arange(0, L)
        m = (k[:, None] < (N - 1)) & (i[None, :] < INNER)
        a = tl.load(y_ptr + k[:, None] * INNER + i[None, :], mask=m, other=0.0).to(tl.float32)
        cs = tl.cumsum(a, axis=0)
        tl.store(out_ptr + k[:, None] * INNER + i[None, :], cs, mask=m)

    for bi, L in ((128, 1024), (32, 128), (64, 256)):
        try:
            k_ax0[(triton.cdiv(INNER, bi),)](yy, out, N, INNER, BI=bi, L=L)
            torch.npu.synchronize()
            r = torch.cumulative_trapezoid(yy, dx=0.5, dim=0)
            res["info"][f"ax0_bi{bi}_L{L}"] = "OK maxabs=%.4f" % (out.float() - r.float()).abs().max().item()
        except Exception as e:
            res["errors"].append({"what": f"ax0_bi{bi}_L{L}", "err": repr(e)[:1500]})

    @triton.jit
    def k_trans(y_ptr, out_ptr, N, INNER, BI: tl.constexpr, L: tl.constexpr):
        i = tl.program_id(0) * BI + tl.arange(0, BI)
        k = tl.arange(0, L)
        m = (k[None, :] < (N - 1)) & (i[:, None] < INNER)
        a = tl.load(y_ptr + k[None, :] * INNER + i[:, None], mask=m, other=0.0).to(tl.float32)
        cs = tl.cumsum(a, axis=1)
        tl.store(out_ptr + k[None, :] * INNER + i[:, None], cs, mask=m)

    for bi, L in ((128, 1024), (128, 256), (64, 512)):
        try:
            k_trans[(triton.cdiv(INNER, bi),)](yy, out, N, INNER, BI=bi, L=L)
            torch.npu.synchronize()
            r = torch.cumulative_trapezoid(yy, dx=0.5, dim=0)
            res["info"][f"trans_bi{bi}_L{L}"] = "OK maxabs=%.4f" % (out.float() - r.float()).abs().max().item()
            res["wall"][f"trans_bi{bi}_L{L}"] = wall(
                lambda bi=bi, L=L: k_trans[(triton.cdiv(INNER, bi),)](yy, out, N, INNER, BI=bi, L=L))
        except Exception as e:
            res["errors"].append({"what": f"trans_bi{bi}_L{L}", "err": repr(e)[:1500]})

    with open(os.path.join(ART, "bench2.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:26000])


if __name__ == "__main__":
    main()
