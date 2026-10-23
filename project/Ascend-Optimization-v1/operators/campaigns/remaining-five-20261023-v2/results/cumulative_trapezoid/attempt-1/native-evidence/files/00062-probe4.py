"""Probe: (a) symbols inside cann vec/mem/math op modules, (b) dim=0 (inner>1)
kernel variants for [1024,128]. Component probe, not a formal evaluation."""
import json
import os
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu"
OUT = {"timings_ms": {}, "errors": {}, "numerics": {}, "syms": {}}


def flush():
    with open(os.path.join(ART, "probe4.json"), "w") as f:
        json.dump(OUT, f, indent=2, default=str)


def say(*a):
    print(*a, flush=True)


# ---- current fallback: one program per column, runtime stride -------------
@triton.jit
def k_cur(y_ptr, o_ptr, N, inner, sri, sro, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    o = pid // inner
    i = pid % inner
    ib = o * sri + i
    ob = o * sro + i
    k = tl.arange(0, BLOCK)
    m = k < (N - 1)
    a = tl.load(y_ptr + ib + k * inner, mask=m, other=0.0).to(tl.float32)
    b = tl.load(y_ptr + ib + (k + 1) * inner, mask=m, other=0.0).to(tl.float32)
    cs = tl.cumsum((a + b) * 0.5, axis=0)
    tl.store(o_ptr + ob + k * inner, cs, mask=m)


# ---- variant A: [BI, BN] tile, constexpr inner stride, scan on last axis ---
@triton.jit
def k_bi_bn(y_ptr, o_ptr, N: tl.constexpr, INNER: tl.constexpr,
            BI: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BI + tl.arange(0, BI)
    k = tl.arange(0, BN)
    m = (k[None, :] < (N - 1)) & (i[:, None] < INNER)
    off = i[:, None] + k[None, :] * INNER
    a = tl.load(y_ptr + off, mask=m, other=0.0).to(tl.float32)
    b = tl.load(y_ptr + off + 1, mask=m, other=0.0).to(tl.float32)
    cs = tl.cumsum((a + b) * 0.5, axis=1)
    tl.store(o_ptr + i[:, None] + k[None, :] * INNER, cs, mask=m)


# ---- variant B: [BN, BI] tile (coalesced inner), scan on axis 0 -----------
@triton.jit
def k_bn_bi(y_ptr, o_ptr, N: tl.constexpr, INNER: tl.constexpr,
            BI: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BI + tl.arange(0, BI)
    k = tl.arange(0, BN)
    m = (k[:, None] < (N - 1)) & (i[None, :] < INNER)
    off = k[:, None] * INNER + i[None, :]
    a = tl.load(y_ptr + off, mask=m, other=0.0).to(tl.float32)
    b = tl.load(y_ptr + off + 1, mask=m, other=0.0).to(tl.float32)
    cs = tl.cumsum((a + b) * 0.5, axis=0)
    tl.store(o_ptr + off, cs, mask=m)


def bench(fn, iters=20):
    fn()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) / iters


def timed(key, fn):
    try:
        OUT["timings_ms"][key] = round(bench(fn), 4)
        say(key, OUT["timings_ms"][key])
    except Exception as ex:  # noqa: BLE001
        OUT["errors"][key] = str(traceback.format_exception_only(type(ex), ex))[:400]
        say(key, "ERROR", OUT["errors"][key][:250])
    flush()


def main():
    N, INNER = 1024, 128          # timing-2/5/8: y is [1024,128], dim=0
    torch.manual_seed(0)
    y = torch.randn(N, INNER, device=DEV, dtype=torch.float16)
    o = torch.empty(N - 1, INNER, device=DEV, dtype=torch.float16)

    # (a) module symbols
    try:
        import triton.language.extra.cann.extension as ext
        for nm in ("vec_ops", "mem_ops", "math_ops", "builtin_custom_ops",
                   "aux_ops", "custom", "core", "builtin"):
            mod = getattr(ext, nm, None)
            OUT["syms"][nm] = sorted(s for s in dir(mod) if not s.startswith("_")) \
                if mod is not None else "MISSING"
        try:
            import triton.language.extra.cann.extension.vec_ops as vo
            OUT["syms"]["vec_ops_attrs"] = sorted(
                s for s in dir(vo) if not s.startswith("_"))[:120]
        except Exception as ex:  # noqa: BLE001
            OUT["syms"]["vec_ops_attrs"] = repr(ex)
    except Exception as ex:  # noqa: BLE001
        OUT["errors"]["syms"] = repr(ex)
    flush()
    say("syms", json.dumps(OUT["syms"], indent=1)[:2500])

    timed("cur_1d", lambda: k_cur[(INNER,)](y, o, N, INNER, N * INNER,
                                            (N - 1) * INNER, 1024))
    for bi in (32, 64, 128):
        timed(f"bi_bn_bi{bi}", lambda bi=bi: k_bi_bn[(INNER // bi,)](
            y, o, N, INNER, bi, 1024))
    for bi in (32, 128):
        timed(f"bn_bi_bi{bi}", lambda bi=bi: k_bn_bi[(INNER // bi,)](
            y, o, N, INNER, bi, 1024))

    try:
        yc = y.cpu().double()
        exact = torch.cumsum((yc[:-1, :] + yc[1:, :]) * 0.5, dim=0)
        for key, fn in (("cur_1d", lambda: k_cur[(INNER,)](
                y, o, N, INNER, N * INNER, (N - 1) * INNER, 1024)),
                ("bi_bn", lambda: k_bi_bn[(INNER // 32,)](y, o, N, INNER, 32, 1024)),
                ("bn_bi", lambda: k_bn_bi[(INNER // 32,)](y, o, N, INNER, 32, 1024))):
            fn()
            torch.npu.synchronize()
            OUT["numerics"][key] = float(
                (o.cpu().double() - exact).abs().max())
    except Exception as ex:  # noqa: BLE001
        OUT["errors"]["numerics"] = str(traceback.format_exception_only(type(ex), ex))[:400]
    flush()
    say("numerics", json.dumps(OUT["numerics"], indent=2))
    say("errors", json.dumps(OUT["errors"], indent=2)[:1200])


if __name__ == "__main__":
    main()
