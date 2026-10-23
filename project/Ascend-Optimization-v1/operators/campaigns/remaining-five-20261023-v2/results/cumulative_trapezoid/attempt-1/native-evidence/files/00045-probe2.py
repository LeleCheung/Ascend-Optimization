"""Test a Cube(tl.dot)-based prefix scan against tl.cumsum on Ascend.

Component probe, not a formal evaluation. For each variant: device time for
[1024, 4096] and max abs error vs an FP64 CPU prefix of the same increments.
"""
import json
import os
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu"


@triton.jit
def k_cumsum(y_ptr, o_ptr, L: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, L)
    a = tl.load(y_ptr + pid * L + cols).to(tl.float32)
    b = tl.load(y_ptr + pid * L + cols + 1).to(tl.float32)
    cs = tl.cumsum((a + b) * 0.5, axis=0)
    tl.store(o_ptr + pid * L + cols, cs)


@triton.jit
def k_dot_scan(y_ptr, o_ptr, M, L: tl.constexpr, BM: tl.constexpr,
               C: tl.constexpr, NC: tl.constexpr, DT: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    kc = tl.arange(0, C)
    tri = tl.where(kc[:, None] >= kc[None, :], 1.0, 0.0).to(DT)
    carry = tl.zeros((BM,), dtype=tl.float32)
    for c in tl.static_range(NC):
        off = rows[:, None] * L + (c * C + kc)[None, :]
        a = tl.load(y_ptr + off).to(tl.float32)
        b = tl.load(y_ptr + off + 1).to(tl.float32)
        t = (a + b) * 0.5
        cs = tl.dot(t.to(DT), tri).to(tl.float32) + carry[:, None]
        tl.store(o_ptr + off, cs)
        carry += tl.sum(t, axis=1)


@triton.jit
def k_chunksums(y_ptr, o_ptr, M, L: tl.constexpr, BM: tl.constexpr,
                C: tl.constexpr, NC: tl.constexpr):
    """Only the per-chunk reductions (upper bound on a 3-kernel split)."""
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    kc = tl.arange(0, C)
    tot = tl.zeros((BM, NC), dtype=tl.float32)
    for c in tl.static_range(NC):
        off = rows[:, None] * L + (c * C + kc)[None, :]
        a = tl.load(y_ptr + off).to(tl.float32)
        b = tl.load(y_ptr + off + 1).to(tl.float32)
        s = tl.sum((a + b) * 0.5, axis=1)
        tot += tl.where(tl.arange(0, NC)[None, :] == c, s[:, None], 0.0)
    tl.store(o_ptr + rows[:, None] * NC + tl.arange(0, NC)[None, :], tot)


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


def main():
    out = {"timings_ms": {}, "errors": {}, "numerics": {}}
    M, L = 1024, 4096
    res = {}
    for dt, tdt in ((torch.float16, tl.float16), (torch.bfloat16, tl.bfloat16),
                    (torch.float32, tl.float32)):
        name = str(dt).split(".")[-1]
        torch.manual_seed(0)
        yf = torch.randn(M * L + 64, device=DEV, dtype=dt)
        y = yf[: M * L].view(M, L)
        o = torch.empty(M * L, device=DEV, dtype=dt)

        def t(key, fn):
            try:
                out["timings_ms"][f"{name}:{key}"] = round(bench(fn), 4)
            except Exception as ex:  # noqa: BLE001
                out["errors"][f"{name}:{key}"] = str(
                    traceback.format_exception_only(type(ex), ex))[:600]

        t("cumsum", lambda: k_cumsum[(M,)](y, o, L))
        best = None
        for bm, c in ((16, 128), (32, 128), (32, 256), (64, 128)):
            nc = L // c
            key = f"dot_bm{bm}_c{c}"
            t(key, lambda bm=bm, c=c, nc=nc: k_dot_scan[(M // bm,)](
                y, o, M, L, bm, c, nc, tdt))
        t("chunksums_bm32_c128", lambda: k_chunksums[(M // 32,)](
            y, o, M, L, 32, 128, L // 128))

        # numerical check of the dot path vs fp64 CPU prefix of the same increments
        try:
            bm, c = 32, 128
            y.zero_()
            torch.manual_seed(1)
            y.copy_(torch.randn(M, L, device=DEV, dtype=dt))
            k_dot_scan[(M // bm,)](y, o, M, L, bm, c, L // c, tdt)
            torch.npu.synchronize()
            yc = y.cpu().double()
            inc = (yc[:, :-1] + yc[:, 1:]) * 0.5
            ref_exact = torch.cumsum(inc, dim=1)
            got = o.view(M, L)[:, : L - 1].cpu().double()
            k_cumsum[(M,)](y, o, L)
            torch.npu.synchronize()
            got_cs = o.view(M, L)[:, : L - 1].cpu().double()
            out["numerics"][name] = {
                "dot_max_abs_vs_exact": float((got - ref_exact).abs().max()),
                "cumsum_max_abs_vs_exact": float((got_cs - ref_exact).abs().max()),
                "dot_vs_cumsum_max_abs": float((got - got_cs).abs().max()),
            }
        except Exception as ex:  # noqa: BLE001
            out["errors"][f"{name}:numerics"] = str(
                traceback.format_exception_only(type(ex), ex))[:600]

    # IR of the plain cumsum kernel (small L for readability)
    try:
        ysmall = torch.randn(64 * 64 + 8, device=DEV, dtype=torch.float16)
        osmall = torch.empty(64 * 64, device=DEV, dtype=torch.float16)
        k_cumsum[(64,)](ysmall, osmall, 64)
        torch.npu.synchronize()
        kern = None
        for v in k_cumsum.device_caches[torch.npu.current_device()][0].values():
            kern = v
        ta = str(kern.asm.get("ttadapter", ""))
        out["ir"] = {
            "ttadapter_len": len(ta),
            "scf_for": ta.count("scf.for"),
            "scf_while": ta.count("scf.while"),
            "vadd": ta.count("vadd"),
            "hivm": ta.count("hivm"),
            "head": ta[:4000],
        }
    except Exception as ex:  # noqa: BLE001
        out["errors"]["ir"] = str(traceback.format_exception_only(type(ex), ex))[:800]

    with open(os.path.join(ART, "probe2.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out["timings_ms"], indent=2))
    print("numerics", json.dumps(out["numerics"], indent=2))
    print("ir", json.dumps({k: v for k, v in out.get("ir", {}).items()
                            if k != "head"}, indent=2))
    print("errors", json.dumps(out["errors"], indent=2)[:3000])


if __name__ == "__main__":
    main()
