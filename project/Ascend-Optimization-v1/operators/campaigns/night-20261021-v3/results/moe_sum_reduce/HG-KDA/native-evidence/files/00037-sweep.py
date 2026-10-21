"""Sweep kernel structures/configs to find the bandwidth limiter (diagnostic only)."""
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

RES = {"host": {}, "dev4096": {}, "dev512": {}, "dev64": {}, "errors": {}}
SCALE = 2.5


def bench(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return round((time.perf_counter() - t0) / n * 1e6, 2)


@triton.jit
def kStore(out_ptr, scale, DN: tl.constexpr):
    pn = tl.program_id(0) * DN + tl.arange(0, DN)
    tl.store(out_ptr + pn, tl.zeros((DN,), dtype=tl.float32) * scale)


@triton.jit
def kA(in_ptr, out_ptr, scale, NTOK: tl.constexpr, NHID: tl.constexpr,
       TK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0) * BM + tl.arange(0, BM)
    pn = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = (pm[:, None] < NTOK) & (pn[None, :] < NHID)
    base = in_ptr + pm[:, None] * (TK * NHID) + pn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(TK):
        v = tl.load(base + k * NHID, mask=mask, other=0.0)
        acc += v.to(tl.float32)
    tl.store(out_ptr + pm[:, None] * NHID + pn[None, :], acc * scale, mask=mask)


@triton.jit
def kB(in_ptr, out_ptr, scale, NTOK: tl.constexpr, NHID: tl.constexpr,
       TK: tl.constexpr, DN: tl.constexpr):
    pid = tl.program_id(0)
    pn = tl.program_id(1) * DN + tl.arange(0, DN)
    mask = pn[None, :] < NHID
    offs = tl.arange(0, TK)[:, None] * NHID + pn[None, :]
    x = tl.load(in_ptr + pid * (TK * NHID) + offs, mask=mask, other=0.0)
    s = tl.sum(x.to(tl.float32), axis=0) * scale
    tl.store(out_ptr + pid * NHID + pn, s, mask=mask)


@triton.jit
def kP(in_ptr, out_ptr, scale, NCHUNK, NTOK: tl.constexpr, NHID: tl.constexpr,
       TK: tl.constexpr, DN: tl.constexpr):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    NBLK: tl.constexpr = (NHID + DN - 1) // DN
    koff = tl.arange(0, TK)[:, None] * NHID
    for c in range(pid, NCHUNK, npg):
        tok = c // NBLK
        nb = c % NBLK
        pn = nb * DN + tl.arange(0, DN)
        mask = pn[None, :] < NHID
        x = tl.load(in_ptr + tok * (TK * NHID) + koff + pn[None, :], mask=mask, other=0.0)
        s = tl.sum(x.to(tl.float32), axis=0) * scale
        tl.store(out_ptr + tok * NHID + pn, s, mask=mask)


@triton.jit
def kT(in_ptr, out_ptr, scale, NTOK: tl.constexpr, NHID: tl.constexpr,
       TK: tl.constexpr, DN: tl.constexpr, NCH: tl.constexpr):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    koff = tl.arange(0, TK)[:, None] * NHID
    for tok in range(pid, NTOK, npg):
        for c in tl.static_range(0, NCH):
            pn = c * DN + tl.arange(0, DN)
            mask = pn[None, :] < NHID
            x = tl.load(in_ptr + tok * (TK * NHID) + koff + pn[None, :], mask=mask, other=0.0)
            s = tl.sum(x.to(tl.float32), axis=0) * scale
            tl.store(out_ptr + tok * NHID + pn, s, mask=mask)


def run_a(x, out, bm, bn, w, st=1):
    nt, tk, nh = x.shape
    kA[(triton.cdiv(nt, bm), triton.cdiv(nh, bn))](x, out, SCALE, nt, nh, tk, bm, bn,
                                                   num_warps=w, num_stages=st)


def run_b(x, out, dn, w, st=1):
    nt, tk, nh = x.shape
    kB[(nt, triton.cdiv(nh, dn))](x, out, SCALE, nt, nh, tk, dn, num_warps=w, num_stages=st)


def run_p(x, out, dn, nprog, w):
    nt, tk, nh = x.shape
    nblk = triton.cdiv(nh, dn)
    kP[(nprog,)](x, out, SCALE, nt * nblk, nt, nh, tk, dn, num_warps=w, num_stages=1)


def run_t(x, out, dn, nch, nprog, w):
    nt, tk, nh = x.shape
    kT[(nprog,)](x, out, SCALE, nt, nh, tk, dn, nch, num_warps=w, num_stages=1)


def main():
    props = torch.npu.get_device_properties(0)
    dinfo = {"name": props.name}
    for k in ("num_aicore", "num_vectorcore", "aicore_num", "vectorcore_num"):
        if hasattr(props, k):
            dinfo[k] = getattr(props, k)
    try:
        from triton.runtime import driver as tdriver
        dinfo["triton_dev_props"] = {k: v for k, v in vars(
            tdriver.active.utils.get_device_properties(torch.npu.current_device())).items()
            if isinstance(v, (int, float, str, bool))}
    except Exception as exc:
        dinfo["triton_dev_props_err"] = repr(exc)
    RES["device"] = dinfo
    ndev = RES.get("device", {})

    # ---- host floor probes (1 token shape) ----
    tmp = torch.empty(16 * 1024, dtype=torch.float32, device="npu")
    x1 = torch.randn((1, 8, 7168), dtype=torch.float32, device="npu").to(torch.bfloat16)
    o1 = torch.empty((1, 7168), dtype=torch.bfloat16, device="npu")
    try:
        RES["host"]["store_1blk"] = bench(lambda: kStore[(1,)](tmp, SCALE, DN=512), 200)
        RES["host"]["store_14blk"] = bench(lambda: kStore[(14,)](tmp, SCALE, DN=512), 200)
        RES["host"]["empty"] = bench(
            lambda: torch.empty((4096, 7168), dtype=torch.bfloat16, device="npu"), 200)
        RES["host"]["b_d1024_1tok"] = bench(lambda: run_b(x1, o1, 1024, 2), 200)
        RES["host"]["b_d1024_1tok_noalloc"] = bench(lambda: run_b(x1, o1, 1024, 2), 200)
        RES["host"]["a_seed_1tok"] = bench(lambda: run_a(x1, o1, 2, 512, 8), 200)
        RES["host"]["t_40prog_1tok"] = bench(lambda: run_t(x1, o1, 4096, 2, 40, 2), 200)
    except Exception:
        RES["errors"]["host"] = traceback.format_exc()

    # ---- device sweep on 4096 tokens ----
    x = torch.randn((4096, 8, 7168), dtype=torch.float32, device="npu").to(torch.bfloat16)
    out = torch.empty((4096, 7168), dtype=torch.bfloat16, device="npu")
    ref = x.float().sum(dim=1).mul(SCALE).to(x.dtype)

    fams = {}
    for bm, bn, w, st in [(1, 2048, 2, 1), (1, 2048, 4, 1), (1, 2048, 2, 2),
                          (1, 4096, 4, 1), (1, 1024, 2, 1), (2, 2048, 2, 1)]:
        fams[f"A_bm{bm}_bn{bn}_w{w}_st{st}"] = (lambda a=bm, b=bn, c=w, d=st: run_a(x, out, a, b, c, d))
    for dn, w in [(1024, 2), (1024, 4), (2048, 2), (2048, 4), (512, 2), (2048, 8)]:
        fams[f"B_dn{dn}_w{w}"] = (lambda a=dn, b=w: run_b(x, out, a, b))
    npg = ndev.get("num_vectorcore") or ndev.get("triton_dev_props", {}).get("num_vector_core") or 40
    for dn, np_, w in [(1024, npg, 2), (2048, npg, 2), (2048, npg * 2, 4)]:
        fams[f"P_dn{dn}_np{np_}_w{w}"] = (lambda a=dn, b=np_, c=w: run_p(x, out, a, b, c))
    for dn, nch, np_, w in [(4096, 2, npg, 2), (8192, 1, npg, 2)]:
        fams[f"T_dn{dn}_nch{nch}_np{np_}_w{w}"] = (lambda a=dn, b=nch, c=np_, d=w: run_t(x, out, a, b, c, d))

    ok = {}
    for name, fn in fams.items():
        try:
            fn()
            torch.npu.synchronize()
            good = bool(torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.02))
            t = bench(fn, 20)
            RES["dev4096"][name] = {"us": t, "gbps": round(528.5 * 1e6 / t / 1e3, 1), "ok": good}
            if good:
                ok[name] = t
        except Exception as exc:
            RES["errors"][name] = repr(exc)[:300]
            torch.npu.synchronize()

    # ---- apply best families to 512 and 64 tokens ----
    for ntok in (512, 64):
        key = f"dev{ntok}"
        xs = torch.randn((ntok, 8, 7168), dtype=torch.float32, device="npu").to(torch.bfloat16)
        os_ = torch.empty((ntok, 7168), dtype=torch.bfloat16, device="npu")
        refs = xs.float().sum(dim=1).mul(SCALE).to(xs.dtype)
        cands = {}
        cands["A_1_2048_w2"] = lambda: run_a(xs, os_, 1, 2048, 2)
        cands["A_2_512_w8"] = lambda: run_a(xs, os_, 2, 512, 8)
        cands["B_1024_w2"] = lambda: run_b(xs, os_, 1024, 2)
        cands["B_2048_w2"] = lambda: run_b(xs, os_, 2048, 2)
        cands["P_1024_w2"] = lambda: run_p(xs, os_, 1024, npg, 2)
        cands["T_4096_w2"] = lambda: run_t(xs, os_, 4096, 2, npg, 2)
        for name, fn in cands.items():
            try:
                fn()
                torch.npu.synchronize()
                good = bool(torch.allclose(os_.float(), refs.float(), atol=0.05, rtol=0.02))
                t = bench(fn, 50)
                nbytes = ntok * 7168 * 2 * 9
                RES[key][name] = {"us": t, "gbps": round(nbytes / t / 1e3, 1), "ok": good}
            except Exception as exc:
                RES["errors"][f"{key}_{name}"] = repr(exc)[:300]
                torch.npu.synchronize()
        del xs, os_, refs

    path = os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "sweep.json")
    with open(path, "w") as f:
        json.dump(RES, f, indent=2)
    print(json.dumps(RES, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL", traceback.format_exc())
        try:
            with open(os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "sweep.json"), "w") as f:
                json.dump(RES, f, indent=2)
        except Exception:
            pass
