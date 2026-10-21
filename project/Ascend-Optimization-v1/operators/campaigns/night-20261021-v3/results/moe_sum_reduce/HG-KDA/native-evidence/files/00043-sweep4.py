"""Round 4 diagnostic: pid ordering (DRAM locality), mask-free specialization, larger tiles."""
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

RES = {"dev": {}, "errors": {}}
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
def kL(in_ptr, out_ptr, scale, NTOK: tl.constexpr, NHID: tl.constexpr,
       TK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0) * BM + tl.arange(0, BM)
    pn = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = (pm[:, None] < NTOK) & (pn[None, :] < NHID)
    base = in_ptr + pm[:, None] * (TK * NHID) + pn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(TK):
        acc += tl.load(base + k * NHID, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + pm[:, None] * NHID + pn[None, :], acc * scale, mask=mask)


@triton.jit
def kS(in_ptr, out_ptr, scale, NTOK: tl.constexpr, NHID: tl.constexpr,
       TK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    # pid0 = n-block (fastest), pid1 = token group -> contiguous DRAM footprint
    pm = tl.program_id(1) * BM + tl.arange(0, BM)
    pn = tl.program_id(0) * BN + tl.arange(0, BN)
    mask = (pm[:, None] < NTOK) & (pn[None, :] < NHID)
    base = in_ptr + pm[:, None] * (TK * NHID) + pn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(TK):
        acc += tl.load(base + k * NHID, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + pm[:, None] * NHID + pn[None, :], acc * scale, mask=mask)


@triton.jit
def kN(in_ptr, out_ptr, scale, NTOK: tl.constexpr, NHID: tl.constexpr,
       TK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    # no masks: requires NHID % BN == 0 and NTOK % BM == 0
    pm = tl.program_id(1) * BM + tl.arange(0, BM)
    pn = tl.program_id(0) * BN + tl.arange(0, BN)
    base = in_ptr + pm[:, None] * (TK * NHID) + pn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(TK):
        acc += tl.load(base + k * NHID).to(tl.float32)
    tl.store(out_ptr + pm[:, None] * NHID + pn[None, :], acc * scale)


def run_l(x, out, bm, bn, w, st=1):
    nt, tk, nh = x.shape
    kL[(triton.cdiv(nt, bm), triton.cdiv(nh, bn))](x, out, SCALE, nt, nh, tk, bm, bn,
                                                   num_warps=w, num_stages=st)


def run_s(x, out, bm, bn, w, st=1):
    nt, tk, nh = x.shape
    kS[(triton.cdiv(nh, bn), triton.cdiv(nt, bm))](x, out, SCALE, nt, nh, tk, bm, bn,
                                                   num_warps=w, num_stages=st)


def run_n(x, out, bm, bn, w, st=1):
    nt, tk, nh = x.shape
    kN[(nh // bn, nt // bm)](x, out, SCALE, nt, nh, tk, bm, bn,
                             num_warps=w, num_stages=st)


def main():
    for ntok in (4096, 512):
        x = torch.randn((ntok, 8, 7168), dtype=torch.float32, device="npu").to(torch.bfloat16)
        out = torch.empty((ntok, 7168), dtype=torch.bfloat16, device="npu")
        ref = x.float().sum(dim=1).mul(SCALE).to(x.dtype)
        nbytes = ntok * 7168 * 2 * 9
        res = {}
        n = 25 if ntok >= 512 else 100
        cfg = [(16, 1024, 4, 1), (16, 1024, 8, 1), (16, 1024, 4, 2), (16, 1024, 4, 3),
               (32, 512, 4, 1), (16, 512, 4, 1), (8, 2048, 4, 2)]
        trials = [(f"L{bm}_{bn}_w{w}_st{st}", lambda a=bm, b=bn, c=w, d=st: run_l(x, out, a, b, c, d))
                  for bm, bn, w, st in cfg]
        trials += [(f"S{bm}_{bn}_w{w}_st{st}", lambda a=bm, b=bn, c=w, d=st: run_s(x, out, a, b, c, d))
                   for bm, bn, w, st in [(16, 1024, 4, 1), (8, 2048, 4, 1), (4, 4096, 8, 1)]]
        trials += [("N16_1024_w4", lambda: run_n(x, out, 16, 1024, 4)),
                   ("N8_2048_w4", lambda: run_n(x, out, 8, 2048, 4)),
                   ("N4_4096_w8", lambda: run_n(x, out, 4, 4096, 8))]
        for name, fn in trials:
            try:
                fn()
                torch.npu.synchronize()
                good = bool(torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.02))
                t = bench(fn, n)
                res[name] = {"us": t, "gbps": round(nbytes / t / 1e3, 1), "ok": good}
            except Exception as exc:
                RES["errors"][f"{ntok}_{name}"] = str(exc)[-160:]
                torch.npu.synchronize()
        RES["dev"][str(ntok)] = res
        del x, out, ref
        torch.npu.synchronize()

    path = os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "sweep4.json")
    with open(path, "w") as f:
        json.dump(RES, f, indent=2)
    print(json.dumps(RES, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL", traceback.format_exc())
        try:
            with open(os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "sweep4.json"), "w") as f:
                json.dump(RES, f, indent=2)
        except Exception:
            pass
