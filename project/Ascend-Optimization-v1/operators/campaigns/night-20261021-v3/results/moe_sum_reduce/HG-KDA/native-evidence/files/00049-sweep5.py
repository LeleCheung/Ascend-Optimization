"""Round 5 diagnostic: mask-free (EVEN) vs masked, pid order, stages; strict correctness."""
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

RES = {"dev": {}, "e2e": {}, "errors": {}}
SCALE = 2.5


def bench(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    nosync = (time.perf_counter() - t0) / n * 1e6
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
        torch.npu.synchronize()
    return round((time.perf_counter() - t0) / n * 1e6, 2), round(nosync, 2)


@triton.jit
def kF(in_ptr, out_ptr, scale, NTOK: tl.constexpr, NHID: tl.constexpr, TK: tl.constexpr,
       BM: tl.constexpr, BN: tl.constexpr, NFIRST: tl.constexpr, EVEN: tl.constexpr):
    if NFIRST:
        pm = tl.program_id(1) * BM + tl.arange(0, BM)
        pn = tl.program_id(0) * BN + tl.arange(0, BN)
    else:
        pm = tl.program_id(0) * BM + tl.arange(0, BM)
        pn = tl.program_id(1) * BN + tl.arange(0, BN)
    base = in_ptr + pm[:, None] * (TK * NHID) + pn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if EVEN:
        for k in range(TK):
            acc += tl.load(base + k * NHID).to(tl.float32)
        tl.store(out_ptr + pm[:, None] * NHID + pn[None, :], acc * scale)
    else:
        mask = (pm[:, None] < NTOK) & (pn[None, :] < NHID)
        for k in range(TK):
            acc += tl.load(base + k * NHID, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + pm[:, None] * NHID + pn[None, :], acc * scale, mask=mask)


def launch(x, out, bm, bn, w, st, nfirst, even):
    nt, tk, nh = x.shape
    g = (triton.cdiv(nh, bn), triton.cdiv(nt, bm)) if nfirst else (triton.cdiv(nt, bm), triton.cdiv(nh, bn))
    kF[g](x, out, SCALE, nt, nh, tk, bm, bn, nfirst, even, num_warps=w, num_stages=st)


CFG = [
    (16, 1024, 4, 2, False, True),   # mask-free, m-fastest
    (16, 1024, 4, 2, True, True),    # mask-free, n-fastest
    (16, 1024, 4, 2, False, False),  # masked control
    (16, 1024, 8, 2, False, True),
    (8, 1024, 4, 2, False, True),
    (4, 1024, 4, 2, False, True),
    (32, 1024, 4, 2, False, True),
    (16, 1024, 4, 3, False, True),
    (16, 1024, 4, 1, False, True),
    (8, 512, 4, 2, False, True),
    (64, 512, 4, 2, False, True),
]


def main():
    for ntok in (4096, 512, 64):
        x = torch.randn((ntok, 8, 7168), dtype=torch.float32, device="npu").to(torch.bfloat16)
        ref = x.float().sum(dim=1).mul(SCALE).to(x.dtype)
        nbytes = ntok * 7168 * 2 * 9
        res = {}
        n = 25 if ntok >= 512 else 100
        for bm, bn, w, st, nf, ev in CFG:
            name = f"bm{bm}_bn{bn}_w{w}_st{st}_{'N' if nf else 'M'}{'E' if ev else 'K'}"
            try:
                out = torch.full((ntok, 7168), float("nan"), dtype=torch.bfloat16, device="npu")
                launch(x, out, bm, bn, w, st, nf, ev)
                torch.npu.synchronize()
                good = bool(torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.02)
                            and not torch.isnan(out.float()).any())
                if good:
                    t_sync, t_ns = bench(lambda: launch(x, out, bm, bn, w, st, nf, ev), n)
                    res[name] = {"sync": t_sync, "ns": t_ns, "gbps": round(nbytes / t_ns / 1e3, 1)}
                else:
                    res[name] = {"ok": False}
                del out
            except Exception as exc:
                RES["errors"][f"{ntok}_{name}"] = str(exc)[-160:]
                torch.npu.synchronize()
        RES["dev"][str(ntok)] = res
        del x, ref
        torch.npu.synchronize()

    # end-to-end: planned wrapper vs reference
    def cand(x, s, bm, bn, w, st):
        nt, tk, nh = x.shape
        out = torch.empty((nt, nh), dtype=x.dtype, device=x.device)
        launch(x, out, bm, bn, w, st, False, True)
        return out

    for ntok in (1, 8, 64, 512, 4096):
        x = torch.randn((ntok, 8, 7168), dtype=torch.float32, device="npu").to(torch.bfloat16)
        n = 25 if ntok >= 512 else 100
        row = {}
        for tag, fn in (("cand", lambda: cand(x, SCALE, min(16, max(1, ntok)), 1024, 4, 2)),
                        ("ref", lambda: x.float().sum(dim=1).mul(SCALE).to(x.dtype))):
            try:
                row[tag] = bench(fn, n, warmup=5)
            except Exception as exc:
                row[tag] = str(exc)[-120:]
        RES["e2e"][str(ntok)] = row
        del x
        torch.npu.synchronize()

    path = os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "sweep5.json")
    with open(path, "w") as f:
        json.dump(RES, f, indent=2)
    print(json.dumps(RES, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL", traceback.format_exc())
        try:
            with open(os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "sweep5.json"), "w") as f:
                json.dump(RES, f, indent=2)
        except Exception:
            pass
