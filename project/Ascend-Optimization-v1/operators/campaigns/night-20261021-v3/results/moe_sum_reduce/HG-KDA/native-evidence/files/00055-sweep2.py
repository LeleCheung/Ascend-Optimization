"""Round 2 diagnostic: fixed 2D-load variants, wider A-config sweep, host launch cost."""
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

RES = {"host": {}, "dev": {}, "errors": {}}
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
def kmin3(in_ptr, out_ptr, scale, NHID: tl.constexpr, TK: tl.constexpr, DN: tl.constexpr):
    pid = tl.program_id(0)
    pn = pid * DN + tl.arange(0, DN)
    v = tl.load(in_ptr + pid * (TK * NHID) + pn, mask=pn < NHID, other=0.0)
    tl.store(out_ptr + pid * NHID + pn, v.to(tl.float32) * scale, mask=pn < NHID)


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
    nmask = pn < NHID
    offs = tl.arange(0, TK)[:, None] * NHID + pn[None, :]
    x = tl.load(in_ptr + pid * (TK * NHID) + offs, mask=nmask[None, :], other=0.0)
    s = tl.sum(x.to(tl.float32), axis=0) * scale
    tl.store(out_ptr + pid * NHID + pn, s, mask=nmask)


@triton.jit
def kU(in_ptr, out_ptr, scale, NTOK: tl.constexpr, NHID: tl.constexpr,
       TK: tl.constexpr, DN: tl.constexpr):
    pid = tl.program_id(0)
    pn = tl.program_id(1) * DN + tl.arange(0, DN)
    nmask = pn < NHID
    base = in_ptr + pid * (TK * NHID) + pn
    acc = tl.load(base, mask=nmask, other=0.0).to(tl.float32)
    acc += tl.load(base + 1 * NHID, mask=nmask, other=0.0).to(tl.float32)
    acc += tl.load(base + 2 * NHID, mask=nmask, other=0.0).to(tl.float32)
    acc += tl.load(base + 3 * NHID, mask=nmask, other=0.0).to(tl.float32)
    acc += tl.load(base + 4 * NHID, mask=nmask, other=0.0).to(tl.float32)
    acc += tl.load(base + 5 * NHID, mask=nmask, other=0.0).to(tl.float32)
    acc += tl.load(base + 6 * NHID, mask=nmask, other=0.0).to(tl.float32)
    acc += tl.load(base + 7 * NHID, mask=nmask, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid * NHID + pn, acc * scale, mask=nmask)


def run_a(x, out, bm, bn, w, st=1):
    nt, tk, nh = x.shape
    kA[(triton.cdiv(nt, bm), triton.cdiv(nh, bn))](x, out, SCALE, nt, nh, tk, bm, bn,
                                                   num_warps=w, num_stages=st)


def run_b(x, out, dn, w, st=1):
    nt, tk, nh = x.shape
    kB[(nt, triton.cdiv(nh, dn))](x, out, SCALE, nt, nh, tk, dn, num_warps=w, num_stages=st)


def run_u(x, out, dn, w):
    nt, tk, nh = x.shape
    kU[(nt, triton.cdiv(nh, dn))](x, out, SCALE, nt, nh, tk, dn, num_warps=w, num_stages=1)


A_CFG = [(2, 2048, 2, 1), (2, 4096, 4, 1), (2, 4096, 8, 1), (4, 2048, 4, 1),
         (4, 2048, 8, 1), (1, 8192, 4, 1), (2, 2048, 2, 2), (2, 2048, 4, 2),
         (8, 1024, 4, 1), (4, 4096, 8, 1)]
B_CFG = [(2048, 2), (2048, 4), (4096, 4), (4096, 8), (8192, 4), (8192, 8)]
U_CFG = [(2048, 2), (4096, 4), (8192, 4)]


def main():
    # ---------- host cost probes at 1 token ----------
    x1 = torch.randn((1, 8, 7168), dtype=torch.float32, device="npu").to(torch.bfloat16)
    o1 = torch.empty((1, 7168), dtype=torch.bfloat16, device="npu")
    RES["host"]["kmin3_raw"] = bench(lambda: kmin3[(14,)](x1, o1, SCALE, 7168, 8, 512), 200)
    RES["host"]["kA_13arg"] = bench(lambda: run_a(x1, o1, 1, 512, 8), 200)
    RES["host"]["kA_13arg_fullrow"] = bench(lambda: run_a(x1, o1, 1, 8192, 4), 200)
    RES["host"]["kB_7arg"] = bench(lambda: run_b(x1, o1, 2048, 2), 200)
    # wrapper-only (alloc + grid math), no launch
    def wrapper_only():
        out = torch.empty((1, 7168), dtype=torch.bfloat16, device="npu")
        grid = (1, 4)
        return out, grid
    RES["host"]["wrapper_only"] = bench(wrapper_only, 200)
    # direct CompiledKernel launch path
    try:
        kmin3[(14,)](x1, o1, SCALE, 7168, 8, 512)
        torch.npu.synchronize()
        dev = torch.npu.current_device()
        caches = kmin3.device_caches[dev][0]
        ck = list(caches.values())[0]
        stream = torch.npu.current_stream().npu_stream
        fast = ck[(14, 1, 1)]
        RES["host"]["kmin3_direct"] = bench(lambda: fast(x1, o1, SCALE), 200)
        RES["host"]["ck_type"] = type(ck).__name__
    except Exception:
        RES["errors"]["direct_launch"] = traceback.format_exc()[-500:]
    torch.npu.synchronize()

    # ---------- device sweep ----------
    for ntok in (4096, 512, 64):
        x = torch.randn((ntok, 8, 7168), dtype=torch.float32, device="npu").to(torch.bfloat16)
        out = torch.empty((ntok, 7168), dtype=torch.bfloat16, device="npu")
        ref = x.float().sum(dim=1).mul(SCALE).to(x.dtype)
        nbytes = ntok * 7168 * 2 * 9
        res = {}
        n = 20 if ntok >= 512 else 100
        trials = []
        if ntok == 4096:
            trials += [(f"A{bm}_{bn}_w{w}_st{st}", lambda a=bm, b=bn, c=w, d=st: run_a(x, out, a, b, c, d))
                       for bm, bn, w, st in A_CFG]
            trials += [(f"B{dn}_w{w}", lambda a=dn, b=w: run_b(x, out, a, b)) for dn, w in B_CFG]
            trials += [(f"U{dn}_w{w}", lambda a=dn, b=w: run_u(x, out, a, b)) for dn, w in U_CFG]
        else:
            for bm, bn, w, st in [(2, 2048, 2, 1), (2, 4096, 4, 1), (1, 8192, 4, 1), (4, 2048, 4, 1)]:
                trials.append((f"A{bm}_{bn}_w{w}_st{st}",
                               lambda a=bm, b=bn, c=w, d=st: run_a(x, out, a, b, c, d)))
            for dn, w in [(2048, 2), (4096, 4)]:
                trials.append((f"B{dn}_w{w}", lambda a=dn, b=w: run_b(x, out, a, b)))
            trials.append(("U4096_w4", lambda: run_u(x, out, 4096, 4)))
        for name, fn in trials:
            try:
                fn()
                torch.npu.synchronize()
                good = bool(torch.allclose(out.float(), ref.float(), atol=0.05, rtol=0.02))
                t = bench(fn, n)
                res[name] = {"us": t, "gbps": round(nbytes / t / 1e3, 1), "ok": good}
            except Exception as exc:
                RES["errors"][f"{ntok}_{name}"] = repr(exc)[-200:]
                torch.npu.synchronize()
        RES["dev"][str(ntok)] = res
        del x, out, ref
        torch.npu.synchronize()

    path = os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "sweep2.json")
    with open(path, "w") as f:
        json.dump(RES, f, indent=2)
    print(json.dumps(RES, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL", traceback.format_exc())
        try:
            with open(os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "sweep2.json"), "w") as f:
                json.dump(RES, f, indent=2)
        except Exception:
            pass
