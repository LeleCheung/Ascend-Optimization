"""Round 3 diagnostic: push tile size on large shapes; lean-arg host cost."""
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


# lean kernel: 3 runtime args, 5 constexprs (candidate shape)
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
def kU(in_ptr, out_ptr, scale, NTOK: tl.constexpr, NHID: tl.constexpr,
       TK: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    pn = tl.program_id(1) * BN + tl.arange(0, BN)
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


def run_l(x, out, bm, bn, w, st=1):
    nt, tk, nh = x.shape
    kL[(triton.cdiv(nt, bm), triton.cdiv(nh, bn))](x, out, SCALE, nt, nh, tk, bm, bn,
                                                   num_warps=w, num_stages=st)


def run_u(x, out, bn, w):
    nt, tk, nh = x.shape
    kU[(nt, triton.cdiv(nh, bn))](x, out, SCALE, nt, nh, tk, bn, num_warps=w, num_stages=1)


L_CFG = [(4, 4096, 8, 1), (4, 4096, 4, 1), (8, 2048, 4, 1), (8, 2048, 8, 1),
         (8, 4096, 8, 1), (16, 1024, 4, 1), (4, 4096, 8, 2), (8, 1024, 4, 1),
         (8, 1024, 8, 1), (16, 2048, 8, 1), (2, 8192, 8, 1)]
U_CFG = [(4096, 4), (4096, 8)]


def main():
    # ---- host cost of lean kernel at 1 token ----
    x1 = torch.randn((1, 8, 7168), dtype=torch.float32, device="npu").to(torch.bfloat16)
    o1 = torch.empty((1, 7168), dtype=torch.bfloat16, device="npu")
    RES["host"]["lean_1tok"] = bench(lambda: run_l(x1, o1, 4, 2048, 4), 200)
    RES["host"]["lean_1tok_u"] = bench(lambda: run_u(x1, o1, 4096, 4), 200)

    for ntok in (4096, 512, 64):
        x = torch.randn((ntok, 8, 7168), dtype=torch.float32, device="npu").to(torch.bfloat16)
        out = torch.empty((ntok, 7168), dtype=torch.bfloat16, device="npu")
        ref = x.float().sum(dim=1).mul(SCALE).to(x.dtype)
        nbytes = ntok * 7168 * 2 * 9
        res = {}
        n = 20 if ntok >= 512 else 100
        trials = []
        if ntok == 4096:
            trials += [(f"L{bm}_{bn}_w{w}_st{st}", lambda a=bm, b=bn, c=w, d=st: run_l(x, out, a, b, c, d))
                       for bm, bn, w, st in L_CFG]
            trials += [(f"U{bn}_w{w}", lambda a=bn, b=w: run_u(x, out, a, b)) for bn, w in U_CFG]
        elif ntok == 512:
            for bm, bn, w, st in [(4, 4096, 8, 1), (4, 4096, 4, 1), (8, 2048, 4, 1),
                                  (8, 1024, 4, 1), (16, 1024, 4, 1), (4, 2048, 4, 1),
                                  (2, 2048, 2, 1)]:
                trials.append((f"L{bm}_{bn}_w{w}_st{st}",
                               lambda a=bm, b=bn, c=w, d=st: run_l(x, out, a, b, c, d)))
            trials += [(f"U{bn}_w{w}", lambda a=bn, b=w: run_u(x, out, a, b)) for bn, w in U_CFG]
        else:
            for bm, bn, w, st in [(4, 4096, 8, 1), (8, 1024, 4, 1), (2, 2048, 2, 1)]:
                trials.append((f"L{bm}_{bn}_w{w}_st{st}",
                               lambda a=bm, b=bn, c=w, d=st: run_l(x, out, a, b, c, d)))
            trials.append(("U4096_w4", lambda: run_u(x, out, 4096, 4)))
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

    path = os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "sweep3.json")
    with open(path, "w") as f:
        json.dump(RES, f, indent=2)
    print(json.dumps(RES, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FATAL", traceback.format_exc())
        try:
            with open(os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "sweep3.json"), "w") as f:
                json.dump(RES, f, indent=2)
        except Exception:
            pass
