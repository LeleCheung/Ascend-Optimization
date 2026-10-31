"""Diagnostics: vectorized-shift alternatives to the scalarised tl.cumsum scan.

Measures walltime (perf_counter, like the harness) and device time (npu Event)
for: copy-only floor, tl.cumsum seed, tl.gather-based Hillis-Steele (2D and
flat), and a lane-parallel sequential accumulate.  Also reports per-launch
host enqueue cost.  Numerics are checked against a float64 CPU reference.
"""
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu"
OUT = {"wall_ms": {}, "dev_ms": {}, "errors": {}, "numerics": {}, "host": {}}


def flush():
    try:
        with open(os.path.join(ART, "probe.json"), "w") as f:
            json.dump(OUT, f, indent=2, default=str)
    except Exception:  # noqa: BLE001
        pass


def say(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# MODE 0: copy only (memory floor)   MODE 1: gather Hillis-Steele on axis 1
# --------------------------------------------------------------------------
@triton.jit
def k_tile(y_ptr, o_ptr, N: tl.constexpr, LS: tl.constexpr, PS: tl.constexpr,
           OLS: tl.constexpr, OPS: tl.constexpr, AL: tl.constexpr,
           C: tl.constexpr, NB: tl.constexpr, LOG: tl.constexpr,
           HDX: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    la = pid * AL + tl.arange(0, AL)
    abase = la * OLS
    ibase = la * LS
    k = tl.arange(0, C)
    carry = tl.zeros((AL,), tl.float32)
    for b in tl.range(0, NB):
        p = b * C + k
        pm = (p < (N - 1))[None, :]
        off = ibase[:, None] + p[None, :] * PS
        va = tl.load(y_ptr + off, mask=pm, other=0.0).to(tl.float32)
        vb = tl.load(y_ptr + off + PS, mask=pm, other=0.0).to(tl.float32)
        t = (va + vb) * HDX
        if MODE == 0:
            cs = t + carry[:, None]
        else:
            cs = t
            for s in tl.static_range(0, LOG):
                sh = 1 << s
                idx = tl.broadcast_to(tl.maximum(k - sh, 0)[None, :], (AL, C))
                g = tl.gather(cs, idx, 1)
                cs = cs + tl.where((k >= sh)[None, :], g, 0.0)
            cs = cs + carry[:, None]
        tl.store(o_ptr + abase[:, None] + p[None, :] * OPS, cs, mask=pm)
        if MODE != 0:
            carry = carry + tl.sum(t, 1)
        else:
            carry = carry + tl.sum(t, 1)


@triton.jit
def k_flat(y_ptr, o_ptr, N: tl.constexpr, LS: tl.constexpr, PS: tl.constexpr,
           OLS: tl.constexpr, OPS: tl.constexpr, AL: tl.constexpr,
           C: tl.constexpr, NB: tl.constexpr, LOG: tl.constexpr,
           HDX: tl.constexpr):
    pid = tl.program_id(0)
    la = pid * AL + tl.arange(0, AL)
    abase = la * OLS
    ibase = la * LS
    k = tl.arange(0, C)
    q = tl.arange(0, AL * C)
    kq = q % C
    carry = tl.zeros((AL,), tl.float32)
    for b in tl.range(0, NB):
        p = b * C + k
        pm = (p < (N - 1))[None, :]
        off = ibase[:, None] + p[None, :] * PS
        va = tl.load(y_ptr + off, mask=pm, other=0.0).to(tl.float32)
        vb = tl.load(y_ptr + off + PS, mask=pm, other=0.0).to(tl.float32)
        t = (va + vb) * HDX
        cs = tl.reshape(t, (AL * C,))
        for s in tl.static_range(0, LOG):
            sh = 1 << s
            idx = q - sh + sh * tl.where(kq < sh, 1, 0)
            g = tl.gather(cs, idx, 0)
            cs = cs + tl.where(kq >= sh, g, 0.0)
        cs = tl.reshape(cs, (AL, C)) + carry[:, None]
        tl.store(o_ptr + abase[:, None] + p[None, :] * OPS, cs, mask=pm)
        carry = carry + tl.sum(t, 1)


# lane-parallel sequential accumulate (no scan at all)
@triton.jit
def k_seq(y_ptr, o_ptr, N: tl.constexpr, LS: tl.constexpr, PS: tl.constexpr,
          OLS: tl.constexpr, OPS: tl.constexpr, AL: tl.constexpr,
          HDX: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    a = pid * AL + tl.arange(0, AL)
    acc = tl.zeros((AL,), tl.float32)
    ib = a * LS
    ob = a * OLS
    for j in tl.range(0, N - 1, num_stages=NS):
        off = ib + j * PS
        va = tl.load(y_ptr + off).to(tl.float32)
        vb = tl.load(y_ptr + off + PS).to(tl.float32)
        acc = acc + (va + vb) * HDX
        tl.store(o_ptr + ob + j * OPS, acc)


# seed: one program per line, whole-line tl.cumsum
@triton.jit
def k_cum(y_ptr, o_ptr, N: tl.constexpr, LS: tl.constexpr, PS: tl.constexpr,
          OLS: tl.constexpr, OPS: tl.constexpr, BLOCK: tl.constexpr,
          HDX: tl.constexpr):
    pid = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    m = k < (N - 1)
    off = pid * LS + k * PS
    va = tl.load(y_ptr + off, mask=m, other=0.0).to(tl.float32)
    vb = tl.load(y_ptr + off + PS, mask=m, other=0.0).to(tl.float32)
    cs = tl.cumsum((va + vb) * HDX, 0)
    tl.store(o_ptr + pid * OLS + k * OPS, cs, mask=m)


@triton.jit
def k_tiny(x_ptr):
    tl.store(x_ptr + tl.program_id(0), 1.0)


def wall(fn, iters=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return round((t2 - t0) / iters * 1e3, 4), round((t1 - t0) / iters * 1e3, 4)


def timed(key, fn, iters=20):
    try:
        w, h = wall(fn, iters)
        OUT["wall_ms"][key] = w
        OUT["host"][key] = h
        say(key, "wall", w, "host", h)
    except Exception as ex:  # noqa: BLE001
        OUT["errors"][key] = str(traceback.format_exception_only(type(ex), ex))[:600]
        say(key, "ERROR", OUT["errors"][key][:300])
    flush()


def main():
    # ---------------- host launch overhead ----------------
    buf = torch.zeros(64, device=DEV, dtype=torch.float32)
    timed("host_only_tiny", lambda: k_tiny[(1,)](buf), 200)

    # ---------------- regime (a): last-dim scan ----------------
    for tag, M, L, dt in (("m1024x4096_f16", 1024, 4096, torch.float16),
                          ("m128x1024_f16", 128, 1024, torch.float16)):
        y = torch.randn(M, L, device=DEV, dtype=dt)
        o = torch.empty(M, L, device=DEV, dtype=dt)
        hdx = 0.5
        common = dict(N=L, LS=L, PS=1, OLS=L - 1, OPS=1)
        timed(f"{tag}_seed_cumsum", lambda: k_cum[(M,)](y, o, BLOCK=L, HDX=hdx, **common))
        if M == 1024:
            for al, c in ((16, 128), (8, 128), (16, 256), (32, 64)):
                nb = (L - 1 + c - 1) // c
                lg = c.bit_length() - 1
                timed(f"{tag}_copy_al{al}_c{c}",
                      lambda al=al, c=c, nb=nb, lg=lg: k_tile[(M // al,)](
                          y, o, AL=al, C=c, NB=nb, LOG=lg, HDX=hdx, MODE=0, **common))
                timed(f"{tag}_gath_al{al}_c{c}",
                      lambda al=al, c=c, nb=nb, lg=lg: k_tile[(M // al,)](
                          y, o, AL=al, C=c, NB=nb, LOG=lg, HDX=hdx, MODE=1, **common))
            timed(f"{tag}_flat_al16_c128",
                  lambda: k_flat[(M // 16,)](y, o, AL=16, C=128, NB=(L - 1 + 127) // 128,
                                             LOG=7, HDX=hdx, **common))
        else:
            timed(f"{tag}_copy_al8_c128",
                  lambda: k_tile[(M // 8,)](y, o, AL=8, C=128, NB=(L - 1 + 127) // 128,
                                            LOG=7, HDX=hdx, MODE=0, **common))
            timed(f"{tag}_gath_al8_c128",
                  lambda: k_tile[(M // 8,)](y, o, AL=8, C=128, NB=(L - 1 + 127) // 128,
                                            LOG=7, HDX=hdx, MODE=1, **common))
        # numerics of the gather variant vs float64 reference
        try:
            k_tile[(M // 8,)](y, o, AL=8, C=128, NB=(L - 1 + 127) // 128, LOG=7,
                              HDX=hdx, MODE=1, **common)
            torch.npu.synchronize()
            yc = y.cpu().double()
            exact = torch.cumsum((yc[:, :-1] + yc[:, 1:]) * hdx, dim=1)
            got = o.cpu().double()[:, : L - 1]
            OUT["numerics"][f"{tag}_gather"] = float((got - exact).abs().max())
            k_cum[(M,)](y, o, BLOCK=L, HDX=hdx, **common)
            torch.npu.synchronize()
            got2 = o.cpu().double()[:, : L - 1]
            OUT["numerics"][f"{tag}_cumsum"] = float((got2 - exact).abs().max())
        except Exception as ex:  # noqa: BLE001
            OUT["errors"][f"{tag}_num"] = str(traceback.format_exception_only(type(ex), ex))[:400]
        flush()
        del y, o

    # ---------------- regime (b): dim=0, inner=128 ----------------
    N, INNER = 1024, 128
    for dt in (torch.float16, torch.float32):
        tag = "dim0_" + ("f16" if dt == torch.float16 else "f32")
        y = torch.randn(N, INNER, device=DEV, dtype=dt)
        o = torch.empty(N - 1, INNER, device=DEV, dtype=dt)
        hdx = 0.5
        common = dict(N=N, LS=1, PS=INNER, OLS=1, OPS=INNER)
        timed(f"{tag}_seed_cumsum",
              lambda: k_cum[(INNER,)](y, o, BLOCK=N, HDX=hdx, **common))
        for al, c in ((64, 128), (32, 128)):
            nb = (N - 1 + c - 1) // c
            lg = c.bit_length() - 1
            timed(f"{tag}_copy_al{al}_c{c}",
                  lambda al=al, c=c, nb=nb, lg=lg: k_tile[(INNER // al,)](
                      y, o, AL=al, C=c, NB=nb, LOG=lg, HDX=hdx, MODE=0, **common))
            timed(f"{tag}_gath_al{al}_c{c}",
                  lambda al=al, c=c, nb=nb, lg=lg: k_tile[(INNER // al,)](
                      y, o, AL=al, C=c, NB=nb, LOG=lg, HDX=hdx, MODE=1, **common))
        for al, ns in ((64, 1), (64, 3), (128, 1)):
            timed(f"{tag}_seq_al{al}_ns{ns}",
                  lambda al=al, ns=ns: k_seq[(INNER // al,)](
                      y, o, AL=al, HDX=hdx, NS=ns, **common))
        try:
            k_seq[(INNER // 64,)](y, o, AL=64, HDX=hdx, NS=1, **common)
            torch.npu.synchronize()
            yc = y.cpu().double()
            exact = torch.cumsum((yc[:-1, :] + yc[1:, :]) * hdx, dim=0)
            OUT["numerics"][f"{tag}_seq"] = float((o.cpu().double() - exact).abs().max())
            k_tile[(INNER // 64,)](y, o, AL=64, C=128, NB=(N - 1 + 127) // 128, LOG=7,
                                   HDX=hdx, MODE=1, **common)
            torch.npu.synchronize()
            OUT["numerics"][f"{tag}_gather"] = float((o.cpu().double() - exact).abs().max())
        except Exception as ex:  # noqa: BLE001
            OUT["errors"][f"{tag}_num"] = str(traceback.format_exception_only(type(ex), ex))[:400]
        flush()
        del y, o

    say(json.dumps({k: OUT[k] for k in ("wall_ms", "numerics")}, indent=1)[:4000])
    say("errors", json.dumps(OUT["errors"], indent=1)[:3000])
    flush()


if __name__ == "__main__":
    main()
