"""Diagnostics 2: (a) Triton host launch path cost and how to bypass it,
(b) shift-free Blelloch scan via tl.split/tl.join on a whole-line tile."""
import inspect
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
OUT = {"wall_ms": {}, "host_ms": {}, "errors": {}, "numerics": {}, "info": {}}


def flush():
    try:
        with open(os.path.join(ART, "probe2.json"), "w") as f:
            json.dump(OUT, f, indent=2, default=str)
    except Exception:  # noqa: BLE001
        pass


def say(*a):
    print(*a, flush=True)


# ---------------- whole-line tile, copy only (memory floor) ----------------
@triton.jit
def k_copy(y_ptr, o_ptr, N: tl.constexpr, LS: tl.constexpr, PS: tl.constexpr,
           OLS: tl.constexpr, OPS: tl.constexpr, AL: tl.constexpr,
           HDX: tl.constexpr):
    pid = tl.program_id(0)
    la = pid * AL + tl.arange(0, AL)
    k = tl.arange(0, N)
    m = (k < (N - 1))[None, :]
    off = la[:, None] * LS + k[None, :] * PS
    va = tl.load(y_ptr + off, mask=m, other=0.0).to(tl.float32)
    vb = tl.load(y_ptr + off + PS, mask=m, other=0.0).to(tl.float32)
    t = (va + vb) * HDX
    tl.store(o_ptr + la[:, None] * OLS + k[None, :] * OPS, t, mask=m)


# ---------------- whole-line tile, Blelloch scan via split/join -------------
@triton.jit
def k_blel(y_ptr, o_ptr, N: tl.constexpr, LS: tl.constexpr, PS: tl.constexpr,
           OLS: tl.constexpr, OPS: tl.constexpr, AL: tl.constexpr,
           LOGN: tl.constexpr, HDX: tl.constexpr):
    pid = tl.program_id(0)
    la = pid * AL + tl.arange(0, AL)
    k = tl.arange(0, N)
    m = (k < (N - 1))[None, :]
    off = la[:, None] * LS + k[None, :] * PS
    va = tl.load(y_ptr + off, mask=m, other=0.0).to(tl.float32)
    vb = tl.load(y_ptr + off + PS, mask=m, other=0.0).to(tl.float32)
    t = (va + vb) * HDX

    ups = []
    cur = t
    for s in tl.static_range(0, LOGN):
        a, b = tl.split(tl.reshape(cur, (AL, N >> (s + 1), 2)))
        ups.append(a)
        cur = a + b

    c = tl.zeros((AL, 1), tl.float32)
    for s in tl.static_range(0, LOGN):
        a = ups[LOGN - 1 - s]
        c = tl.reshape(tl.join(c, c + a), (AL, N >> (LOGN - 1 - s)))

    cs = t + c
    tl.store(o_ptr + la[:, None] * OLS + k[None, :] * OPS, cs, mask=m)


@triton.jit
def k_tiny(x_ptr):
    tl.store(x_ptr + tl.program_id(0), 1.0)


@triton.jit
def k_tiny2(x_ptr, n):
    tl.store(x_ptr + tl.program_id(0), n)


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
        OUT["host_ms"][key] = h
        say(key, "wall", w, "host", h)
    except Exception as ex:  # noqa: BLE001
        OUT["errors"][key] = str(traceback.format_exception_only(type(ex), ex))[:700]
        say(key, "ERR", OUT["errors"][key][:200])
    flush()


def flat_arg():
    return None


def main():
    OUT["info"]["triton_version"] = triton.__version__
    buf = torch.zeros(64, device=DEV, dtype=torch.float32)
    y1 = torch.randn(1024 * 4096, device=DEV, dtype=torch.float16)
    o0 = torch.empty(1024 * 4096, device=DEV, dtype=torch.float16)

    # ---------------- (a) host launch path ----------------
    timed("host_tiny_1arg", lambda: k_tiny[(1,)](buf), 200)
    timed("host_tiny_2arg", lambda: k_tiny2[(1,)](buf, 3), 200)
    timed("host_torch_add_", lambda: buf.add_(1.0), 200)
    try:
        OUT["info"]["jitrun_src"] = inspect.getsource(
            triton.runtime.jit.JITFunction.run)[:3500]
    except Exception as ex:  # noqa: BLE001
        OUT["errors"]["jitrun_src"] = repr(ex)
    flush()

    # discover the compiled-kernel object behind a launch
    try:
        h = k_tiny.warmup(buf, grid=(1,))
        OUT["info"]["warmup_type"] = str(type(h))
        OUT["info"]["warmup_attrs"] = [a for a in dir(h) if not a.startswith("_")][:80]
    except Exception as ex:  # noqa: BLE001
        OUT["errors"]["warmup"] = repr(ex)[:400]
    try:
        dc = getattr(k_tiny, "device_caches", None)
        OUT["info"]["device_caches"] = str(dc)[:600]
        for kk, v in (dc or {}).items():
            OUT["info"]["dc0"] = str(kk) + " -> " + str(type(v))
            for k2, v2 in (v[0] if isinstance(v, tuple) else {}).items():
                OUT["info"]["ck"] = str(k2)[:300] + " -> " + str(type(v2))
                OUT["info"]["ck_attrs"] = [a for a in dir(v2) if not a.startswith("_")][:80]
                break
            break
    except Exception as ex:  # noqa: BLE001
        OUT["errors"]["dc"] = repr(ex)[:400]
    flush()

    # ---------------- (b) device kernels ----------------
    for tag, M, L, dt, al in (("m1024x4096_f16", 1024, 4096, torch.float16, 1),
                              ("m1024x4096_f16_al2", 1024, 4096, torch.float16, 2),
                              ("m128x1024_f16", 128, 1024, torch.float16, 4),
                              ("m1024x4096_f32", 1024, 4096, torch.float32, 1)):
        y = y1.view(M, L) if dt == torch.float16 else y1.float().view(M, L)
        o = o0.view(M, L) if dt == torch.float16 else o0.float().view(M, L)
        cm = dict(N=L, LS=L, PS=1, OLS=L - 1, OPS=1, AL=al, HDX=0.5)
        lg = L.bit_length() - 1
        timed(f"{tag}_copy", lambda: k_copy[(M // al,)](y, o, **cm))
        timed(f"{tag}_blel", lambda: k_blel[(M // al,)](y, o, LOGN=lg, **cm))
        try:
            k_blel[(M // al,)](y, o, LOGN=lg, **cm)
            torch.npu.synchronize()
            yc = y.cpu().double()
            exact = torch.cumsum((yc[:, :-1] + yc[:, 1:]) * 0.5, dim=1)
            OUT["numerics"][tag] = float((o.cpu().double()[:, : L - 1] - exact).abs().max())
        except Exception as ex:  # noqa: BLE001
            OUT["errors"][f"{tag}_num"] = str(traceback.format_exception_only(type(ex), ex))[:400]
        flush()
        del y, o

    say(json.dumps(OUT["wall_ms"], indent=1)[:2500])
    say("info", json.dumps(OUT["info"], indent=1)[:3000])
    say("num", json.dumps(OUT["numerics"], indent=1)[:800])
    say("errors", json.dumps(OUT["errors"], indent=1)[:2500])
    flush()


if __name__ == "__main__":
    main()
