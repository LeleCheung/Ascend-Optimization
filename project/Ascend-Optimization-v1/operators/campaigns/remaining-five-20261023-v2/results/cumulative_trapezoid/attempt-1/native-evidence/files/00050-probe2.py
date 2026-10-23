"""Lean probe: is a Cube (tl.dot) lower-triangular prefix faster than tl.cumsum?

fp16 only, few variants, results flushed after each step so a timeout still
leaves evidence. Component probe, not a formal evaluation.
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
OUT = {"timings_ms": {}, "errors": {}, "numerics": {}}


def flush():
    with open(os.path.join(ART, "probe2.json"), "w") as f:
        json.dump(OUT, f, indent=2, default=str)


def say(*a):
    print(*a, flush=True)


@triton.jit
def k_cumsum(y_ptr, o_ptr, L: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, L)
    a = tl.load(y_ptr + pid * L + cols).to(tl.float32)
    b = tl.load(y_ptr + pid * L + cols + 1).to(tl.float32)
    cs = tl.cumsum((a + b) * 0.5, axis=0)
    tl.store(o_ptr + pid * L + cols, cs)


@triton.jit
def k_dot_scan(y_ptr, o_ptr, L: tl.constexpr, BM: tl.constexpr,
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


def bench(fn, iters=10):
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
        OUT["errors"][key] = str(traceback.format_exception_only(type(ex), ex))[:500]
        say(key, "ERROR", OUT["errors"][key][:200])
    flush()


def main():
    M, L = 1024, 4096
    torch.manual_seed(0)
    yf = torch.randn(M * L + 64, device=DEV, dtype=torch.float16)
    y = yf[: M * L].view(M, L)
    o = torch.empty(M * L, device=DEV, dtype=torch.float16)
    flush()

    timed("cumsum", lambda: k_cumsum[(M,)](y, o, L))
    timed("dot_bm32_c128", lambda: k_dot_scan[(M // 32,)](y, o, L, 32, 128, L // 128, tl.float16))
    timed("dot_bm32_c256", lambda: k_dot_scan[(M // 32,)](y, o, L, 32, 256, L // 256, tl.float16))
    timed("dot_bm64_c128", lambda: k_dot_scan[(M // 64,)](y, o, L, 64, 128, L // 128, tl.float16))

    # numerics of the dot path vs an exact fp64 prefix of the same increments
    try:
        k_dot_scan[(M // 32,)](y, o, L, 32, 128, L // 128, tl.float16)
        torch.npu.synchronize()
        yc = y.cpu().double()
        exact = torch.cumsum((yc[:, :-1] + yc[:, 1:]) * 0.5, dim=1)
        dot = o.view(M, L)[:, : L - 1].cpu().double()
        k_cumsum[(M,)](y, o, L)
        torch.npu.synchronize()
        cs = o.view(M, L)[:, : L - 1].cpu().double()
        OUT["numerics"]["fp16"] = {
            "dot_max_abs_vs_exact": float((dot - exact).abs().max()),
            "cumsum_max_abs_vs_exact": float((cs - exact).abs().max()),
            "dot_vs_cumsum_max_abs": float((dot - cs).abs().max()),
        }
    except Exception as ex:  # noqa: BLE001
        OUT["errors"]["numerics"] = str(traceback.format_exception_only(type(ex), ex))[:500]
    flush()
    say("numerics", json.dumps(OUT["numerics"], indent=2))
    say("errors", json.dumps(OUT["errors"], indent=2)[:2000])


if __name__ == "__main__":
    main()
