"""Bounded diagnosis 2: Triton launch-path breakdown + device tile sweep.

Diagnostic evidence only. Questions:
 A. Where do the ~100 us of host time per call go, and can a cached
    CompiledKernel be invoked directly (bypassing JITFunction.run)?
 B. Which BLOCK / num_warps / rows-per-program minimizes device time on the
    big contiguous case and on a small-d many-row case?
"""

import inspect
import json
import os
import time

import torch
import triton
import triton.language as tl
from triton.runtime import driver
from triton.runtime.jit import JITFunction

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {}


def rec(key, value):
    OUT[key] = value
    print(key, "=", str(value)[:2000], flush=True)


DEV = torch.npu.current_device()


@triton.jit
def _rows_kernel(x_ptr, o_ptr, d, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    row = tl.program_id(0)
    ct = tl.program_id(1)
    cols = ct * BLOCK + tl.arange(0, BLOCK)
    in_base = row * (2 * d)
    if EVEN:
        gate = tl.load(x_ptr + in_base + cols)
        up = tl.load(x_ptr + in_base + d + cols)
    else:
        mask = cols < d
        gate = tl.load(x_ptr + in_base + cols, mask=mask, other=0.0)
        up = tl.load(x_ptr + in_base + d + cols, mask=mask, other=0.0)
    gf = gate.to(tl.float32)
    uf = up.to(tl.float32)
    act = gf * (0.5 * (1.0 + tl.erf(gf * 0.7071067811865475)))
    res = act * uf
    out_base = row * d
    if EVEN:
        tl.store(o_ptr + out_base + cols, res.to(o_ptr.dtype.element_ty))
    else:
        tl.store(o_ptr + out_base + cols, res.to(o_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _multi_row_kernel(
    x_ptr, o_ptr, d, rows, BLOCK: tl.constexpr, EVEN: tl.constexpr, R: tl.constexpr
):
    row0 = tl.program_id(0) * R
    ct = tl.program_id(1)
    cols = ct * BLOCK + tl.arange(0, BLOCK)
    n = tl.minimum(R, rows - row0)
    for r in range(0, n):
        row = row0 + r
        in_base = row * (2 * d)
        if EVEN:
            gate = tl.load(x_ptr + in_base + cols)
            up = tl.load(x_ptr + in_base + d + cols)
        else:
            mask = cols < d
            gate = tl.load(x_ptr + in_base + cols, mask=mask, other=0.0)
            up = tl.load(x_ptr + in_base + d + cols, mask=mask, other=0.0)
        gf = gate.to(tl.float32)
        uf = up.to(tl.float32)
        act = gf * (0.5 * (1.0 + tl.erf(gf * 0.7071067811865475)))
        out_base = row * d
        if EVEN:
            tl.store(o_ptr + out_base + cols, (act * uf).to(o_ptr.dtype.element_ty))
        else:
            tl.store(
                o_ptr + out_base + cols,
                (act * uf).to(o_ptr.dtype.element_ty),
                mask=cols < d,
            )


def wall(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    st = torch.npu.Event(enable_timing=True)
    en = torch.npu.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.npu.synchronize()
    return st.elapsed_time(en) / iters


def cpu(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    return (t1 - t0) / iters * 1e3


# ---------------- A. host launch path ----------------
hs = torch.randn(64, 2048, dtype=torch.float32, device="npu").to(torch.bfloat16)
d = 1024
out = torch.empty((64, d), dtype=torch.bfloat16, device="npu")

_rows_kernel[(64, 1)](hs, out, d, 1024, True, num_warps=4)
torch.npu.synchronize()

rec("full_launch_cpu_ms", cpu(lambda: _rows_kernel[(64, 1)](hs, out, d, 1024, True, num_warps=4)))
rec("stream_lookup_cpu_ms", cpu(lambda: driver.active.get_current_stream(DEV)))
rec("current_stream_cpu_ms", cpu(lambda: torch.npu.current_stream()))
rec("empty_cpu_ms", cpu(lambda: torch.empty((64, d), dtype=torch.bfloat16, device="npu")))
rec("shape_cpu_ms", cpu(lambda: hs.stride(0)))

try:
    OUT["jit_run_source"] = inspect.getsource(JITFunction.run)
except Exception as exc:
    OUT["jit_run_source"] = repr(exc)

caches = getattr(_rows_kernel, "device_caches", None)
try:
    per_dev = caches[DEV][0]
    rec("cache_size", len(per_dev))
    ck = list(per_dev.values())[0]
    rec("compiled_kernel_type", type(ck))
    rec("compiled_kernel_attrs", [a for a in dir(ck) if not a.startswith("_")])
    rec("launcher_type", type(ck.run))
    rec("launcher_sig", str(inspect.signature(ck.run)) if callable(ck.run) else "n/a")
    rec(
        "ck_fields",
        {
            k: str(v)[:200]
            for k, v in vars(ck).items()
            if k in ("function", "packed_metadata", "metadata", "src", "module")
        },
    )
except Exception as exc:
    OUT["cache_introspection_error"] = repr(exc)
    ck = None

# try direct launcher invocations
if ck is not None:
    stream = driver.active.get_current_stream(DEV)

    def attempt(name, fn):
        try:
            fn()
            torch.npu.synchronize()
            ref = torch.nn.functional.gelu(hs[:, :d].float()) * hs[:, d:].float()
            ok = bool(torch.allclose(out.float(), ref, atol=0.05, rtol=0.02))
            OUT[name] = {"ok": True, "close": ok, "cpu_ms": cpu(fn)}
        except Exception as exc:
            OUT[name] = {"ok": False, "error": repr(exc)[:400]}
        print(name, OUT[name], flush=True)

    variants = {
        "direct_v1": lambda: ck.run(
            ck, 64, 1, 1, stream, ck.function, ck.packed_metadata, None, None, None,
            hs, out, d, 1024, True,
        ),
        "direct_v2": lambda: ck.run(
            64, 1, 1, stream, ck.function, ck.packed_metadata, None, None, None,
            hs, out, d, 1024, True,
        ),
        "direct_v3": lambda: ck.run(
            ck, 64, 1, 1, stream, ck.function, ck.packed_metadata, None, None, None,
            hs, out, d,
        ),
    }
    for name, fn in variants.items():
        attempt(name, fn)

# ---------------- B. device tile sweep ----------------
shapes = [(4096, 16384), (4096, 2048)]
sweep = []
for shape in shapes:
    rows, doubled = shape
    dd = doubled // 2
    x = torch.randn(rows, doubled, dtype=torch.float32, device="npu").to(torch.bfloat16)
    o = torch.empty((rows, dd), dtype=torch.bfloat16, device="npu")
    ref = torch.nn.functional.gelu(x[:, :dd].float()) * x[:, dd:].float()
    entry = {"shape": list(shape), "d": dd, "variants": {}}
    for block in (1024, 2048, 4096, 8192):
        if block > dd and block > 32:
            continue
        even = dd % block == 0 and block <= dd
        nct = triton.cdiv(dd, block)
        for nw in (1, 2, 4, 8):
            name = "b%d_nw%d" % (block, nw)
            try:
                f = lambda block=block, nw=nw, nct=nct, even=even: _rows_kernel[(rows, nct)](
                    x, o, dd, block, even, num_warps=nw
                )
                f()
                torch.npu.synchronize()
                err = float((o.float() - ref).abs().max())
                entry["variants"][name] = {"wall_ms": wall(f), "max_err": err}
            except Exception as exc:
                entry["variants"][name] = {"error": repr(exc)[:200]}
            print(shape, name, entry["variants"][name], flush=True)
    for rr in (2, 4, 8):
        for block in (1024, 2048, 4096):
            if block > dd:
                continue
            even = dd % block == 0
            nct = triton.cdiv(dd, block)
            name = "multi_r%d_b%d" % (rr, block)
            try:
                f = lambda rr=rr, block=block, nct=nct, even=even: _multi_row_kernel[
                    (triton.cdiv(rows, rr), nct)
                ](x, o, dd, rows, block, even, rr, num_warps=4)
                f()
                torch.npu.synchronize()
                err = float((o.float() - ref).abs().max())
                entry["variants"][name] = {"wall_ms": wall(f), "max_err": err}
            except Exception as exc:
                entry["variants"][name] = {"error": repr(exc)[:200]}
            print(shape, name, entry["variants"][name], flush=True)
    sweep.append(entry)
    del x, o, ref

OUT["sweep"] = sweep

# host cost of each launch form (tiny case, host dominates)
x = torch.randn(1, 2048, dtype=torch.float32, device="npu").to(torch.bfloat16)
o = torch.empty((1, 1024), dtype=torch.bfloat16, device="npu")
OUT["tiny_rows_b1024_nw4"] = {
    "wall_ms": wall(lambda: _rows_kernel[(1, 1)](x, o, 1024, 1024, True, num_warps=4)),
    "cpu_ms": cpu(lambda: _rows_kernel[(1, 1)](x, o, 1024, 1024, True, num_warps=4)),
}
for nw in (1, 2, 8):
    OUT["tiny_nw%d_cpu_ms" % nw] = cpu(
        lambda nw=nw: _rows_kernel[(1, 1)](x, o, 1024, 1024, True, num_warps=nw)
    )

with open(os.path.join(ART, "probe2.json"), "w") as fh:
    json.dump(OUT, fh, indent=2, default=str)
print("WROTE probe2.json")
