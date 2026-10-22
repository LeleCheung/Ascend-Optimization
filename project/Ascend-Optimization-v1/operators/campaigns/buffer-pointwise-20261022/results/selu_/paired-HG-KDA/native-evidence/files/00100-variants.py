"""Interleaved A/B of shippable host-path variants for selu_ (fp16 1024 case).

Each variant is a complete, admissible launch path (except where marked FLOOR).
Also verifies numerical correctness of the plain-int launcher entry.
"""
import importlib.util
import json
import os
import statistics
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
out = {}


def save():
    with open(os.path.join(ART, "variants.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)


spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

numel_m = torch.Tensor.numel
dptr_m = torch.Tensor.data_ptr
contig_m = torch.Tensor.is_contiguous

x = torch.randn(1024, dtype=torch.float16, device="npu")
n = x.numel()
ptr = x.data_ptr()
block = 1024
assert n % block == 0
grid = n // block

compiled = cand._selu_nomask.warmup(x, BLOCK=block, grid=(1,))
CALL = compiled.run.launch
FUNC = compiled.function
META = compiled.packed_metadata
STREAM = cand._resolve_stream(x.device.index)
ADDR = cand._hgkda_address_type(n)


# ---------------------------------------------------------------- variants
def m0_current(A):
    return cand.run(A)


def m1_plain_int(A):
    """Prebound methods, dict keyed by numel, plain int pointer, single frame."""
    p = dptr_m(A)
    e = _TAB.get(numel_m(A))
    if e is not None and e[0] is A.dtype and contig_m(A):
        e[1 + ((p & 15) != 0)](p)
        return A
    return cand._dispatch(A, numel_m(A), p)


def m2_int_subclass(A):
    """Same as m1 but keeps the profiler-compatible address subclass."""
    p = dptr_m(A)
    e = _TAB.get(numel_m(A))
    if e is not None and e[0] is A.dtype and contig_m(A):
        e[2 + ((p & 15) != 0)](p)
        return A
    return cand._dispatch(A, numel_m(A), p)


def m3_no_contig(A):
    """MARGINAL-COST PROBE (inadmissible): drops the contiguity check."""
    p = dptr_m(A)
    e = _TAB.get(numel_m(A))
    if e is not None and e[0] is A.dtype:
        e[1 + ((p & 15) != 0)](p)
        return A
    return cand._dispatch(A, numel_m(A), p)


def m4_no_meta(A):
    """MARGINAL-COST PROBE (inadmissible): no metadata reads at all."""
    CALL(grid, 1, 1, STREAM(), FUNC, META, None, None, None, ptr, None)
    return A


def m5_torch(A):
    return torch.selu_(A)


def ref_style(A):
    """Reference wrapper shape: locals().copy() + dict dispatch."""
    kwargs = locals().copy()
    fns = {"selu_": torch.selu_}
    return fns["selu_"](kwargs["A"])


# table: numel -> (dtype, launch_aligned_plain, launch_unaligned_plain,
#                  launch_aligned_sub, launch_unaligned_sub)


def _mk(subclass):
    def launcher(kind):
        def go(p, grid=grid, call=CALL, s=STREAM, f=FUNC, m=META,
               at=ADDR if subclass else None):
            if subclass:
                call(grid, 1, 1, s(), f, m, None, None, None, at(p), None)
            else:
                call(grid, 1, 1, s(), f, m, None, None, None, p, None)
        return go
    return launcher


_TAB = {}
_plain = _mk(False)(None)
_plain_u = _plain  # alignment only selects the launcher kind; same kernel here
_sub = _mk(True)(None)
_TAB[n] = (x.dtype, _plain, _plain_u, _sub, _sub)

# ------------------------------------------------------- correctness spot check
check = {}
ref = torch.nn.functional.selu(x.clone().float()).half()
for name, fn in [("m1_plain_int", m1_plain_int), ("m2_int_subclass", m2_int_subclass)]:
    y = x.clone()
    fn(y)
    torch.npu.synchronize()
    diff = (y.float() - ref.float()).abs().max().item()
    check[name] = {"max_abs_diff_vs_ref": diff}
out["correctness"] = check
save()

# ------------------------------------------------------------- interleaved A/B
def one_pass(fn, iters=300, warm=30):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    return (t1 - t0) / iters * 1e6


order = [
    ("m0_current", m0_current),
    ("m1_plain_int", m1_plain_int),
    ("m2_int_subclass", m2_int_subclass),
    ("m3_no_contig", m3_no_contig),
    ("m4_no_meta", m4_no_meta),
    ("m5_torch", m5_torch),
    ("ref_style", ref_style),
]
res = {name: [] for name, _ in order}
try:
    for rnd in range(3):
        for name, fn in order:
            res[name].append(one_pass(lambda f=fn: f(x)))
        print(f"round {rnd}: " + json.dumps({k: [round(v, 2) for v in vs] for k, vs in res.items()}), flush=True)
except Exception:
    out["error"] = traceback.format_exc()[-2000:]
out["host_us_median"] = {k: round(statistics.median(v), 3) for k, v in res.items() if v}
out["host_us_all"] = {k: [round(x, 3) for x in v] for k, v in res.items()}
save()
print(json.dumps(out, indent=2, default=str), flush=True)
