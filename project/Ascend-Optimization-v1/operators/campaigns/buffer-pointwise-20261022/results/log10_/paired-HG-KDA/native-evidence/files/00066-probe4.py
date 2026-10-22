"""Probe 4: is the large-shape gap memory-bound or math-bound? Copy floor vs log variants."""
import importlib.util
import json
import os
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
out = {}


def bench(fn, iters):
    try:
        for _ in range(10):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        t1 = time.perf_counter()
        torch.npu.synchronize()
        t2 = time.perf_counter()
        return {"host_us": round((t1 - t0) / iters * 1e6, 3),
                "total_us": round((t2 - t0) / iters * 1e6, 3)}
    except Exception as exc:  # noqa
        return {"error": repr(exc)[:300]}


spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)
ADDR = cand._hgkda_address_type
ST = cand._resolve_stream(0)


def mkfast(kernel, t, scalars, consts, grid):
    a = ADDR(t.numel())(t.data_ptr())
    compiled = kernel.warmup(t, *scalars, **consts, grid=(1,))
    rn = compiled.run
    call, fn, md = rn.launch, compiled.function, compiled.packed_metadata

    def go():
        call(grid, 1, 1, ST(), fn, md, None, None, None, a, *scalars, None)
    return go


@triton.jit
def _k(X, MODE: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    if MODE == 0:
        y = x
    elif MODE == 1:
        y = tl.log(x) * 0.4342944819032518
    elif MODE == 2:
        y = tl.log2(x) * 0.3010299956639812
    elif MODE == 3:
        y = tl.math.log10(x)
    else:
        y = tl.math.log2(x) * 0.3010299956639812
    tl.store(X + offs, y.to(X.dtype.element_ty))


MODES = {0: "copy", 1: "log_mul", 2: "log2_mul", 3: "math_log10", 4: "math_log2"}


def make(dtype, n, seed):
    return (torch.randn(n, generator=torch.Generator().manual_seed(seed)).abs() + 0.05).to(dtype).npu()


for dname, dtype, seed in (("fp16", torch.float16, 11), ("fp32", torch.float32, 12)):
    r = {}
    n = 4096 * 4096
    for mode, label in MODES.items():
        for block in ((32768,) if mode else (16384, 32768, 65536)):
            t = make(dtype, n, seed)
            try:
                go = mkfast(_k, t, (), {"MODE": mode, "BLOCK": block}, n // block)
                key = f"{label}_b{block}"
                r[key] = bench(go, 200)
                # accuracy on a fresh tensor
                tt = make(dtype, n, seed + 100)
                ref = torch.log10(tt.clone().double())
                go2 = mkfast(_k, tt, (), {"MODE": mode, "BLOCK": block}, n // block)
                go2()
                torch.npu.synchronize()
                d = (tt.double() - ref)
                small = ref.abs() < 1e-3
                r[key]["max_abs_err"] = float(d.abs().max().item())
                r[key]["max_rel_err_nonsmall"] = float((d[~small].abs() / ref[~small].abs()).max().item()) if bool((~small).any()) else 0.0
            except Exception as exc:  # noqa
                r[f"{label}_b{block}"] = {"error": repr(exc)[:250]}
    r["torch"] = bench(lambda: torch.log10_(make(dtype, n, seed)), 200)
    r["torch_fixed"] = bench((lambda t: (lambda: torch.log10_(t)))(make(dtype, n, seed)), 200)
    out[dname] = r

# small-shape math cost (1M, host-bound so compare host_us only)
r = {}
n = 1048576
for mode, label in MODES.items():
    t = make(torch.float16, n, 21)
    try:
        go = mkfast(_k, t, (), {"MODE": mode, "BLOCK": 32768}, n // 32768)
        r[label] = bench(go, 2000)
    except Exception as exc:  # noqa
        r[label] = {"error": repr(exc)[:200]}
out["fp16_1M"] = r

path = os.path.join(ART, "probe4.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1)[:9000])
print("ARTIFACT", path)
