"""Introspect the flagtree-ascend launch path and test direct compiled-kernel calls."""
import json
import linecache
import os
import time
import traceback

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
DEV = "npu"
res = {}


def seg(path, a, b):
    lines = linecache.getlines(path)
    if not lines:
        return f"<missing {path}>"
    return "".join(f"{i}: {ln}" for i, ln in enumerate(lines[a - 1:b], start=a))


TR = "/usr/local/python3.11.15/lib/python3.11/site-packages/triton"
res["compiler_runner"] = seg(f"{TR}/spec/ascend/compiler/compiler.py", 440, 600)
res["driver_launch"] = seg(f"{TR}/backends/ascend/driver.py", 100, 260)


@triton.jit
def neg_1arg(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    tl.store(X + offs, -x)


@triton.jit
def neg_2arg(X, n, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(X + offs)
        tl.store(X + offs, -x)
    else:
        m = offs < n
        x = tl.load(X + offs, mask=m)
        tl.store(X + offs, -x, mask=m)


def bench_wall(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


x = torch.randn([1024], device=DEV, dtype=torch.float16)
tim = {}
tim["jit_2arg"] = bench_wall(lambda: neg_2arg[(1,)](x, 1024, BLOCK=1024, EVEN=True), 500)
tim["jit_1arg"] = bench_wall(lambda: neg_1arg[(1,)](x, BLOCK=1024), 500)
tim["torch_neg_"] = bench_wall(lambda: torch.neg_(x), 500)

# direct compiled-kernel calls
try:
    ck = neg_2arg.warmup(x, 1024, BLOCK=1024, EVEN=True, grid=(1,))
    ck._init_handles()
    res["ck_attrs"] = sorted(a for a in dir(ck) if not a.startswith("__"))
    res["ck_metadata"] = repr(getattr(ck, "metadata", None))[:2000]
    res["ck_packed_metadata"] = repr(getattr(ck, "packed_metadata", None))[:2000]
    res["ck_function"] = repr(getattr(ck, "function", None))[:500]
    res["ck_launch"] = repr(getattr(ck, "launch", None))[:500]
    stream = torch.npu.current_stream().npu_stream
    res["stream"] = stream
    variants = {
        "9plus2": (1, 1, 1, stream, ck.function, ck.packed_metadata, None, None, None, x, 1024),
        "11plus2": (1, 1, 1, stream, ck.function, ck.packed_metadata, None, None, None, 0, 0, x, 1024),
        "11plus2b": (1, 1, 1, stream, ck.function, ck.packed_metadata, None, None, None, None, None, x, 1024),
        "10plus2": (1, 1, 1, stream, ck.function, ck.packed_metadata, None, None, 0, x, 1024),
        "12plus2": (1, 1, 1, stream, ck.function, ck.packed_metadata, None, None, None, 0, 0, 0, x, 1024),
    }
    for name, args in variants.items():
        try:
            args[3]
            torch.manual_seed(0)
            expect = x.clone()
            torch.neg_(expect)
            ck.run(*args)
            torch.npu.synchronize()
            ok = bool(torch.equal(x, expect))
            tim[f"direct_{name}"] = bench_wall(lambda a=args: ck.run(*a), 500)
            res[f"direct_{name}_status"] = f"OK ok={ok} nargs={len(args)}"
        except Exception:
            res[f"direct_{name}_status"] = traceback.format_exc()[-500:]
except Exception:
    res["ck"] = traceback.format_exc()[-1500:]

# 1-arg compiled kernel via warmup + direct
try:
    ck1 = neg_1arg.warmup(x, BLOCK=1024, grid=(1,))
    ck1._init_handles()
    stream = torch.npu.current_stream().npu_stream
    for name, args in {
        "a": (1, 1, 1, stream, ck1.function, ck1.packed_metadata, None, None, None, x),
        "b": (1, 1, 1, stream, ck1.function, ck1.packed_metadata, None, None, None, 0, 0, x),
    }.items():
        try:
            ck1.run(*args)
            torch.npu.synchronize()
            res[f"direct1_{name}"] = "OK"
            tim[f"direct1_{name}"] = bench_wall(lambda a=args: ck1.run(*a), 500)
        except Exception:
            res[f"direct1_{name}"] = traceback.format_exc()[-400:]
except Exception:
    res["ck1"] = traceback.format_exc()[-1200:]

# what does the JIT wrapper spend time on?  time a repeated same-arg call path
try:
    jf = neg_2arg
    res["jit_binder"] = repr(getattr(jf, "binder", None))[:300]
except Exception:
    pass

res["timing_ms"] = tim
with open(os.path.join(ART, "launch_probe.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({"timing_ms": tim, "status": {k: v for k, v in res.items() if "status" in k or k.startswith("direct1_") or k == "ck_attrs"}}, indent=1))
