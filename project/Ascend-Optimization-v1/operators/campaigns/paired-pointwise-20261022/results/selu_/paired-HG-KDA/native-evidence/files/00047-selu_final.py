"""Three-way probe: native-precision accuracy, loop pipelining, host launch cost.

Established: device cost of these elementwise kernels is a fixed ~1.5us per TILE,
independent of bytes and dtype (fp16 and fp32 take the same time; a 16384 tile
costs the same as an 8192 tile).  So the levers are (a) fewer tiles, (b) overlap
tiles, (c) host launch cost for the launch-bound small shapes.

Checks:
  1. numerics of native fp16/bf16 math vs torch.selu (tolerance is 1.5% rel/abs)
  2. num_stages software pipelining for the fp32 path (blocked from 16384 by UB)
  3. where the ~14.7us raw-launch host cost goes (stream query vs launch binding)

Writes $KGS_DEBUG_ARTIFACTS/selu_final.json
"""

import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
OUT = {}


@triton.jit
def k_selu(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_selu_nat(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_selu_pipe(X, NT, BLOCK: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    g = tl.num_programs(0)
    for i in tl.range(0, NT, num_stages=NS):
        t = pid + i * g
        offs = t * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        xf = x.to(tl.float32)
        y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
        tl.store(X + offs, y.to(X.dtype.element_ty))


def bench(fn, iters=60, warmup=15):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    host_us = (time.perf_counter() - t0) / iters * 1e6
    torch.npu.synchronize()
    total_us = (time.perf_counter() - t0) / iters * 1e6
    return {"total_us": round(total_us, 2), "host_us": round(host_us, 2)}


def safe(fn):
    try:
        return bench(fn)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc).replace("\n", " ")[-160:]}


def accuracy(dtype, block, native):
    """max abs / rel error of the candidate formula against torch.selu."""
    n = 1 << 20
    src = torch.randn(n, dtype=dtype, device="npu:0")
    if dtype == torch.float16:
        src = src.clamp(-8.0, 8.0)
    ref = torch.selu(src.clone())
    got = src.clone()
    kern = k_selu_nat if native else k_selu
    kern[(n // block,)](got, BLOCK=block)
    torch.npu.synchronize()
    d = (got.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
    return {
        "max_abs": round(d.max().item(), 7),
        "max_rel": round((d / denom).max().item(), 7),
        "atol": 0.015 if dtype != torch.float32 else 1e-4,
        "rtol": 0.015 if dtype != torch.float32 else 1e-4,
        "bitwise_equal": bool(torch.equal(got, ref)),
    }


def main():
    n = 4096 * 4096

    # 1. numerics of the native path (the speedup depends on it being acceptable)
    OUT["acc_fp16_b16384_native"] = accuracy(torch.float16, 16384, True)
    OUT["acc_fp16_b8192_fp32math"] = accuracy(torch.float16, 8192, False)
    OUT["acc_bf16_b16384_native"] = accuracy(torch.bfloat16, 16384, True)
    OUT["acc_bf16_b8192_fp32math"] = accuracy(torch.bfloat16, 8192, False)

    # 2. device geometry
    for dtype, block, nat in ((torch.bfloat16, 16384, True), (torch.float32, 8192, False)):
        tag = str(dtype).replace("torch.", "")
        x = torch.randn(n, dtype=dtype, device="npu:0")
        rows = {}
        kern = k_selu_nat if nat else k_selu
        rows[f"selu_b{block}"] = safe(lambda: kern[(n // block,)](x, BLOCK=block))
        for g, ns in ((32, 2), (32, 3), (64, 2), (128, 2), (256, 2)):
            ntile = n // block
            if ntile % g:
                continue
            nt = ntile // g
            rows[f"pipe_b{block}_g{g}_ns{ns}"] = safe(
                lambda b=block, gg=g, nn=nt, s=ns: k_selu_pipe[(gg,)](x, nn, BLOCK=b, NS=s))
        OUT[f"geom_{tag}"] = rows
        del x

    # 3. host launch cost breakdown
    x = torch.randn(1024, dtype=torch.float32, device="npu:0")
    compiled = k_selu.warmup(x, BLOCK=1024, grid=(1,))
    dev = torch.npu.current_device()
    raw = torch_npu._C._npu_getCurrentRawStreamNoWait
    call = compiled.run.launch
    function = compiled.function
    metadata = compiled.packed_metadata
    stream_val = raw(dev)

    def stream_only():
        raw(dev)

    def launch_fixed_stream():
        call(1, 1, 1, stream_val, function, metadata, None, None, None, x, None)

    def launch_with_stream():
        call(1, 1, 1, raw(dev), function, metadata, None, None, None, x, None)

    def jit_wrapper():
        k_selu[(1,)](x, BLOCK=1024)

    def torch_selu():
        torch.selu_(x)

    OUT["host_breakdown"] = {
        "stream_query_only": bench(stream_only, iters=2000),
        "raw_launch_fixed_stream": bench(launch_fixed_stream, iters=600),
        "raw_launch_with_stream_query": bench(launch_with_stream, iters=600),
        "jit_wrapper_launch": bench(jit_wrapper, iters=600),
        "torch_selu_inplace": bench(torch_selu, iters=600),
    }

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "selu_final.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
