"""Tile-cost mechanism probe for in-place SELU on Ascend.

Findings so far: cost scales ~linearly with the NUMBER OF TILES and is nearly
independent of tile size and dtype, and a grid-stride loop with few programs is
not faster.  Two competing explanations remain:
  (A) fixed per-tile latency (~1.5us) that does not overlap, so fewer+bigger
      tiles win;
  (B) throughput limit that happens to be tile-count proportional.
This probes (A) by testing bigger tiles, multiple tiles per program, and
software-pipelined loops, and resolves device time cleanly by launching through
the cheap raw launcher (low host overhead) instead of the JIT wrapper.

Writes $KGS_DEBUG_ARTIFACTS/selu_tile.json
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


# ---- kernels: constants inlined (module globals are not allowed in @triton.jit)
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
    """Native-precision math: no fp32 tile, so BLOCK can double inside the UB."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = 1.0507009873554805 * tl.where(x > 0, x, 1.6732632423543772 * (tl.exp(x) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_selu_x2(X, BLOCK: tl.constexpr):
    """Two unrolled tiles per program (same tile size, half the programs)."""
    pid = tl.program_id(0)
    base = pid * (2 * BLOCK)
    for i in tl.static_range(2):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        xf = x.to(tl.float32)
        y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
        tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_selu_pipe(X, NT, BLOCK: tl.constexpr, NS: tl.constexpr):
    """Grid-stride loop with explicit software pipelining depth NS."""
    pid = tl.program_id(0)
    g = tl.num_programs(0)
    for i in tl.range(0, NT, num_stages=NS):
        t = pid + i * g
        offs = t * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        xf = x.to(tl.float32)
        y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
        tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_copy(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(X + offs, tl.load(X + offs))


_RT = {}


def _stream_getter():
    if "g" not in _RT:
        dev = torch.npu.current_device()
        raw = torch_npu._C._npu_getCurrentRawStreamNoWait
        _RT["g"] = lambda: raw(dev)
    return _RT["g"]


def make_raw(compiled, grid):
    """Bind the Ascend launcher ABI: (grid,1,1,stream,fn,meta,None,None,None,*params,None)."""
    call = compiled.run.launch
    function = compiled.function
    metadata = compiled.packed_metadata
    stream = _stream_getter()

    def invoke(*params):
        call(grid, 1, 1, stream(), function, metadata, None, None, None, *params, None)

    return invoke


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


def raw_variant(kern, x, grid, nparams, block):
    """Compile then launch through the raw path so host cost stays tiny."""
    shape_arg = [x] if nparams == 1 else [x, 0]
    try:
        compiled = kern.warmup(*shape_arg, BLOCK=block, grid=(1,))
        invoke = make_raw(compiled, grid)
        if nparams == 1:
            fn = lambda: invoke(x)
        else:
            fn = lambda: invoke(x, 0)
        return fn, "raw"
    except Exception as exc:  # noqa: BLE001
        return None, repr(exc)[-200:]


def safe(fn):
    try:
        return bench(fn)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc).replace("\n", " ")[-200:]}


def main():
    n = 4096 * 4096
    OUT["n"] = n

    for dtype in (torch.float16, torch.float32):
        tag = str(dtype).replace("torch.", "")
        x = torch.randn(n, dtype=dtype, device="npu:0")
        rows = {}

        def plain(kern, block, np_=1, extra=None):
            g = n // block
            if np_ == 1:
                return safe(lambda: kern[(g,)](x, BLOCK=block))
            return safe(lambda: kern[(g,)](x, extra, BLOCK=block))

        rows["plain_selu_b8192"] = plain(k_selu, 8192)
        rows["plain_selu_b16384"] = plain(k_selu, 16384)
        rows["plain_selu_nat_b16384"] = plain(k_selu_nat, 16384)
        rows["plain_selu_nat_b32768"] = plain(k_selu_nat, 32768)
        rows["plain_selu_x2_b8192"] = safe(lambda: k_selu_x2[(n // 16384,)](x, BLOCK=8192))
        rows["plain_copy_b8192"] = safe(lambda: k_copy[(n // 8192,)](x, BLOCK=8192))
        rows["plain_copy_b16384"] = safe(lambda: k_copy[(n // 16384,)](x, BLOCK=16384))

        # raw-launched variants: device time is then total_us almost directly
        for name, kern, block, np_ in (
            ("raw_selu_b8192", k_selu, 8192, 1),
            ("raw_selu_b16384", k_selu, 16384, 1),
            ("raw_selu_nat_b16384", k_selu_nat, 16384, 1),
            ("raw_selu_nat_b32768", k_selu_nat, 32768, 1),
            ("raw_selu_x2_b8192", k_selu_x2, 8192, 1),
            ("raw_copy_b8192", k_copy, 8192, 1),
            ("raw_copy_b16384", k_copy, 16384, 1),
        ):
            mult = 2 if kern is k_selu_x2 else 1
            grid = n // (block * mult) if block * mult <= n else 1
            if kern in (k_selu, k_selu_nat):
                grid = n // block
            fn, note = raw_variant(kern, x, grid, np_, block)
            rows[name] = safe(fn) if fn is not None else {"error": note}

        # pipelined grid-stride loop, 40 programs (one per vector core)
        for block, g, ns in ((8192, 40, 2), (8192, 40, 3), (8192, 80, 2), (4096, 40, 3)):
            ntile = n // block
            if ntile % g:
                rows[f"plain_pipe_b{block}_g{g}_ns{ns}"] = "not divisible"
                continue
            nt = ntile // g
            rows[f"plain_pipe_b{block}_g{g}_ns{ns}"] = safe(
                lambda b=block, gg=g, nn=nt, s=ns: k_selu_pipe[(gg,)](x, nn, BLOCK=b, NS=s))

        OUT[tag] = rows
        del x

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "selu_tile.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
