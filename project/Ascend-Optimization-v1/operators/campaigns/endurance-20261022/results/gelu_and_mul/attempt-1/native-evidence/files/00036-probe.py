"""Bounded diagnosis for gelu_and_mul on Ascend (diagnostic evidence only).

Questions:
 1. How much of the seed's latency is host-side launch/wrapper cost vs device?
 2. Is `tl.erf` the per-element bottleneck?
 3. Does a contiguous, mask-free, 2D-grid row-tile kernel vectorize?
"""

import json
import os
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
os.makedirs(ART, exist_ok=True)
OUT = {}


def rec(key, value):
    OUT[key] = value
    print(key, "=", value, flush=True)


# ---------------- environment ----------------
rec("torch", torch.__version__)
rec("triton", triton.__version__)
try:
    import torch_npu

    rec("torch_npu", torch_npu.__version__)
except Exception as exc:  # pragma: no cover
    rec("torch_npu", "import-failed: %r" % (exc,))

props = {}
try:
    from triton.runtime import driver

    raw = driver.active.utils.get_device_properties(torch.npu.current_device())
    for k, v in raw.items():
        props[k] = v if isinstance(v, (int, float, str, bool)) else str(v)
except Exception as exc:
    props["error"] = repr(exc)
rec("device_properties", props)
rec("current_device", torch.npu.current_device())


# ---------------- seed (round 1 candidate) ----------------
@triton.jit
def _seed_kernel(
    hidden_states_ptr,
    output_ptr,
    hidden_size,
    batch_size,
    input_stride_batch,
    input_stride_hidden,
    output_stride_batch,
    output_stride_hidden,
    num_ctas,
    tiles_per_cta,
    num_column_tiles,
    total_tiles,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    lanes = tl.arange(0, BLOCK_ROWS * BLOCK_COLUMNS)
    rows_in_tile = lanes // BLOCK_COLUMNS
    columns_in_tile = lanes - rows_in_tile * BLOCK_COLUMNS
    for iteration in range(0, tiles_per_cta):
        tile_id = pid + iteration * num_ctas
        batch_tile = tile_id // num_column_tiles
        column_tile = tile_id - batch_tile * num_column_tiles
        batch_index = batch_tile * BLOCK_ROWS + rows_in_tile
        columns = column_tile * BLOCK_COLUMNS + columns_in_tile
        mask = (
            (tile_id < total_tiles)
            & (batch_index < batch_size)
            & (columns < hidden_size)
        )
        input_row = batch_index * input_stride_batch
        gate = tl.load(
            hidden_states_ptr + input_row + columns * input_stride_hidden,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up = tl.load(
            hidden_states_ptr
            + input_row
            + (hidden_size + columns) * input_stride_hidden,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        activated = gate * (0.5 * (1.0 + tl.erf(gate * 0.7071067811865475)))
        output_offsets = batch_index * output_stride_batch + columns * output_stride_hidden
        tl.store(output_ptr + output_offsets, activated * up, mask=mask)


def seed_wrapper(hidden_states):
    batch_size, doubled = hidden_states.shape
    d = doubled // 2
    output = torch.empty((batch_size, d), dtype=hidden_states.dtype, device=hidden_states.device)
    block_columns = min(1024, triton.next_power_of_2(d))
    block_rows = min(1024 // block_columns, triton.next_power_of_2(batch_size))
    nct = triton.cdiv(d, block_columns)
    nbt = triton.cdiv(batch_size, block_rows)
    nt = nbt * nct
    num_ctas = min(nt, 48)
    tpc = triton.cdiv(nt, num_ctas)
    _seed_kernel[(num_ctas,)](
        hidden_states, output, d, batch_size,
        hidden_states.stride(0), hidden_states.stride(1),
        output.stride(0), output.stride(1),
        num_ctas, tpc, nct, nt, block_rows, block_columns,
        num_warps=4,
    )
    return output


# ---------------- new row-tile kernel ----------------
@triton.jit
def _row_kernel(
    x_ptr,
    o_ptr,
    d,
    in_row_stride,
    out_row_stride,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    GELU_MODE: tl.constexpr,
    TANH: tl.constexpr,
):
    row = tl.program_id(0)
    ct = tl.program_id(1)
    cols = ct * BLOCK + tl.arange(0, BLOCK)
    base = row * in_row_stride
    if EVEN:
        g = tl.load(x_ptr + base + cols)
        u = tl.load(x_ptr + base + d + cols)
    else:
        mask = cols < d
        g = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
        u = tl.load(x_ptr + base + d + cols, mask=mask, other=0.0)
    gf = g.to(tl.float32)
    uf = u.to(tl.float32)
    if GELU_MODE == 0:
        act = gf * (0.5 * (1.0 + tl.erf(gf * 0.7071067811865475)))
    elif GELU_MODE == 1:
        if TANH == 0:
            inner = 0.7978845608028654 * (gf + 0.044715 * gf * gf * gf)
            t2 = 1.0 - 2.0 / (tl.exp(2.0 * inner) + 1.0)
            act = 0.5 * gf * (1.0 + t2)
        else:
            act = 0.5 * gf * (1.0 + tl.tanh(0.7978845608028654 * (gf + 0.044715 * gf * gf * gf)))
    else:
        ax = tl.abs(gf)
        t = 1.0 / (1.0 + 0.3275911 * ax)
        y = t * (
            0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
        )
        y = 1.0 - y * tl.exp(-ax * ax)
        e = tl.where(gf < 0, -y, y)
        act = gf * (0.5 * (1.0 + e))
    res = act * uf
    orow = row * out_row_stride
    if EVEN:
        tl.store(o_ptr + orow + cols, res.to(o_ptr.dtype.element_ty))
    else:
        tl.store(o_ptr + orow + cols, res.to(o_ptr.dtype.element_ty), mask=mask)


def row_wrapper(hidden_states, block, gelu_mode, tanh=1):
    rows, doubled = hidden_states.shape
    d = doubled // 2
    out = torch.empty((rows, d), dtype=hidden_states.dtype, device=hidden_states.device)
    nct = triton.cdiv(d, block)
    _row_kernel[(rows, nct)](
        hidden_states, out, d,
        hidden_states.stride(0), out.stride(0),
        block, (d % block == 0), gelu_mode, tanh,
        num_warps=4,
    )
    return out


# ---------------- timing helpers ----------------
def bench_wall(fn, iters=50, warmup=10):
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


def bench_host(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    return (t1 - t0) / iters * 1e3


def ref_out(hs):
    d = hs.shape[-1] // 2
    x1, x3 = hs[..., :d], hs[..., d:]
    o = F.gelu(x1.float(), approximate="none") * x3.float()
    return o.to(hs.dtype)


def maxerr(got, exp):
    a = got.float()
    b = exp.float()
    return float((a - b).abs().max()), float(((a - b).abs() / (b.abs() + 1e-6)).max())


SHAPES = [(1, 2048), (8, 2048), (48, 6144), (4096, 16384)]
results = []

torch.manual_seed(0)
for shape in SHAPES:
    hs = torch.randn(*shape, dtype=torch.float32, device="npu").to(torch.bfloat16)
    rows, doubled = shape
    d = doubled // 2
    exp = ref_out(hs)
    entry = {"shape": list(shape), "d": d, "variants": {}}

    def add(name, fn, out=None):
        wall = bench_wall(fn)
        host = bench_host(fn)
        e = {"wall_ms": wall, "host_ms": host}
        if out is not None:
            me, mre = maxerr(out, exp)
            e["max_abs_err"] = me
            e["max_rel_err"] = mre
        entry["variants"][name] = e
        print(shape, name, e, flush=True)

    if shape[0] * d <= 4096 * 8192:
        out = seed_wrapper(hs)
        add("seed", lambda: seed_wrapper(hs), out)

    for block in (1024, 2048, 4096):
        if block > 4 * d:
            continue
        for mode, tname in ((0, "tl_erf"), (1, "tanh"), (2, "exp_poly_erf")):
            try:
                out = row_wrapper(hs, block, mode)
                add(
                    "row_b%d_%s" % (block, tname),
                    lambda block=block, mode=mode: row_wrapper(hs, block, mode),
                    out,
                )
            except Exception as exc:
                entry["variants"]["row_b%d_%s" % (block, tname)] = {"error": repr(exc)[:400]}
                print("ERR", shape, block, tname, repr(exc)[:200], flush=True)

    results.append(entry)

OUT["shapes"] = results

# pure allocation + host cost of empty
hs = torch.randn(1, 2048, dtype=torch.float32, device="npu").to(torch.bfloat16)
OUT["empty_host_ms"] = bench_host(
    lambda: torch.empty((1, 1024), dtype=torch.bfloat16, device=hs.device)
)
OUT["shape_host_ms"] = bench_host(lambda: hs.shape)

with open(os.path.join(ART, "probe.json"), "w") as fh:
    json.dump(OUT, fh, indent=2, default=str)
print("WROTE", os.path.join(ART, "probe.json"))
