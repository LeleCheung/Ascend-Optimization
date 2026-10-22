"""Bounded diagnostic: separate GELU arithmetic cost from addressing cost, and
identify which GELU formula the NPU reference actually uses.

No candidate/protected state is modified. Results go to $KGS_DEBUG_ARTIFACTS.
"""

import importlib.util
import json
import math
import os
import time

import torch
import triton
import triton.language as tl

torch.manual_seed(0)
DEV = "npu"

RESULTS = {}


# ---------------------------------------------------------------- variants


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
    MODE: tl.constexpr,
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

        if MODE == 0:
            activated = gate
        elif MODE == 1:
            activated = gate * (0.5 * (1.0 + tl.erf(gate * 0.7071067811865475)))
        elif MODE == 2:
            inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
            activated = gate * (0.5 * (1.0 + inner * (0.0 + 2.0 / (1.0 + tl.exp(-2.0 * inner))) * 0.5))
        else:
            activated = gate * tl.sigmoid(1.702 * gate)

        output_offsets = (
            batch_index * output_stride_batch + columns * output_stride_hidden
        )
        tl.store(output_ptr + output_offsets, activated * up, mask=mask)


@triton.jit
def _flat_kernel(
    x_ptr,
    out_ptr,
    half_hidden,
    total_out,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_out
    row = offs // half_hidden
    col = offs - row * half_hidden
    base = row * (2 * half_hidden) + col
    gate = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + base + half_hidden, mask=mask, other=0.0).to(tl.float32)

    if MODE == 0:
        activated = gate
    elif MODE == 1:
        activated = gate * (0.5 * (1.0 + tl.erf(gate * 0.7071067811865475)))
    elif MODE == 2:
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        activated = gate * (0.5 * (1.0 + inner * (0.0 + 2.0 / (1.0 + tl.exp(-2.0 * inner))) * 0.5))
    else:
        activated = gate * tl.sigmoid(1.702 * gate)

    tl.store(out_ptr + offs, activated * up, mask=mask)


def launch_seed(x, out, mode):
    batch_size, doubled = x.shape
    hidden = doubled // 2
    bc = min(1024, triton.next_power_of_2(hidden))
    br = min(1024 // bc, triton.next_power_of_2(batch_size))
    nct = triton.cdiv(hidden, bc)
    nbt = triton.cdiv(batch_size, br)
    ntiles = nbt * nct
    ncta = min(ntiles, 48)
    tpc = triton.cdiv(ntiles, ncta)
    _seed_kernel[(ncta,)](
        x,
        out,
        hidden,
        batch_size,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        ncta,
        tpc,
        nct,
        ntiles,
        br,
        bc,
        mode,
        num_warps=4,
    )


def launch_flat(x, out, mode, block=1024):
    hidden = x.shape[-1] // 2
    total = out.numel()
    grid = (triton.cdiv(total, block),)
    _flat_kernel[grid](x, out, hidden, total, block, mode, num_warps=4)


def time_call(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    # host-pipelined wall time per call (what the evaluator's walltime sees)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    pipelined = (time.perf_counter() - t0) / iters
    # device-serialized per-call time
    per = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        per.append(time.perf_counter() - t0)
    per.sort()
    return {
        "pipelined_ms": pipelined * 1e3,
        "serialized_median_ms": per[len(per) // 2] * 1e3,
        "serialized_min_ms": per[0] * 1e3,
    }


def main():
    results = {"variants": {}, "reference_formula": {}, "torch_probe": {}}

    # 1) variant timing on three representative shapes
    shapes = [(1, 2048), (512, 16384), (4096, 16384)]
    for shape in shapes:
        x = torch.randn(shape, dtype=torch.bfloat16, device=DEV)
        out = torch.empty((shape[0], shape[1] // 2), dtype=torch.bfloat16, device=DEV)
        key = f"{shape[0]}x{shape[1]}"
        results["variants"][key] = {}
        for mode, name in [(0, "no_gelu"), (1, "erf"), (2, "tanh"), (3, "sigmoid")]:
            iters = 20 if shape[0] <= 512 else 8
            results["variants"][key][f"seed_{name}"] = time_call(
                lambda m=mode: launch_seed(x, out, m), iters=iters
            )
            results["variants"][key][f"flat_{name}"] = time_call(
                lambda m=mode: launch_flat(x, out, m), iters=iters
            )
        # reference torch chain for the same shape (context only)
        def torch_chain():
            d = x.shape[-1] // 2
            a, b = x[..., :d], x[..., d:]
            return torch.nn.functional.gelu(a.float(), approximate="none") * b.float()

        results["variants"][key]["torch_chain"] = time_call(
            torch_chain, iters=20 if shape[0] <= 512 else 8
        )
        del x, out

    # 2) which formula does the NPU reference use?
    for scale in (1.0, 3.0):
        x = (torch.randn(1 << 20, dtype=torch.float32, device=DEV) * scale)
        ref_none = torch.nn.functional.gelu(x, approximate="none")
        ref_tanh = torch.nn.functional.gelu(x, approximate="tanh")
        xc = x.float().cpu().double()
        exact = 0.5 * xc * (1.0 + torch.erf(xc / math.sqrt(2.0)))
        tanh_approx = 0.5 * xc * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (xc + 0.044715 * xc.pow(3))
            )
        )
        a = ref_none.float().cpu().double()
        b = ref_tanh.float().cpu().double()
        entry = {
            "npu_none_vs_cpu_exact_maxabs": float((a - exact).abs().max()),
            "npu_none_vs_cpu_tanh_maxabs": float((a - tanh_approx).abs().max()),
            "npu_tanh_vs_cpu_tanh_maxabs": float((b - tanh_approx).abs().max()),
            "npu_tanh_vs_cpu_exact_maxabs": float((b - exact).abs().max()),
            "npu_none_equals_npu_tanh": bool(torch.equal(ref_none, ref_tanh)),
            "out_dtype": str(ref_none.dtype),
            "input_absmax": float(x.abs().max()),
        }
        if torch.equal(ref_none, ref_tanh):
            entry["note"] = "npu none and tanh outputs are bit-identical"
        results["reference_formula"][f"scale_{scale}"] = entry
        del x, ref_none, ref_tanh, xc, exact, tanh_approx

    # 3) cost of the torch-side cast chain parts (context only)
    x = torch.randn((4096, 16384), dtype=torch.bfloat16, device=DEV)
    d = x.shape[-1] // 2
    a = x[..., :d]
    results["torch_probe"]["cast_a_fp32"] = time_call(lambda: a.float())
    results["torch_probe"]["gelu_fp32"] = time_call(
        lambda: torch.nn.functional.gelu(a.float(), approximate="none")
    )
    results["torch_probe"]["gelu_tanh_fp32"] = time_call(
        lambda: torch.nn.functional.gelu(a.float(), approximate="tanh")
    )
    results["torch_probe"]["sigmoid_bf16"] = time_call(lambda: a * torch.sigmoid(a))
    results["torch_probe"]["pure_mul_bf16"] = time_call(lambda: a * a)
    results["torch_probe"]["copy_bf16"] = time_call(lambda: a.clone())
    del x, a
    torch.npu.empty_cache()

    out_dir = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "probe_variants.json"), "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
