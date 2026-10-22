# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused exact-GELU gating for Ascend (FlagOS, gelu_and_mul).

Architecture
------------
The reference evaluates ``x1.float() -> gelu(exact) -> * x3.float() -> bf16``
with five separate aclnn kernels (three of them pure casts).  This module
performs the whole expression in one vector kernel, so HBM traffic drops from
five passes over the tensor to one read of the gated pair plus one write.

Device layout
-------------
Rows are addressed as ``row * (2 * d)`` (input) and ``row * d`` (output) so a
row-tile is a purely affine, contiguous range: the compiler emits bulk MTE
moves instead of the per-element gather that a runtime stride multiply forces.
Each program covers ``ROWS_PER_PROG`` whole rows with a dynamic-trip-count
scalar loop, so a small hidden size does not degenerate into thousands of tiny
one-row programs while every loop iteration keeps affine addressing.

Host layout
-----------
The measurement is a full-call walltime, and on this target a Triton call
spends several times longer inside ``JITFunction.run`` (binder, specialization
key, cache lookup, metadata) than the reference spends launching five aclnn
kernels.  The first call for each launch signature compiles normally and keeps
the resulting ``CompiledKernel``; later calls invoke that kernel's launcher
directly with the *current* tensor pointers, the *current* stream and the same
runtime values.  The cache key carries every Triton-specialized scalar
(dtype, hidden size, row count, strides, input pointer alignment, device) so a
cached entry can never be reused for a differently specialized launch.  No
tensor, address or output is cached across calls.  An installed launch hook
(profiling) disables the shortcut, and any exception or missing entry falls
back to ``kernel[grid](...)``.

Numerics
--------
Gate and up operands are loaded in the input dtype and widened to FP32, the
exact GELU ``x * 0.5 * (1 + erf(x / sqrt(2)))`` is evaluated in FP32, the
product is formed in FP32 and rounded once to the output dtype -- the same
rounding order as the reference.

Copyright 2023-2024 SGLang Team
Copyright 2026 FlagOS Contributors
SPDX-License-Identifier: Apache-2.0
"""

import torch
import triton
import triton.language as tl
from triton.runtime import driver

try:  # pragma: no cover - import guard: no compiled handle, no shortcut
    from triton.compiler.compiler import CompiledKernel as _CompiledKernel
except Exception:  # pragma: no cover
    _CompiledKernel = None

__all__ = ["gelu_and_mul"]


@triton.jit
def _gelu_and_mul_rows_kernel(
    x_ptr,
    o_ptr,
    d,
    rows,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    """Row-regular (unit last-dim stride) path: whole rows per program."""
    row0 = tl.program_id(0) * ROWS_PER_PROG
    column_tile = tl.program_id(1)
    columns = column_tile * BLOCK + tl.arange(0, BLOCK)
    n_rows = tl.minimum(ROWS_PER_PROG, rows - row0)

    for step in range(0, n_rows):
        row = row0 + step
        in_base = row * (2 * d)
        if EVEN:
            gate = tl.load(x_ptr + in_base + columns)
            up = tl.load(x_ptr + in_base + d + columns)
        else:
            mask = columns < d
            gate = tl.load(x_ptr + in_base + columns, mask=mask, other=0.0)
            up = tl.load(x_ptr + in_base + d + columns, mask=mask, other=0.0)

        gate_f = gate.to(tl.float32)
        up_f = up.to(tl.float32)
        activated = gate_f * (0.5 * (1.0 + tl.erf(gate_f * 0.7071067811865475)))
        result = activated * up_f

        out_base = row * d
        if EVEN:
            tl.store(o_ptr + out_base + columns, result.to(o_ptr.dtype.element_ty))
        else:
            tl.store(
                o_ptr + out_base + columns,
                result.to(o_ptr.dtype.element_ty),
                mask=columns < d,
            )


@triton.jit
def _gelu_and_mul_generic_kernel(
    hidden_states_ptr,
    output_ptr,
    hidden_size,
    batch_size,
    input_stride_batch,
    input_stride_hidden,
    output_stride_batch,
    output_stride_hidden,
    BLOCK: tl.constexpr,
):
    """Layout fallback: explicit strides, one row-tile per program."""
    row = tl.program_id(0)
    column_tile = tl.program_id(1)
    columns = column_tile * BLOCK + tl.arange(0, BLOCK)
    mask = (row < batch_size) & (columns < hidden_size)

    gate = tl.load(
        hidden_states_ptr + row * input_stride_batch + columns * input_stride_hidden,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        hidden_states_ptr
        + row * input_stride_batch
        + (hidden_size + columns) * input_stride_hidden,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    activated = gate * (0.5 * (1.0 + tl.erf(gate * 0.7071067811865475)))
    tl.store(
        output_ptr + row * output_stride_batch + columns * output_stride_hidden,
        (activated * up).to(output_ptr.dtype.element_ty),
        mask=mask,
    )


_MAX_BLOCK = 8192
_ROWS_TARGET_ELEMS = 32768
_MAX_ROWS_PER_PROG = 8
_NUM_WARPS = 4
# hidden size -> (block, even, num_column_tiles, rows_per_prog); metadata only.
_PLAN = {}
# launch signature -> CompiledKernel; contains no tensor and no address.
_LAUNCHER = {}


def _plan_for(hidden_size):
    plan = _PLAN.get(hidden_size)
    if plan is not None:
        return plan
    block = 1
    while block * 2 <= _MAX_BLOCK and hidden_size % (block * 2) == 0:
        block *= 2
    if block >= 32:
        even = True
    else:
        block = min(_MAX_BLOCK, triton.next_power_of_2(hidden_size))
        even = hidden_size % block == 0
    num_column_tiles = triton.cdiv(hidden_size, block)
    rows_per_prog = max(1, min(_MAX_ROWS_PER_PROG, _ROWS_TARGET_ELEMS // block))
    plan = (block, even, num_column_tiles, rows_per_prog)
    _PLAN[hidden_size] = plan
    return plan


def _shortcut_allowed():
    """False as soon as any Triton launch hook is installed (profiling)."""
    if _CompiledKernel is None:
        return False
    return (
        getattr(_CompiledKernel, "launch_enter_hook", None) is None
        and getattr(_CompiledKernel, "launch_exit_hook", None) is None
    )


def _run_compiled(compiled, grid0, grid1, device_index, args):
    """Invoke an already-compiled kernel with current pointers and stream."""
    compiled.run(
        grid0,
        grid1,
        1,
        driver.active.get_current_stream(device_index),
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        *args,
    )


def gelu_and_mul(hidden_states):
    shape = hidden_states.shape
    doubled_hidden_size = shape[-1]
    hidden_size = doubled_hidden_size // 2
    leading = hidden_states.numel() // doubled_hidden_size
    dtype = hidden_states.dtype
    dev = hidden_states.device
    if len(shape) == 2:
        output = torch.empty((shape[0], hidden_size), dtype=dtype, device=dev)
    else:
        output = torch.empty(shape[:-1] + (hidden_size,), dtype=dtype, device=dev)

    block, even, num_column_tiles, rows_per_prog = _plan_for(hidden_size)

    rank = hidden_states.dim()
    if hidden_states.stride(-1) != 1:
        row_regular = False
    elif rank <= 1:
        row_regular = True
    elif rank == 2:
        row_regular = hidden_states.stride(0) == doubled_hidden_size
    else:
        row_regular = hidden_states.is_contiguous()

    device_index = dev.index
    if device_index is None:
        device_index = torch.npu.current_device()
    allowed = _shortcut_allowed()

    if row_regular:
        grid = (triton.cdiv(leading, rows_per_prog), num_column_tiles)
        key = (
            0,
            dtype,
            hidden_size,
            leading,
            hidden_states.data_ptr() & 15,
            device_index,
        )
        compiled = _LAUNCHER.get(key)
        if compiled is not None and allowed:
            try:
                _run_compiled(
                    compiled,
                    grid[0],
                    grid[1],
                    device_index,
                    (
                        hidden_states,
                        output,
                        hidden_size,
                        leading,
                        block,
                        even,
                        rows_per_prog,
                    ),
                )
                return output
            except Exception:
                _LAUNCHER.pop(key, None)

        cache = _gelu_and_mul_rows_kernel.device_caches[device_index][0]
        known = set(cache)
        _gelu_and_mul_rows_kernel[grid](
            hidden_states,
            output,
            hidden_size,
            leading,
            block,
            even,
            rows_per_prog,
            num_warps=_NUM_WARPS,
        )
        fresh = set(cache) - known
        if len(fresh) == 1:
            _LAUNCHER[key] = cache[fresh.pop()]
        return output

    in_row = hidden_states.stride(-2) if rank > 1 else 0
    in_col = hidden_states.stride(-1)
    out_row = output.stride(-2) if output.dim() > 1 else 0
    out_col = output.stride(-1)
    grid = (leading, num_column_tiles)
    key = (
        1,
        dtype,
        hidden_size,
        leading,
        in_row,
        in_col,
        out_row,
        out_col,
        hidden_states.data_ptr() & 15,
        device_index,
    )
    compiled = _LAUNCHER.get(key)
    if compiled is not None and allowed:
        try:
            _run_compiled(
                compiled,
                grid[0],
                grid[1],
                device_index,
                (
                    hidden_states,
                    output,
                    hidden_size,
                    leading,
                    in_row,
                    in_col,
                    out_row,
                    out_col,
                    block,
                ),
            )
            return output
        except Exception:
            _LAUNCHER.pop(key, None)

    cache = _gelu_and_mul_generic_kernel.device_caches[device_index][0]
    known = set(cache)
    _gelu_and_mul_generic_kernel[grid](
        hidden_states,
        output,
        hidden_size,
        leading,
        in_row,
        in_col,
        out_row,
        out_col,
        block,
        num_warps=_NUM_WARPS,
    )
    fresh = set(cache) - known
    if len(fresh) == 1:
        _LAUNCHER[key] = cache[fresh.pop()]
    return output


run = gelu_and_mul
