# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained Triton implementation of ``logaddexp2``.

``log2(2**x + 2**y) = m + log2(1 + 2**(-|x - y|))`` with ``m = max(x, y)``.
The grid is capped at the CANN ``coreDim`` limit (65535) and each program
strides over several tiles, so every workload size launches successfully.
"""

import torch
import triton
import triton.language as tl

# CANN rejects a launch whose coreDim exceeds 65535.
_GRID_CAP = 65535
_BLOCK = 1024


@triton.jit
def _logaddexp2_kernel(
    x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr, WIDE: tl.constexpr
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if WIDE:
        n = n_elements.to(tl.int64)
        start = pid.to(tl.int64) * BLOCK
        step = nprog.to(tl.int64) * BLOCK
    else:
        n = n_elements
        start = pid * BLOCK
        step = nprog * BLOCK
    arange = tl.arange(0, BLOCK)
    for off in tl.range(start, n, step):
        offs = off + arange
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        y_f32 = y.to(tl.float32)
        m = tl.maximum(x_f32, y_f32)
        delta = x_f32 - y_f32
        res = m + tl.log2(1.0 + tl.exp2(-tl.abs(delta)))
        # delta is NaN when x and y are equal infinities (inf - inf); then res = m.
        res = tl.where(delta != delta, m, res)
        # Genuine NaN inputs must still propagate NaN.
        res = tl.where((x_f32 != x_f32) | (y_f32 != y_f32), float("nan"), res)
        tl.store(out_ptr + offs, res.to(out_ptr.dtype.element_ty), mask=mask)


def run(self, other):
    x = self
    y = other
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, device=y.device)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y, device=x.device)

    shape = torch.broadcast_shapes(x.shape, y.shape)
    out = torch.empty(shape, dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    n_elements = out.numel()
    if n_elements == 0:
        return out

    if x.shape != shape:
        x = x.expand(shape)
    if y.shape != shape:
        y = y.expand(shape)
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()

    grid = min(triton.cdiv(n_elements, _BLOCK), _GRID_CAP)
    _logaddexp2_kernel[(grid,)](
        x,
        y,
        out,
        n_elements,
        BLOCK=_BLOCK,
        WIDE=n_elements > 2**30,
    )
    return out
