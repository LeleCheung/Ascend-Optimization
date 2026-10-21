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

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    output_ptr,
    routed_scaling_factor,
    NTOK: tl.constexpr,
    NHID: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    base = input_ptr + offs_m[:, None] * (TOPK * NHID) + offs_n[None, :]
    out_ptrs = output_ptr + offs_m[:, None] * NHID + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if EVEN:
        for k in range(TOPK):
            acc += tl.load(base + k * NHID).to(tl.float32)
        tl.store(out_ptrs, acc * routed_scaling_factor)
    else:
        mask = (offs_m[:, None] < NTOK) & (offs_n[None, :] < NHID)
        for k in range(TOPK):
            acc += tl.load(base + k * NHID, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptrs, acc * routed_scaling_factor, mask=mask)


_CONFIGS = {}


def _make_config(num_tokens, hidden_dim):
    block_n = 1024
    while block_n > 1 and hidden_dim % block_n:
        block_n >>= 1
    block_m = 16
    while block_m > 1 and num_tokens % block_m:
        block_m >>= 1
    even = (hidden_dim % block_n == 0) and (num_tokens % block_m == 0)
    if num_tokens > 512:
        warps, stages = 4, 2
    else:
        warps, stages = 4, 1
    grid = (
        (num_tokens + block_m - 1) // block_m,
        (hidden_dim + block_n - 1) // block_n,
    )
    cfg = (block_m, block_n, warps, stages, even, grid)
    _CONFIGS[(num_tokens, hidden_dim)] = cfg
    return cfg


def moe_sum_reduce(input: torch.Tensor, routed_scaling_factor: float) -> torch.Tensor:
    num_tokens, top_k, hidden_dim = input.shape
    if not input.is_contiguous():
        input = input.contiguous()
    output = torch.empty(
        (num_tokens, hidden_dim), dtype=input.dtype, device=input.device
    )
    cfg = _CONFIGS.get((num_tokens, hidden_dim))
    if cfg is None:
        cfg = _make_config(num_tokens, hidden_dim)
    block_m, block_n, warps, stages, even, grid = cfg
    _moe_sum_reduce_kernel[grid](
        input,
        output,
        routed_scaling_factor,
        num_tokens,
        hidden_dim,
        top_k,
        block_m,
        block_n,
        even,
        num_warps=warps,
        num_stages=stages,
    )
    return output


__all__ = ["moe_sum_reduce"]

run = moe_sum_reduce


# Codex独立HG布局修复；原始两轮生成源码按上述字节保留。
_hgkda_original_moe_sum_reduce = moe_sum_reduce


def moe_sum_reduce(input, routed_scaling_factor):
    num_tokens, top_k, hidden_dim = input.shape
    if num_tokens == 0 or hidden_dim == 0:
        return torch.empty((num_tokens, hidden_dim), dtype=input.dtype, device=input.device)
    if input.is_contiguous() and hidden_dim % 512 == 0:
        return _hgkda_original_moe_sum_reduce(input, routed_scaling_factor)
    if not input.is_contiguous():
        input = input.contiguous()
    # 奇数行宽用有mask的列块，避免原配置退化为一列一个program。
    block_n = min(256, triton.next_power_of_2(hidden_dim))
    output = torch.empty((num_tokens, hidden_dim), dtype=input.dtype, device=input.device)
    _moe_sum_reduce_kernel[(num_tokens, triton.cdiv(hidden_dim, block_n))](
        input, output, routed_scaling_factor, num_tokens, hidden_dim, top_k,
        1, block_n, hidden_dim % block_n == 0, num_warps=4, num_stages=1,
    )
    return output


run = moe_sum_reduce
