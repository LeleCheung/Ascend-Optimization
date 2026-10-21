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
def _moe_sum_reduce_contig_kernel(
    input_ptr,
    output_ptr,
    routed_scaling_factor,
    K: tl.constexpr,
    H: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """Mask-free flat [T, K, H] contiguous reduce over the K axis.

    Each program owns BM whole tokens for one BN-wide column slice, so a
    program streams a large contiguous window of the activation instead of a
    single token row.  On the target this is what moves the kernel off the
    per-program overhead regime and up to the memory-system ceiling.
    """
    offsets_m = tl.program_id(0) * BM + tl.arange(0, BM)
    offsets_n = tl.program_id(1) * BN + tl.arange(0, BN)
    input_offsets = offsets_m[:, None] * (K * H) + offsets_n[None, :]
    reduced = tl.zeros((BM, BN), dtype=tl.float32)
    for topk_index in tl.static_range(0, K):
        reduced += tl.load(input_ptr + input_offsets + topk_index * H).to(
            tl.float32
        )
    output_offsets = offsets_m[:, None] * H + offsets_n[None, :]
    tl.store(output_ptr + output_offsets, reduced * routed_scaling_factor)


@triton.jit
def _moe_sum_reduce_strided_kernel(
    input_ptr,
    output_ptr,
    routed_scaling_factor,
    num_tokens: tl.constexpr,
    top_k: tl.constexpr,
    hidden_dim: tl.constexpr,
    input_stride_token,
    input_stride_topk,
    input_stride_hidden,
    output_stride_token,
    output_stride_hidden,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Generic masked, strided fallback (not used by the benchmarked shapes)."""
    offsets_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offsets_m[:, None] < num_tokens) & (
        offsets_n[None, :] < hidden_dim
    )
    input_offsets = (
        offsets_m[:, None] * input_stride_token
        + offsets_n[None, :] * input_stride_hidden
    )
    reduced = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for topk_index in tl.static_range(0, top_k):
        reduced += tl.load(
            input_ptr + input_offsets + topk_index * input_stride_topk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
    reduced *= routed_scaling_factor
    output_offsets = (
        offsets_m[:, None] * output_stride_token
        + offsets_n[None, :] * output_stride_hidden
    )
    tl.store(output_ptr + output_offsets, reduced, mask=mask)


def moe_sum_reduce(
    input: torch.Tensor, routed_scaling_factor: float
) -> torch.Tensor:
    num_tokens, top_k, hidden_dim = input.shape
    output = torch.empty(
        (num_tokens, hidden_dim), dtype=input.dtype, device=input.device
    )
    if num_tokens == 0 or hidden_dim == 0:
        return output

    block_m = 16
    while block_m > 1 and num_tokens % block_m:
        block_m >>= 1
    block_n = 1024
    while block_n > 1 and hidden_dim % block_n:
        block_n >>= 1

    if block_n > 1 and input.is_contiguous():
        _moe_sum_reduce_contig_kernel[
            (num_tokens // block_m, hidden_dim // block_n)
        ](
            input,
            output,
            routed_scaling_factor,
            K=top_k,
            H=hidden_dim,
            BM=block_m,
            BN=block_n,
            num_warps=4,
            num_stages=2,
        )
    else:
        block_n = 512 if hidden_dim >= 512 else 1 << (hidden_dim - 1).bit_length()
        _moe_sum_reduce_strided_kernel[
            (num_tokens, (hidden_dim + block_n - 1) // block_n)
        ](
            input,
            output,
            routed_scaling_factor,
            num_tokens,
            top_k,
            hidden_dim,
            *input.stride(),
            *output.stride(),
            BLOCK_M=1,
            BLOCK_N=block_n,
            num_warps=4,
            num_stages=1,
        )
    return output


__all__ = ["moe_sum_reduce"]

run = moe_sum_reduce
