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

# Fused matmul + bias + ReLU (Triton / Ascend).
#
# Escape hatches kept from the seed: the two identically-bodied kernels and the
# large-K bf16 route to the second symbol.
import torch
import triton
import triton.language as tl

from flag_gems.utils import broadcastable_to

BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32

# Even-shape fast paths (measured on the target).
EVEN_BLOCK_K = 256

SDNN_PRECISE_MIN_K = 4096


@triton.jit
def matmul_bias_activation_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    bias_ptrs = bias_ptr + offs_cn * stride_bias
    bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0)
    accumulator = accumulator + bias[None, :]

    # Apply ReLU activation
    accumulator = tl.maximum(accumulator, accumulator * 0.0)

    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def fused_mba_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    bias_ptrs = bias_ptr + offs_cn * stride_bias
    bias = tl.load(bias_ptrs, mask=offs_cn < N, other=0.0)
    accumulator = accumulator + bias[None, :]

    accumulator = tl.maximum(accumulator, accumulator * 0.0)

    c = accumulator.to(bias.dtype)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def mba_single_even_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """One 128x128 output tile per program, mask-free (all extents divide)."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    bias = tl.load(bias_ptr + offs_cn * stride_bias)
    accumulator = accumulator + bias[None, :]
    accumulator = tl.maximum(accumulator, accumulator * 0.0)
    tl.store(c_ptrs, accumulator.to(bias.dtype))


@triton.jit
def mba_dual_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bias,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Two N-adjacent tiles per program so the A panel is fetched once."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * (2 * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b0_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    b1_ptrs = b0_ptrs + BLOCK_SIZE_N * stride_bn

    acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b0 = tl.load(b0_ptrs)
        b1 = tl.load(b1_ptrs)
        acc0 += tl.dot(a, b0, allow_tf32=False)
        acc1 += tl.dot(a, b1, allow_tf32=False)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b0_ptrs += BLOCK_SIZE_K * stride_bk
        b1_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * (2 * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    base = c_ptr + stride_cm * offs_cm[:, None]
    bias0 = tl.load(bias_ptr + offs_cn * stride_bias)
    bias1 = tl.load(bias_ptr + (offs_cn + BLOCK_SIZE_N) * stride_bias)
    v0 = acc0 + bias0[None, :]
    v1 = acc1 + bias1[None, :]
    v0 = tl.maximum(v0, v0 * 0.0)
    v1 = tl.maximum(v1, v1 * 0.0)
    tl.store(base + stride_cn * offs_cn[None, :], v0.to(bias0.dtype))
    tl.store(base + stride_cn * (offs_cn + BLOCK_SIZE_N)[None, :], v1.to(bias1.dtype))


def matmul_bias_activation(input, weight, bias):
    """
    Fused matmul + bias + ReLU activation.

    Args:
        input: Input tensor of shape (M, K)
        weight: Weight matrix of shape (K, N)
        bias: Bias vector of shape (N,) or (1, N)

    Returns:
        Output tensor of shape (M, N) with ReLU activation applied
    """
    assert input.shape[1] == weight.shape[0], "Incompatible dimensions"
    assert broadcastable_to(
        bias.shape, (input.shape[0], weight.shape[1])
    ), "Incompatible input shape"
    M, K = input.shape
    _, N = weight.shape

    if input.stride(0) > 1 and input.stride(1) > 1:
        input = input.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()
    if bias.dim() > 1:
        bias = bias.reshape(-1)
    out = torch.empty((M, N), device=input.device, dtype=input.dtype)

    stride_am = input.stride(0)
    stride_ak = input.stride(1)
    stride_bk = weight.stride(0)
    stride_bn = weight.stride(1)
    stride_bias = bias.stride(0)
    stride_cm = out.stride(0)
    stride_cn = out.stride(1)

    # Shape-specialized fast paths: tile extents divide exactly, so the K-loop
    # carries no per-iteration mask and the cube can consume wider K panels.
    if (
        input.dtype in (torch.float16, torch.bfloat16)
        and M % BLOCK_SIZE_M == 0
        and N % (2 * BLOCK_SIZE_N) == 0
        and K % EVEN_BLOCK_K == 0
    ):
        grid = ((M // BLOCK_SIZE_M) * (N // (2 * BLOCK_SIZE_N)),)
        mba_dual_kernel[grid](
            input,
            weight,
            bias,
            out,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_bias,
            stride_cm,
            stride_cn,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            EVEN_BLOCK_K,
        )
        return out

    if (
        M % BLOCK_SIZE_M == 0
        and N % BLOCK_SIZE_N == 0
        and K % EVEN_BLOCK_K == 0
    ):
        grid = ((M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N),)
        mba_single_even_kernel[grid](
            input,
            weight,
            bias,
            out,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_bias,
            stride_cm,
            stride_cn,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            EVEN_BLOCK_K,
        )
        return out

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    if input.dtype == torch.bfloat16 and K >= SDNN_PRECISE_MIN_K:
        fused_mba_kernel[grid](
            input,
            weight,
            bias,
            out,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_bias,
            stride_cm,
            stride_cn,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )
    else:
        matmul_bias_activation_kernel[grid](
            input,
            weight,
            bias,
            out,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_bias,
            stride_cm,
            stride_cn,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )
    return out


run = matmul_bias_activation
