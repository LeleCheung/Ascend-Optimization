# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl

_BLOCK = 1024

@triton.jit
def _transform_bias_rescale_qkv_kernel(qkv_ptr, bias_ptr, q_ptr, k_ptr, v_ptr, batch: tl.constexpr, seq_len: tl.constexpr, num_heads: tl.constexpr, head_dim: tl.constexpr, scale_factor: tl.constexpr, stride_qkv_b: tl.constexpr, stride_qkv_s: tl.constexpr, stride_qkv_d: tl.constexpr, stride_out_b: tl.constexpr, stride_out_h: tl.constexpr, stride_out_s: tl.constexpr, stride_out_d: tl.constexpr, BIAS_STRIDE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    total_elements = batch * num_heads * seq_len * head_dim
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    head_dim_idx = offsets % head_dim
    temp = offsets // head_dim
    seq_idx = temp % seq_len
    temp = temp // seq_len
    head_idx = temp % num_heads
    batch_idx = temp // num_heads
    hidden_dim = num_heads * head_dim
    base_offset = batch_idx * stride_qkv_b + seq_idx * stride_qkv_s
    head_offset = head_idx * head_dim + head_dim_idx
    q_input_offset = base_offset + (0 * hidden_dim + head_offset) * stride_qkv_d
    q_bias_offset = (0 * hidden_dim + head_offset) * BIAS_STRIDE
    q_val = tl.load(qkv_ptr + q_input_offset, mask=mask, other=0.0)
    q_bias_val = tl.load(bias_ptr + q_bias_offset, mask=mask, other=0.0)
    q_result = (q_val + q_bias_val) * scale_factor
    q_output_offset = batch_idx * stride_out_b + head_idx * stride_out_h + seq_idx * stride_out_s + head_dim_idx * stride_out_d
    tl.store(q_ptr + q_output_offset, q_result, mask=mask)
    k_input_offset = base_offset + (1 * hidden_dim + head_offset) * stride_qkv_d
    k_bias_offset = (1 * hidden_dim + head_offset) * BIAS_STRIDE
    k_val = tl.load(qkv_ptr + k_input_offset, mask=mask, other=0.0)
    k_bias_val = tl.load(bias_ptr + k_bias_offset, mask=mask, other=0.0)
    k_result = k_val + k_bias_val
    k_output_offset = batch_idx * stride_out_b + head_idx * stride_out_h + seq_idx * stride_out_s + head_dim_idx * stride_out_d
    tl.store(k_ptr + k_output_offset, k_result, mask=mask)
    v_input_offset = base_offset + (2 * hidden_dim + head_offset) * stride_qkv_d
    v_bias_offset = (2 * hidden_dim + head_offset) * BIAS_STRIDE
    v_val = tl.load(qkv_ptr + v_input_offset, mask=mask, other=0.0)
    v_bias_val = tl.load(bias_ptr + v_bias_offset, mask=mask, other=0.0)
    v_result = v_val + v_bias_val
    v_output_offset = batch_idx * stride_out_b + head_idx * stride_out_h + seq_idx * stride_out_s + head_dim_idx * stride_out_d
    tl.store(v_ptr + v_output_offset, v_result, mask=mask)

def run(qkv, qkv_bias, num_heads):
    if qkv.ndim != 3 or qkv_bias.ndim != 1:
        raise ValueError('expected 3D qkv and 1D bias')
    batch, seq_len, total_dim = qkv.shape
    if num_heads <= 0 or total_dim % (3 * num_heads) or qkv_bias.numel() != total_dim:
        raise ValueError('inconsistent hidden dimension or head count')
    head_dim = total_dim // (3 * num_heads)
    if head_dim == 0:
        raise ValueError('head dimension must be positive')
    packed = torch.empty((3, batch, num_heads, seq_len, head_dim), dtype=qkv.dtype, device=qkv.device)
    q, key, value = packed.unbind(0)
    count = batch * num_heads * seq_len * head_dim
    if count:
        _transform_bias_rescale_qkv_kernel[(triton.cdiv(count, _BLOCK),)](
            qkv, qkv_bias, q, key, value,
            batch, seq_len, num_heads, head_dim, head_dim ** -0.5,
            *qkv.stride(), *q.stride(), qkv_bias.stride(0), BLOCK_SIZE=_BLOCK)
    return q, key, value
