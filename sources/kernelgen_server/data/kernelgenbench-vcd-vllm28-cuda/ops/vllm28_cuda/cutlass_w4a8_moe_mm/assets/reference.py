"""vLLM 参考实现：cutlass_w4a8_moe_mm。CUTLASS W4A8 (int4 权重 / fp8 激活) 分组 MoE GEMM，in-place 写 out_tensors。CUDA 算子。"""
from vllm._custom_ops import cutlass_w4a8_moe_mm as _vllm_cutlass_w4a8_moe_mm


def _baseline_cutlass_w4a8_moe_mm(
    out_tensors,
    a_tensors,
    b_tensors,
    a_scales,
    b_scales,
    b_group_scales,
    b_group_size,
    expert_offsets,
    problem_sizes,
    a_strides,
    b_strides,
    c_strides,
    group_scale_strides,
    maybe_schedule=None,
):
    _vllm_cutlass_w4a8_moe_mm(
        out_tensors,
        a_tensors,
        b_tensors,
        a_scales,
        b_scales,
        b_group_scales,
        b_group_size,
        expert_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        group_scale_strides,
        maybe_schedule,
    )


def cutlass_w4a8_moe_mm(*args, **kwargs):
    return _baseline_cutlass_w4a8_moe_mm(*args, **kwargs)
