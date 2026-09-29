"""vLLM 参考实现：cutlass_moe_mm。CUTLASS fp8 分组 MoE grouped GEMM，in-place 写 out_tensors。CUDA 算子。"""
from vllm._custom_ops import cutlass_moe_mm as _vllm_cutlass_moe_mm


def _baseline_cutlass_moe_mm(
    out_tensors,
    a_tensors,
    b_tensors,
    a_scales,
    b_scales,
    expert_offsets,
    problem_sizes,
    a_strides,
    b_strides,
    c_strides,
    per_act_token,
    per_out_ch,
):
    _vllm_cutlass_moe_mm(
        out_tensors,
        a_tensors,
        b_tensors,
        a_scales,
        b_scales,
        expert_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        per_act_token,
        per_out_ch,
    )


def cutlass_moe_mm(*args, **kwargs):
    return _baseline_cutlass_moe_mm(*args, **kwargs)
