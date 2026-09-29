"""vLLM 参考实现：marlin_gemm。Marlin 量化 GEMM (uint4/uint8)。CUDA 算子。"""
from vllm._custom_ops import marlin_gemm as _vllm_marlin_gemm


def _baseline_marlin_gemm(
    a,
    c,
    b_q_weight,
    b_bias,
    b_scales,
    a_scales,
    global_scale,
    b_zeros,
    g_idx,
    perm,
    workspace,
    b_q_type,
    size_m,
    size_n,
    size_k,
    is_k_full=True,
    use_atomic_add=False,
    use_fp32_reduce=False,
    is_zp_float=False,
):
    return _vllm_marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias,
        b_scales,
        a_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        workspace,
        b_q_type,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )


def marlin_gemm(*args, **kwargs):
    return _baseline_marlin_gemm(*args, **kwargs)
