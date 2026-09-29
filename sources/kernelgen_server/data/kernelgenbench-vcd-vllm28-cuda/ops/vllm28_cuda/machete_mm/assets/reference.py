"""vLLM 参考实现：machete_mm。Machete 量化 GEMM (uint4/uint8)。CUDA 算子。"""
from vllm._custom_ops import machete_mm as _vllm_machete_mm


def _baseline_machete_mm(
    a,
    b_q,
    b_type,
    out_type=None,
    b_group_scales=None,
    b_group_zeros=None,
    b_group_size=None,
    b_channel_scales=None,
    a_token_scales=None,
    schedule=None,
):
    return _vllm_machete_mm(
        a,
        b_q,
        b_type,
        out_type,
        b_group_scales,
        b_group_zeros,
        b_group_size,
        b_channel_scales,
        a_token_scales,
        schedule,
    )


def machete_mm(*args, **kwargs):
    return _baseline_machete_mm(*args, **kwargs)
