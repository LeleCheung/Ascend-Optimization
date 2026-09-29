"""vLLM 参考实现：cutlass_encode_and_reorder_int4b_grouped。分组(MoE)int4 权重编码/重排为 CUTLASS w4a8 布局，确定性重排。CUDA 算子。"""
from vllm._custom_ops import (
    cutlass_encode_and_reorder_int4b_grouped as _vllm_cutlass_encode_and_reorder_int4b_grouped,
)


def _baseline_cutlass_encode_and_reorder_int4b_grouped(b_tensors):
    return _vllm_cutlass_encode_and_reorder_int4b_grouped(b_tensors)


def cutlass_encode_and_reorder_int4b_grouped(*args, **kwargs):
    return _baseline_cutlass_encode_and_reorder_int4b_grouped(*args, **kwargs)
