"""vLLM 参考实现：cutlass_encode_and_reorder_int4b。将 int4 打包权重编码/重排为 CUTLASS w4a8 布局，确定性重排。CUDA 算子。"""
from vllm._custom_ops import cutlass_encode_and_reorder_int4b as _vllm_cutlass_encode_and_reorder_int4b


def _baseline_cutlass_encode_and_reorder_int4b(b):
    return _vllm_cutlass_encode_and_reorder_int4b(b)


def cutlass_encode_and_reorder_int4b(*args, **kwargs):
    return _baseline_cutlass_encode_and_reorder_int4b(*args, **kwargs)
