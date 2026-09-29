"""vLLM 参考实现：triton_dequantize_nvfp4。NVFP4 反量化（Triton，swizzle=False）。"""
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _triton_dequantize_nvfp4,
)


def _baseline_triton_dequantize_nvfp4(
    tensor_fp4,
    tensor_sf,
    global_scale,
    dtype,
    block_size=16,
):
    return _triton_dequantize_nvfp4(
        tensor_fp4, tensor_sf, global_scale, dtype, block_size
    )


def triton_dequantize_nvfp4(*args, **kwargs):
    return _baseline_triton_dequantize_nvfp4(*args, **kwargs)
