"""vLLM 参考实现：downcast_to_mxfp。沿 axis 将张量量化为 packed e2m1 (MXFP4)。"""
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    downcast_to_mxfp as _vllm_downcast_to_mxfp,
)


def _baseline_downcast_to_mxfp(
    src_tensor, axis, out_quant_tensor=None, out_scale=None,
    BLOCK_OUT_DIM=128, BLOCK_QUANT_DIM=32,
):
    return _vllm_downcast_to_mxfp(
        src_tensor, axis, out_quant_tensor, out_scale,
        BLOCK_OUT_DIM, BLOCK_QUANT_DIM,
    )


def downcast_to_mxfp(*args, **kwargs):
    return _baseline_downcast_to_mxfp(*args, **kwargs)
