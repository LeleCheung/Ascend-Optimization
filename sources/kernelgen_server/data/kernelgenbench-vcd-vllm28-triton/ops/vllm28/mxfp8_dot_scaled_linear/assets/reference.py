"""vLLM 参考实现：mxfp8_dot_scaled_linear。MXFP8 (e4m3) scaled linear（Triton）。"""
from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (
    _mxfp8_dot_scaled_linear,
)


def _baseline_mxfp8_dot_scaled_linear(x, w, w_scale):
    return _mxfp8_dot_scaled_linear(x, w, w_scale)


def mxfp8_dot_scaled_linear(*args, **kwargs):
    return _baseline_mxfp8_dot_scaled_linear(*args, **kwargs)
