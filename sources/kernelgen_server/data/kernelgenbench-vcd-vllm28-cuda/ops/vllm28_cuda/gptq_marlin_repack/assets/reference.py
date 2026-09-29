"""vLLM 参考实现：gptq_marlin_repack。将 GPTQ 打包权重重排为 Marlin 格式，确定性重排。CUDA 算子。"""
from vllm._custom_ops import gptq_marlin_repack as _vllm_gptq_marlin_repack


def _baseline_gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit=False):
    return _vllm_gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit)


def gptq_marlin_repack(*args, **kwargs):
    return _baseline_gptq_marlin_repack(*args, **kwargs)
