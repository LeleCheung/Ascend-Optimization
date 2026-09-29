"""vLLM 参考实现：concat_mla_q。拼接 MLA/DSA attention 的 query nope 与 rope。CUDA 算子。"""
from vllm._custom_ops import concat_mla_q as _vllm_concat_mla_q


def _baseline_concat_mla_q(ql_nope, q_pe, q_out):
    # in-place：写入 q_out，返回该张量供比较
    _vllm_concat_mla_q(ql_nope, q_pe, q_out)
    return q_out


def concat_mla_q(*args, **kwargs):
    return _baseline_concat_mla_q(*args, **kwargs)
