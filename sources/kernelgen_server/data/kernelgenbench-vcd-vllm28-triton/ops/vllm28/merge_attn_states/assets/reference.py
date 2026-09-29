"""vLLM 参考实现：merge_attn_states。LSE 加权合并 prefix/suffix 注意力输出（Triton）。"""
from vllm.v1.attention.ops.triton_merge_attn_states import (
    merge_attn_states as _vllm_merge_attn_states,
)


def _baseline_merge_attn_states(
    output,
    prefix_output,
    prefix_lse,
    suffix_output,
    suffix_lse,
    output_lse=None,
):
    _vllm_merge_attn_states(
        output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
    )


def merge_attn_states(*args, **kwargs):
    return _baseline_merge_attn_states(*args, **kwargs)
