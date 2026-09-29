"""vLLM 参考实现：decode_attention_fwd。分页 KV 解码注意力（Triton, SGLang 风格）。"""
from vllm.v1.attention.ops.triton_decode_attention import (
    decode_attention_fwd as _vllm_decode_attention_fwd,
)


def _baseline_decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size=1,
    logit_cap=0.0,
    k_scale=None,
    v_scale=None,
    is_mla=False,
):
    _vllm_decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o,
        lse,
        req_to_token,
        b_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
        k_scale,
        v_scale,
        is_mla,
    )


def decode_attention_fwd(*args, **kwargs):
    return _baseline_decode_attention_fwd(*args, **kwargs)
