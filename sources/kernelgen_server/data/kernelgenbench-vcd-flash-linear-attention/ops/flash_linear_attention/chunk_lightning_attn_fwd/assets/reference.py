"""flash-linear-attention 参考实现：chunk_lightning_attn_fwd。
"""
from fla.ops.lightning_attn import chunk_lightning_attn


def _baseline_chunk_lightning_attn_fwd(q, k, v, g, layer_idx, num_layers):
    o, _ = chunk_lightning_attn(q, k, v, layer_idx, num_layers, output_final_state=False)
    return o


def chunk_lightning_attn_fwd(*args, **kwargs):
    return _baseline_chunk_lightning_attn_fwd(*args, **kwargs)
