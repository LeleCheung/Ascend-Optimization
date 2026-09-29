"""flash-linear-attention 参考实现：chunk_mesa_net_fwd。
"""
from fla.ops import chunk_mesa_net as _fla_chunk_mesa_net


def _baseline_chunk_mesa_net_fwd(q, k, v, g, beta, lamb):
    o = _fla_chunk_mesa_net(q, k, v, g, beta, lamb, max_CG_iteration=30, output_final_state=False)[0]
    return o


def chunk_mesa_net_fwd(*args, **kwargs):
    return _baseline_chunk_mesa_net_fwd(*args, **kwargs)
