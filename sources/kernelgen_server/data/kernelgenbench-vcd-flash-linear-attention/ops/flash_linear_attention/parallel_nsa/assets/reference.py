"""flash-linear-attention 参考实现：parallel_nsa。
"""
import torch
from fla.ops import parallel_nsa as _fla_parallel_nsa


def _baseline_parallel_nsa(q, k, v, block_indices, block_size, block_counts):
    # parallel_nsa 需要 g_slc 门控，这里设为1（不门控）
    return _fla_parallel_nsa(q, k, v, block_indices=block_indices, block_counts=block_counts,
                       block_size=block_size, g_slc=torch.ones(q.shape[0], q.shape[1], q.shape[2], device=q.device))


def parallel_nsa(*args, **kwargs):
    return _baseline_parallel_nsa(*args, **kwargs)
