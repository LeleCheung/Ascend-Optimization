"""vLLM 参考实现：prepare_dcp_local_seq_lens。填充持久化 DCP 本地 seq_lens 缓冲（Triton, in-place）。"""
import torch
try:
    from vllm.v1.worker.gpu.cp_utils import (
        prepare_dcp_local_seq_lens as _vllm_prepare_dcp_local_seq_lens,
    )
except ModuleNotFoundError:
    _vllm_prepare_dcp_local_seq_lens = None


def _baseline_prepare_dcp_local_seq_lens(
    dcp_local_seq_lens, seq_lens, num_reqs, dcp_size, dcp_rank, cp_interleave
):
    if _vllm_prepare_dcp_local_seq_lens is None:
        raise RuntimeError("vLLM not installed or module not found")
    return _vllm_prepare_dcp_local_seq_lens(
        dcp_local_seq_lens, seq_lens, num_reqs, dcp_size, dcp_rank, cp_interleave
    )


def prepare_dcp_local_seq_lens(*args, **kwargs):
    return _baseline_prepare_dcp_local_seq_lens(*args, **kwargs)
