REFERENCE_DEVICE = 'target'

import torch
def run(attn_output, A_buf, batch_info):
    S, H, kv_lora_rank = attn_output.shape
    rank = A_buf.shape[1]
    out = attn_output.new_zeros(S, H, rank)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    lora_ranks = batch_info.lora_ranks
    permutation = batch_info.permutation

    for b in range(batch_info.bs):
        start = int(seg_indptr[b].item())
        end = int(seg_indptr[b + 1].item())
        if start == end:
            continue
        w_idx = int(weight_indices[b].item())
        if int(lora_ranks[w_idx].item()) == 0:
            continue
        rows = permutation[start:end].long() if permutation is not None else torch.arange(
            start, end, device=attn_output.device
        )

        a_slice = A_buf[w_idx].float()
        for h in range(H):
            x_slice = attn_output[rows, h, :].float()
            out[rows, h, :] = (x_slice @ a_slice.t()).to(attn_output.dtype)

    return out
def _build_case(
    seg_lens, num_lora, h, kv_lora_rank, rank, permutation="none", dtype=torch.bfloat16, seed=0
):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    s = sum(seg_lens)

    attn_output = torch.randn(
        s, h, kv_lora_rank, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    a_buf = torch.randn(
        num_lora, rank, kv_lora_rank, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    weight_indices = [i % num_lora for i in range(len(seg_lens))]
    batch_info = make_batch_info(
        seg_lens, weight_indices, lora_ranks=[rank] * num_lora, permutation=permutation
    )
    return dict(attn_output=attn_output, A_buf=a_buf, batch_info=batch_info)

def make_batch_info(seg_lens, weight_indices, lora_ranks, scalings=None, permutation="none"):
    """``permutation``: "none" (SORTED_BY_ADAPTER=False; tokens contiguous per
    segment), "identity" (an explicit but order-preserving permutation, for
    kernels that always require one), or "shuffled" (exercises real
    re-ordering)."""
    device=_DEVICE
    bs = len(seg_lens)
    seg_lens_t = torch.tensor(seg_lens, dtype=torch.int32, device=device)
    seg_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
    seg_indptr[1:] = torch.cumsum(seg_lens_t, dim=0)
    weight_indices_t = torch.tensor(weight_indices, dtype=torch.int32, device=device)
    lora_ranks_t = torch.tensor(lora_ranks, dtype=torch.int32, device=device)
    if scalings is None:
        scalings = [1.0] * len(lora_ranks)
    scalings_t = torch.tensor(scalings, dtype=torch.float32, device=device)
    max_len = max(seg_lens) if seg_lens else 0
    total_tokens = int(seg_indptr[-1].item())

    if permutation == "none":
        permutation_t = None
    elif permutation == "identity":
        permutation_t = torch.arange(total_tokens, device=device, dtype=torch.int32)
    elif permutation == "shuffled":
        permutation_t = torch.randperm(total_tokens, device=device).to(torch.int32)
    else:
        raise ValueError(permutation)

    return LoRABatchInfo(
        use_cuda_graph=False,
        bs=bs,
        num_segments=bs,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices_t,
        lora_ranks=lora_ranks_t,
        scalings=scalings_t,
        max_len=max_len,
        seg_lens=seg_lens_t,
        permutation=permutation_t,
    )




_DEVICE = None


def _materialize_arg(value, device):
    if isinstance(value, dict) and "__tensor__" in value:
        dtype = value.get("dtype", "float32")
        return torch.tensor(
            value["__tensor__"],
            dtype=getattr(torch, dtype),
            device=device,
        )
    return value


def gen_inputs(ctx, device):
    global _DEVICE
    _DEVICE = device
    case_args = ctx["inputs"].get("_case_args", {})
    args = [_materialize_arg(value, device) for value in case_args.get("args", [])]
    kwargs = {
        key: _materialize_arg(value, device)
        for key, value in case_args.get("kwargs", {}).items()
    }
    dtype = kwargs.get("dtype")
    if isinstance(dtype, str) and dtype:
        kwargs["dtype"] = getattr(torch, dtype)
    constructor = case_args.get("constructor", "_case")
    if constructor == '_case':
        built = _build_case(*args, **kwargs)
    parameters = {'attn_output', 'A_buf', 'batch_info'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class LoRABatchInfo:
    use_cuda_graph: bool
    bs: int
    num_segments: int
    seg_indptr: torch.Tensor
    weight_indices: torch.Tensor
    lora_ranks: torch.Tensor
    scalings: torch.Tensor
    max_len: Optional[int]
    seg_lens: Optional[torch.Tensor]
    permutation: Optional[torch.Tensor]
    expected_tokens: Optional[int] = None
    has_active_lora: bool = False
    req_seg_indptr: Optional[torch.Tensor] = None
    req_weight_indices: Optional[torch.Tensor] = None
    moe_lora_info: object = None
