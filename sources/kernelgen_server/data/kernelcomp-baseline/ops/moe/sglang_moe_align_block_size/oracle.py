REFERENCE_DEVICE = 'target'

import torch
def run(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    pad_sorted_token_ids,
):
    # ``num_experts`` counts the trailing "filtered expert" slot, which never
    # receives tokens and therefore gets no blocks; only 0..num_experts-2 do.
    num_routed = num_experts - 1
    flat = topk_ids.flatten()
    numel = flat.numel()
    sorted_ids = torch.full_like(sorted_token_ids, numel)
    eids = expert_ids.clone()

    offset = 0
    for e in range(num_routed):
        idx = (flat == e).nonzero().flatten()
        n = idx.numel()
        aligned = ((n + block_size - 1) // block_size) * block_size
        if n:
            sorted_ids[offset : offset + n] = idx.to(sorted_ids.dtype)
        nblocks = aligned // block_size
        if nblocks:
            beg = offset // block_size
            eids[beg : beg + nblocks] = e
        offset += aligned

    npost = torch.full_like(num_tokens_post_pad, offset)
    return sorted_ids, eids, npost
import triton
def _build_case(num_tokens, top_k, num_experts, block_size, seed=0):
    """``num_experts`` is the model's routed-expert count; ids run in
    ``[0, num_experts)``. The kernel is called with ``num_experts + 1`` — the
    extra slot is the EP "filtered expert" bucket and gets no blocks, which is
    exactly how ``moe_runner.triton_utils`` calls it."""
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=_DEVICE, generator=g
    )
    numel = topk_ids.numel()
    arg_experts = num_experts + 1
    if numel < arg_experts + 1:
        max_padded = numel * block_size
    else:
        max_padded = numel + (arg_experts + 1) * (block_size - 1)
    max_blocks = triton.cdiv(max_padded, block_size)

    return dict(
        topk_ids=topk_ids,
        num_experts=arg_experts,
        block_size=block_size,
        sorted_token_ids=torch.full(
            (max_padded,), numel, dtype=torch.int32, device=_DEVICE
        ),
        expert_ids=torch.zeros(max_blocks, dtype=torch.int32, device=_DEVICE),
        num_tokens_post_pad=torch.zeros(1, dtype=torch.int32, device=_DEVICE),
        cumsum_buffer=torch.zeros(arg_experts + 1, dtype=torch.int32, device=_DEVICE),
        pad_sorted_token_ids=True,
        check=_make_check(block_size, numel),
    )

def _make_check(block_size, pad_value):
    def _check(actual, expected):
        a_sorted, a_eids, a_npost = actual
        e_sorted, e_eids, e_npost = expected
        torch.testing.assert_close(a_npost, e_npost)

        n = int(e_npost[0])
        nb = n // block_size
        torch.testing.assert_close(a_eids[:nb], e_eids[:nb])

        # Tokens are placed inside their expert's range by an atomic cursor, so
        # the order within one expert is not deterministic. Compare each
        # expert's range as a multiset instead.
        for beg, end in _expert_runs(e_eids, nb):
            lo, hi = beg * block_size, end * block_size
            torch.testing.assert_close(
                torch.sort(a_sorted[lo:hi]).values,
                torch.sort(e_sorted[lo:hi]).values,
            )
        # Everything past num_tokens_post_pad stays at the padding sentinel.
        assert bool((a_sorted[n:] == pad_value).all()), "tail not left as padding"

    return _check

def _expert_runs(expert_ids, num_blocks):
    """Consecutive block ranges that belong to the same expert."""
    runs = []
    beg = 0
    ids = expert_ids[:num_blocks].tolist()
    for i in range(1, num_blocks + 1):
        if i == num_blocks or ids[i] != ids[beg]:
            runs.append((beg, i))
            beg = i
    return runs




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
    parameters = {'topk_ids', 'num_experts', 'block_size', 'sorted_token_ids', 'expert_ids', 'num_tokens_post_pad', 'cumsum_buffer', 'pad_sorted_token_ids'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

