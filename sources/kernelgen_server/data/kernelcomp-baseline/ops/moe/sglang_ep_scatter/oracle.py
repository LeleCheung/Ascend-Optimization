REFERENCE_DEVICE = 'target'

import torch
def run(
    recv_x,
    recv_x_scale,
    recv_topk,
    num_recv_tokens_per_expert,
    num_valid_tokens_per_expert,
    expert_start_loc,
    output_tensor,
    output_tensor_scale,
    m_indices,
    output_index,
):
    num_experts = num_recv_tokens_per_expert.shape[0]
    t, topk = recv_topk.shape

    # Pass 1: padded-per-expert start offsets, and the m_indices map (expert id
    # for the valid rows of each expert's slab, -1 for its padding tail).
    counts = num_recv_tokens_per_expert.to(torch.int64)
    starts = torch.cumsum(counts, 0) - counts
    m_out = m_indices.clone()
    for e in range(num_experts):
        s = int(starts[e])
        pad = int(counts[e])
        valid = int(num_valid_tokens_per_expert[e])
        idx = torch.arange(pad, device=m_indices.device)
        m_out[s : s + pad] = torch.where(
            idx < valid,
            torch.full_like(idx, e, dtype=m_out.dtype),
            torch.full_like(idx, -1, dtype=m_out.dtype),
        )

    # Pass 2: every (token, slot) with a non-negative expert claims the next
    # free row of that expert's slab. The kernel claims via atomic_add, so the
    # order inside an expert is not deterministic; this reference fills in
    # token-major order and the check compares each expert's range as a
    # multiset (same convention as moe/moe_align_block_size).
    cursor = starts.clone()
    out = output_tensor.clone()
    out_s = output_tensor_scale.clone()
    out_idx = output_index.clone()
    ids = recv_topk.to(torch.int64)
    for ti in range(t):
        for j in range(topk):
            e = int(ids[ti, j])
            if e < 0:
                continue
            dest = int(cursor[e])
            cursor[e] += 1
            out[dest] = recv_x[ti]
            if recv_x_scale is not None:
                out_s[dest] = recv_x_scale[ti]
            out_idx[ti, j] = dest

    return cursor.to(expert_start_loc.dtype), out, out_s, m_out, out_idx
_BLOCK_E = 128
_QUANT_BLOCK = 128
def _build_case(t, topk, hidden, num_experts, dtype=torch.float8_e4m3fn, seed=0, drop=0.0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)

    ids = torch.randint(
        0, num_experts, (t, topk), device=_DEVICE, generator=g, dtype=torch.int32
    )
    if drop:
        mask = torch.rand(ids.shape, device=_DEVICE, generator=g) < drop
        ids = torch.where(mask, torch.full_like(ids, -1), ids)

    valid = torch.zeros(num_experts, dtype=torch.int32, device=_DEVICE)
    for e in range(num_experts):
        valid[e] = int((ids == e).sum())
    # Each expert's slab is padded up to a multiple of BLOCK_E.
    padded = ((valid + _BLOCK_E - 1) // _BLOCK_E * _BLOCK_E).to(torch.int32)
    total = int(padded.sum())

    recv_x = (
        torch.randn(t, hidden, dtype=torch.float32, device=_DEVICE, generator=g)
    ).to(dtype)
    recv_x_scale = torch.rand(
        t, hidden // _QUANT_BLOCK, dtype=torch.float32, device=_DEVICE, generator=g
    )

    return dict(
        recv_x=recv_x.contiguous(),
        recv_x_scale=recv_x_scale.contiguous(),
        recv_topk=ids.contiguous(),
        num_recv_tokens_per_expert=padded,
        num_valid_tokens_per_expert=valid,
        expert_start_loc=torch.zeros(num_experts, dtype=torch.int32, device=_DEVICE),
        output_tensor=torch.zeros((total, hidden), dtype=dtype, device=_DEVICE),
        output_tensor_scale=torch.zeros(
            (total, hidden // _QUANT_BLOCK), dtype=torch.float32, device=_DEVICE
        ),
        m_indices=torch.zeros(total, dtype=torch.int32, device=_DEVICE),
        output_index=torch.full(
            (t, topk), -1, dtype=torch.int32, device=_DEVICE
        ),
        check=_make_check(ids, recv_x, recv_x_scale, padded, valid),
    )

def _make_check(ids, recv_x, recv_x_scale, padded, valid):
    """The kernel hands out destination rows with `atomic_add`, so the order
    inside one expert's slab is not part of the contract. Check the structure
    instead of the exact permutation."""

    def _check(actual, expected):
        a_loc, a_out, a_scale, a_m, a_idx = actual
        e_loc, _, _, e_m, _ = expected

        # Deterministic parts: the post-scatter cursor and the expert map.
        assert_close(a_loc, e_loc)
        assert_close(a_m, e_m)

        starts = torch.cumsum(padded.to(torch.int64), 0) - padded.to(torch.int64)
        flat_ids = ids.reshape(-1).to(torch.int64)
        flat_slots = a_idx.reshape(-1).to(torch.int64)
        src = torch.arange(ids.shape[0], device=ids.device).repeat_interleave(
            ids.shape[1]
        )

        live = flat_ids >= 0
        # 1. Every dispatched slot lands inside its expert's valid range.
        lo = starts[flat_ids[live]]
        hi = lo + valid.to(torch.int64)[flat_ids[live]]
        assert bool(
            ((flat_slots[live] >= lo) & (flat_slots[live] < hi)).all()
        ), "a slot landed outside its expert's valid range"
        # 2. No two dispatched slots share a destination row.
        assert (
            flat_slots[live].unique().numel() == int(live.sum())
        ), "destination rows are not unique"
        # 3. The row actually written matches that token's payload and scale.
        assert_close(a_out[flat_slots[live]].to(torch.float32), recv_x[src[live]].to(torch.float32))
        assert_close(a_scale[flat_slots[live]], recv_x_scale[src[live]], dtype=torch.float32)
        # 4. Undispatched slots keep the sentinel.
        assert bool((flat_slots[~live] == -1).all()), "an undispatched slot was written"

    return _check




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
    parameters = {'recv_x', 'recv_x_scale', 'recv_topk', 'num_recv_tokens_per_expert', 'num_valid_tokens_per_expert', 'expert_start_loc', 'output_tensor', 'output_tensor_scale', 'm_indices', 'output_index'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

