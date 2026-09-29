REFERENCE_DEVICE = 'target'

import torch
import torch.nn.functional as F
def run(
    q_extend, k_extend, v_extend, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices,
    max_len_extend,
):
    B = qo_indptr.size(0) - 1
    _, H_Q, D = q_extend.shape
    _, H_KV, _ = k_extend.shape
    group_size = H_Q // H_KV
    scale = 1.0 / D**0.5

    o = torch.empty_like(q_extend, dtype=torch.float32)
    for i in range(B):
        q_start, q_end = int(qo_indptr[i].item()), int(qo_indptr[i + 1].item())
        kv_start, kv_end = int(kv_indptr[i].item()), int(kv_indptr[i + 1].item())

        prefix_indices = kv_indices[kv_start:kv_end]
        k_prefix = k_buffer[prefix_indices]
        v_prefix = v_buffer[prefix_indices]

        k_ext = k_extend[q_start:q_end]
        v_ext = v_extend[q_start:q_end]
        q_ext = q_extend[q_start:q_end]

        k_full = torch.cat([k_prefix, k_ext], dim=0).float()
        v_full = torch.cat([v_prefix, v_ext], dim=0).float()
        if group_size != 1:
            k_full = k_full.repeat_interleave(group_size, dim=1)
            v_full = v_full.repeat_interleave(group_size, dim=1)

        prefix_len = k_prefix.size(0)
        extend_len = k_ext.size(0)
        total_len = prefix_len + extend_len

        pos_keys = torch.arange(total_len, device=q_extend.device)
        t = prefix_len + torch.arange(extend_len, device=q_extend.device)
        causal_mask = pos_keys.unsqueeze(0) <= t.unsqueeze(1)

        attn_scores = (
            torch.einsum("qhd,khd->qhk", q_ext.float(), k_full) * scale
        )
        attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        o[q_start:q_end] = torch.einsum("qhk,khd->qhd", attn_weights, v_full)

    return o
def _build_case(B, n_ctx, H_Q, H_KV, D, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)

    def rnd_int(lo, hi, n):
        return torch.randint(lo, hi, (n,), dtype=torch.int32, device=_DEVICE, generator=g)

    b_seq_len_prefix = rnd_int(1, max(2, n_ctx // 2), B)
    b_seq_len_extend = rnd_int(1, max(2, n_ctx // 2), B)
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_start_loc = torch.zeros((B,), dtype=torch.int32, device=_DEVICE)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device=_DEVICE)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=_DEVICE)
    kv_indptr[1:] = torch.cumsum(b_seq_len_prefix, 0)
    kv_indices = torch.zeros((int(b_seq_len_prefix.sum()),), dtype=torch.int32, device=_DEVICE)
    for i in range(B):
        kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
            b_start_loc[i].item(), (b_start_loc[i] + b_seq_len_prefix[i]).item(), device=_DEVICE
        )

    total_token_num = int(b_seq_len.sum())
    extend_token_num = int(b_seq_len_extend.sum())
    k_buffer = torch.randn(
        total_token_num, H_KV, D, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    v_buffer = torch.randn(
        total_token_num, H_KV, D, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)

    k_extend = torch.zeros((extend_token_num, H_KV, D), dtype=dtype, device=_DEVICE)
    v_extend = torch.zeros((extend_token_num, H_KV, D), dtype=dtype, device=_DEVICE)
    q_extend = torch.zeros((extend_token_num, H_Q, D), dtype=dtype, device=_DEVICE)
    for i in range(B):
        eib = (b_start_loc[i] + b_seq_len_prefix[i]).item()
        eie = (b_start_loc[i] + b_seq_len[i]).item()
        es = b_start_loc_extend[i].item()
        ee = (b_start_loc_extend[i] + b_seq_len_extend[i]).item()
        k_extend[es:ee] = k_buffer[eib:eie]
        v_extend[es:ee] = v_buffer[eib:eie]
        q_extend[es:ee] = torch.randn(
            (ee - es, H_Q, D), generator=g, device=_DEVICE, dtype=torch.float32
        ).to(dtype)

    qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=_DEVICE)
    qo_indptr[1:] = torch.cumsum(b_seq_len_extend, 0)
    max_len_extend = int(b_seq_len_extend.max())

    return dict(
        q_extend=q_extend,
        k_extend=k_extend,
        v_extend=v_extend,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        max_len_extend=max_len_extend,
        check=_check,
    )

def _check(actual, expected):
    assert_close(actual.to(torch.float32), expected, atol=1e-2, rtol=1e-2)




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
    parameters = {'q_extend', 'k_extend', 'v_extend', 'k_buffer', 'v_buffer', 'qo_indptr', 'kv_indptr', 'kv_indices', 'max_len_extend'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

