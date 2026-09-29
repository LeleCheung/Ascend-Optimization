REFERENCE_DEVICE = 'target'
import math
import torch.nn.functional as F

import torch
def run(B, x, dt, dA_cumsum, cu_seqlens, chunk_states):
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.numel() - 1
    ratio = nheads // ngroups

    states = torch.zeros(batch, nheads, headdim, dstate, dtype=chunk_states.dtype, device=x.device)

    for bidx in range(batch):
        start = int(cu_seqlens[bidx].item())
        end = int(cu_seqlens[bidx + 1].item())
        pid_c = (end - 1) // chunk_size
        chunk_start_tok = pid_c * chunk_size
        start_rel = start - chunk_start_tok
        end_rel = end - chunk_start_tok

        for h in range(nheads):
            g = h // ratio
            dA_cs_last = dA_cumsum[h, pid_c, end_rel - 1].float()
            x_seg = x[start:end, h, :].float()
            b_seg = B[start:end, g, :].float()
            dt_seg = dt[h, pid_c, start_rel:end_rel].float()
            dA_seg = dA_cumsum[h, pid_c, start_rel:end_rel].float()
            scale = torch.exp(dA_cs_last - dA_seg) * dt_seg
            b_scaled = b_seg * scale.unsqueeze(-1)
            states[bidx, h] = (x_seg.t() @ b_scaled).to(chunk_states.dtype)

    return states
def _build_case(seq_lens, chunk_size, nheads, ngroups, headdim, dstate, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)

    # Tight-packed cu_seqlens, with `seq_lens` chosen (by the caller) so that
    # every running cumulative sum which ends a chunk lands exactly on a
    # chunk_size multiple -- i.e. no sequence straddles a chunk boundary.
    # That keeps every sequence within a single physical chunk, matching
    # this problem's documented scope (no initial-state / cross-chunk path).
    cu_list = [0]
    for length in seq_lens:
        cu_list.append(cu_list[-1] + length)
    total_seqlen = cu_list[-1]
    assert total_seqlen % chunk_size == 0
    nchunks = total_seqlen // chunk_size
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int32, device=_DEVICE)

    x = torch.randn(
        total_seqlen, nheads, headdim, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    b = torch.randn(
        total_seqlen, ngroups, dstate, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)
    raw_dt = torch.rand(1, total_seqlen, nheads, generator=g, device=_DEVICE, dtype=torch.float32)
    a = -torch.rand(nheads, generator=g, device=_DEVICE, dtype=torch.float32) - 0.1
    dt_out, dA_cumsum = chunk_cumsum_reference(raw_dt, a, chunk_size)
    dt_out = dt_out.squeeze(0)
    dA_cumsum = dA_cumsum.squeeze(0)

    chunk_states = torch.randn(
        nchunks, nheads, headdim, dstate, generator=g, device=_DEVICE, dtype=torch.float32
    ).to(dtype)

    return dict(
        B=b,
        x=x,
        dt=dt_out,
        dA_cumsum=dA_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_states=chunk_states,
        check=_check,
    )

def _check(actual, expected):
    assert_close(actual, expected, atol=3e-2, rtol=3e-2)

def chunk_cumsum_reference(dt, A, chunk_size, dt_bias=None, dt_softplus=False):
    batch, seqlen, nheads = dt.shape
    nchunks = math.ceil(seqlen / chunk_size)

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        dt_f = torch.where(dt_f <= 20.0, F.softplus(dt_f), dt_f)
    dt_f = dt_f.clamp(min=0.0)

    dt_out = dt_f.reshape(batch, nchunks, chunk_size, nheads).permute(0, 3, 1, 2).contiguous()
    dA = dt_out * A.float().view(1, nheads, 1, 1)
    dA_cumsum = dA.cumsum(dim=-1)
    return dt_out, dA_cumsum




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
    parameters = {'B', 'x', 'dt', 'dA_cumsum', 'cu_seqlens', 'chunk_states'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

