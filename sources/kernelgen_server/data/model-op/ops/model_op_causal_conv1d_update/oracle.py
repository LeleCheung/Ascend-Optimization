REFERENCE_DEVICE = 'target'

import torch
import torch.nn.functional as F


def _conv1d_fp32(x, weight, bias, groups):
    """Depthwise conv1d, valid padding, evaluated in true float32.

    Ascend's conv path defaults to HF32 (torch.npu.conv.allow_hf32 is True),
    which costs ~2e-3 of accuracy on float32 and exceeds the float32 tolerance
    the delivery's own test uses (rtol=3e-4, atol=1e-3). Rather than mutate that
    global device flag, the depthwise convolution is accumulated tap by tap in
    float32, which matches a float64 CPU reference to ~1e-6.
    """
    dim, width = weight.shape
    if groups != dim:
        raise ValueError(f"expected depthwise conv (groups == dim), got {groups}")
    x32 = x.to(torch.float32)
    w32 = weight.to(torch.float32)
    out_len = x32.shape[-1] - (width - 1)
    acc = torch.zeros(
        x32.shape[0], dim, out_len, dtype=torch.float32, device=x32.device
    )
    for tap in range(width):
        acc = acc + x32[..., tap : tap + out_len] * w32[:, tap].unsqueeze(-1)
    if bias is not None:
        acc = acc + bias.to(torch.float32).unsqueeze(-1)
    return acc.to(weight.dtype)


def _causal_conv1d_update_ref(
    x, conv_state, weight, bias=None, activation=None, cache_seqlens=None
):
    """Per-sequence reference, transcribed from the PR's causal_conv1d_update_ref.

    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), state_len >= width - 1; updated in place
    by keeping the last state_len columns of concat(conv_state, x).
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(weight.dtype)
        conv_state.copy_(x_new[:, :, -state_len:])
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = (
            torch.remainder(width_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        )
        x_new = torch.cat([conv_state.gather(2, width_idx), x], dim=-1).to(weight.dtype)
        copy_idx = torch.arange(
            seqlen, dtype=torch.long, device=x.device
        ).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx, state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = _conv1d_fp32(x_new, weight, bias, groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(dtype=dtype_in)


def run(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    pad_slot_id=-1,
):
    """Decode causal conv1d with paged conv-state, PR #725 semantics.

    The delivery ships `causal_conv1d_update_ref` (used verbatim above) plus the
    batch-gather driver in
    tests/test_causal_conv1d_update.py::test_causal_conv1d_update_with_batch_gather.
    That test passes a padded x whose trailing `padding` rows map to
    pad_slot_id, gathers the live slots out of the paged conv_state, calls the
    reference on just those rows, and then checks:

      * out[:batch_size] against the reference output, and
      * conv_state[conv_state_indices] against the reference's in-place state.

    So padded rows produce output values that are never compared. This reference
    computes the live rows through the shipped reference, writes their updated
    state back into the paged conv_state, and leaves the padded rows of the
    output as zeros -- deterministic, and outside the region the source test
    treats as meaningful.
    """
    if conv_state_indices is None:
        state = conv_state
        out = _causal_conv1d_update_ref(
            x, state, weight, bias, activation=activation
        )
        return out

    indices = [int(v) for v in conv_state_indices.detach().cpu().tolist()]
    live = [i for i, slot in enumerate(indices) if slot != pad_slot_id]
    live_slots = [indices[i] for i in live]

    out = torch.zeros_like(x)
    if not live:
        return out

    row_index = torch.tensor(live, dtype=torch.long, device=x.device)
    slot_index = torch.tensor(live_slots, dtype=torch.long, device=x.device)

    gathered = conv_state[slot_index].detach().clone()
    live_out = _causal_conv1d_update_ref(
        x.index_select(0, row_index).contiguous(),
        gathered,
        weight,
        bias,
        activation=activation,
    )
    out.index_copy_(0, row_index, live_out.to(out.dtype))
    conv_state[slot_index] = gathered.to(conv_state.dtype)
    return out



def _scalars(ctx):
    """Unwrap ctx["inputs"] recipe entries to plain values.

    The harness passes each input as its raw recipe mapping (e.g.
    {"type": "scalar", "value": 4}), so scalar knobs must be read out of the
    "value" field; non-recipe entries are already plain.
    """
    out = {}
    for key, raw in ctx["inputs"].items():
        if isinstance(raw, dict) and "value" in raw:
            out[key] = raw["value"]
        else:
            out[key] = raw
    return out


def gen_inputs(ctx, device):
    """Build the paged conv-state layout the recipe cannot express.

    Transcribed from test_causal_conv1d_update_with_batch_gather: x and
    conv_state are both created transposed (contiguous along dim), the live
    slots are a random permutation of the cache lines, and `padding` trailing
    rows carry pad_slot_id.
    """
    spec = _scalars(ctx)
    batch = int(spec["batch"])
    dim = int(spec["dim"])
    seqlen = int(spec["seqlen"])
    width = int(spec["width"])
    has_bias = bool(spec["has_bias"])
    silu = bool(spec["silu_activation"])
    with_padding = bool(spec["with_padding"])
    dtype = getattr(torch, spec["dtype"])
    pad_slot_id = int(spec.get("pad_slot_id", -1))

    torch.manual_seed(int(ctx["seed"]))

    padding = 5 if with_padding else 0
    padded_batch_size = batch + padding
    total_entries = 10 * batch

    x = torch.randn(
        padded_batch_size, seqlen, dim, device=device, dtype=dtype
    ).transpose(1, 2)

    conv_state_indices = torch.randperm(total_entries)[:batch].to(
        dtype=torch.int32, device=device
    )
    padded_state_indices = torch.concat(
        [
            conv_state_indices,
            torch.as_tensor(
                [pad_slot_id] * padding, dtype=torch.int32, device=device
            ),
        ],
        dim=0,
    )

    conv_state = torch.randn(
        total_entries, width - 1, dim, device=device, dtype=dtype
    ).transpose(1, 2)

    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None

    return {
        "x": x,
        "conv_state": conv_state,
        "weight": weight,
        "bias": bias,
        "activation": "silu" if silu else None,
        "conv_state_indices": padded_state_indices,
        "pad_slot_id": pad_slot_id,
    }
