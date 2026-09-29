REFERENCE_DEVICE = 'target'

import torch
import torch.nn.functional as F


def _conv1d_fp32(x, weight, bias, padding, groups):
    """Depthwise conv1d evaluated in true float32.

    Ascend's conv path defaults to HF32 (`torch.npu.conv.allow_hf32` is True),
    which drops roughly 2e-3 of accuracy on float32 inputs -- more than the
    float32 tolerance the delivery's own test uses (rtol=3e-4, atol=1e-3). The
    flag is global device state, so instead of mutating it this computes the
    depthwise convolution as an explicit tap-by-tap accumulation in float32,
    which is exact to ~1e-6 against a float64 CPU reference and needs no conv
    kernel at all.

    Inputs are (batch, dim, length) and weight is (dim, width); `groups` always
    equals dim here, i.e. the convolution is depthwise.
    """
    dim, width = weight.shape
    if groups != dim:
        raise ValueError(f"expected depthwise conv (groups == dim), got {groups}")
    x32 = x.to(torch.float32)
    w32 = weight.to(torch.float32)
    if padding:
        x32 = F.pad(x32, (padding, padding))
    out_len = x32.shape[-1] - (width - 1)
    acc = torch.zeros(
        x32.shape[0], dim, out_len, dtype=torch.float32, device=x32.device
    )
    for tap in range(width):
        acc = acc + x32[..., tap : tap + out_len] * w32[:, tap].unsqueeze(-1)
    if bias is not None:
        acc = acc + bias.to(torch.float32).unsqueeze(-1)
    return acc.to(weight.dtype)


def _causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation="silu",
):
    """Per-sequence reference, transcribed from the PR's own causal_conv1d_ref.

    x: (batch, dim, seqlen); weight: (dim, width); bias: (dim,)
    initial_states / final_states_out: (batch, dim, width - 1)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = _conv1d_fp32(x, weight, bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = _conv1d_fp32(x, weight, bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def run(
    x,
    weight,
    bias,
    conv_states,
    query_start_loc,
    cache_indices=None,
    has_initial_state=None,
    activation="silu",
    pad_slot_id=-1,
):
    """Varlen causal conv1d, FlagGems-vllm PR #724 semantics.

    The delivery ships `causal_conv1d_ref` (used verbatim above) plus the varlen
    driver loop in tests/test_causal_conv1d_fn.py::test_causal_conv1d_varlen.
    This reference reproduces that loop:

      * sequence i spans x[:, query_start_loc[i]:query_start_loc[i+1]];
      * a sequence whose cache_indices[i] == pad_slot_id is skipped entirely and
        contributes nothing to the output and no state write;
      * a sequence seeds from conv_states[cache_indices[i]] when
        has_initial_state[i], otherwise from implicit left zero padding;
      * after each sequence the trailing width-1 columns (of the initial-state
        concatenated input) are written back into conv_states[cache_indices[i]]
        in place, so state writes are visible to later sequences sharing a slot;
      * the per-sequence outputs are concatenated in order along the token axis.

    The test only compares `out[:, :out_ref.shape[-1]]`, i.e. the region covered
    by non-padded sequences, and only checks conv_states at the live
    state_indices. This reference returns exactly that covered region: an output
    of width sum(len(seq) for non-padded seq). Padded (pad_slot_id) sequences
    occupy no output columns, matching the reference concatenation.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2-D (dim, cu_seq_len), got {x.dim()}-D")
    starts = [int(v) for v in query_start_loc.detach().cpu().tolist()]
    batch = len(starts) - 1
    width = weight.shape[1]

    indices = (
        [int(v) for v in cache_indices.detach().cpu().tolist()]
        if cache_indices is not None
        else list(range(batch))
    )
    initial_flags = (
        [bool(v) for v in has_initial_state.detach().cpu().tolist()]
        if has_initial_state is not None
        else [False] * batch
    )

    pieces = []
    for i in range(batch):
        slot = indices[i]
        if slot == pad_slot_id:
            continue
        seq = x[:, starts[i] : starts[i + 1]].unsqueeze(0)
        state_slot = conv_states[slot].unsqueeze(0)
        out_i, _ = _causal_conv1d_ref(
            seq,
            weight,
            bias,
            activation=activation,
            return_final_states=True,
            final_states_out=state_slot,
            initial_states=state_slot.clone() if initial_flags[i] else None,
        )
        pieces.append(out_i)

    if not pieces:
        return x.new_empty(x.shape[0], 0)
    return torch.cat(pieces, dim=2).squeeze(0)



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
    """Build the interdependent varlen tensors the recipe cannot express.

    Transcribed from test_causal_conv1d_varlen's setup so the shapes and the
    padding/pad-slot structure match the delivery's own test:

      * the total seqlen is randomly split into batch + padding sequences;
      * x is sliced out of a wider tensor (transpose + narrow) so the reference
        sees the same non-contiguous layout the kernel is given;
      * conv_states is built transposed, as in the test;
      * cache_indices is a random permutation of slots for the real sequences,
        followed by `padding` entries of pad_slot_id.
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

    padding = 3 if with_padding else 0
    padded_batch_size = batch + padding
    nsplits = padded_batch_size - 1

    eos_pos = torch.randperm(seqlen - 1)[:nsplits].sort().values
    seqlens = torch.diff(
        torch.cat([torch.tensor([-1]), eos_pos, torch.tensor([seqlen - 1])])
    ).tolist()

    total_entries = batch * 10
    cumsum = torch.cumsum(torch.tensor(seqlens), dim=0).to(torch.int32)
    query_start_loc = torch.concat(
        [torch.tensor([0], dtype=torch.int32), cumsum], dim=0
    )

    x = torch.randn(1, seqlen, 4096 + dim + 64, device=device, dtype=dtype)
    x = x.transpose(1, 2)[:, 4096 : 4096 + dim, :]

    weight = torch.randn(dim, width, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None

    conv_states = torch.randn(
        total_entries, width - 1, dim, device=device, dtype=dtype
    ).transpose(1, 2)

    has_initial_state = torch.randint(
        0, 2, (query_start_loc.shape[0] - 1,), dtype=torch.bool, device=device
    )
    state_indices = torch.randperm(total_entries, dtype=torch.int32, device=device)[
        :batch
    ]
    cache_indices = torch.concat(
        [
            state_indices,
            torch.as_tensor(
                [pad_slot_id] * padding, dtype=torch.int32, device=device
            ),
        ],
        dim=-1,
    )

    return {
        "x": x.squeeze(0),
        "weight": weight,
        "bias": bias,
        "conv_states": conv_states,
        "query_start_loc": query_start_loc.to(device),
        "cache_indices": cache_indices,
        "has_initial_state": has_initial_state,
        "activation": "silu" if silu else None,
        "pad_slot_id": pad_slot_id,
    }
