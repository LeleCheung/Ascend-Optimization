REFERENCE_DEVICE = 'target'

import torch
import torch.nn.functional as F
def run(gateup_output, activation="silu", swiglu_limit=None):
    hidden_size = gateup_output.shape[1]
    half = hidden_size // 2
    gate = gateup_output[:, :half].float()
    up = gateup_output[:, half:].float()

    if swiglu_limit is not None:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)

    if activation == "silu":
        act = F.silu(gate)
    elif activation == "gelu":
        act = F.gelu(gate, approximate="tanh")
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    out = (act.to(gateup_output.dtype) * up.to(gateup_output.dtype)).to(gateup_output.dtype)
    return out
