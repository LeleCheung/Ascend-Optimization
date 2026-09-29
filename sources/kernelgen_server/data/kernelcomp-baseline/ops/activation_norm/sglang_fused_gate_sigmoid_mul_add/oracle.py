REFERENCE_DEVICE = 'target'

import torch
def run(hidden_states, gate_weight, shared_output, final_hidden_states):
    gate = (hidden_states.float() * gate_weight.float()).sum(dim=-1)
    out = final_hidden_states.float() + torch.sigmoid(gate)[:, None] * shared_output.float()
    return out.to(final_hidden_states.dtype)
