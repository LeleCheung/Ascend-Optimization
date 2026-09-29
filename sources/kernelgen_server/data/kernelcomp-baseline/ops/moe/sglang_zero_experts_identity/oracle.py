REFERENCE_DEVICE = 'target'

import torch
def run(expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states):
    zero_scales = torch.where(
        expert_indices >= num_experts, expert_scales, torch.zeros_like(expert_scales)
    )
    total = zero_scales.float().sum(dim=-1, keepdim=True)
    return (hidden_states.float() * total).to(hidden_states.dtype)
