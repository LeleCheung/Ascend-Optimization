REFERENCE_DEVICE = 'target'

import math
import torch

def _legacy_gen_inputs(ctx, device):
    dim = int(ctx["inv_freq__shape"][0]) * 2
    max_seq_len = int(ctx["position_angles__shape"][0])
    base = 10000.0
    inv_freq = 1.0 / (
        base ** (
            torch.arange(0, dim, 2, dtype=torch.float32) / dim
        )
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    positions_norm = positions / max_seq_len * (2 * math.pi)
    position_angles = positions_norm.unsqueeze(-1) * inv_freq
    position_angles = position_angles.repeat_interleave(2, dim=-1)
    return {
        "inv_freq": inv_freq.to(device),
        "position_angles": position_angles.to(device),
    }

def run(timestamps, seq_len, inv_freq, position_angles):
    max_seq_len = 256
    batch_positions = torch.arange(
        timestamps.shape[0],
        device=inv_freq.device,
        dtype=inv_freq.dtype,
    )
    batch_positions = batch_positions / max_seq_len
    batch_freqs = batch_positions.unsqueeze(-1) * inv_freq
    batch_freqs = batch_freqs.repeat_interleave(2, dim=-1)
    batch_freqs = batch_freqs[:, None, :]
    time_freqs = position_angles[:seq_len][None, :, :]
    batch_freqs, time_freqs = torch.broadcast_tensors(
        batch_freqs, time_freqs
    )
    freqs = torch.cat((batch_freqs, time_freqs), dim=-1)
    angle = (-timestamps * 2 * math.pi).to(freqs)
    freqs = freqs * angle.unsqueeze(-1)
    return freqs.cos(), freqs.sin()


def _legacy_context(ctx):
    result = {"__workload_seed": ctx["seed"]}
    for name, spec in ctx["inputs"].items():
        kind = spec.get("type") if isinstance(spec, dict) else None
        if kind in {"random", "custom"}:
            result[f"{name}__shape"] = spec["shape"]
            result[f"{name}__dtype"] = spec["dtype"]
            for key, value in spec.items():
                if key not in {"type", "shape", "dtype"}:
                    result[f"{name}__{key}"] = value
        elif kind in {"scalar", "literal"}:
            result[name] = spec["value"]
        else:
            raise ValueError(f"unsupported legacy input recipe: {name}")
    return result


def gen_inputs(ctx, device):
    return _legacy_gen_inputs(_legacy_context(ctx), device)
