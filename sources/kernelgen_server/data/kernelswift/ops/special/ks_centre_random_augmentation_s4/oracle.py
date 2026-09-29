REFERENCE_DEVICE = 'target'

import math
import torch

def _seed_all(seed):
    torch.manual_seed(seed)
    for name in ("gcu", "cuda", "npu", "mlu"):
        module = getattr(torch, name, None)
        if module is None:
            continue
        try:
            if module.is_available():
                module.manual_seed_all(seed)
        except Exception:
            pass

def _legacy_gen_inputs(ctx, device):
    seed = int(ctx["__workload_seed"])
    _seed_all(seed)
    x_input_coords = torch.randn(
        tuple(ctx["x_input_coords__shape"]),
        dtype=getattr(torch, ctx["x_input_coords__dtype"]),
        device=device,
    )
    mask = torch.ones(
        tuple(ctx["mask__shape"]),
        dtype=getattr(torch, ctx["mask__dtype"]),
        device=device,
    )
    # Official correctness resets the case seed immediately before forward.
    _seed_all(seed)
    return {"x_input_coords": x_input_coords, "mask": mask}

def _random_rotation_matrices(n, device, dtype):
    u1 = torch.rand(n, device=device, dtype=dtype)
    u2 = torch.rand(n, device=device, dtype=dtype)
    u3 = torch.rand(n, device=device, dtype=dtype)
    q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
    q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(n, 3, 3)

def _rot_vec_mul(rotation, vector):
    x, y, z = torch.unbind(vector, dim=-1)
    return torch.stack(
        [
            rotation[..., 0, 0] * x
            + rotation[..., 0, 1] * y
            + rotation[..., 0, 2] * z,
            rotation[..., 1, 0] * x
            + rotation[..., 1, 1] * y
            + rotation[..., 1, 2] * z,
            rotation[..., 2, 0] * x
            + rotation[..., 2, 1] * y
            + rotation[..., 2, 2] * z,
        ],
        dim=-1,
    )

def run(x_input_coords, mask):
    n_sample = 4
    s_trans = 1.0
    dtype = x_input_coords.dtype
    m = mask.to(dtype=dtype).unsqueeze(-1)
    center = (x_input_coords * m).sum(
        dim=-2, keepdim=True
    ) / (m.sum(dim=-2, keepdim=True) + 1e-12)
    x = x_input_coords - center
    x = x.unsqueeze(0).expand(n_sample, -1, -1).contiguous()
    rotation = _random_rotation_matrices(
        n_sample, x.device, dtype
    )
    translation = s_trans * torch.randn(
        n_sample, 3, device=x.device, dtype=dtype
    )
    x = _rot_vec_mul(
        rotation[:, None, :, :].expand(
            -1, x.shape[1], -1, -1
        ),
        x,
    ) + translation[:, None, :]
    return x * mask.to(dtype=dtype)[None, :, None]


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
