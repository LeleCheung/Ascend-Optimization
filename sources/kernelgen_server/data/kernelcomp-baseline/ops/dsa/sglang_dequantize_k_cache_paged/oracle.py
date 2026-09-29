REFERENCE_DEVICE = 'target'

import torch
def run(quant_k_cache, page_table_1_flattened, group_size=128):
    quant_k_cache = quant_k_cache.view(-1, quant_k_cache.shape[-1])
    dim_nope = 512
    dim_rope = 64
    num_tiles = dim_nope // group_size

    idx = page_table_1_flattened.long()
    gathered = quant_k_cache[idx]

    output = torch.empty(
        gathered.shape[0], dim_nope + dim_rope, dtype=torch.bfloat16, device=quant_k_cache.device
    )
    input_nope_q = gathered[:, :dim_nope]
    input_nope_s = gathered[:, dim_nope : dim_nope + num_tiles * 4].view(torch.float32)
    input_rope = gathered[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)
    output[:, dim_nope:] = input_rope

    for tile_idx in range(num_tiles):
        cur_nope = input_nope_q[:, tile_idx * group_size : (tile_idx + 1) * group_size].to(
            torch.float32
        )
        cur_scale = input_nope_s[:, tile_idx].unsqueeze(-1)
        output[:, tile_idx * group_size : (tile_idx + 1) * group_size] = cur_nope * cur_scale

    return output.unsqueeze(1)
def _build_case(num_blocks, num_tokens, group_size=128, seed=0):
    quant_k_cache = _build_quant_k_cache(num_blocks, group_size=group_size, seed=seed).unsqueeze(
        1
    )
    g = torch.Generator(device=_DEVICE).manual_seed(seed + 1)
    page_table_1_flattened = torch.randint(
        0, num_blocks, (num_tokens,), generator=g, device=_DEVICE, dtype=torch.int32
    )
    return dict(
        quant_k_cache=quant_k_cache,
        page_table_1_flattened=page_table_1_flattened,
        group_size=group_size,
    )

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = 448.0
def _build_quant_k_cache(num_tokens, dim_nope=512, dim_rope=64, group_size=128, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    k_nope = (torch.randn(num_tokens, dim_nope, generator=g, device=_DEVICE) * 2).to(
        torch.bfloat16
    )
    k_rope = torch.randn(num_tokens, dim_rope, generator=g, device=_DEVICE).to(torch.bfloat16)
    num_tiles = dim_nope // group_size

    buf = torch.zeros(
        num_tokens, dim_nope + num_tiles * 4 + dim_rope * 2, dtype=_FP8_DTYPE, device=_DEVICE
    )
    nope_q = buf[:, :dim_nope]
    nope_s = buf[:, dim_nope : dim_nope + num_tiles * 4].view(torch.float32)
    rope = buf[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)
    rope[:] = k_rope

    for t in range(num_tiles):
        tile = k_nope[:, t * group_size : (t + 1) * group_size].float()
        scale = tile.abs().amax(dim=-1) / _FP8_MAX
        scale = scale.clamp(min=1e-12)
        nope_s[:, t] = scale
        nope_q[:, t * group_size : (t + 1) * group_size] = (tile / scale.unsqueeze(-1)).to(
            _FP8_DTYPE
        )

    return buf




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
    parameters = {'quant_k_cache', 'page_table_1_flattened', 'group_size'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

