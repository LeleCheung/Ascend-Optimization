REFERENCE_DEVICE = 'target'

import torch
_DIM_NOPE = 512
_DIM_ROPE = 64
def run(quant_k_cache, page_table_1_flattened, group_size, extra_rows, out):
    cache = quant_k_cache.view(-1, quant_k_cache.shape[-1])
    num_tokens = page_table_1_flattened.shape[0]
    num_tiles = _DIM_NOPE // group_size
    total_rows = num_tokens + extra_rows

    idx = page_table_1_flattened.to(torch.int64)
    # 656 bytes/token: [512 nope fp8 | 4 fp32 group scales | 64 bf16 rope]
    nope_q = cache[idx, :_DIM_NOPE].to(torch.float32)
    scales = cache[idx, _DIM_NOPE : _DIM_NOPE + num_tiles * 4].view(torch.float32)
    rope = cache[idx, _DIM_NOPE + num_tiles * 4 :].view(torch.bfloat16)

    # dequant per group, then requant to per-tensor fp8 with scale 1.0
    nope = nope_q * scales.repeat_interleave(group_size, dim=-1)

    output = torch.zeros(
        (total_rows, 1, _DIM_NOPE + _DIM_ROPE),
        dtype=torch.float8_e4m3fn,
        device=cache.device,
    )
    output[:num_tokens, 0, :_DIM_NOPE] = nope.to(torch.float8_e4m3fn)
    output[:num_tokens, 0, _DIM_NOPE:] = rope.to(torch.float8_e4m3fn)
    # Rows [num_tokens, total_rows) stay zero: the -1-sentinel landing pad.
    return output
_DIM = 656
_GROUP = 128
def _build_case(total_cached, num_tokens, extra_rows, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    # Build the packed 656-byte rows: fp8 nope, fp32 group scales, bf16 rope.
    cache = torch.zeros(
        (total_cached, 1, _DIM), dtype=torch.float8_e4m3fn, device=_DEVICE
    )
    flat = cache.view(total_cached, _DIM)
    flat[:, :512] = (
        torch.randn(total_cached, 512, dtype=torch.float32, device=_DEVICE, generator=g)
    ).to(torch.float8_e4m3fn)
    scales = 0.05 * (
        1.0 + torch.rand(total_cached, 4, dtype=torch.float32, device=_DEVICE, generator=g)
    )
    flat[:, 512:528] = scales.view(torch.float8_e4m3fn)
    rope = torch.randn(
        total_cached, 64, dtype=torch.float32, device=_DEVICE, generator=g
    ).to(torch.bfloat16)
    flat[:, 528:] = rope.view(torch.float8_e4m3fn)

    page_table = torch.randint(
        0, total_cached, (num_tokens,), device=_DEVICE, generator=g, dtype=torch.int32
    )
    # A dirty persistent destination: the kernel must overwrite every byte,
    # including the landing-pad rows.
    out = torch.full(
        (num_tokens + extra_rows, 1, 576), 3.0, dtype=torch.float32, device=_DEVICE
    ).to(torch.float8_e4m3fn)
    return dict(
        quant_k_cache=cache,
        page_table_1_flattened=page_table,
        group_size=_GROUP,
        extra_rows=extra_rows,
        out=out,
        check=_check,
    )

def _check(actual, expected):
    assert_close(actual.view(torch.uint8), expected.view(torch.uint8))




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
    parameters = {'quant_k_cache', 'page_table_1_flattened', 'group_size', 'extra_rows', 'out'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

