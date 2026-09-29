REFERENCE_DEVICE = 'target'

import torch
def _run_impl(x, cache_x, padding, keep_cache_t):
    pw_l, pw_r, ph_t, ph_b, pt_front, pt_back = padding
    cache_t = 0 if cache_x is None else cache_x.shape[2]

    cat = x if cache_x is None else torch.cat([cache_x, x], dim=2)
    # Cache frames are consumed by the temporal front padding; any remainder
    # is zero-filled, matching the aten fallback.
    front = max(pt_front - cache_t, 0)
    out = torch.nn.functional.pad(cat, (pw_l, pw_r, ph_t, ph_b, front, pt_back))
    out = out.contiguous(memory_format=torch.channels_last_3d)

    if keep_cache_t <= 0:
        return out
    # The compact next-chunk cache: the unpadded interior of the last
    # keep_cache_t frames.
    keep_t = min(keep_cache_t, out.shape[2])
    h0, h1 = ph_t, ph_t + x.shape[3]
    w0, w1 = pw_l, pw_l + x.shape[4]
    cache = out[:, :, out.shape[2] - keep_t :, h0:h1, w0:w1]
    return out, cache.contiguous(memory_format=torch.channels_last_3d)


def run(x, cache_x, padding, keep_cache_t):
    result = _run_impl(x, cache_x, padding, keep_cache_t)
    if isinstance(result, tuple):
        return result
    return result, None
def _build_case(b, c, t, h, w, cache_t, padding, keep=0, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)

    def _t(*shape):
        return (
            torch.randn(*shape, dtype=torch.float32, device=_DEVICE, generator=g)
            .to(dtype)
            .contiguous(memory_format=torch.channels_last_3d)
        )

    return dict(
        x=_t(b, c, t, h, w),
        cache_x=_t(b, c, cache_t, h, w) if cache_t else None,
        padding=padding,
        keep_cache_t=keep,
        check=_check,
    )

def _check(actual, expected):
    if isinstance(expected, tuple):
        for a, e in zip(actual, expected):
            assert_close(a, e)
    else:
        assert_close(actual, expected)




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
    parameters = {'x', 'cache_x', 'padding', 'keep_cache_t'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

