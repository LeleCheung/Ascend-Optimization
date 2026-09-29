REFERENCE_DEVICE = 'target'

import torch
def run(key, value, key_cache, value_cache, slot_mapping, swa_slot_mapping, k_scale, v_scale):
    kc = key_cache.clone()
    vc = value_cache.clone()
    block_size = key_cache.shape[1]

    slots = slot_mapping.to(torch.int64)
    if swa_slot_mapping is not None:
        # Second-stage remap: the first table indexes into the SWA table.
        slots = swa_slot_mapping.to(torch.int64)[slots]

    live = slots >= 0
    k = key[live].to(torch.float32)
    v = value[live].to(torch.float32)
    if k_scale is not None:
        k = k / k_scale
        v = v / v_scale

    dest = slots[live]
    kc.view(-1, *kc.shape[2:])[dest] = k.to(kc.dtype)
    vc.view(-1, *vc.shape[2:])[dest] = v.to(vc.dtype)
    return kc, vc
def _build_case(tokens, heads, head_size, blocks, block_size=16, seed=0, dtype=torch.bfloat16,
          swa=False, scale=False, drop=0.0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)

    def _t(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=_DEVICE, generator=g
        ).to(dtype)

    total_slots = blocks * block_size
    slots = torch.randperm(total_slots, generator=g, device=_DEVICE)[:tokens].to(
        torch.int64
    )
    swa_table = None
    if swa:
        # The first table indexes a second remap table.
        swa_table = torch.randperm(total_slots, generator=g, device=_DEVICE).to(
            torch.int64
        )
    if drop:
        mask = torch.rand(slots.shape, device=_DEVICE, generator=g) < drop
        slots = torch.where(mask, torch.full_like(slots, -1), slots)

    return dict(
        key=_t(tokens, heads, head_size).contiguous(),
        value=_t(tokens, heads, head_size).contiguous(),
        key_cache=_t(blocks, block_size, heads, head_size).contiguous(),
        value_cache=_t(blocks, block_size, heads, head_size).contiguous(),
        slot_mapping=slots,
        swa_slot_mapping=swa_table,
        k_scale=torch.full((1,), 0.5, dtype=torch.float32, device=_DEVICE) if scale else None,
        v_scale=torch.full((1,), 2.0, dtype=torch.float32, device=_DEVICE) if scale else None,
        check=_check,
    )

def _check(actual, expected):
    for a, e in zip(actual, expected):
        assert_close(a, e)




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
    parameters = {'key', 'value', 'key_cache', 'value_cache', 'slot_mapping', 'swa_slot_mapping', 'k_scale', 'v_scale'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

