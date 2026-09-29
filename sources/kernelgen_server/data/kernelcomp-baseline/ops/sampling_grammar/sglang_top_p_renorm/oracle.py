REFERENCE_DEVICE = 'target'

import torch
def run(probs, top_p):
    probs = probs.float().contiguous()
    batch, vocab = probs.shape
    if isinstance(top_p, torch.Tensor):
        top_ps = top_p.to(device=probs.device, dtype=torch.float32).reshape(-1)
        if top_ps.numel() == 1:
            top_ps = top_ps.expand(batch)
    else:
        top_ps = torch.full((batch,), float(top_p), device=probs.device, dtype=torch.float32)

    # FlashInfer's threshold semantics: sort ascending, drop the prefix whose
    # cumulative mass is below 1 - p, keep every tie at the pivot.
    sorted_probs = torch.sort(probs, dim=-1).values
    cdf = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(cdf, (1.0 - top_ps).unsqueeze(1), right=False).squeeze(1)
    cutoff = cutoff.clamp(max=vocab - 1)
    pivots = sorted_probs.gather(1, cutoff.unsqueeze(1))

    kept = torch.where(probs >= pivots, probs, torch.zeros_like(probs))
    return kept / kept.sum(dim=-1, keepdim=True)
def _build_case(batch, vocab, top_p, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)
    logits = torch.randn(batch, vocab, dtype=torch.float32, device=_DEVICE, generator=g)
    probs = torch.softmax(logits, dim=-1)
    return dict(probs=probs.contiguous(), top_p=top_p, check=_check)

def _check(actual, expected):
    assert_close(actual, expected, dtype=torch.float32, atol=1e-5, rtol=1e-5)




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
    parameters = {'probs', 'top_p'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

