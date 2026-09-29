REFERENCE_DEVICE = 'target'

import torch
def run(
    candidates,
    retrive_index,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    num_slots,
):
    B, S = candidates.shape
    V = target_probs.shape[-1]

    predicts = torch.zeros(num_slots, dtype=candidates.dtype, device=candidates.device)
    accept_index = torch.full(
        (B, S), -1, dtype=retrive_index.dtype, device=candidates.device
    )
    accept_token_num = torch.zeros(B, dtype=torch.int32, device=candidates.device)

    for b in range(B):
        root = int(retrive_index[b, 0].item())
        accept_index[b, 0] = root
        last_slot = root
        cur_row = 0
        num_accept = 0
        step = 1
        all_accepted = True

        while step < S:
            draft_token = int(candidates[b, step].item())
            p = target_probs[b, cur_row, draft_token]
            q = draft_probs[b, cur_row, draft_token]
            coin = uniform_samples[b, step - 1]
            if coin * q < p:
                num_accept += 1
                predicts[last_slot] = draft_token
                cur_row = step
                curr_slot = int(retrive_index[b, step].item())
                accept_index[b, num_accept] = curr_slot
                last_slot = curr_slot
                step += 1
            else:
                all_accepted = False
                break
        accept_token_num[b] = num_accept

        coin_final = uniform_samples_for_final_sampling[b]
        p_row = target_probs[b, cur_row]
        if all_accepted:
            val = p_row.clone()
        else:
            q_row = torch.nan_to_num(draft_probs[b, cur_row], nan=0.0)
            val = (p_row - q_row).clamp(min=0.0)

        norm_sum = val.sum()
        target_u = coin_final * norm_sum
        cumsum = torch.cumsum(val, dim=0)
        match = cumsum > target_u
        final_token = int(match.float().argmax().item()) if match.any() else V - 1
        predicts[last_slot] = final_token

    return predicts, accept_index, accept_token_num
def _build_case(B, S, V, seed=0):
    g = torch.Generator(device=_DEVICE).manual_seed(seed)

    candidates = torch.randint(0, V, (B, S), dtype=torch.int32, device=_DEVICE, generator=g)
    retrive_index = torch.arange(B * S, dtype=torch.int64, device=_DEVICE).reshape(B, S)
    uniform_samples = torch.rand(B, S - 1, generator=g, device=_DEVICE)
    uniform_samples_for_final_sampling = torch.rand(B, generator=g, device=_DEVICE)

    target_logits = torch.randn(B, S, V, generator=g, device=_DEVICE)
    target_probs = torch.softmax(target_logits, dim=-1)
    draft_logits = torch.randn(B, S - 1, V, generator=g, device=_DEVICE)
    draft_probs = torch.softmax(draft_logits, dim=-1)

    return dict(
        candidates=candidates,
        retrive_index=retrive_index,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        # Precomputed here (not derived via .item() inside baseline/solution)
        # since retrive_index = arange(B*S) makes it exact, and a host sync
        # inside the timed function breaks CUDA-graph capture in bench.py.
        num_slots=B * S,
        check=_check,
    )

def _check(actual, expected):
    a_pred, a_idx, a_num = actual
    e_pred, e_idx, e_num = expected
    assert torch.equal(a_pred, e_pred), "predicts mismatch"
    assert torch.equal(a_idx, e_idx), "accept_index mismatch"
    assert torch.equal(a_num, e_num), "accept_token_num mismatch"




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
    parameters = {'candidates', 'retrive_index', 'uniform_samples', 'uniform_samples_for_final_sampling', 'target_probs', 'draft_probs', 'num_slots'}
    return {
        name: value
        for name, value in built.items()
        if name in parameters and name != "check"
    }

