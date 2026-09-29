REFERENCE_DEVICE = 'target'

import torch

def _legacy_gen_inputs(ctx, device):
    torch.manual_seed(0)
    mixes = torch.randn(
        tuple(ctx["mixes__shape"]),
        dtype=getattr(torch, ctx["mixes__dtype"]),
    )
    hc_scale = torch.tensor([0.5, 0.25, 1.0], dtype=torch.float32)
    hc_base = torch.randn(
        tuple(ctx["hc_base__shape"]), dtype=torch.float32
    ) * 0.1
    return {
        "mixes": mixes.to(device),
        "hc_scale": hc_scale.to(device),
        "hc_base": hc_base.to(device),
    }

def run(mixes, hc_scale, hc_base):
    hc = 4
    sinkhorn_iters = 20
    eps = 1e-6
    batch, sequence, mix_hc = mixes.shape
    expected = (2 + hc) * hc
    if mix_hc != expected:
        raise ValueError(
            f"expected mix dim {expected}, got {mix_hc}"
        )
    x = mixes.reshape(-1, mix_hc).to(dtype=torch.float32)
    base = hc_base.to(dtype=torch.float32)
    s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]
    pre = torch.sigmoid(
        x[:, :hc] * s0 + base[:hc].unsqueeze(0)
    ) + eps
    post = 2 * torch.sigmoid(
        x[:, hc:2 * hc] * s1
        + base[hc:2 * hc].unsqueeze(0)
    )
    raw = x[:, 2 * hc:2 * hc + hc * hc]
    comb = raw.view(-1, hc, hc) * s2 + base[
        2 * hc:2 * hc + hc * hc
    ].view(1, hc, hc)
    row_max = comb.amax(dim=-1, keepdim=True)
    comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return (
        pre.view(batch, sequence, hc),
        post.view(batch, sequence, hc),
        comb.view(batch, sequence, hc, hc),
    )

def _legacy_valid(
    reference_outputs,
    candidate_outputs,
    candidate_args,
    ctx,
):
    del candidate_args, ctx
    metrics = {}
    if not (
        isinstance(reference_outputs, list)
        and isinstance(candidate_outputs, list)
        and len(reference_outputs) == len(candidate_outputs) == 3
    ):
        return {
            "passed": False,
            "message": "expected a three-tensor output tuple",
            "metrics": metrics,
        }

    atol = 1e-2
    rtol = 1e-2
    max_abs_error = 0.0
    for index, (reference, candidate) in enumerate(
        zip(reference_outputs, candidate_outputs)
    ):
        if not (
            isinstance(reference, torch.Tensor)
            and isinstance(candidate, torch.Tensor)
        ):
            return {
                "passed": False,
                "message": f"output[{index}] is not a tensor",
                "metrics": metrics,
            }
        if reference.shape != candidate.shape:
            return {
                "passed": False,
                "message": f"output[{index}] shape differs",
                "metrics": metrics,
            }
        if reference.dtype != candidate.dtype:
            return {
                "passed": False,
                "message": f"output[{index}] dtype differs",
                "metrics": metrics,
            }
        if reference.numel():
            error = (candidate.float() - reference.float()).abs()
            max_abs_error = max(
                max_abs_error,
                float(error.max().item()),
            )
        if not torch.allclose(
            reference,
            candidate,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        ):
            metrics["official_max_abs_error"] = max_abs_error
            return {
                "passed": False,
                "message": f"output[{index}] exceeds official tolerance",
                "metrics": metrics,
            }

    reference_comb = reference_outputs[2].float()
    candidate_comb = candidate_outputs[2].float()

    def residuals(value):
        row = float(
            (value.sum(dim=-1) - 1.0).abs().max().item()
        )
        column = float(
            (value.sum(dim=-2) - 1.0).abs().max().item()
        )
        return row, column

    reference_row, reference_column = residuals(reference_comb)
    candidate_row, candidate_column = residuals(candidate_comb)
    quality_floor = 1e-5
    quality_factor = 4.0
    allowed_row = max(
        quality_floor,
        quality_factor * reference_row,
    )
    allowed_column = max(
        quality_floor,
        quality_factor * reference_column,
    )
    metrics.update({
        "official_max_abs_error": max_abs_error,
        "reference_row_residual": reference_row,
        "candidate_row_residual": candidate_row,
        "allowed_row_residual": allowed_row,
        "reference_column_residual": reference_column,
        "candidate_column_residual": candidate_column,
        "allowed_column_residual": allowed_column,
    })
    converged = (
        candidate_row <= allowed_row
        and candidate_column <= allowed_column
    )
    return {
        "passed": converged,
        "message": (
            ""
            if converged
            else (
                "candidate does not reach reference "
                "Sinkhorn convergence quality"
            )
        ),
        "metrics": metrics,
    }


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


VALID_OWNS_RETURN_CONTRACT = True

def valid(ref_outputs, sol_outputs, inputs, ctx):
    ordered = [inputs[name] for name in ['mixes', 'hc_scale', 'hc_base']]
    return _legacy_valid(
        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)
    )
