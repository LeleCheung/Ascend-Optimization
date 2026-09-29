REFERENCE_DEVICE = 'target'

import torch
from kernelgenbench.dataset.baseline.cublas.cublasDtrsmBatched import cublasDtrsmBatched as _baseline

def _legacy_gen_inputs(ctx, device):
    m = ctx['m']
    n = ctx['n']
    batchCount = ctx['batchCount']
    uplo = ctx['uplo']
    dtype = torch.float64
    A_list = []
    B_list = []
    for _ in range(batchCount):
        diag_vals = 1.0 + torch.rand(m, dtype=dtype, device=device)
        A = torch.diag(diag_vals)
        if uplo == 1:
            A = A + torch.triu(torch.randn(m, m, dtype=dtype, device=device), diagonal=1) * 0.1
        else:
            A = A + torch.tril(torch.randn(m, m, dtype=dtype, device=device), diagonal=-1) * 0.1
        B_rm = torch.randn(m, n, dtype=dtype, device=device)
        B_cm = B_rm.t().contiguous()
        A_list.append(A)
        B_list.append(B_cm)
    Aarray = torch.tensor([a.data_ptr() for a in A_list], dtype=torch.int64, device=device)
    Barray = torch.tensor([b.data_ptr() for b in B_list], dtype=torch.int64, device=device)
    global _keepalive
    _keepalive = tuple((value for key, value in list(locals().items()) if key.endswith('_list')))
    return {'Aarray': Aarray, 'Barray': Barray, '_B_template': B_list}

def run(side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray, ldb, batchCount, _B_template):
    B_list = [b.clone() for b in _B_template]
    Barray_cloned = torch.tensor([b.data_ptr() for b in B_list], dtype=torch.int64, device=Aarray.device)
    _baseline(side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray_cloned, ldb, batchCount)
    return B_list

def _legacy_valid(ref_outs, cand_outs, inputs, ctx):
    import torch
    ref_list = ref_outs[0] if isinstance(ref_outs[0], list) else ref_outs
    cand_list = cand_outs[0] if isinstance(cand_outs[0], list) else cand_outs
    if not isinstance(ref_list, list) or not isinstance(cand_list, list):
        return {'passed': False, 'message': 'outputs are not lists', 'metrics': {}}
    batchCount = ctx['batchCount']
    if len(ref_list) != batchCount or len(cand_list) != batchCount:
        return {'passed': False, 'message': f'batch count mismatch', 'metrics': {}}
    m = ctx['m']
    max_abs_err = 0.0
    max_rel_err = 0.0
    for i in range(batchCount):
        ref_b = ref_list[i].float()
        cand_b = cand_list[i].float()
        if ref_b.shape != cand_b.shape:
            return {'passed': False, 'message': f'batch {i} shape mismatch', 'metrics': {}}
        diff = (ref_b - cand_b).abs()
        abs_err = diff.max().item()
        rel_err = (diff / (ref_b.abs() + 1e-08)).max().item()
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)
    rtol = 0.0001 * m ** 0.5
    atol = 1e-05
    passed = max_abs_err <= atol or max_rel_err <= rtol
    return {'passed': passed, 'message': f'abs_err={max_abs_err:.6e}, rel_err={max_rel_err:.6e}', 'metrics': {'max_absolute_error': max_abs_err, 'max_relative_error': max_rel_err}}


def _legacy_context(ctx):
    result = {}
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
    ordered = [inputs[name] for name in ['side', 'uplo', 'trans', 'diag', 'm', 'n', 'alpha', 'Aarray', 'lda', 'Barray', 'ldb', 'batchCount', '_B_template']]
    return _legacy_valid(
        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)
    )
