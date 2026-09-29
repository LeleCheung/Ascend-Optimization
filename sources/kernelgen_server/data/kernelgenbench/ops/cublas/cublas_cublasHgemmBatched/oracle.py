REFERENCE_DEVICE = 'target'

import torch
from kernelgenbench.dataset.baseline.cublas.cublasHgemmBatched import cublasHgemmBatched as _baseline

def _legacy_gen_inputs(ctx, device):
    M = ctx['m']
    N = ctx['n']
    K = ctx['k']
    transa = ctx['transa']
    transb = ctx['transb']
    batchCount = ctx['batchCount']
    dtype = torch.float16
    A_list = []
    B_list = []
    C_list = []
    for _ in range(batchCount):
        A_shape = (K, M) if transa == 'T' else (M, K)
        B_shape = (N, K) if transb == 'T' else (K, N)
        A_list.append(torch.randn(A_shape, dtype=dtype, device=device))
        B_list.append(torch.randn(B_shape, dtype=dtype, device=device))
        C_list.append(torch.randn(M, N, dtype=dtype, device=device))
    Aarray = torch.tensor([a.data_ptr() for a in A_list], dtype=torch.int64, device=device)
    Barray = torch.tensor([b.data_ptr() for b in B_list], dtype=torch.int64, device=device)
    Carray = torch.tensor([c.data_ptr() for c in C_list], dtype=torch.int64, device=device)
    global _keepalive
    _keepalive = tuple((value for key, value in list(locals().items()) if key.endswith('_list')))
    return {'Aarray': Aarray, 'Barray': Barray, 'Carray': Carray, '_C_template': C_list}

def run(transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount, _C_template):
    C_list = [c.clone() for c in _C_template]
    Carray_cloned = torch.tensor([c.data_ptr() for c in C_list], dtype=torch.int64, device=Aarray.device)
    _baseline(transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray_cloned, ldc, batchCount)
    return C_list

def _legacy_valid(ref_outs, cand_outs, inputs, ctx):
    import torch
    ref_list = ref_outs[0] if isinstance(ref_outs[0], list) else ref_outs
    cand_list = cand_outs[0] if isinstance(cand_outs[0], list) else cand_outs
    if not isinstance(ref_list, list) or not isinstance(cand_list, list):
        return {'passed': False, 'message': 'outputs are not lists', 'metrics': {}}
    batchCount = ctx['batchCount']
    if len(ref_list) != batchCount or len(cand_list) != batchCount:
        return {'passed': False, 'message': f'batch count mismatch', 'metrics': {}}
    K = ctx['k']
    max_abs_err = 0.0
    max_rel_err = 0.0
    for i in range(batchCount):
        ref_c = ref_list[i].float()
        cand_c = cand_list[i].float()
        if ref_c.shape != cand_c.shape:
            return {'passed': False, 'message': f'batch {i} shape mismatch', 'metrics': {}}
        diff = (ref_c - cand_c).abs()
        abs_err = diff.max().item()
        rel_err = (diff / (ref_c.abs() + 1e-08)).max().item()
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)
    rtol = 0.001 * K ** 0.5
    atol = 0.001
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
    ordered = [inputs[name] for name in ['transa', 'transb', 'm', 'n', 'k', 'alpha', 'Aarray', 'lda', 'Barray', 'ldb', 'beta', 'Carray', 'ldc', 'batchCount', '_C_template']]
    return _legacy_valid(
        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)
    )
