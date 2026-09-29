REFERENCE_DEVICE = 'target'

import torch
from kernelgenbench.dataset.baseline.cublas.cublasZgemvBatched import cublasZgemvBatched as _baseline

def _legacy_gen_inputs(ctx, device):
    m = ctx['m']
    n = ctx['n']
    trans = ctx['trans']
    incx = ctx['incx']
    incy = ctx['incy']
    batchCount = ctx['batchCount']
    dtype = torch.complex128
    Lx = n if trans == 'N' else m
    Ly = m if trans == 'N' else n
    A_list = []
    x_list = []
    y_list = []
    for _ in range(batchCount):
        A_desired = torch.randn((m, n), dtype=dtype, device=device)
        A_buf = A_desired.t().contiguous()
        A_list.append(A_buf)
        x_len = (Lx - 1) * incx + 1 if Lx > 0 else 1
        y_len = (Ly - 1) * incy + 1 if Ly > 0 else 1
        x_list.append(torch.randn(x_len, dtype=dtype, device=device))
        y_list.append(torch.randn(y_len, dtype=dtype, device=device))
    Aarray = torch.tensor([a.data_ptr() for a in A_list], dtype=torch.int64, device=device)
    xarray = torch.tensor([x.data_ptr() for x in x_list], dtype=torch.int64, device=device)
    yarray = torch.tensor([y.data_ptr() for y in y_list], dtype=torch.int64, device=device)
    global _keepalive
    _keepalive = tuple((value for key, value in list(locals().items()) if key.endswith('_list')))
    return {'Aarray': Aarray, 'xarray': xarray, 'yarray': yarray, '_y_template': y_list}

def run(trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount, _y_template):
    y_list = [y.clone() for y in _y_template]
    yarray_cloned = torch.tensor([y.data_ptr() for y in y_list], dtype=torch.int64, device=Aarray.device)
    _baseline(trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray_cloned, incy, batchCount)
    return y_list

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
    n = ctx['n']
    trans = ctx['trans']
    K = n if trans == 'N' else m
    max_abs_err = 0.0
    for i in range(batchCount):
        diff = (ref_list[i] - cand_list[i]).abs()
        max_abs_err = max(max_abs_err, diff.max().item())
    rtol = 0.0001 * K ** 0.5
    atol = 1e-05
    passed = max_abs_err <= atol
    return {'passed': passed, 'message': f'abs_err={max_abs_err:.6e}', 'metrics': {'max_absolute_error': max_abs_err}}


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
    ordered = [inputs[name] for name in ['trans', 'm', 'n', 'alpha', 'Aarray', 'lda', 'xarray', 'incx', 'beta', 'yarray', 'incy', 'batchCount', '_y_template']]
    return _legacy_valid(
        ref_outputs, sol_outputs, ordered, _legacy_context(ctx)
    )
