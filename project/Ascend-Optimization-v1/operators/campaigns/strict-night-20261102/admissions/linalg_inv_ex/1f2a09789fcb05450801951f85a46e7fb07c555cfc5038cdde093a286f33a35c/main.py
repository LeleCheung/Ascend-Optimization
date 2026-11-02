"""Self-contained Triton implementation of linalg_inv_ex.

The reference for this operator is `aten::linalg_inv_ex`, which on this target
falls back to the CPU and therefore runs LAPACK.  Two facts drive the design:

* The reference factors the **transpose** of the input.  The workload generator
  makes the last row of the singular cases equal to the first row; factorising
  the matrix directly cancels those rows exactly and yields an exact zero pivot
  (info = n), whereas the reference reports info = 0 together with a large
  finite inverse.  Factorising ``A^T`` reproduces that.
* The reference uses a **blocked** factorisation rather than a plain
  right-looking sweep.  For a panel of width w the trailing block is updated by
  an in-place fused multiply-add over the panel columns, while the panel rows
  are completed against the right block by a unit-lower triangular solve whose
  product is rounded before subtraction.  On a rank-deficient matrix the last
  pivot is a pure rounding residual, so this ordering (not merely the tolerance)
  decides the returned values.

Both effects can be written as a single flat sweep over columns: at column j the
panel-internal rank-1 update is applied first, then the triangular solve for row
j, then the in-place trailing update driven by column j.  Because later panel
columns never touch the right block, and the trailing rows are disjoint from the
panel rows, this interleaving is identical to the blocked algorithm.

The factorisation and the triangular solve run in two kernels so each stays
inside the on-chip buffer budget.
"""
from collections import namedtuple

import torch
import triton
import triton.language as tl

LinalgInvExResult = namedtuple('LinalgInvExResult', ['inverse', 'info'])

@triton.jit
def _FMA(a, b, c):
    """Correctly rounded a*b + c using only single-rounding float32 ops.

    The reference performs its panel and trailing-block updates with a true
    fused multiply-add, but this backend never contracts a multiply with an add
    (``tl.math.fma`` lowers to a separate multiply and add as well), so the
    fused result has to be reconstructed.  ``p`` is the rounded product,
    ``p + e`` is the product exactly (Dekker/Veltkamp two-product), and
    ``s + err`` is ``p + c`` exactly (Knuth two-sum), so the exact value is
    ``s + err + e``; it is then re-rounded with one correction step.
    """
    p = a * b
    ta = a * 4097.0
    ah = ta - (ta - a)
    al = a - ah
    tb = b * 4097.0
    bh = tb - (tb - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    s = p + c
    bb = s - p
    err = (p - (s - bb)) + (c - bb)
    t = err + e
    r = s + t
    v = (s - r) + t
    return r + v


@triton.jit
def _linalg_lu_kernel(A, LU, INFO, PIVOTS, ONE, N: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLK: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < N) & (cols[None, :] < N)
    base = pid * N * N
    # factor A^T: work[r, c] = A[c, r]
    work = tl.load(A + base + cols[None, :] * N + rows[:, None], mask=mask, other=0.0)
    work = work.to(tl.float32)
    pivot_base = pid * N
    singular = 0
    live = rows < N

    for j in tl.range(0, N):
        start = (j // BLK) * BLK
        pend = start + BLK
        col_j = tl.sum(tl.where(cols[None, :] == j, work, 0.0), axis=1)
        abs_col = tl.where((rows >= j) & live, tl.abs(col_j), -1.0)
        pivot_val = tl.max(abs_col, axis=0)
        pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_N), axis=0)
        tl.store(PIVOTS + pivot_base + j, pivot_row)
        pivot = tl.sum(tl.where(rows == pivot_row, col_j, 0.0), axis=0)
        diag_j = tl.sum(tl.where(rows == j, col_j, 0.0), axis=0)
        row_j = tl.sum(tl.where(rows[:, None] == j, work, 0.0), axis=0)
        row_p = tl.sum(tl.where(rows[:, None] == pivot_row, work, 0.0), axis=0)
        work = tl.where((rows[:, None] == j) & (cols[None, :] < N), row_p[None, :], work)
        work = tl.where((rows[:, None] == pivot_row) & (cols[None, :] < N), row_j[None, :], work)
        rec = tl.where(pivot != 0.0, 1.0 / pivot, 1.0)
        work = tl.where((rows[:, None] > j) & (cols[None, :] == j) & live[:, None],
                        work * rec, work)
        singular = tl.where((pivot == 0.0) & (singular == 0), j + 1, singular)
        # column j below the pivot after the interchange and scaling, from vectors
        x_col = tl.where(rows == j, pivot, col_j)
        x_col = tl.where(rows == pivot_row, diag_j, x_col)
        x_col = tl.where(rows > j, x_col * rec, x_col)
        # panel-internal rank-1 update (fused).  Only BLK-1 columns of the
        # current panel are still to the right of column j, so this is done with
        # vector ops and costs no tile-width working buffer.
        for kk in tl.static_range(BLK - 1):
            kc = start + 1 + kk
            act = (j <= start + kk) & (kc < pend) & (kc < N)
            kcc = tl.minimum(kc, N - 1)
            vcol = tl.sum(tl.where(cols[None, :] == kcc, work, 0.0), axis=1)
            uu = tl.sum(tl.where(cols == kcc, row_p, 0.0), axis=0)
            newv = _FMA(x_col, -uu, vcol)
            work = tl.where(act & (rows[:, None] > j) & (cols[None, :] == kcc) & live[:, None],
                            newv[:, None], work)
        # unit-lower triangular solve: finish panel row j against the right block.
        # The reference rounds the product before subtracting, but this backend
        # contracts a multiply and add into one fused op.  Multiplying by an
        # opaque unit scalar keeps the two roundings separate: even if the outer
        # multiply is contracted with the subtraction the result is
        # round(round(l*row_k) - target).
        u_after = row_p
        one = tl.load(ONE)
        for kk in tl.static_range(BLK - 1):
            k = start + kk
            kc = tl.minimum(k, N - 1)
            row_k = tl.sum(tl.where(rows[:, None] == kc, work, 0.0), axis=0)
            ljk = tl.sum(tl.where(cols == kc, row_p, 0.0), axis=0)
            ljk = tl.where(k < j, ljk, 0.0)
            u_after = u_after - (ljk * row_k) * one
        work = tl.where((rows[:, None] == j) & (cols[None, :] >= pend) & (cols[None, :] < N),
                        u_after[None, :], work)
        # trailing block: in-place fused update driven by panel column j
        gem = ((rows[:, None] >= pend) & (cols[None, :] >= pend)
               & (cols[None, :] < N) & live[:, None])
        work = tl.where(gem, _FMA(-x_col[:, None], u_after[None, :], work), work)

    tl.store(LU + base + cols[None, :] * N + rows[:, None], work, mask=mask)
    tl.store(INFO + pid, singular)


@triton.jit
def _linalg_solve_kernel(LU, INV, PIVOTS, N: tl.constexpr, BLOCK_N: tl.constexpr,
                         COL_TILE: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < N) & (cols[None, :] < N)
    base = pid * N * N
    work = tl.load(LU + base + cols[None, :] * N + rows[:, None], mask=mask, other=0.0)
    work = work.to(tl.float32)
    pivot_base = pid * N
    tile_cols = tl.arange(0, COL_TILE)
    for tile_start in tl.range(0, N, COL_TILE):
        rhs = tl.where(rows[:, None] == tile_start + tile_cols[None, :], 1.0, 0.0)
        for j_perm in tl.range(0, N):
            piv_row = tl.load(PIVOTS + pivot_base + j_perm)
            rhs_row_j = tl.sum(tl.where(rows[:, None] == j_perm, rhs, 0.0), axis=0)
            rhs_row_p = tl.sum(tl.where(rows[:, None] == piv_row, rhs, 0.0), axis=0)
            rhs = tl.where(rows[:, None] == j_perm, rhs_row_p[None, :], rhs)
            rhs = tl.where(rows[:, None] == piv_row, rhs_row_j[None, :], rhs)
        for j_fwd in tl.range(0, N):
            l_col = tl.sum(tl.where(cols[None, :] == j_fwd, work, 0.0), axis=1)
            l_col = tl.where(rows > j_fwd, l_col, 0.0)
            rhs_row_j = tl.sum(tl.where(rows[:, None] == j_fwd, rhs, 0.0), axis=0)
            rhs = rhs - l_col[:, None] * rhs_row_j[None, :]
        for j_rev in tl.range(0, N):
            j_back = N - 1 - j_rev
            diag_col = tl.sum(tl.where(cols[None, :] == j_back, work, 0.0), axis=1)
            diag_val = tl.sum(tl.where(rows == j_back, diag_col, 0.0), axis=0)
            rhs_row_j = tl.sum(tl.where(rows[:, None] == j_back, rhs, 0.0), axis=0) / diag_val
            rhs = tl.where(rows[:, None] == j_back, rhs_row_j[None, :], rhs)
            u_col = tl.sum(tl.where(cols[None, :] == j_back, work, 0.0), axis=1)
            u_col = tl.where(rows < j_back, u_col, 0.0)
            rhs = rhs - u_col[:, None] * rhs_row_j[None, :]
        out_offsets = base + (tile_start + tile_cols[None, :]) * N + rows[:, None]
        out_mask = (rows[:, None] < N) & (tile_start + tile_cols[None, :] < N)
        tl.store(INV + out_offsets, rhs, mask=out_mask)


def linalg_inv_ex(A, *, check_errors=False):
    """Compute the inverse of a square matrix.

    Args:
        A: Input tensor of shape (*, n, n).
        check_errors: If True, raises an error if the matrix is singular.

    Returns:
        A namedtuple (inverse, info).
    """
    assert A.ndim >= 2, 'Input must be at least 2D'
    n = A.shape[-1]
    assert A.shape[-2] == n, 'Input must be a square matrix'
    device = A.device
    dtype = A.dtype
    batch_shape = A.shape[:-2]
    batch_size = 1
    for s in batch_shape:
        batch_size *= s
    A_flat = _strict_reshape(_strict_contiguous(A), batch_size, n, n)
    block_n = max(triton.next_power_of_2(n), 16)
    nw = 4 if block_n <= 32 else 8
    col_tile = block_n if block_n <= 32 else 32
    blk = 2 if n > 2 else n
    lu = torch.empty(batch_size, n, n, dtype=torch.float32, device=device)
    inverse = torch.empty(batch_size, n, n, dtype=torch.float32, device=device)
    info = torch.empty(batch_size, dtype=torch.int32, device=device)
    pivots = torch.empty(batch_size, n, dtype=torch.int32, device=device)
    unit = _strict_ones(1, dtype=torch.float32, device=device)
    _linalg_lu_kernel[batch_size,](
        A_flat, lu, info, pivots, unit,
        N=n, BLOCK_N=block_n, BLK=blk, num_warps=nw, num_stages=1,
    )
    _linalg_solve_kernel[batch_size,](
        lu, inverse, pivots, N=n, BLOCK_N=block_n, COL_TILE=col_tile, num_warps=nw, num_stages=1,
    )
    inverse = _strict_cast(_strict_reshape(inverse, *batch_shape, n, n), dtype)
    info = _strict_reshape(info, list(batch_shape))
    if check_errors:
        for v in info.tolist():
            if v != 0:
                raise RuntimeError(
                    'torch.linalg.inv_ex: The diagonal element of the LU decomposition is zero.'
                )
    return LinalgInvExResult(inverse, info)


run = linalg_inv_ex


"""候选内嵌的Triton存储操作：复制、连续化、填充、转换均执行Triton核。"""


@triton.jit
def _strict_copy_kernel(X, Y, N: tl.constexpr, SHAPE: tl.constexpr, XS: tl.constexpr, YS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remain = i
    xp = tl.full((BLOCK,), 0, tl.int64)
    yp = tl.full((BLOCK,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = remain % SHAPE[d]
        remain = remain // SHAPE[d]
        xp += coord * XS[d]
        yp += coord * YS[d]
    value = tl.load(X + xp, i < N, other=0)
    tl.store(Y + yp, value, i < N)


@triton.jit
def _strict_fill_kernel(Y, N: tl.constexpr, VALUE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, VALUE, i < N)


def _strict_empty_like(x, *, dtype=None, **kwargs):
    return torch.empty(tuple(x.shape), device=x.device, dtype=x.dtype if dtype is None else dtype)


def _strict_copy_into(dst, src):
    if tuple(dst.shape) != tuple(src.shape):
        raise ValueError('strict copy requires equal shapes')
    if dst.numel():
        _strict_copy_kernel[(triton.cdiv(dst.numel(), 1024),)](
            src, dst, dst.numel(), tuple(dst.shape), tuple(src.stride()),
            tuple(dst.stride()), 1024,
        )
    return dst


def _strict_clone(x, **kwargs):
    return _strict_copy_into(_strict_empty_like(x), x)


def _strict_contiguous(x, *args, **kwargs):
    return x if x.is_contiguous() else _strict_clone(x)


def _strict_cast(x, dtype):
    if dtype == x.dtype:
        return x
    return _strict_copy_into(_strict_empty_like(x, dtype=dtype), x)


def _strict_reshape(x, *shape):
    return _strict_contiguous(x).view(*shape)


def _strict_flatten(x, start_dim=0, end_dim=-1):
    rank = x.ndim
    first = start_dim % rank
    last = end_dim % rank
    size = 1
    for n in x.shape[first:last + 1]:
        size *= n
    shape = tuple(x.shape[:first]) + (size,) + tuple(x.shape[last + 1:])
    return _strict_contiguous(x).view(shape)


def _strict_full(shape, value, **kwargs):
    if isinstance(shape, int):
        shape = (shape,)
    out = torch.empty(shape, **kwargs)
    if out.numel():
        _strict_fill_kernel[(triton.cdiv(out.numel(), 1024),)](out, out.numel(), value, 1024)
    return out


def _strict_zeros(*shape, **kwargs):
    shape = shape[0] if len(shape) == 1 else shape
    return _strict_full(shape, 0, **kwargs)


def _strict_ones(*shape, **kwargs):
    shape = shape[0] if len(shape) == 1 else shape
    return _strict_full(shape, 1, **kwargs)
