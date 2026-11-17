import logging
from collections import namedtuple
import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
logger = logging.getLogger(__name__)
LinalgInvExResult = namedtuple('LinalgInvExResult', ['inverse', 'info'])

@libentry()
@triton.jit
def _linalg_inv_kernel(A, INV, INFO, PIVOTS, N: tl.constexpr, BLOCK_N: tl.constexpr, COL_TILE: tl.constexpr):
    """Fused LU factorization + triangular solve for matrix inversion.

    The factorization runs on the TRANSPOSE of A.  The reference path on this
    target (aten::linalg_inv_ex, NPU -> CPU fallback) hands LAPACK the
    column-major buffer built as `A.mT().contiguous()`, so LAPACK's (i, j)
    element is A(j, i).  That matters numerically: when the input has two
    identical rows, elimination on A cancels them exactly (U[n-1,n-1] == 0.0,
    info = n), whereas elimination on A^T leaves a rounding residual, so the
    last pivot is a tiny non-zero value, info stays 0 and the inverse has large
    finite entries -- exactly what the reference shows.

    info keeps LAPACK semantics: the 1-based index of the first exactly-zero
    pivot, or 0.
    """
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < N) & (cols[None, :] < N)
    base = pid * N * N
    work = tl.load(A + base + cols[None, :] * N + rows[:, None], mask=mask, other=0.0)
    work = work.to(tl.float32)
    pivot_base = pid * N
    singular = 0
    for j_ind in tl.range(0, N):
        col_j = tl.sum(tl.where(cols[None, :] == j_ind, work, 0.0), axis=1)
        abs_col = tl.abs(col_j)
        abs_col = tl.where(rows >= j_ind, abs_col, -1.0)
        abs_col = tl.where(rows < N, abs_col, -1.0)
        pivot_val = tl.max(abs_col, axis=0)
        pivot_row = tl.min(tl.where(abs_col == pivot_val, rows, BLOCK_N), axis=0)
        tl.store(PIVOTS + pivot_base + j_ind, pivot_row)
        pivot = tl.sum(tl.where(rows == pivot_row, col_j, 0.0), axis=0)
        row_j = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)
        row_p = tl.sum(tl.where(rows[:, None] == pivot_row, work, 0.0), axis=0)
        work = tl.where((rows[:, None] == j_ind) & (cols[None, :] < N), row_p[None, :], work)
        work = tl.where((rows[:, None] == pivot_row) & (cols[None, :] < N), row_j[None, :], work)
        rec = tl.where(pivot != 0.0, 1.0 / pivot, 1.0)
        scale_mask = (rows[:, None] > j_ind) & (cols[None, :] == j_ind)
        work = tl.where(scale_mask, work * rec, work)
        singular = tl.where((pivot == 0.0) & (singular == 0), j_ind + 1, singular)
        x_col = tl.sum(tl.where(cols[None, :] == j_ind, work, 0.0), axis=1)
        u_row = tl.sum(tl.where(rows[:, None] == j_ind, work, 0.0), axis=0)
        update_mask = (rows[:, None] > j_ind) & (cols[None, :] > j_ind)
        work = tl.where(update_mask, work - x_col[:, None] * u_row[None, :], work)
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
    tl.store(INFO + pid, singular)

def linalg_inv_ex(A, *, check_errors=False):
    """Compute the inverse of a square matrix.

    Args:
        A: Input tensor of shape (*, n, n).
        check_errors: If True, raises an error if the matrix is singular.

    Returns:
        A namedtuple (inverse, info) where:
        - inverse: The inverse matrix of shape (*, n, n).
        - info: An int32 tensor. 0 indicates success; a positive integer i
          indicates that the i-th diagonal element of U in the LU
          decomposition is exactly zero.
    """
    logger.debug('GEMS LINALG_INV_EX')
    assert A.ndim >= 2, 'Input must be at least 2D'
    n = A.shape[-1]
    assert A.shape[-2] == n, 'Input must be a square matrix'
    assert A.dtype in (torch.float32, torch.float64), f'linalg_inv_ex: unsupported dtype {A.dtype}, requires float32 or float64'
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
    inverse = torch.empty(batch_size, n, n, dtype=torch.float32, device=device)
    info = torch.empty(batch_size, dtype=torch.int32, device=device)
    pivots = torch.empty(batch_size, n, dtype=torch.int32, device=device)
    with torch_device_fn.device(device):
        _linalg_inv_kernel[batch_size,](A_flat, inverse, info, pivots, N=n, BLOCK_N=block_n, COL_TILE=col_tile, num_warps=nw)
    inverse = _strict_cast(_strict_reshape(inverse, *batch_shape, n, n), dtype)
    info = _strict_reshape(info, list(batch_shape))
    if check_errors and any((v != 0 for v in info.tolist())):
        raise torch.linalg.LinAlgError('torch.linalg.inv_ex: The diagonal element of the LU decomposition is zero.')
    return LinalgInvExResult(inverse, info)
run = linalg_inv_ex

"""候选内嵌的Triton存储操作：复制、连续化、填充、转换均执行Triton核。"""
import torch
import triton
import triton.language as tl

@triton.jit
def _strict_copy_kernel(X,Y,N:tl.constexpr,SHAPE:tl.constexpr,XS:tl.constexpr,YS:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    remain=i;xp=tl.full((BLOCK,),0,tl.int64);yp=tl.full((BLOCK,),0,tl.int64)
    for d in tl.static_range(len(SHAPE)-1,-1,-1):
        coord=remain%SHAPE[d];remain=remain//SHAPE[d]
        xp+=coord*XS[d];yp+=coord*YS[d]
    value=tl.load(X+xp,i<N,other=0)
    tl.store(Y+yp,value,i<N)

@triton.jit
def _strict_fill_kernel(Y,N:tl.constexpr,VALUE:tl.constexpr,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    tl.store(Y+i,VALUE,i<N)

@triton.jit
def _strict_random_kernel(Y,N:tl.constexpr,SEED,OFFSET,BLOCK:tl.constexpr):
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    # 备用随机采样路径；全部随机数计算在Triton内，生成器仅在宿主管理状态。
    value=tl.rand(SEED,(i.to(tl.uint64)+OFFSET).to(tl.uint32))
    tl.store(Y+i,value,i<N)

def _strict_empty_like(x,*,dtype=None,**kwargs):
    return torch.empty(tuple(x.shape),device=x.device,dtype=x.dtype if dtype is None else dtype)

def _strict_copy_into(dst,src):
    if tuple(dst.shape)!=tuple(src.shape):raise ValueError('strict copy requires equal shapes')
    if dst.numel():
        _strict_copy_kernel[(triton.cdiv(dst.numel(),1024),)](src,dst,dst.numel(),tuple(dst.shape),tuple(src.stride()),tuple(dst.stride()),1024)
    return dst

def _strict_clone(x,**kwargs):
    return _strict_copy_into(_strict_empty_like(x),x)

def _strict_contiguous(x,*args,**kwargs):
    return x if x.is_contiguous() else _strict_clone(x)

def _strict_cast(x,dtype):
    if dtype==x.dtype:return x
    return _strict_copy_into(_strict_empty_like(x,dtype=dtype),x)

def _strict_reshape(x,*shape):
    # view只操作元数据，输入连续化由上面的Triton复制完成。
    return _strict_contiguous(x).view(*shape)

def _strict_flatten(x,start_dim=0,end_dim=-1):
    rank=x.ndim;first=start_dim%rank;last=end_dim%rank
    size=1
    for n in x.shape[first:last+1]:size*=n
    shape=tuple(x.shape[:first])+(size,)+tuple(x.shape[last+1:])
    return _strict_contiguous(x).view(shape)

def _strict_full(shape,value,**kwargs):
    if isinstance(shape,int):shape=(shape,)
    out=torch.empty(shape,**kwargs)
    if out.numel():_strict_fill_kernel[(triton.cdiv(out.numel(),1024),)](out,out.numel(),value,1024)
    return out

def _strict_zeros(*shape,**kwargs):
    shape=shape[0] if len(shape)==1 else shape
    return _strict_full(shape,0,**kwargs)

def _strict_ones(*shape,**kwargs):
    shape=shape[0] if len(shape)==1 else shape
    return _strict_full(shape,1,**kwargs)

def _strict_rand(*shape,**kwargs):
    shape=shape[0] if len(shape)==1 else shape
    generator=kwargs.pop('generator',None)
    out=torch.empty(shape,**kwargs)
    if out.numel():
        if generator is None:
            import torch_npu
            generator=torch.npu.default_generators[out.device.index or 0]
        seed=generator.initial_seed();offset=generator.get_offset()
        advance=triton.cdiv(out.numel(),4)*4
        if offset+advance>=2**32:raise RuntimeError('备用随机路径超过32位计数范围，明确拒绝而非静默重复')
        generator.set_offset(offset+advance)
        _strict_random_kernel[(triton.cdiv(out.numel(),256),)](out,out.numel(),seed,offset,256)
    return out
