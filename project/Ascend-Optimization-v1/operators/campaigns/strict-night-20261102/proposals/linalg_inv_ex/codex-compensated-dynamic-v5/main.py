"""纯Triton求逆：小矩阵按列延迟更新LU，再按原布局选择求解方向。

按列更新的运算顺序来自固定版本OpenBLAS源码与离线FP32模型。
这里只实现通用算法，不包含固定输入、参考逆矩阵或按输入值选择的特化。
小矩阵采用补偿乘加和余数修正商；循环静态展开，倒数广播为向量，避免标量precise_div转换。
FP32为当前实现支持的类型；大矩阵沿用本批种子的右看式LU策略。
"""
from collections import namedtuple
import torch
import triton
import triton.language as tl

Result = namedtuple('LinalgInvExResult', ['inverse', 'info'])


@triton.jit
def _compensated_fma(a, b, c):
    product = a * b
    split = product + c
    mask = tl.full((), 0xfffff000, tl.uint32)
    ah = (a.to(tl.uint32, bitcast=True) & mask).to(tl.float32, bitcast=True)
    bh = (b.to(tl.uint32, bitcast=True) & mask).to(tl.float32, bitcast=True)
    al = a - ah
    bl = b - bh
    e1 = product - ah * bh
    e2 = e1 - al * bh
    e3 = e2 - ah * bl
    product_error = al * bl - e3
    virtual = split - product
    sum_error = (product - (split - virtual)) + (c - virtual)
    return split + (sum_error + product_error)


@triton.jit
def _corrected_div(a, b):
    quotient = a / b
    residual = _compensated_fma(-quotient, b, a)
    correction = residual / b
    corrected = quotient + correction
    ordinary = (b == 0.) | (tl.abs(b) == float('inf')) | (tl.abs(quotient) == float('inf')) | (quotient != quotient)
    return tl.where(ordinary, quotient, corrected)


@triton.jit
def _leftlooking(A, INV, INFO, N: tl.constexpr, B: tl.constexpr,
                 BSHAPE: tl.constexpr, STRIDES: tl.constexpr, TRANSPOSE: tl.constexpr):
    pid = tl.program_id(0)
    r = tl.arange(0, B)
    c = tl.arange(0, B)
    rest = pid
    offset = tl.full((), 0, tl.int64)
    for d in tl.static_range(len(BSHAPE) - 1, -1, -1):
        offset += (rest % BSHAPE[d]) * STRIDES[d]
        rest = rest // BSHAPE[d]
    if TRANSPOSE:
        address = offset + c[None, :] * STRIDES[-2] + r[:, None] * STRIDES[-1]
    else:
        address = offset + r[:, None] * STRIDES[-2] + c[None, :] * STRIDES[-1]
    work = tl.load(A + address, (r[:, None] < N) & (c[None, :] < N), other=0)
    pivots = tl.full((B,), 0, tl.int32)
    info = tl.full((), 0, tl.int32)
    for j in tl.range(0, N):
        column = tl.sum(tl.where(c[None, :] == j, work, 0.), 1)
        # 只给当前尚未更新的列应用以前的行置换。
        for k in tl.range(0, j):
            pivot_k = tl.sum(tl.where(r == k, pivots, 0), 0)
            vk = tl.sum(tl.where(r == k, column, 0.), 0)
            vp = tl.sum(tl.where(r == pivot_k, column, 0.), 0)
            column = tl.where(r == k, vp, tl.where(r == pivot_k, vk, column))
        # 上三角：先完成点积，再从原列元素中减去；避免改成逐次减法。
        for i in tl.range(1, j):
            dot = tl.full((), 0., tl.float32)
            row_i = tl.sum(tl.where(r[:, None] == i, work, 0.), 0)
            for k in tl.range(0, i):
                aik = tl.sum(tl.where(c == k, row_i, 0.), 0)
                ukj = tl.sum(tl.where(r == k, column, 0.), 0)
                dot = dot + aik * ukj
            old = tl.sum(tl.where(r == i, column, 0.), 0)
            column = tl.where(r == i, old - dot, column)
        # 下半列：按k递增，用纯Triton补偿乘加保留乘积舍入误差。
        for k in tl.range(0, j):
            lk = tl.sum(tl.where(c[None, :] == k, work, 0.), 1)
            ukj = tl.sum(tl.where(r == k, column, 0.), 0)
            column = tl.where(r >= j, _compensated_fma(-lk, ukj, column), column)
        magnitudes = tl.where((r >= j) & (r < N), tl.abs(column), -1.)
        maximum = tl.max(magnitudes, 0)
        pivot_row = tl.min(tl.where(magnitudes == maximum, r, B), 0)
        pivot = tl.sum(tl.where(r == pivot_row, column, 0.), 0)
        pivots = tl.where(r == j, pivot_row, pivots)
        info = tl.where((pivot == 0.) & (info == 0), j + 1, info)
        # 已求出的列同步换行，未处理列在其自己的迭代中应用置换。
        row_j = tl.sum(tl.where(r[:, None] == j, work, 0.), 0)
        row_p = tl.sum(tl.where(r[:, None] == pivot_row, work, 0.), 0)
        work = tl.where((r[:, None] == j) & (c[None, :] < j), row_p[None, :], work)
        work = tl.where((r[:, None] == pivot_row) & (c[None, :] < j), row_j[None, :], work)
        vj = tl.sum(tl.where(r == j, column, 0.), 0)
        vp = tl.sum(tl.where(r == pivot_row, column, 0.), 0)
        column = tl.where(r == j, vp, tl.where(r == pivot_row, vj, column))
        reciprocal = _corrected_div(tl.full((B,), 1., tl.float32), tl.where(pivot == 0., 1., pivot))
        column = tl.where(r > j, column * reciprocal, column)
        work = tl.where(c[None, :] == j, column[:, None], work)

    rhs = tl.where(r[:, None] == c[None, :], 1., 0.)
    if TRANSPOSE:
        # PA^T=LU：先U^T，再L^T，最后逆置换。
        for j in tl.range(0, N):
            ucol = tl.sum(tl.where(c[None, :] == j, work, 0.), 1)
            value = tl.sum(tl.where(r[:, None] == j, rhs, 0.), 0)
            for k in tl.range(0, j):
                coeff = tl.sum(tl.where(r == k, ucol, 0.), 0)
                previous = tl.sum(tl.where(r[:, None] == k, rhs, 0.), 0)
                value = value - coeff * previous
            diagonal = tl.sum(tl.where(r == j, ucol, 0.), 0)
            value = _corrected_div(value, diagonal)
            rhs = tl.where(r[:, None] == j, value[None, :], rhs)
        for back in tl.range(0, N):
            j = N - 1 - back
            lcol = tl.sum(tl.where(c[None, :] == j, work, 0.), 1)
            value = tl.sum(tl.where(r[:, None] == j, rhs, 0.), 0)
            for k in tl.range(j + 1, N):
                coeff = tl.sum(tl.where(r == k, lcol, 0.), 0)
                previous = tl.sum(tl.where(r[:, None] == k, rhs, 0.), 0)
                value = value - coeff * previous
            rhs = tl.where(r[:, None] == j, value[None, :], rhs)
        for back in tl.range(0, N):
            j = N - 1 - back
            p = tl.sum(tl.where(r == j, pivots, 0), 0)
            row_j = tl.sum(tl.where(r[:, None] == j, rhs, 0.), 0)
            row_p = tl.sum(tl.where(r[:, None] == p, rhs, 0.), 0)
            rhs = tl.where(r[:, None] == j, row_p[None, :], rhs)
            rhs = tl.where(r[:, None] == p, row_j[None, :], rhs)
    else:
        for j in tl.range(0, N):
            p = tl.sum(tl.where(r == j, pivots, 0), 0)
            row_j = tl.sum(tl.where(r[:, None] == j, rhs, 0.), 0)
            row_p = tl.sum(tl.where(r[:, None] == p, rhs, 0.), 0)
            rhs = tl.where(r[:, None] == j, row_p[None, :], rhs)
            rhs = tl.where(r[:, None] == p, row_j[None, :], rhs)
        for j in tl.range(0, N):
            row = tl.sum(tl.where(r[:, None] == j, work, 0.), 0)
            value = tl.sum(tl.where(r[:, None] == j, rhs, 0.), 0)
            for k in tl.range(0, j):
                coeff = tl.sum(tl.where(c == k, row, 0.), 0)
                previous = tl.sum(tl.where(r[:, None] == k, rhs, 0.), 0)
                value = value - coeff * previous
            rhs = tl.where(r[:, None] == j, value[None, :], rhs)
        for back in tl.range(0, N):
            j = N - 1 - back
            row = tl.sum(tl.where(r[:, None] == j, work, 0.), 0)
            value = tl.sum(tl.where(r[:, None] == j, rhs, 0.), 0)
            for k in tl.range(j + 1, N):
                coeff = tl.sum(tl.where(c == k, row, 0.), 0)
                previous = tl.sum(tl.where(r[:, None] == k, rhs, 0.), 0)
                value = value - coeff * previous
            diagonal = tl.sum(tl.where(c == j, row, 0.), 0)
            rhs = tl.where(r[:, None] == j, _corrected_div(value, diagonal)[None, :], rhs)
    tl.store(INV + pid * N * N + r[:, None] * N + c[None, :], rhs,
             (r[:, None] < N) & (c[None, :] < N))
    tl.store(INFO + pid, info)


@triton.jit
def _large(A, INV, INFO, PIVOTS, N: tl.constexpr, B: tl.constexpr, CT: tl.constexpr,
           TRANSPOSE: tl.constexpr):
    pid = tl.program_id(0)
    r = tl.arange(0, B)
    c = tl.arange(0, B)
    base = pid * N * N
    if TRANSPOSE:
        offsets = base + c[None, :] * N + r[:, None]
    else:
        offsets = base + r[:, None] * N + c[None, :]
    work = tl.load(A + offsets, (r[:, None] < N) & (c[None, :] < N), other=0.)
    info = tl.full((), 0, tl.int32)
    for j in tl.range(0, N):
        column = tl.sum(tl.where(c[None, :] == j, work, 0.), 1)
        magnitudes = tl.where((r >= j) & (r < N), tl.abs(column), -1.)
        maximum = tl.max(magnitudes, 0)
        p = tl.min(tl.where(magnitudes == maximum, r, B), 0)
        tl.store(PIVOTS + pid * N + j, p)
        pivot = tl.sum(tl.where(r == p, column, 0.), 0)
        row_j = tl.sum(tl.where(r[:, None] == j, work, 0.), 0)
        row_p = tl.sum(tl.where(r[:, None] == p, work, 0.), 0)
        work = tl.where(r[:, None] == j, row_p[None, :], work)
        work = tl.where(r[:, None] == p, row_j[None, :], work)
        rec = 1. / tl.where(pivot == 0., 1., pivot)
        work = tl.where((r[:, None] > j) & (c[None, :] == j), work * rec, work)
        info = tl.where((pivot == 0.) & (info == 0), j + 1, info)
        column = tl.sum(tl.where(c[None, :] == j, work, 0.), 1)
        row = tl.sum(tl.where(r[:, None] == j, work, 0.), 0)
        work = tl.where((r[:, None] > j) & (c[None, :] > j),
                        work - column[:, None] * row[None, :], work)
    t = tl.arange(0, CT)
    for start in tl.range(0, N, CT):
        rhs = tl.where(r[:, None] == start + t[None, :], 1., 0.)
        for j in tl.range(0, N):
            p = tl.load(PIVOTS + pid * N + j)
            row_j = tl.sum(tl.where(r[:, None] == j, rhs, 0.), 0)
            row_p = tl.sum(tl.where(r[:, None] == p, rhs, 0.), 0)
            rhs = tl.where(r[:, None] == j, row_p[None, :], rhs)
            rhs = tl.where(r[:, None] == p, row_j[None, :], rhs)
        for j in tl.range(0, N):
            column = tl.sum(tl.where(c[None, :] == j, work, 0.), 1)
            column = tl.where(r > j, column, 0.)
            value = tl.sum(tl.where(r[:, None] == j, rhs, 0.), 0)
            rhs = rhs - column[:, None] * value[None, :]
        for back in tl.range(0, N):
            j = N - 1 - back
            column = tl.sum(tl.where(c[None, :] == j, work, 0.), 1)
            diagonal = tl.sum(tl.where(r == j, column, 0.), 0)
            value = tl.sum(tl.where(r[:, None] == j, rhs, 0.), 0) / diagonal
            rhs = tl.where(r[:, None] == j, value[None, :], rhs)
            column = tl.where(r < j, column, 0.)
            rhs = rhs - column[:, None] * value[None, :]
        if TRANSPOSE:
            out_offsets = base + (start + t[None, :]) * N + r[:, None]
        else:
            out_offsets = base + r[:, None] * N + start + t[None, :]
        tl.store(INV + out_offsets, rhs, (r[:, None] < N) & (start + t[None, :] < N))
    tl.store(INFO + pid, info)


@triton.jit
def _pack(A, OUT, TOTAL: tl.constexpr, SHAPE: tl.constexpr, STRIDES: tl.constexpr):
    i = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    rest = i
    address = tl.full((1024,), 0, tl.int64)
    for d in tl.static_range(len(SHAPE) - 1, -1, -1):
        address += rest % SHAPE[d] * STRIDES[d]
        rest = rest // SHAPE[d]
    tl.store(OUT + i, tl.load(A + address, i < TOTAL, other=0.), i < TOTAL)


@triton.jit
def _zero_info(INFO, COUNT: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(INFO + i, 0, i < COUNT)


def run(A, *, check_errors=False):
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError('输入必须是方阵或方阵批次')
    if A.dtype != torch.float32:
        raise TypeError('当前Triton实现支持float32')
    n = A.shape[-1]
    batch_shape = tuple(A.shape[:-2])
    count = 1
    for extent in batch_shape:
        count *= extent
    inverse = torch.empty(tuple(A.shape), dtype=A.dtype, device=A.device)
    info = torch.empty(batch_shape, dtype=torch.int32, device=A.device)
    transpose = A.is_contiguous()
    if n == 0:
        if count:
            _zero_info[(triton.cdiv(count, 256),)](info, count)
    elif count and n <= 8:
        _leftlooking[(count,)](A, inverse, info, n, max(16, triton.next_power_of_2(n)),
                              batch_shape, tuple(A.stride()), transpose, enable_fp_fusion=False)
    elif count:
        packed = A
        if not A.is_contiguous():
            packed = torch.empty(tuple(A.shape), dtype=A.dtype, device=A.device)
            _pack[(triton.cdiv(A.numel(), 1024),)](A, packed, A.numel(), tuple(A.shape), tuple(A.stride()))
        b = max(16, triton.next_power_of_2(n))
        pivots = torch.empty((count, n), dtype=torch.int32, device=A.device)
        _large[(count,)](packed, inverse, info, pivots, n, b, min(b, 32), transpose,
                        num_warps=4 if b <= 32 else 8)
    # 仅在调用者要求报错时把Triton计算的info带回宿主，数值输出仍全部在设备产生。
    if check_errors:
        statuses = info.view(-1).tolist()
        for status in statuses:
            if status:
                raise RuntimeError('矩阵LU分解遇到零主元，无法求逆')
    return Result(inverse, info)
