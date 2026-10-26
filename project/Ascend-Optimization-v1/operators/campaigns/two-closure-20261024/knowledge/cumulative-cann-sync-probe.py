"""Triton融合增量＋CANN扫描的同步数值探针；同步开销计入，不作为性能交付。"""
import ctypes as ct
import torch
import triton
import triton.language as tl
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _increments(Y, Z, TOTAL: tl.constexpr, SHAPE: tl.constexpr,
                STRIDES: tl.constexpr, AXIS: tl.constexpr,
                SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remainder = offsets
    address = tl.full((BLOCK,), 0, tl.int64)
    for i in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = remainder % SHAPE[i]
        remainder = remainder // SHAPE[i]
        address += coord * STRIDES[i]
    left = tl.load(Y + address, offsets < TOTAL, other=0).to(tl.float32)
    right = tl.load(Y + address + STRIDES[AXIS], offsets < TOTAL, other=0).to(tl.float32)
    summed = (left + right).to(Y.dtype.element_ty).to(tl.float32)
    scaled = summed * SCALE
    tl.store(Z + offsets, scaled.to(Z.dtype.element_ty), offsets < TOTAL)


_LIB = None


def _library():
    global _LIB
    if _LIB is not None:
        return _LIB
    lib = ct.CDLL('libopapi.so')
    i64p = ct.POINTER(ct.c_int64)
    lib.aclCreateTensor.argtypes = [i64p, ct.c_uint64, ct.c_int, i64p,
        ct.c_int64, ct.c_int, i64p, ct.c_uint64, ct.c_void_p]
    lib.aclCreateTensor.restype = ct.c_void_p
    lib.aclDestroyTensor.argtypes = [ct.c_void_p]
    lib.aclDestroyTensor.restype = ct.c_int
    lib.aclnnCumsumGetWorkspaceSize.argtypes = [ct.c_void_p, ct.c_int64,
        ct.c_int, ct.c_void_p, ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_void_p)]
    lib.aclnnCumsumGetWorkspaceSize.restype = ct.c_int
    lib.aclnnCumsum.argtypes = [ct.c_void_p, ct.c_uint64, ct.c_void_p, ct.c_void_p]
    lib.aclnnCumsum.restype = ct.c_int
    _LIB = lib
    return lib


def _check(code, operation):
    if code:
        raise RuntimeError('%s returned %s' % (operation, code))


def _descriptor(lib, tensor, dtype):
    assert tensor.is_contiguous() and tensor.storage_offset() == 0
    rank = tensor.dim()
    dims = (ct.c_int64 * rank)(*tensor.shape)
    strides = (ct.c_int64 * rank)(*tensor.stride())
    descriptor = lib.aclCreateTensor(dims, rank, dtype, strides, 0, 2,
        dims, rank, ct.c_void_p(tensor.data_ptr()))
    if not descriptor:
        raise RuntimeError('aclCreateTensor returned NULL')
    return descriptor


def run(y, *, dx=1, dim=-1):
    if y.ndim == 0 or not -y.ndim <= dim < y.ndim:
        raise IndexError('cumulative_trapezoid: invalid dimension')
    axis = dim % y.ndim
    shape = list(y.shape)
    shape[axis] = max(0, shape[axis] - 1)
    out = torch.empty(shape, dtype=y.dtype, device=y.device)
    if out.numel() == 0:
        return out
    dtype = {torch.float32: 0, torch.float16: 1, torch.float64: 11, torch.bfloat16: 27}[y.dtype]
    increments = torch.empty_like(out)
    stream = torch.npu.current_stream(y.device)
    _increments[(triton.cdiv(out.numel(), 2048),)](y, increments, out.numel(),
        tuple(shape), tuple(y.stride()), axis, float(dx) / 2., 2048)
    # torch_npu may enqueue launches on its task queue.  This diagnostic first
    # synchronizes before direct CANN enqueue, separating ordering from numerics.
    stream.synchronize()
    lib = _library()
    desc_in = desc_out = None
    workspace = None
    try:
        desc_in = _descriptor(lib, increments, dtype)
        desc_out = _descriptor(lib, out, dtype)
        size = ct.c_uint64()
        executor = ct.c_void_p()
        _check(lib.aclnnCumsumGetWorkspaceSize(desc_in, axis, dtype, desc_out,
            ct.byref(size), ct.byref(executor)), 'aclnnCumsumGetWorkspaceSize')
        workspace = torch.empty((size.value,), dtype=torch.uint8, device=y.device)
        _check(lib.aclnnCumsum(ct.c_void_p(workspace.data_ptr()), size.value,
            executor, ct.c_void_p(stream.npu_stream)), 'aclnnCumsum')
        stream.synchronize()
    finally:
        if desc_out:
            _check(lib.aclDestroyTensor(desc_out), 'aclDestroyTensor(out)')
        if desc_in:
            _check(lib.aclDestroyTensor(desc_in), 'aclDestroyTensor(in)')
    return out
