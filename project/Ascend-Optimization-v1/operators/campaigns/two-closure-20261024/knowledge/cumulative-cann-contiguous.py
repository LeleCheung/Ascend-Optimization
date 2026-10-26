"""Triton融合增量＋CANN扫描异步候选；资源仅保留至当前stream的完成事件。"""
import ctypes as ct
from collections import deque
import threading
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




@libentry()
@triton.jit
def _increments_last(Y,Z,N:tl.constexpr,SCALE:tl.constexpr,BLOCK:tl.constexpr):
    tile=tl.program_id(0)
    row=tl.program_id(1)
    extended=tile*BLOCK+tl.arange(0,2*BLOCK)
    values=tl.load(Y+row*N+extended,extended<N,other=0).to(tl.float32)
    k=tl.arange(0,BLOCK)
    a=tl.gather(values,k,0)
    z=tl.gather(values,k+1,0)
    added=(a+z).to(Y.dtype.element_ty).to(tl.float32)
    off=tile*BLOCK+k
    tl.store(Z+row*(N-1)+off,(added*SCALE).to(Z.dtype.element_ty),off<N-1)


@libentry()
@triton.jit
def _increments_first(Y,Z,TOTAL:tl.constexpr,INNER:tl.constexpr,SCALE:tl.constexpr,BLOCK:tl.constexpr):
    off=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    a=tl.load(Y+off,off<TOTAL,other=0).to(tl.float32)
    z=tl.load(Y+off+INNER,off<TOTAL,other=0).to(tl.float32)
    added=(a+z).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Z+off,(added*SCALE).to(Z.dtype.element_ty),off<TOTAL)

_LIB = None
_PENDING = deque()
_LOCK = threading.Lock()


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
    increments = torch.empty(shape, dtype=y.dtype, device=y.device)
    stream = torch.npu.current_stream(y.device)
    if y.is_contiguous() and axis == y.ndim - 1:
        n = int(y.shape[axis])
        _increments_last[(triton.cdiv(n-1,1024),out.numel()//(n-1))](y,increments,n,float(dx)/2.,1024)
    elif y.is_contiguous() and axis == 0:
        inner = y.numel() // int(y.shape[axis])
        _increments_first[(triton.cdiv(out.numel(),2048),)](y,increments,out.numel(),inner,float(dx)/2.,2048)
    else:
        _increments[(triton.cdiv(out.numel(), 2048),)](y, increments, out.numel(),
            tuple(shape), tuple(y.stride()), axis, float(dx) / 2., 2048)
    lib = _library()
    # Never reuse an input, output, descriptor or executor for a new call.
    # These references only keep asynchronous resources alive until completion.
    with _LOCK:
        remaining = deque()
        while _PENDING:
            event, tensors, descriptors = _PENDING.popleft()
            if event.query():
                for descriptor in descriptors:
                    _check(lib.aclDestroyTensor(descriptor), 'aclDestroyTensor(completed)')
            else:
                remaining.append((event, tensors, descriptors))
        _PENDING.extend(remaining)
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
        done = torch.npu.Event()
        done.record(stream)
        with _LOCK:
            _PENDING.append((done, (y, increments, out, workspace), (desc_out, desc_in)))
        desc_in = desc_out = None
    except Exception:
        # Error cleanup cannot free resources before their device work finishes.
        stream.synchronize()
        raise
    finally:
        if desc_out:
            _check(lib.aclDestroyTensor(desc_out), 'aclDestroyTensor(out)')
        if desc_in:
            _check(lib.aclDestroyTensor(desc_in), 'aclDestroyTensor(in)')
    return out
