# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

# Kernel constants must be tl.constexpr objects so that @triton.jit bodies can
# read them (plain Python globals are rejected by the Ascend code generator).
_SCALE = tl.constexpr(1.0507009873554805)
# alpha * scale folded so the negative branch needs one multiply instead of two
_ALPHA_SCALE = tl.constexpr(1.7580993408473766)

_BLOCK_F16 = 8192
_BLOCK_F32 = 4096
_BLOCK_STRIDED = 2048
_MAX_RANK = 8


@triton.jit
def _selu_flat_nomask(x_ptr, BLOCK: tl.constexpr, F16MATH: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    if F16MATH:
        y = tl.where(x > 0, x * _SCALE, (tl.exp(x) - 1.0) * _ALPHA_SCALE)
    else:
        xf = x.to(tl.float32)
        yf = tl.where(xf > 0, xf * _SCALE, (tl.exp(xf) - 1.0) * _ALPHA_SCALE)
        y = yf.to(x.dtype)
    tl.store(x_ptr + offs, y.to(x.dtype))


@triton.jit
def _selu_flat_masked(x_ptr, n_elements, BLOCK: tl.constexpr, F16MATH: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if F16MATH:
        y = tl.where(x > 0, x * _SCALE, (tl.exp(x) - 1.0) * _ALPHA_SCALE)
    else:
        xf = x.to(tl.float32)
        yf = tl.where(xf > 0, xf * _SCALE, (tl.exp(xf) - 1.0) * _ALPHA_SCALE)
        y = yf.to(x.dtype)
    tl.store(x_ptr + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _selu_strided(
    x_ptr,
    n_elements,
    base,
    BLOCK: tl.constexpr,
    s0: tl.constexpr, s1: tl.constexpr, s2: tl.constexpr, s3: tl.constexpr,
    s4: tl.constexpr, s5: tl.constexpr, s6: tl.constexpr, s7: tl.constexpr,
    t0: tl.constexpr, t1: tl.constexpr, t2: tl.constexpr, t3: tl.constexpr,
    t4: tl.constexpr, t5: tl.constexpr, t6: tl.constexpr, t7: tl.constexpr,
    RANK: tl.constexpr,
    IDX64: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    if IDX64:
        rem = idx.to(tl.int64)
        off = tl.zeros([BLOCK], dtype=tl.int64)
    else:
        rem = idx
        off = tl.zeros([BLOCK], dtype=tl.int32)
    o = rem % s0
    rem = rem // s0
    off += o * t0
    if RANK > 1:
        o = rem % s1
        rem = rem // s1
        off += o * t1
    if RANK > 2:
        o = rem % s2
        rem = rem // s2
        off += o * t2
    if RANK > 3:
        o = rem % s3
        rem = rem // s3
        off += o * t3
    if RANK > 4:
        o = rem % s4
        rem = rem // s4
        off += o * t4
    if RANK > 5:
        o = rem % s5
        rem = rem // s5
        off += o * t5
    if RANK > 6:
        o = rem % s6
        rem = rem // s6
        off += o * t6
    if RANK > 7:
        o = rem % s7
        off += o * t7
    ptr = x_ptr + (off + base)
    x = tl.load(ptr, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    yf = tl.where(xf > 0, xf * _SCALE, (tl.exp(xf) - 1.0) * _ALPHA_SCALE)
    tl.store(ptr, yf.to(x.dtype), mask=mask)


def _collapse(shape, stride):
    """Drop size-1 dims and merge neighbouring dims that are already contiguous."""
    dims = [(int(s), int(t)) for s, t in zip(shape, stride) if int(s) != 1]
    out = []
    for s, t in dims:
        if out and out[-1][1] == t * s:
            ps, pt = out.pop()
            out.append((ps * s, t))
        else:
            out.append((s, t))
    return out


def _strided_launch(x, base, n_elements, dims):
    if len(dims) > _MAX_RANK:
        # Host-side slicing of the outermost dim keeps the kernel rank <= 8.
        s0, t0 = dims[0]
        inner = 1
        for s, _ in dims[1:]:
            inner *= s
        for i in range(s0):
            _strided_launch(x, base + i * t0, inner, dims[1:])
        return
    span = base + 1
    for s, t in dims:
        span += (s - 1) * t
    idx64 = span > 0x7FFFFFFF
    shapes = [d[0] for d in dims] + [1] * (_MAX_RANK - len(dims))
    strides = [d[1] for d in dims] + [1] * (_MAX_RANK - len(dims))
    rank = len(dims)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    with torch_npu.npu.device(x.device):
        _selu_strided[grid](
            x,
            n_elements,
            base,
            BLOCK=_BLOCK_STRIDED,
            s0=shapes[0], s1=shapes[1], s2=shapes[2], s3=shapes[3],
            s4=shapes[4], s5=shapes[5], s6=shapes[6], s7=shapes[7],
            t0=strides[0], t1=strides[1], t2=strides[2], t3=strides[3],
            t4=strides[4], t5=strides[5], t6=strides[6], t7=strides[7],
            RANK=rank,
            IDX64=idx64,
        )


def _selu_inplace(x):
    n_elements = x.numel()
    if n_elements == 0:
        return x
    f16_math = x.dtype == torch.float16
    if x.is_contiguous():
        block = _BLOCK_F16 if f16_math else _BLOCK_F32
        if n_elements < block:
            # Shrink the block for short tensors so the tail case still uses
            # few enough lanes; pure-Python bit_length avoids extra imports.
            block = 1 << max(7, (n_elements - 1).bit_length())
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
        with torch_npu.npu.device(x.device):
            if n_elements % block == 0:
                _selu_flat_nomask[grid](x, BLOCK=block, F16MATH=f16_math)
            else:
                _selu_flat_masked[grid](
                    x, n_elements, BLOCK=block, F16MATH=f16_math
                )
        return x
    dims = _collapse(x.shape, x.stride())
    if not dims:
        # Every dim is size 1 -> single element; the base tensor pointer already
        # points at the view's first element, so the relative base stays 0.
        dims = [(1, 1)]
    # The tensor pointer passed to Triton already includes storage_offset, so the
    # top-level base must be 0 (relative to the view; recursion adds offsets).
    _strided_launch(x, 0, n_elements, dims)
    return x


def selu_(*args, **kwargs):
    x = None
    if len(args) > 0 and torch.is_tensor(args[0]):
        x = args[0]
    elif "input" in kwargs and torch.is_tensor(kwargs["input"]):
        x = kwargs["input"]
    elif "self" in kwargs and torch.is_tensor(kwargs["self"]):
        x = kwargs["self"]
    elif "x" in kwargs and torch.is_tensor(kwargs["x"]):
        x = kwargs["x"]
    else:
        raise ValueError(
            "selu_ expects a Tensor as the first argument or under 'input'/'self'/'x' keyword."
        )
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("selu_ Triton path supports float16/bfloat16/float32 only.")
    return _selu_inplace(x)


def run(A):
    return selu_(A)
