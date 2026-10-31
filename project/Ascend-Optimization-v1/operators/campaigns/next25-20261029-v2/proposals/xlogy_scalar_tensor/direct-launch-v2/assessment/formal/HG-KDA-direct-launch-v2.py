"""xlogy_scalar_tensor : out = xlogy(self_scalar, other), self-contained Triton.

Reference semantics (``torch.xlogy(scalar, tensor)``, fp32 math):
  * ``y != y``            -> NaN      (checked first, matches aten precedence)
  * ``self == 0``         -> +0.0     (even for y == 0, y < 0, y == +-inf)
  * otherwise             -> ``float(self) * log(y)``
  * output dtype/shape always follow ``other``; the scalar path never mutates it.

For ``self != 0`` the two special cases collapse into plain IEEE arithmetic
(``s * log(NaN)`` is already NaN), so the timed path is a single fused
multiply-by-log.  For ``self == 0`` the transcendental is dropped entirely and
only the NaN positions of ``y`` survive.

Layout: contiguous inputs take a flat kernel; non-contiguous views take an
index-math kernel driven by their real element strides, so ``base[..., 0]``
style views are read correctly.
"""

import torch
import torch_npu
import triton
import triton.language as tl

_BLOCK_MIN = 1024
_BLOCK_CAP_2B = 16384      # 32 KiB source tile for 2-byte dtypes
_BLOCK_CAP_4B = 8192       # 32 KiB source tile for 4-byte dtypes
_TARGET_PROGS = 32
_BLOCK_STRIDED = 512


@triton.jit
def _xlogy_flat(Y, OUT, n, s, BLOCK: tl.constexpr, HAS_LOG: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    y = tl.load(Y + offs, mask=mask, other=1.0).to(tl.float32)
    if HAS_LOG:
        val = s * tl.log(y)
    else:
        val = tl.where(y != y, float("nan"), 0.0)
    tl.store(OUT + offs, val.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _xlogy_strided(Y, OUT, n, s, SHAPE: tl.constexpr, YS: tl.constexpr,
                   BLOCK: tl.constexpr, HAS_LOG: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n
    rem = idx
    off = tl.zeros((BLOCK,), dtype=tl.int32)
    for ax in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = rem % SHAPE[ax]
        rem = rem // SHAPE[ax]
        off += coord * YS[ax]
    y = tl.load(Y + off, mask=mask, other=1.0).to(tl.float32)
    if HAS_LOG:
        val = s * tl.log(y)
    else:
        val = tl.where(y != y, float("nan"), 0.0)
    tl.store(OUT + idx, val.to(OUT.dtype.element_ty), mask=mask)


def _pick_block(n, esize):
    cap = _BLOCK_CAP_2B if esize < 4 else _BLOCK_CAP_4B
    target = -(-n // _TARGET_PROGS)
    block = _BLOCK_MIN
    while block < target:
        block <<= 1
    return block if block < cap else cap


_ENTRIES = {}
_RAW = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)


def _stream(device):
    if _RAW is not None:
        try:return _RAW(device)
        except TypeError:return _RAW()
    return torch.npu.current_stream(device).npu_stream


def run(self, other):
    y = other
    out = torch.empty_like(y, memory_format=torch.contiguous_format)
    n = y.numel()
    if n == 0:
        return out
    s = float(self)
    has_log = s != 0.0
    if not y.is_contiguous():
        _xlogy_strided[((n + _BLOCK_STRIDED - 1) // _BLOCK_STRIDED,)](
            y, out, n, s, SHAPE=tuple(y.shape), YS=tuple(y.stride()),
            BLOCK=_BLOCK_STRIDED, HAS_LOG=has_log)
        return out
    device = y.device.index
    key = (device, y.dtype, n, has_log)
    entry = _ENTRIES.get(key)
    if entry is None:
        block = _pick_block(n, y.element_size())
        grid = (n + block - 1) // block
        launch = None
        try:
            compiled = _xlogy_flat.warmup(y, out, n, s, BLOCK=block, HAS_LOG=has_log, grid=(1,))
            launcher = compiled.run
            if not (getattr(launcher, 'compile_only', False) or
                    getattr(launcher, 'enable_msprof_register_tensor', False) or
                    getattr(launcher.metadata, 'debug_enabled', False) or
                    getattr(launcher.metadata, 'debug', False)):
                launch = (launcher.launch, compiled.function, compiled.packed_metadata)
        except Exception:
            pass
        entry = (grid, block, launch)
        _ENTRIES[key] = entry
    launch = entry[2]
    if launch and not ((y.data_ptr() | out.data_ptr()) & 15):
        launch[0](entry[0], 1, 1, _stream(device), launch[1], launch[2], None, None, None,
                  y.data_ptr(), out.data_ptr(), n, s, entry[1], has_log)
    else:
        _xlogy_flat[(entry[0],)](y, out, n, s, BLOCK=entry[1], HAS_LOG=has_log)
    return out
