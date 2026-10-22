# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-place base-10 logarithm (log10_) for Ascend with Triton.

log10_ is a single elementwise pass, so the per-call cost for every measured
shape up to 1M elements is the host launch path, not the device work.  Measured
on the target with this batch's paired method:

* FlagTree's Python dispatch (``kernel[grid](...)``) costs ~52us per call;
* the compiled launcher entry itself costs ~6.5-8us, and ``torch.log10_``
  dispatches in ~5.5-9.5us, so only the compiled launcher is competitive;
* building the pointer wrapper class per call costs ~18us (class creation),
  while reusing one instance costs nothing, so each (numel, pointer)
  specialization is built once and its wrapper object is reused;
* the current stream must be read per call, which costs ~0.26us through
  ``_npu_getCurrentRawStreamNoWait`` and is kept on the steady-state path.

Steady state is therefore one ``data_ptr``, one ``numel``, one dict lookup, one
dtype identity check, one contiguity check, one stream read and the launch.
The address is part of the cache key, so a different tensor with the same extent
never reuses a stale pointer.  Cold, misaligned and non-contiguous calls keep
the ordinary Triton path.
"""

def _hgkda_address_type(n):

    class DeviceAddress(int):

        def size(self):
            return (n,)
    return DeviceAddress
import torch
import torch_npu
import triton
import triton.language as tl
_BLOCK_MAX = 32768
_BLOCK_MIN = 1024
_BLOCK_TARGET_PROGRAMS = 8
_BLOCK_STRIDED = 1024
_FAST_LIMIT = 4096

@triton.jit
def _log10_nomask(X, BLOCK: tl.constexpr):
    """numel % BLOCK == 0 (host-guarded), so the tail mask can be dropped."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log(x) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty))

@triton.jit
def _log10_masked(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.log(x) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty), mask=mask)

@triton.jit
def _log10_strided(X, N: tl.constexpr, SHAPE: tl.constexpr, STRIDES: tl.constexpr, BLOCK: tl.constexpr):
    """Non-contiguous input: logical index -> storage offset, written in place."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remainder = index
    offset = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = remainder % SHAPE[axis]
        remainder = remainder // SHAPE[axis]
        offset += coord.to(tl.int64) * STRIDES[axis]
    x = tl.load(X + offset, index < N, other=1)
    y = tl.log(x) * 0.4342944819032518
    tl.store(X + offset, y.to(X.dtype.element_ty), index < N)
_FAST = {}
_fast_get = _FAST.get
_STREAM = {}

def _pick_block(n):
    """Largest tile that spreads work over cores without exceeding the UB budget."""
    if n >= _BLOCK_MAX * _BLOCK_TARGET_PROGRAMS:
        return _BLOCK_MAX
    target = -(-n // _BLOCK_TARGET_PROGRAMS)
    block = _BLOCK_MIN
    while block < target:
        block <<= 1
    return min(block, _BLOCK_MAX)

def _resolve_stream(device):
    """Cheapest faithful current-stream accessor for this device, resolved once."""
    getter = _STREAM.get(device)
    if getter is not None:
        return getter
    raw = getattr(torch_npu._C, '_npu_getCurrentRawStreamNoWait', None)
    if raw is not None:
        try:
            want = int(torch.npu.current_stream().npu_stream)
            try:
                if int(raw(device)) == want:
                    getter = lambda: raw(device)
            except TypeError:
                if int(raw()) == want:
                    getter = raw
        except Exception:
            getter = None
    if getter is None:
        from triton.runtime import driver
        getter = lambda: driver.active.get_current_stream(device)
    _STREAM[device] = getter
    return getter

def _standard_launch(n, block, masked):
    """Ordinary Triton launch: no alignment assumption, safe for any pointer."""
    if masked:

        def launch(X, ptr, n=n, block=block):
            _log10_masked[triton.cdiv(n, block),](X, n, BLOCK=block)
    else:

        def launch(X, ptr, n=n, block=block):
            _log10_nomask[n // block,](X, BLOCK=block)
    return launch

def _make_launch(A, n, ptr):
    """Build the per-specialization launcher, degrading to a plain Triton launch."""
    block = _pick_block(n)
    masked = bool(n % block)
    if ptr & 15:
        return _standard_launch(n, block, masked)
    try:
        if masked:
            compiled = _log10_masked.warmup(A, n, BLOCK=block, grid=(1,))
            grid = triton.cdiv(n, block)
        else:
            compiled = _log10_nomask.warmup(A, BLOCK=block, grid=(1,))
            grid = n // block
        launcher = compiled.run
        if getattr(launcher, 'compile_only', False) or getattr(launcher, 'enable_msprof_register_tensor', False) or getattr(launcher.metadata, 'debug_enabled', False):
            return _standard_launch(n, block, masked)
        call = launcher.launch
        function = compiled.function
        metadata = compiled.packed_metadata
        stream = _resolve_stream(A.device.index)
    except Exception:
        return _standard_launch(n, block, masked)
    address = _hgkda_address_type(n)(ptr)
    if masked:

        def launch(X, ptr, grid=grid, call=call, function=function, metadata=metadata, stream=stream, address=address, n=n):
            call(grid, 1, 1, stream(), function, metadata, None, None, None, address, n, None)
    else:

        def launch(X, ptr, grid=grid, call=call, function=function, metadata=metadata, stream=stream, address=address):
            call(grid, 1, 1, stream(), function, metadata, None, None, None, address, None)
    return launch

def _dispatch(A, n, ptr):
    if n == 0:
        return A
    if not A.is_contiguous():
        _log10_strided[triton.cdiv(n, _BLOCK_STRIDED),](A, n, tuple(A.shape), tuple(A.stride()), BLOCK=_BLOCK_STRIDED)
        return A
    if len(_FAST) >= _FAST_LIMIT:
        _FAST.clear()
    launch = _make_launch(A, n, ptr)
    _FAST[(n, ptr)] = (A.dtype, launch)
    launch(A, ptr)
    return A

def run(A):
    n = A.numel()
    ptr = A.data_ptr()
    entry = _fast_get((n, ptr))
    if entry is not None and A.dtype is entry[0] and A.is_contiguous():
        entry[1](A, ptr)
        return A
    return _dispatch(A, n, ptr)
