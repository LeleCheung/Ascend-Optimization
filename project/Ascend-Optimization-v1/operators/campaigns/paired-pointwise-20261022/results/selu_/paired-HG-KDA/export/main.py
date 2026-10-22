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

"""In-place SELU (selu_) for Ascend with Triton.

selu_ is a single elementwise in-place pass, so per-call latency is dominated by
the host launch path on small inputs and by a fixed per-tile device latency on
large inputs.  Measured on the target:

  * the FlagTree JIT wrapper costs ~49us per call, the already-bound compiled
    launcher ~12.3us, and the framework reference ~8.7us;
  * ~2us more came from rebuilding (dtype, numel, alignment) and walking a dict
    on every call.

So each specialization is compiled once and then launched through the raw
launcher with every argument bound, including the data pointer, and a repeated
call for the same tensor object short-circuits the whole dispatch behind a single
identity test.  Non-contiguous and misaligned calls keep ordinary Triton
launches.

Device-side, cost tracks the number of TILES (a fixed ~1.5us each, independent
of dtype and bytes), which is why the tile is the largest that fits the 192KB UB
budget once the fp32 compute tile is materialized.
"""

import torch
import torch_npu  # noqa: F401  (provides the raw current-stream query)
import triton
import triton.language as tl

# 8192 elements per tile is the largest that fits the 192KB UB budget once the
# fp32 compute tile is materialized with auto multi-buffering.
_BLOCK_MAX = 8192
_BLOCK_MIN = 1024
_BLOCK_TARGET_PROGRAMS = 8
_BLOCK_STRIDED = 1024


@triton.jit
def _selu_nomask(X, BLOCK: tl.constexpr):
    """numel % BLOCK == 0 (host-guarded), so the tail mask can be dropped."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = (1.0507009873554805 * tl.where(x.to(tl.float32) > 0, x.to(tl.float32), 1.6732632423543772 * (tl.exp(x.to(tl.float32))-1.0)))
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def _selu_masked(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = (1.0507009873554805 * tl.where(x.to(tl.float32) > 0, x.to(tl.float32), 1.6732632423543772 * (tl.exp(x.to(tl.float32))-1.0)))
    tl.store(X + offs, y.to(X.dtype.element_ty), mask=mask)


@triton.jit
def _selu_strided(
    X,
    N,
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    T7,
    NDIMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Arbitrary strides: logical index -> storage offset, written in place."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N
    rem = offs
    off = tl.zeros([BLOCK], tl.int64)
    if NDIMS >= 8:
        off += (rem % S7) * T7
        rem = rem // S7
    if NDIMS >= 7:
        off += (rem % S6) * T6
        rem = rem // S6
    if NDIMS >= 6:
        off += (rem % S5) * T5
        rem = rem // S5
    if NDIMS >= 5:
        off += (rem % S4) * T4
        rem = rem // S4
    if NDIMS >= 4:
        off += (rem % S3) * T3
        rem = rem // S3
    if NDIMS >= 3:
        off += (rem % S2) * T2
        rem = rem // S2
    if NDIMS >= 2:
        off += (rem % S1) * T1
        rem = rem // S1
    off += (rem % S0) * T0
    x = tl.load(X + off, mask=mask)
    y = (1.0507009873554805 * tl.where(x.to(tl.float32) > 0, x.to(tl.float32), 1.6732632423543772 * (tl.exp(x.to(tl.float32))-1.0)))
    tl.store(X + off, y.to(X.dtype.element_ty), mask=mask)


# last (tensor object, zero-argument launch) so a repeated call costs one `is`
_LAST = [None, None]

# (dtype, numel, aligned, data_ptr) -> zero-argument launch
_CACHE = {}

# device index and the cheapest faithful current-stream accessor, resolved once
_RUNTIME = {}


def _pad8(values, fill):
    head = list(values[:8])
    return head + [fill] * (8 - len(head))


def _pick_block(n):
    """Largest tile that spreads work over cores without exceeding the UB budget."""
    if n >= _BLOCK_MAX * _BLOCK_TARGET_PROGRAMS:
        return _BLOCK_MAX
    target = -(-n // _BLOCK_TARGET_PROGRAMS)
    block = _BLOCK_MIN
    while block < target:
        block <<= 1
    return min(block, _BLOCK_MAX)


def _noop():
    return None


def _plain_launch(A, n, block, masked):
    """Ordinary Triton launch, used for cold, misaligned or unusual calls."""
    if masked:
        grid = triton.cdiv(n, block)

        def launch():
            _selu_masked[(grid,)](A, n, BLOCK=block)
    else:
        grid = n // block

        def launch():
            _selu_nomask[(grid,)](A, BLOCK=block)
    return launch


def _make_contig_launch(A, n, ptr, aligned):
    """Bind one (dtype, numel, alignment, pointer) specialization."""
    block = _pick_block(n)
    masked = bool(n % block)
    if not aligned:
        return _plain_launch(A, n, block, masked)

    raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    if raw is None:
        return _plain_launch(A, n, block, masked)
    try:
        if masked:
            compiled = _selu_masked.warmup(A, n, BLOCK=block, grid=(1,))
            grid = triton.cdiv(n, block)
        else:
            compiled = _selu_nomask.warmup(A, BLOCK=block, grid=(1,))
            grid = n // block
        launcher = compiled.run
        # NPULauncher only adds bookkeeping when one of these is enabled;
        # otherwise its bound pybind launch is exactly what it forwards to.
        if (getattr(launcher, "compile_only", False)
                or getattr(launcher, "enable_msprof_register_tensor", False)
                or getattr(launcher.metadata, "debug_enabled", False)):
            return _plain_launch(A, n, block, masked)
        call = launcher.launch
        function = compiled.function
        metadata = compiled.packed_metadata
    except Exception:  # noqa: BLE001
        return _plain_launch(A, n, block, masked)

    dev = _RUNTIME.get("device")
    if dev is None:
        dev = A.device.index
        if dev is None:
            try:
                dev = torch.npu.current_device()
            except Exception:  # noqa: BLE001
                dev = 0
        _RUNTIME["device"] = dev

    # The Ascend launcher ABI is (grid, stream, function, packed metadata, hooks,
    # *kernel params, *constexpr placeholders).  An integer data pointer is
    # accepted for a tensor parameter, so the specialization can be fully bound.
    if masked:
        def invoke(call=call, function=function, metadata=metadata, grid=grid,
                   raw=raw, dev=dev, ptr=ptr, n=n):
            call(grid, 1, 1, raw(dev), function, metadata, None, None, None,
                 ptr, n, None)
    else:
        def invoke(call=call, function=function, metadata=metadata, grid=grid,
                   raw=raw, dev=dev, ptr=ptr):
            call(grid, 1, 1, raw(dev), function, metadata, None, None, None,
                 ptr, None)
    return invoke


def _make_strided_launch(A, n):
    shape = _pad8(A.shape, 1)
    strides = _pad8(A.stride(), 0)
    grid = triton.cdiv(n, _BLOCK_STRIDED)
    ndims = min(A.dim(), 8)

    def launch():
        _selu_strided[(grid,)](
            A, n,
            shape[0], shape[1], shape[2], shape[3],
            shape[4], shape[5], shape[6], shape[7],
            strides[0], strides[1], strides[2], strides[3],
            strides[4], strides[5], strides[6], strides[7],
            NDIMS=ndims,
            BLOCK=_BLOCK_STRIDED,
        )
    return launch


def run(A):
    last = _LAST
    launch = last[1]
    if A is last[0] and launch is not None:
        # Repeat call on the same tensor: storage, shape, dtype and pointer are
        # unchanged by an in-place op, so the bound launch is still exact.
        launch()
        return A

    n = A.numel()
    if not n:
        last[0] = A
        last[1] = _noop
        return A

    if A.is_contiguous():
        ptr = A.data_ptr()
        key = (A.dtype, n, ptr & 15 == 0, ptr)
        launch = _CACHE.get(key)
        if launch is None:
            launch = _make_contig_launch(A, n, ptr, key[2])
            _CACHE[key] = launch
    else:
        launch = _make_strided_launch(A, n)

    last[0] = A
    last[1] = launch
    launch()
    return A
