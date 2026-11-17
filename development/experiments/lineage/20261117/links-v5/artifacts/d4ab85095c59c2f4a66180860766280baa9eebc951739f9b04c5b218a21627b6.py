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

The op is a single elementwise pass, so per-call latency on 910B is dominated by
the host launch path (measured ~52us through the FlagTree JIT wrapper versus
~10us for the framework reference), not by the kernel.  Each (dtype, numel)
specialization is therefore compiled once and afterwards launched through the
already-bound compiled kernel with the current stream, which keeps the per-call
cost to the compiled launcher itself (~13us).  Cold, misaligned and
non-contiguous calls use ordinary Triton launches.
"""

import torch
import torch_npu  # noqa: F401  (provides the raw current-stream query)
import triton
import triton.language as tl

_LOG10E = 0.4342944819032518

# 32768 elements per program keeps per-program dispatch overhead amortized on the
# large shapes (16384 programs at 1024 elements cost ~0.6ms on 4096x4096); larger
# tiles exceed the 192KB UB budget once the fp32 compute buffer is materialized.
_BLOCK_MAX = 32768
_BLOCK_MIN = 1024
_BLOCK_TARGET_PROGRAMS = 8
_BLOCK_STRIDED = 1024


@triton.jit
def _log10_nomask(X, BLOCK: tl.constexpr):
    """numel % BLOCK == 0 (host-guarded), so the tail mask can be dropped."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def _log10_masked(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty), mask=mask)


@triton.jit
def _log10_strided(
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
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + off, y.to(X.dtype.element_ty), mask=mask)


# per-(dtype, numel, alignment) launch entry
_CACHE = {}

# device index and cheapest faithful current-stream accessor, resolved once
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


def _init_runtime():
    """Resolve the device index and the cheapest faithful stream accessor once."""
    try:
        device = torch.npu.current_device()
    except Exception:  # noqa: BLE001
        device = 0
    getter = None
    try:
        # The no-wait variant follows stream switches exactly like the torch
        # accessor but does not stall behind already enqueued work: measured
        # ~0.5us per call versus ~20us for the waiting raw query.
        raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
        if raw is not None:
            current = int(torch.npu.current_stream().npu_stream)
            try:
                if int(raw(device)) == current:
                    getter = lambda: raw(device)  # noqa: E731
            except TypeError:
                if int(raw()) == current:
                    getter = raw
    except Exception:  # noqa: BLE001
        getter = None
    if getter is None:
        from triton.runtime import driver

        getter = lambda: driver.active.get_current_stream(device)  # noqa: E731
    _RUNTIME["device"] = device
    _RUNTIME["stream"] = getter
    return getter


def _standard_launch(n, block, masked):
    """Ordinary Triton launch, used for cold, misaligned or unusual calls."""
    if masked:
        def launch(X, n=n, block=block):
            _log10_masked[(triton.cdiv(n, block),)](X, X.numel(), BLOCK=block)
    else:
        def launch(X, n=n, block=block):
            _log10_nomask[(n // block,)](X, BLOCK=block)
    return launch


def _build_launch(A, n, aligned):
    block = _pick_block(n)
    masked = n % block != 0
    if not aligned:
        return _standard_launch(n, block, masked)
    try:
        if masked:
            compiled = _log10_masked.warmup(A, n, BLOCK=block, grid=(1,))
            grid = triton.cdiv(n, block)
        else:
            compiled = _log10_nomask.warmup(A, BLOCK=block, grid=(1,))
            grid = n // block
        launcher = compiled.run
        # NPULauncher only adds bookkeeping when one of these is enabled;
        # otherwise its bound pybind launch is exactly what it forwards to.
        if getattr(launcher, "compile_only", False) or getattr(
            launcher, "enable_msprof_register_tensor", False
        ):
            call = launcher
        else:
            call = launcher.launch
        function = compiled.function
        metadata = compiled.packed_metadata
    except Exception:  # noqa: BLE001
        return _standard_launch(n, block, masked)

    stream = _RUNTIME.get("stream")
    if stream is None:
        stream = _init_runtime()

    # The Ascend launcher ABI is (grid, stream, function, packed metadata, hooks,
    # *kernel params) with constexpr params passed as placeholders.
    if masked:
        def launch(X, grid=grid, call=call, function=function, metadata=metadata,
                   stream=stream):
            call(grid, 1, 1, stream(), function, metadata, None, None, None,
                 X, X.numel(), None)
    else:
        def launch(X, grid=grid, call=call, function=function, metadata=metadata,
                   stream=stream):
            call(grid, 1, 1, stream(), function, metadata, None, None, None,
                 X, None)
    return launch


def run(A):
    n = A.numel()
    if n == 0:
        return A
    if A.is_contiguous():
        key = (A.dtype, n, A.data_ptr() & 15 == 0)
        launch = _CACHE.get(key)
        if launch is None:
            launch = _build_launch(A, n, key[2])
            _CACHE[key] = launch
        launch(A)
    else:
        shape = _pad8(A.shape, 1)
        strides = _pad8(A.stride(), 0)
        _log10_strided[(triton.cdiv(n, _BLOCK_STRIDED),)](
            A,
            n,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            shape[4],
            shape[5],
            shape[6],
            shape[7],
            strides[0],
            strides[1],
            strides[2],
            strides[3],
            strides[4],
            strides[5],
            strides[6],
            strides[7],
            NDIMS=min(A.dim(), 8),
            BLOCK=_BLOCK_STRIDED,
        )
    return A
