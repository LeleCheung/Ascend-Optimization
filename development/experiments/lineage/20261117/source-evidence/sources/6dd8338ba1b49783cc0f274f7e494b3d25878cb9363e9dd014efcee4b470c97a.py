# Copyright 2026, The FlagOS Contributors.
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

"""In-place negation (``negative_`` / ``neg_``) as a Triton kernel.

Semantics preserved from the reference ``a.neg_()``:
  * mutates ``A``'s storage in place and returns the same tensor object,
  * IEEE negation for every element (NaN -> NaN, inf -> -inf, -0.0 -> 0.0),
  * non-contiguous (strided) views are negated through their real strides,
    so no neighbouring storage (garbage) is touched.

Host path: the whole operator is one device kernel, so the measured cost of
small tensors is the Triton *launch* cost, not the device work.  ``run`` keeps
a per-specialisation handle to the already compiled kernel and launches it
through Triton's own compiled-kernel entry point, which skips the per-call
Python argument binding / cache lookup of the ``kernel[grid](...)`` wrapper.
Every launch still goes through the same Triton launcher, stream and grid; the
device kernel is unchanged.  If the direct entry point is unavailable for any
reason the standard ``kernel[grid](...)`` call is used instead.
"""

import torch
import triton
import triton.language as tl

try:
    from triton.runtime import driver as _triton_driver
except Exception:  # pragma: no cover - standard launch path still works
    _triton_driver = None


@triton.jit
def _neg_exact(X, BLOCK: tl.constexpr):
    """Contiguous negation of one tile; the grid covers the tensor exactly."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(X + o, -tl.load(X + o))


@triton.jit
def _neg_tail(X, n, BLOCK: tl.constexpr):
    """Contiguous negation of one tile with a tail mask."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    tl.store(X + o, -tl.load(X + o, mask=m, other=0.0), mask=m)


@triton.jit
def _neg_strided(X, n, d1, d2, d3, s0, s1, s2, s3, RANK: tl.constexpr, BLOCK: tl.constexpr):
    """Negation of a non-contiguous view, addressed through collapsed strides."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < n
    if RANK == 1:
        off = idx * s0
    elif RANK == 2:
        r0 = idx // d1
        off = r0 * s0 + (idx - r0 * d1) * s1
    elif RANK == 3:
        r0 = idx // (d1 * d2)
        rem = idx - r0 * (d1 * d2)
        r1 = rem // d2
        off = r0 * s0 + r1 * s1 + (rem - r1 * d2) * s2
    else:
        p1 = d1 * d2 * d3
        r0 = idx // p1
        rem0 = idx - r0 * p1
        r1 = rem0 // (d2 * d3)
        rem1 = rem0 - r1 * (d2 * d3)
        r2 = rem1 // d3
        off = r0 * s0 + r1 * s1 + r2 * s2 + (rem1 - r2 * d3) * s3
    tl.store(X + off, -tl.load(X + off, mask=m, other=0.0), mask=m)


# ---------------------------------------------------------------------------
# launch plumbing
# ---------------------------------------------------------------------------

_ENTRIES = {}          # per-specialisation compiled-kernel handles
_DEVICE = [None]
_DIRECT = [True]       # False as soon as the direct entry point is unusable


def _device_index():
    dev = _DEVICE[0]
    if dev is None:
        dev = _triton_driver.active.get_current_device()
        _DEVICE[0] = dev
    return dev


def _entry(kernel, key, t, rt, ce_names, ce_vals):
    """Compile (cached) and keep the launcher handles for one specialisation."""
    entry = _ENTRIES.get(key)
    if entry is None:
        try:
            compiled = kernel.warmup(
                t, *rt, grid=(1,), **dict(zip(ce_names, ce_vals))
            )
            entry = (compiled.run, compiled.function, compiled.packed_metadata,
                     tuple(rt) + tuple(ce_vals))
        except Exception:
            entry = False
        _ENTRIES[key] = entry
    return entry


def _direct_launch(key, kernel, t, rt, ce_names, ce_vals, grid):
    """Launch the precompiled kernel; return False when the ordinary path must be used."""
    if not _DIRECT[0] or _triton_driver is None:
        return False
    entry = _entry(kernel, key, t, rt, ce_names, ce_vals)
    if entry is False:
        return False
    launch, function, packed, tail = entry
    try:
        stream = _triton_driver.active.get_current_stream(_device_index())
        launch(grid, 1, 1, stream, function, packed, None, None, None, t, *tail)
        return True
    except Exception:
        _DIRECT[0] = False
        return False


def _collapse(t):
    """Collapse a shape/stride list to the minimal equivalent (size, stride) pairs."""
    dims = [(sz, st) for sz, st in zip(t.shape, t.stride()) if sz != 1]
    out = []
    for sz, st in reversed(dims):
        if out and out[-1][1] == st * sz:
            out[-1] = (out[-1][0] * sz, st)
        else:
            out.append((sz, st))
    out.reverse()
    return out


def _launch_strided(t):
    dims = _collapse(t)
    if len(dims) > 4:
        for i in range(t.shape[0]):
            _launch_strided(t[i])
        return
    n = t.numel()
    if n == 0:
        return
    pad = dims + [(1, 0)] * (4 - len(dims))
    block = 1024
    grid = ((n + block - 1) // block,)
    _neg_strided[grid](
        t, n,
        pad[1][0], pad[2][0], pad[3][0],
        pad[0][1], pad[1][1], pad[2][1], pad[3][1],
        RANK=len(dims), BLOCK=block,
    )


def _tile(itemsize):
    """One 64 KiB tile per program: the largest element-wise tile that keeps the
    staged input and output buffers inside the vector core's unified buffer."""
    return 65536 // itemsize


def run(A):
    if not A.is_contiguous():
        _launch_strided(A)
        return A
    n = A.numel()
    if n == 0:
        return A
    if n <= 4096:
        block = 1 << max(0, (n - 1).bit_length())
        grid = 1
    else:
        block = _tile(A.element_size())
        grid = (n + block - 1) // block
    if block == n or (n > 4096 and n % block == 0):
        key = ("e", A.dtype, block)
        if _direct_launch(key, _neg_exact, A, (), ("BLOCK",), (block,), grid):
            return A
        _neg_exact[(grid,)](A, BLOCK=block)
    else:
        key = ("t", A.dtype, n, block)
        if _direct_launch(key, _neg_tail, A, (n,), ("BLOCK",), (block,), grid):
            return A
        _neg_tail[(grid,)](A, n, BLOCK=block)
    return A
