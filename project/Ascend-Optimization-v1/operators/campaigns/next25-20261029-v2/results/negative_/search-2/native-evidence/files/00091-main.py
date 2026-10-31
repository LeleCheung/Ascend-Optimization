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

The whole operator is a single device kernel, so for tensors up to a few MiB
the measured cost is the *launch* cost rather than the device work.  ``run``
therefore keeps one pre-bound launcher closure per (dtype, element count):

  * the kernel is compiled once (``warmup``) and launched through Triton's own
    compiled-kernel entry point, so no per-call argument binding or kernel
    cache lookup happens;
  * the tile pointer is passed as its integer address, which skips the
    launcher's per-pointer attribute query;
  * the raw stream handle is read from the current torch stream, validated once
    against the Triton driver;
  * the grid and the trailing arguments are captured in the closure.

Device kernels, tiling and the strided path are unchanged by this; if any part
of the fast path is unavailable the ordinary ``kernel[grid](...)`` launch is
used and stays the semantic reference.
"""

import torch
import triton
import triton.language as tl

try:
    from triton.runtime import driver as _triton_driver
except Exception:  # pragma: no cover - ordinary launch path still works
    _triton_driver = None


# ---------------------------------------------------------------------------
# device kernels
# ---------------------------------------------------------------------------

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
# tiling
# ---------------------------------------------------------------------------

def _tile(itemsize):
    """One 64 KiB tile per program: the largest element-wise tile that keeps a
    staged input and output buffer inside the vector core's unified buffer."""
    return 65536 // itemsize


def _geometry(element_size, n):
    """(tile elements, program count, exact cover) for a contiguous tensor."""
    if n <= 4096:
        block = 1 << max(0, (n - 1).bit_length())
        return block, 1, block == n
    block = _tile(element_size)
    return block, (n + block - 1) // block, n % block == 0


# ---------------------------------------------------------------------------
# host launch path
# ---------------------------------------------------------------------------

_PLANS = {}     # (dtype, numel) -> pre-bound launcher closure, or False
_STREAM = []    # one-time validated stream reader


def _torch_stream():
    return torch.npu.current_stream().npu_stream


def _stream_reader():
    """Read the raw current stream handle as cheaply as possible.

    ``torch.npu.current_stream().npu_stream`` is the stream torch is currently
    submitting to; it is compared once against the Triton driver's value so a
    different torch_npu build falls back to the driver call instead.
    """
    if _STREAM:
        return _STREAM[0]
    reader = False
    if _triton_driver is not None:
        try:
            driver = _triton_driver.active
            device = driver.get_current_device()
            if torch.npu.current_stream().npu_stream == driver.get_current_stream(device):
                reader = _torch_stream
            else:
                def reader(_driver=driver, _device=device):
                    return _driver.get_current_stream(_device)
        except Exception:
            reader = False
    _STREAM.append(reader)
    return reader


def _plan(t, n):
    """Compile once and capture everything the launch needs for this geometry."""
    reader = _stream_reader()
    if reader is False:
        return False
    try:
        if t.device.type != "npu":
            return False
        block, grid, exact = _geometry(t.element_size(), n)
        if exact:
            compiled = _neg_exact.warmup(t, grid=(1,), BLOCK=block)
            tail = (block,)
        else:
            compiled = _neg_tail.warmup(t, n, grid=(1,), BLOCK=block)
            tail = (n, block)
        launch = compiled.run
        function = compiled.function
        packed = compiled.packed_metadata
    except Exception:
        return False

    def call(ptr, _launch=launch, _reader=reader, _function=function,
             _packed=packed, _tail=tail, _grid=grid):
        _launch(_grid, 1, 1, _reader(), _function, _packed,
                None, None, None, ptr, *_tail)

    return call


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


def run(A):
    if not A.is_contiguous():
        _launch_strided(A)
        return A
    n = A.numel()
    if n == 0:
        return A
    key = (A.dtype, n)
    call = _PLANS.get(key)
    if call is None:
        call = _plan(A, n)
        _PLANS[key] = call
    if call is not False:
        try:
            call(A.data_ptr())
            return A
        except Exception:
            _PLANS[key] = False
    block, grid, exact = _geometry(A.element_size(), n)
    if exact:
        _neg_exact[(grid,)](A, BLOCK=block)
    else:
        _neg_tail[(grid,)](A, n, BLOCK=block)
    return A
