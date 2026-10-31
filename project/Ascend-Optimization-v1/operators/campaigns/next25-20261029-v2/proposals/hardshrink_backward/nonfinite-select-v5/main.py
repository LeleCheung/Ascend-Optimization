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

import torch
import triton
import triton.language as tl

_WARPS = 4
_BASE = 1024          # small / tail tile
_CAP_2B = 8192       # 32 KiB staged tile for 2-byte dtypes
_CAP_4B = 4096        # 32 KiB staged tile for 4-byte dtypes

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_CACHE = {}
_RAW_GETTER = None
_RAW_PROBED = False


# --------------------------------------------------------------------------- kernels
@triton.jit
def _hsb_flat(go_ptr, x_ptr, out_ptr, lambd, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    go = tl.load(go_ptr + offs)
    xx = tl.load(x_ptr + offs)
    tl.store(
        out_ptr + offs,
        tl.where(tl.abs(xx) <= lambd, tl.where((tl.abs(go.to(tl.float32)) == float("inf")) | (go != go), float("nan"), 0.0), go.to(tl.float32)),
    )


@triton.jit
def _hsb_masked(go_ptr, x_ptr, out_ptr, n, lambd, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    go = tl.load(go_ptr + offs, mask=m, other=0)
    xx = tl.load(x_ptr + offs, mask=m, other=0)
    tl.store(
        out_ptr + offs,
        tl.where(tl.abs(xx) <= lambd, tl.where((tl.abs(go.to(tl.float32)) == float("inf")) | (go != go), float("nan"), 0.0), go.to(tl.float32)),
        mask=m,
    )


@triton.jit
def _hsb_scaled(go_ptr, x_ptr, out_ptr, n, ss, lambd, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    go = tl.load(go_ptr + offs, mask=m, other=0)
    xx = tl.load(x_ptr + offs * ss, mask=m, other=0)
    tl.store(
        out_ptr + offs,
        tl.where(tl.abs(xx) <= lambd, tl.where((tl.abs(go.to(tl.float32)) == float("inf")) | (go != go), float("nan"), 0.0), go.to(tl.float32)),
        mask=m,
    )


# --------------------------------------------------------------------------- host
def _resolve_raw():
    """Raw current-stream accessor (~0.35 us) instead of torch.npu.current_stream() (~23 us)."""
    global _RAW_GETTER, _RAW_PROBED
    if _RAW_PROBED:
        return _RAW_GETTER
    _RAW_PROBED = True
    try:
        import torch_npu

        raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
        if raw is not None:
            dev = torch.npu.current_device()
            if int(raw(dev)) == int(torch.npu.current_stream().npu_stream):
                _RAW_GETTER = raw
    except Exception:
        _RAW_GETTER = None
    return _RAW_GETTER


def _stream_getter(dev):
    raw = _resolve_raw()
    if raw is not None:
        return lambda: raw(dev)
    return lambda: torch.npu.current_stream().npu_stream


def _uniform_stride(t):
    """Single element stride of a view whose dims collapse into one axis, else None."""
    shape = t.shape
    stride = t.stride()
    dims = []
    for d in range(len(shape) - 1, -1, -1):
        s = shape[d]
        if s == 1:
            continue
        st = stride[d]
        if dims and st == dims[-1][0] * dims[-1][1]:
            dims[-1] = (dims[-1][0] * s, dims[-1][1])
        else:
            dims.append((s, st))
    if not dims:
        return 1
    if len(dims) == 1:
        return dims[0][1]
    return None


def _build(n, esize, sample, dev):
    """Resolve (compile once) the launcher for this (numel, dtype-size, device)."""
    cap = _CAP_4B if esize > 2 else _CAP_2B
    if n % cap == 0:
        jitfn, block, grid, extra = _hsb_flat, cap, n // cap, ()
    else:
        jitfn, block, extra = _hsb_masked, _BASE, (n,)
        grid = -(-n // _BASE)
    lam0 = 0.5
    stream_of = _stream_getter(dev)

    def jit_call(g, x, o, lam, _f=jitfn, _b=block, _e=extra):
        if _e:
            _f[(_e[0] + _b - 1) // _b,](g, x, o, _e[0], lam, BLOCK=_b, num_warps=_WARPS)
        else:
            _f[(g.numel() // _b,)](g, x, o, lam, BLOCK=_b, num_warps=_WARPS)

    try:
        if extra:
            ck = jitfn.warmup(
                sample, sample, sample, extra[0], lam0,
                BLOCK=block, num_warps=_WARPS, grid=(1,),
            )
        else:
            ck = jitfn.warmup(
                sample, sample, sample, lam0,
                BLOCK=block, num_warps=_WARPS, grid=(1,),
            )
        launcher = ck.run
        md = launcher.metadata
        if (
            getattr(launcher, "compile_only", False)
            or getattr(launcher, "enable_msprof_register_tensor", False)
            or getattr(md, "debug_enabled", False)
            or getattr(md, "debug", False)
        ):
            return jit_call
        L, fn, pm = launcher.launch, ck.function, ck.packed_metadata
    except Exception:
        return jit_call

    if extra:
        n0 = extra[0]

        def call(g, x, o, lam,
                 _L=L, _fn=fn, _pm=pm, _g=grid, _sg=stream_of, _n=n0, _b=block,
                 _f=jit_call):
            gp = g.data_ptr()
            xp = x.data_ptr()
            op = o.data_ptr()
            if (gp | xp | op) & 15:
                _f(g, x, o, lam)
                return
            _L(_g, 1, 1, _sg(), _fn, _pm, None, None, None, gp, xp, op, _n, lam, _b)
    else:

        def call(g, x, o, lam,
                 _L=L, _fn=fn, _pm=pm, _g=grid, _sg=stream_of, _b=block,
                 _f=jit_call):
            gp = g.data_ptr()
            xp = x.data_ptr()
            op = o.data_ptr()
            if (gp | xp | op) & 15:
                _f(g, x, o, lam)
                return
            _L(_g, 1, 1, _sg(), _fn, _pm, None, None, None, gp, xp, op, lam, _b)

    return call


def _dispatch(grad_out, self_t, out, n, lambd):
    """Write the hardshrink_backward result of (grad_out, self_t) into ``out``."""
    key = (n, grad_out.dtype, grad_out.get_device())
    call = _CACHE.get(key)
    if call is None:
        call = _build(n, out.element_size(), out, key[2])
        _CACHE[key] = call
    call(grad_out, self_t, out, lambd)


def _layout_fallback(grad_out, self_t, out, n, lambd):
    """Rare layouts that the flat/stride kernels cannot express (never in contract cases)."""
    go = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
    ss = _uniform_stride(self_t)
    if ss is None:
        xx = self_t if self_t.is_contiguous() else self_t.contiguous()
        ss = 1
    else:
        xx = self_t
    grid = -(-n // _BASE)
    if ss == 1:
        _hsb_masked[(grid,)](go, xx, out, n, lambd, BLOCK=_BASE, num_warps=_WARPS)
    else:
        _hsb_scaled[(grid,)](go, xx, out, n, int(ss), lambd, BLOCK=_BASE, num_warps=_WARPS)


def hardshrink_backward(grad_out: torch.Tensor, self_t: torch.Tensor, lambd=0.5):
    dt = grad_out.dtype
    if dt not in _DTYPES:
        raise TypeError(
            f"Unsupported dtype {dt}. Supported dtypes are float16, bfloat16, and float32."
        )
    if dt is not self_t.dtype:
        raise TypeError(
            f"Dtype mismatch: grad_out.dtype={dt}, self.dtype={self_t.dtype}"
        )
    if self_t.shape != grad_out.shape:
        raise ValueError(
            f"Shape mismatch: grad_out.shape={grad_out.shape}, self.shape={self_t.shape}"
        )

    n = grad_out.numel()
    out = torch.empty_like(grad_out)
    if n == 0:
        return out
    lam = float(lambd)
    if grad_out.is_contiguous() and self_t.is_contiguous():
        _dispatch(grad_out, self_t, out, n, lam)
    else:
        _layout_fallback(grad_out, self_t, out, n, lam)
    return out


def hardshrink_backward_grad_input(
    grad_out: torch.Tensor, self_t: torch.Tensor, lambd=0.5, *, grad_input=None
):
    if grad_input is None:
        raise ValueError(
            "Argument 'grad_input' must be provided for hardshrink_backward.grad_input."
        )
    if self_t.shape != grad_out.shape or grad_input.shape != grad_out.shape:
        raise ValueError(
            f"Shape mismatch: grad_out.shape={grad_out.shape}, self.shape={self_t.shape}, "
            f"grad_input.shape={grad_input.shape}"
        )
    grad_input.copy_(hardshrink_backward(grad_out, self_t, lambd))
    return grad_input


run = hardshrink_backward
