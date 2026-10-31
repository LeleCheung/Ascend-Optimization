"""Second bounded probe: cheapest output allocation, launch-path decomposition,
and remaining tile choices for xlogy_scalar_tensor.
"""
import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

OUT = {}
MS = 1e-3
ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")


def dump():
    try:
        with open(os.path.join(ART, "probe2.json"), "w") as fh:
            json.dump(OUT, fh, indent=2, default=str)
    except Exception as exc:
        print("artifact write failed: %s" % exc)


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters / MS


raw = torch_npu._C._npu_getCurrentRawStreamNoWait
OUT["stream_handle"] = int(raw(0))


def sh():
    return raw(0)


# --------------------------------------------------------------- allocation forms
n = 1024
y = (torch.rand(n, device="npu") + 0.05).to(torch.float16)
shape = y.shape
dt = y.dtype
dev = y.device
opts = dict(dtype=dt, device=dev)
big = (torch.rand(4096 * 4096, device="npu") + 0.05).to(torch.float16)
bign = big.numel()
bigdt = big.dtype
bigdev = big.device

OUT["alloc"] = {
    "empty_shape": bench(lambda: torch.empty(shape, **opts)),
    "empty_int": bench(lambda: torch.empty(n, **opts)),
    "empty_int_pos": bench(lambda: torch.empty(n, dt, dev)),
    "empty_like": bench(lambda: torch.empty_like(y)),
    "empty_like_cf": bench(lambda: torch.empty_like(y, memory_format=torch.contiguous_format)),
    "new_empty": bench(lambda: y.new_empty(shape)),
    "aten_empty": bench(lambda: torch.ops.aten.empty.memory_format(
        [n], dtype=dt, device=dev, pin_memory=False)),
    "empty_strided": bench(lambda: torch.empty_strided(shape, (1,), **opts)),
    "big_empty_shape": bench(lambda: torch.empty(big.shape, dtype=bigdt, device=bigdev), iters=20),
    "big_empty_int": bench(lambda: torch.empty(bign, dtype=bigdt, device=bigdev), iters=20),
    "big_empty_like": bench(lambda: torch.empty_like(big), iters=20),
}
dump()


# --------------------------------------------------------------- kernels
@triton.jit
def k_masked(Y, OUTP, n, s, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    v = s * tl.log(tl.load(Y + offs, mask=m, other=1.0).to(tl.float32))
    tl.store(OUTP + offs, v.to(OUTP.dtype.element_ty), mask=m)


@triton.jit
def k_flat(Y, OUTP, s, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = s * tl.log(tl.load(Y + offs).to(tl.float32))
    tl.store(OUTP + offs, v.to(OUTP.dtype.element_ty))


@triton.jit
def k_flat_hint(Y, OUTP, s, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    v = s * tl.log(tl.load(Y + offs).to(tl.float32))
    tl.store(OUTP + offs, v.to(OUTP.dtype.element_ty))


# --------------------------------------------------------------- launch path parts
o = torch.empty_like(y)
ck = k_masked.warmup(y, o, n, 2.0, BLOCK=1024, grid=(1,))
la = ck.run
OUT["launcher_type"] = type(la).__name__
OUT["launcher_dir"] = [a for a in dir(la) if not a.startswith("_")]
OUT["launch_doc"] = (getattr(la.launch, "__doc__", "") or "")[:400]
fn_c, pm = ck.function, ck.packed_metadata
p1, p2 = y.data_ptr(), o.data_ptr()
g = 1
s_ = 2.0
b = 1024
head = (g, 1, 1)
mid = (fn_c, pm, None, None, None)
tail = (n, s_, b)

OUT["launch"] = {
    "full": bench(lambda: la.launch(g, 1, 1, sh(), fn_c, pm, None, None, None, p1, p2, n, s_, b)),
    "full_star": bench(lambda: la.launch(*head, sh(), *mid, p1, p2, *tail)),
    "only_launch_const_ptrs": bench(lambda: la.launch(g, 1, 1, OUT["stream_handle"], fn_c, pm,
                                                   None, None, None, p1, p2, n, s_, b)),
    "stream_only": bench(sh),
    "ptrs_only": bench(lambda: (y.data_ptr(), o.data_ptr())),
    "numel_iscontig_dtype": bench(lambda: (y.numel(), y.is_contiguous(), y.dtype)),
    "get_device": bench(lambda: y.get_device()),
    "dict_lookup_tuple": bench(lambda: d.get((dt, n, 0))),
    "torch_xlogy_1024": bench(lambda: torch.xlogy(2.0, y)),
}
dump()


# --------------------------------------------------------------- tile choices
PLAN = [
    (torch.float32, 4096 * 4096, [("flat", 8192), ("flat", 16384), ("masked", 16384),
                                  ("hint", 8192)]),
    (torch.float16, 4096 * 4096, [("flat", 16384), ("flat", 32768)]),
    (torch.float16, 1024, [("masked", 1024), ("masked", 2048), ("masked", 4096)]),
    (torch.float32, 65536, [("masked", 4096), ("masked", 8192), ("masked", 16384)]),
]
KER = {"flat": k_flat, "masked": k_masked, "hint": k_flat_hint}

dev_entries = {}
for dtype, nn, variants in PLAN:
    yy = (torch.rand(nn, device="npu") + 0.05).to(dtype)
    oo = torch.empty_like(yy)
    ref = torch.xlogy(2.0, yy)
    tag = "%s_%d" % (str(dtype).split(".")[-1], nn)
    entry = {"ref": bench(lambda: torch.xlogy(2.0, yy), warmup=5, iters=30)}
    for kind, bb in variants:
        masked = (kind == "masked") or bool(nn % bb)
        try:
            if masked:
                ckc = k_masked.warmup(yy, oo, nn, 2.0, BLOCK=bb, grid=(1,))
                tl_ = (nn, 2.0, bb)
            else:
                ckc = KER[kind].warmup(yy, oo, 2.0, BLOCK=bb, grid=(1,))
                tl_ = (2.0, bb)
            laa = ckc.run
            f2, p2m = ckc.function, ckc.packed_metadata
            gg = triton.cdiv(nn, bb)
            if masked:
                def call(laa=laa, f2=f2, p2m=p2m, gg=gg, tl_=tl_, yy=yy, oo=oo):
                    laa.launch(gg, 1, 1, sh(), f2, p2m, None, None, None,
                               yy.data_ptr(), oo.data_ptr(), *tl_)
            else:
                def call(laa=laa, f2=f2, p2m=p2m, gg=gg, tl_=tl_, yy=yy, oo=oo):
                    laa.launch(gg, 1, 1, sh(), f2, p2m, None, None, None,
                               yy.data_ptr(), oo.data_ptr(), *tl_)
            oo.zero_()
            call()
            torch.npu.synchronize()
            err = (oo.float() - ref.float()).abs()
            key = "%s_b%d%s" % (kind, bb, "m" if masked else "")
            entry[key] = bench(call, warmup=5, iters=30)
            entry[key + "_err"] = float(err.nan_to_num(1e9).max().item())
        except Exception as exc:
            entry["%s_b%d" % (kind, bb)] = "ERR:%s" % str(exc)[:200]
    dev_entries[tag] = entry
    OUT["dev"] = dev_entries
    dump()

print(json.dumps(OUT, indent=1, default=str))
