"""Bounded probe 3: decompose the per-call cost of the hot path and inspect the
backend launcher for a leaner entry point."""
import inspect
import json
import os
import sys
import time
import traceback

import torch
import torch_npu
import triton
import triton.language as tl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import main as cand  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {}


def flush():
    with open(os.path.join(ART, "probe3.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)


dev = torch.npu.current_device()


def stream_handle(device):
    fn = torch_npu._C._npu_getCurrentRawStreamNoWait
    return fn(device)


def bench_us(fn, iters=300, warmup=30, sync_each=False):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    if sync_each:
        for _ in range(iters):
            fn()
            torch.npu.synchronize()
    else:
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


@triton.jit
def k5(A, B, alpha, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    a = tl.load(A + offs, mask=m, other=0.0)
    b = tl.load(B + offs, mask=m, other=0.0)
    tl.store(A + offs, (a.to(tl.float32) - alpha * b.to(tl.float32)).to(A.dtype.element_ty), mask=m)


@triton.jit
def k2(A, B, ALPHA: tl.constexpr, BLOCK: tl.constexpr):
    """Pointer-only signature: alpha and block folded into the binary."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offs)
    b = tl.load(B + offs)
    tl.store(A + offs, (a.to(tl.float32) - ALPHA * b.to(tl.float32)).to(A.dtype.element_ty))


def section_decompose():
    sec = {}
    A = torch.randn(1024, dtype=torch.float16, device="npu:0")
    B = torch.randn(1024, dtype=torch.float16, device="npu:0")
    A32 = torch.randn(1024, dtype=torch.float32, device="npu:0")
    B32 = torch.randn(1024, dtype=torch.float32, device="npu:0")
    n = A.numel()

    ck = k5.warmup(A, B, 0.5, n, BLOCK=4096, grid=(1,))
    l5 = ck.run.launch
    gx = (n + 4095) // 4096
    stream = stream_handle(dev)
    pa, pb = A.data_ptr(), B.data_ptr()

    def launch5():
        l5(gx, 1, 1, stream, ck.function, ck.packed_metadata,
           None, None, None, pa, pb, 0.5, n, 4096)

    ck2 = k2.warmup(A, B, 0.5, BLOCK=4096, grid=(1,))
    l2 = ck2.run.launch

    def launch2():
        l2(gx, 1, 1, stream, ck2.function, ck2.packed_metadata,
           None, None, None, pa, pb, 4096)

    def python_prelude():
        nn = A.numel()
        d = A.device.index
        e = cand._ENTRIES.get((A.dtype, nn, d))
        ca = A.is_contiguous()
        cb = B.is_contiguous()
        eq = A.shape == B.shape
        qa = A.data_ptr()
        qb = B.data_ptr()
        al = (qa | qb) & 15
        st = stream_handle(d)
        return e, ca, cb, eq, al, st

    sec["launch5_us"] = bench_us(launch5)
    sec["launch2_args_us"] = bench_us(launch2)
    sec["launch5_sync_each_us"] = bench_us(launch5, iters=200, sync_each=True)
    sec["launch2_sync_each_us"] = bench_us(launch2, iters=200, sync_each=True)
    sec["python_prelude_us"] = bench_us(python_prelude)
    sec["stream_only_us"] = bench_us(lambda: stream_handle(dev))
    sec["device_index_us"] = bench_us(lambda: A.device.index)
    sec["numel_us"] = bench_us(lambda: A.numel())
    sec["is_contig_x2_us"] = bench_us(lambda: (A.is_contiguous(), B.is_contiguous()))
    sec["shape_eq_us"] = bench_us(lambda: A.shape == B.shape)
    sec["data_ptr_x2_us"] = bench_us(lambda: (A.data_ptr(), B.data_ptr()))
    sec["dict_get_us"] = bench_us(lambda: cand._ENTRIES.get((A.dtype, 1024, 0)))
    sec["sync_only_us"] = bench_us(lambda: torch.npu.synchronize(), iters=200)
    sec["noop_us"] = bench_us(lambda: None)
    sec["cand_run_f16_us"] = bench_us(lambda: cand.run(A, B, alpha=0.5))
    sec["cand_run_f16_sync_each_us"] = bench_us(lambda: cand.run(A, B, alpha=0.5),
                                                iters=200, sync_each=True)
    sec["cand_run_f32_us"] = bench_us(lambda: cand.run(A32, B32, alpha=0.5))
    sec["torch_ref_f16_us"] = bench_us(lambda: A.sub_(B, alpha=0.5))
    sec["torch_ref_f16_sync_each_us"] = bench_us(lambda: A.sub_(B, alpha=0.5),
                                                 iters=200, sync_each=True)
    sec["torch_ref_f32_us"] = bench_us(lambda: A32.sub_(B32, alpha=0.5))
    # correctness of the 2-arg variant
    A.ref = None
    exp = A.clone().sub_(B, alpha=0.5)
    launch2()
    torch.npu.synchronize()
    sec["launch2_match"] = bool(torch.equal(exp, A))
    OUT["decompose"] = sec


try:
    section_decompose()
except Exception:
    OUT["decompose_error"] = traceback.format_exc()[-1500:]
flush()

# ---------------------------------------------------------- launcher source
src_info = {}
try:
    import triton.backends.ascend as _ba
    import triton.spec.ascend as _sa
    roots = [os.path.dirname(m.__file__) for m in (_ba, _sa) if m is not None]
    hits = []
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    txt = open(p, "r", errors="replace").read()
                except Exception:
                    continue
                if "class NPULauncher" in txt or "def launch(" in txt:
                    hits.append(p)
    src_info["candidate_files"] = hits[:20]
    for p in hits[:20]:
        txt = open(p, "r", errors="replace").read()
        if "class NPULauncher" in txt:
            i = txt.index("class NPULauncher")
            src_info["npu_launcher_source"] = txt[i:i + 6000]
            break
except Exception:
    src_info["error"] = traceback.format_exc()[-1500:]
OUT["launcher_source"] = src_info
flush()
print(json.dumps(OUT, indent=1, default=str))
