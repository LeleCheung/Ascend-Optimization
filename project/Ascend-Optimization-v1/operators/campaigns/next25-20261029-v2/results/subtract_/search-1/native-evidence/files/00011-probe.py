"""Bounded target probe: launcher ABI, device properties, host-path overhead.

Not a measurement of record: only to choose the next candidate.
"""
import inspect
import json
import os
import sys
import time

import torch
import torch_npu
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import main as cand  # noqa: E402  tmp/main.py

OUT = {}
dev = torch.npu.current_device()
OUT["triton_version"] = getattr(triton, "__version__", "?")
OUT["torch_version"] = torch.__version__
try:
    OUT["device_properties"] = {k: v for k, v in
                                triton.runtime.driver.active.utils.get_device_properties(dev).items()}
except Exception as exc:  # pragma: no cover
    OUT["device_properties"] = "ERR %r" % (exc,)


def stream_handle(device):
    raw = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    if raw is not None:
        try:
            return raw(device)
        except TypeError:
            return raw()
    return torch.npu.current_stream(device).npu_stream


def bench_us(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def ref_sub(A, B, alpha=0.5):
    return A.sub_(B, alpha=alpha)


cases = {
    "f16_1024": lambda: (torch.randn(1024, dtype=torch.float16, device="npu:0"),
                         torch.randn(1024, dtype=torch.float16, device="npu:0")),
    "f32_1024": lambda: (torch.randn(1024, dtype=torch.float32, device="npu:0"),
                         torch.randn(1024, dtype=torch.float32, device="npu:0")),
    "f32_4096x4096": lambda: (torch.randn((4096, 4096), dtype=torch.float32, device="npu:0"),
                              torch.randn((4096, 4096), dtype=torch.float32, device="npu:0")),
    "f16_4096x4096": lambda: (torch.randn((4096, 4096), dtype=torch.float16, device="npu:0"),
                              torch.randn((4096, 4096), dtype=torch.float16, device="npu:0")),
}

# ---- Numerics of the current candidate vs the torch reference (incl. +/-inf) ----
numerics = {}
for name, mk in cases.items():
    A, B = mk()
    if name.endswith("1024") or name.endswith("4096"):
        A.flatten()[:7] = torch.tensor([0., -0., float("nan"), float("inf"),
                                        -float("inf"), -1., 1.],
                                       dtype=A.dtype, device="npu:0")
    Aref = A.clone()
    r_ref = ref_sub(Aref, B)
    r_cand = cand.run(A, B, alpha=0.5)
    same_obj = r_cand.data_ptr() == A.data_ptr()
    ok = torch.equal(r_ref.view(-1).float(), r_cand.view(-1).float()) and \
        torch.equal(torch.isnan(r_ref.view(-1)), torch.isnan(r_cand.view(-1)))
    numerics[name] = {"bitwise_equal_ref": bool(ok), "returns_same_storage": bool(same_obj),
                      "max_abs_diff": float((r_ref.float() - r_cand.float()).abs().max())}
    torch.npu.synchronize()

# strided case
Ax = torch.randn((7, 9), dtype=torch.float32, device="npu:0")
Ax.flatten()[:7] = torch.tensor([0., -0., float("nan"), float("inf"), -float("inf"), -1., 1.],
                                dtype=torch.float32, device="npu:0")
base = torch.empty((7, 9, 2), dtype=torch.float32, device="npu:0")
base[..., 0] = Ax
view = base[..., 0]
Bx = torch.randn((7, 9), dtype=torch.float32, device="npu:0")
ref_view = view.clone()
exp = ref_view.sub_(Bx, alpha=0.5)
got = cand.run(view, Bx, alpha=0.5)
numerics["f32_7x9_strided"] = {
    "bitwise_equal_ref": bool(torch.equal(exp.view(-1).float(), got.view(-1).float())),
    "nan_equal": bool(torch.equal(torch.isnan(exp.view(-1)), torch.isnan(got.view(-1)))),
    "same_storage": bool(got.data_ptr() == view.data_ptr()),
    "storage_stride": list(got.stride()),
}
OUT["numerics"] = numerics

# ---- Launcher ABI of the flat kernel ----
A, B = cases["f16_1024"]()
n = A.numel()
ck = cand._sub_flat.warmup(A, B, 0.5, n, BLOCK=4096, grid=(1,))
launcher = ck.run
OUT["launcher_type"] = type(launcher).__name__
for attr in ("compile_only", "enable_msprof_register_tensor"):
    OUT["launcher_" + attr] = repr(getattr(launcher, attr, "<missing>"))
OUT["launcher_metadata_attrs"] = [a for a in dir(launcher.metadata) if "debug" in a.lower()]
try:
    OUT["launch_signature"] = str(inspect.signature(launcher.launch))
except Exception as exc:
    OUT["launch_signature"] = "ERR %r" % (exc,)
OUT["compiled_kernel_attrs"] = [a for a in dir(ck) if a in
                               ("function", "packed_metadata", "metadata", "run", "launch_metadata")]

grid = ((n + 4096 - 1) // 4096,)
stream = stream_handle(dev)


def direct_launch(Aa, Bb, alpha, nn, blk):
    launcher.launch(grid, 1, 1, stream, ck.function, ck.packed_metadata,
                    None, None, None, Aa.data_ptr(), Bb.data_ptr(), alpha, nn, blk)


direct_err = None
try:
    A1, B1 = cases["f16_1024"]()
    A1.flatten()[:7] = torch.tensor([0., -0., float("nan"), float("inf"), -float("inf"), -1., 1.],
                                    dtype=torch.float16, device="npu:0")
    exp1 = A1.clone().sub_(B1, alpha=0.5)
    direct_launch(A1, B1, 0.5, A1.numel(), 4096)
    torch.npu.synchronize()
    OUT["direct_launch_bitwise_ok"] = bool(torch.equal(exp1.view(-1).float(), A1.view(-1).float()))
    # second call with different pointers and a different alpha
    A2, B2 = cases["f16_1024"]()
    exp2 = A2.clone().sub_(B2, alpha=0.25)
    direct_launch(A2, B2, 0.25, A2.numel(), 4096)
    torch.npu.synchronize()
    OUT["direct_launch_second_call_ok"] = bool(
        torch.equal(exp2.view(-1).float(), A2.view(-1).float()))
    # large fp32 through the same compiled entry (n is a runtime arg)
    A3, B3 = cases["f32_4096x4096"]()
    exp3 = A3.clone().sub_(B3, alpha=0.5)
    direct_launch(A3, B3, 0.5, A3.numel(), 4096)
    torch.npu.synchronize()
    OUT["direct_launch_large_ok"] = bool(torch.equal(exp3, A3))
except Exception as exc:
    direct_err = repr(exc)
OUT["direct_launch_error"] = direct_err

# ---- Host-path overhead for the small case ----
A, B = cases["f16_1024"]()
n = A.numel()
small = {}
small["plain_jit_kernel_launch_us"] = bench_us(lambda: cand._sub_flat[grid](A, B, 0.5, n, BLOCK=4096))
if direct_err is None:
    small["direct_launcher_us"] = bench_us(lambda: direct_launch(A, B, 0.5, n, 4096))
small["candidate_run_us"] = bench_us(lambda: cand.run(A, B, alpha=0.5))
small["torch_ref_sub_us"] = bench_us(lambda: ref_sub(A, B))
small["noop_python_us"] = bench_us(lambda: None)
OUT["small_case_host_overhead_us"] = small

# ---- Large-case bandwidth view (device time by sync walltime) ----
big = {}
for name in ("f16_4096x4096", "f32_4096x4096"):
    A, B = cases[name]()
    nn = A.numel()
    g = ((nn + 4096 - 1) // 4096,)
    big[name + "_candidate_us"] = bench_us(lambda: cand.run(A, B, alpha=0.5), iters=50)
    big[name + "_torch_us"] = bench_us(lambda: ref_sub(A, B), iters=50)
    if direct_err is None:
        big[name + "_direct_us"] = bench_us(
            lambda: direct_launch(A, B, 0.5, nn, 4096), iters=50)
    del A, B
OUT["large_case_us"] = big

art = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
with open(os.path.join(art, "probe.json"), "w") as fh:
    json.dump(OUT, fh, indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
