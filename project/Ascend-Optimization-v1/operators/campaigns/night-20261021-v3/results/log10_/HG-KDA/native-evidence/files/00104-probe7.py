"""Compare the candidate's direct pybind launch against NPULauncher/bookkeeping
forms, print msprof/compile-only flags, and quantify host-time noise."""
import json
import os
import sys
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

sys.path.insert(0, os.path.join(os.getcwd(), "tmp"))
import main as cand  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {"bench": {}, "notes": {}, "errors": []}
DEV = "npu:0"


def save():
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "probe7.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=str)


def host_time(fn, iters=400):
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    return (t1 - t0) / iters * 1e6


def record(name, fn, reps=3):
    vals = []
    try:
        for _ in range(reps):
            vals.append(host_time(fn))
    except Exception as exc:  # noqa: BLE001
        OUT["bench"][name] = {"err": f"{type(exc).__name__}: {str(exc)[:250]}"}
    else:
        OUT["bench"][name] = {"host_us": vals, "min": min(vals)}
    save()
    print(name, json.dumps(OUT["bench"][name], default=str))


@triton.jit
def k_nomask(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty))


def main():
    OUT["notes"]["env"] = {
        k: os.environ.get(k)
        for k in ("TRITON_REGISTER_TENSOR_MSPROF", "TRITON_COMPILE_ONLY",
                  "TRITON_ENABLE_TASKQUEUE")
    }
    OUT["notes"]["torch_npu_ver"] = getattr(torch_npu, "__version__", "?")

    g = torch.Generator().manual_seed(0)
    small = torch.randn(1024, generator=g).abs().to(torch.float16).to(DEV) + 0.05
    big = torch.randn(4096 * 4096, generator=g).abs().to(torch.float16).to(DEV) + 0.05

    # candidate's own path
    cand.run(small)
    cand.run(big)
    record("cand_run::fp16::1024", lambda: cand.run(small))
    record("cand_run::fp16::16777216", lambda: cand.run(big))

    ck = k_nomask.warmup(big, BLOCK=32768, grid=(1,))
    launcher = ck.run
    OUT["notes"]["compile_only"] = bool(getattr(launcher, "compile_only", "NA"))
    OUT["notes"]["msprof_register"] = bool(getattr(launcher, "enable_msprof_register_tensor", "NA"))
    OUT["notes"]["launcher_type"] = str(type(launcher))
    st = cand._RUNTIME["stream"]()
    grid = big.numel() // 32768

    pyb = launcher.launch
    record("pybind_cached_stream::fp16::16777216",
           lambda: pyb(grid, 1, 1, st, ck.function, ck.packed_metadata, None, None, None, big, None))
    record("pybind_live_stream::fp16::16777216",
           lambda: pyb(grid, 1, 1, st, ck.function, ck.packed_metadata, None, None, None, big, None))
    record("npulauncher_call::fp16::16777216",
           lambda: launcher(grid, 1, 1, st, ck.function, ck.packed_metadata, None, None, None, big, None))
    closure = ck[(grid, 1, 1)]
    record("closure::fp16::16777216", lambda: closure(big, None, stream=st))
    record("jit_launch::fp16::16777216", lambda: k_nomask[(grid,)](big, BLOCK=32768))

    # same kernel shape but tiny grid, to separate grid-size effects
    ck_s = k_nomask.warmup(small, BLOCK=1024, grid=(1,))
    pyb_s = ck_s.run.launch if hasattr(ck_s.run, "launch") else ck_s.run
    record("pybind_cached_stream::fp16::1024",
           lambda: pyb_s(1, 1, 1, st, ck_s.function, ck_s.packed_metadata, None, None, None, small, None))
    record("jit_launch::fp16::1024", lambda: k_nomask[(1,)](small, BLOCK=1024))
    record("torch_ref::fp16::1024", lambda: torch.log10_(small))

    # misaligned / cold path timing (uses the standard JIT launch)
    base = torch.randn(1024 + 8, generator=g).abs().to(torch.float32).to(DEV) + 0.05
    view = base[1:1025]
    cand.run(view)
    record("cand_misaligned_cold_path::fp32::1024", lambda: cand.run(view))

    save()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        OUT["errors"].append(traceback.format_exc()[-3000:])
    save()
    print("ERRORS", json.dumps(OUT["errors"], indent=1)[:2000])
    print("NOTES", json.dumps(OUT["notes"], indent=1)[:1500])
