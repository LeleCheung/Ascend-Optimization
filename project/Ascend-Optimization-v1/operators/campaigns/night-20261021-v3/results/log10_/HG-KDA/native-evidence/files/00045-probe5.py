"""Recover the Ascend Triton pybind launcher signature, get a direct CompiledKernel
launch working, verify numerics, and sweep device configs."""
import functools
import json
import os
import re
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {"bench": {}, "notes": {}, "errors": []}
TXT = []


def save():
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "probe5.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=str)
    with open(os.path.join(ART, "launcher_src.txt"), "w") as f:
        f.write("\n\n".join(TXT))


@triton.jit
def k_flat(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty), mask=mask)


@triton.jit
def k_nomask(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty))


@triton.jit
def k_nomask_l2(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log2(x.to(tl.float32)) * 0.30102999566398120
    tl.store(X + offs, y.to(X.dtype.element_ty))


def host_time(fn, iters=300):
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) / iters * 1e6, (t2 - t0) / iters * 1e6


def maxerr(x, fn):
    a = x.clone()
    fn(a)
    torch.npu.synchronize()
    return float((a.float() - torch.log10(x.float())).abs().max())


def record(name, fn, err=None):
    e = {}
    try:
        e["host_us"], e["amortized_us"] = host_time(fn)
    except Exception as exc:  # noqa: BLE001
        e["launch_err"] = f"{type(exc).__name__}: {str(exc)[:250]}"
    if err is not None:
        try:
            e["abs_err"] = err()
        except Exception as exc:  # noqa: BLE001
            e["err"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    OUT["bench"][name] = e
    save()
    print(name, json.dumps(e, default=str))


def main():
    from triton.runtime import driver

    dev = "npu:0"
    g = torch.Generator().manual_seed(0)
    ts = {}
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        for n in (1024, 65536, 1048576, 4096 * 4096):
            ts[(str(dt).split(".")[-1], n)] = torch.randn(n, generator=g).abs().to(dt).to(dev) + 0.05

    x = ts[("float32", 4096 * 4096)]
    n = x.numel()
    blk = 32768
    ck = k_nomask.warmup(x, BLOCK=blk, grid=(1,))
    OUT["notes"]["packed_metadata"] = str(ck.packed_metadata)
    OUT["notes"]["workspace_size"] = str(getattr(ck.metadata, "workspace_size", "NA"))
    OUT["notes"]["lock_num"] = str(getattr(ck.metadata, "lock_num", "NA"))
    OUT["notes"]["metadata_fields"] = sorted(ck.metadata._asdict().keys())

    # ---- reconstruct the generated launcher C source ----
    try:
        from triton.backends.ascend.driver import make_launcher

        src = ck.src
        constants = src.constants if hasattr(src, "constants") else {}
        cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
        constants = {cst_key(k): v for k, v in constants.items()}
        signature = {cst_key(k): v for k, v in src.signature.items()}
        OUT["notes"]["src_signature"] = str(signature)
        OUT["notes"]["src_constants"] = str(constants)
        wsrc = make_launcher(constants, signature, ck.metadata)
        TXT.append("=== generated launcher wrapper ===\n" + wsrc)
        idx = wsrc.find("PyArg_ParseTuple")
        OUT["notes"]["parse_ctx"] = wsrc[max(0, idx - 2500):idx + 400] if idx >= 0 else "not found"
        OUT["notes"]["wrapper_len"] = len(wsrc)
    except Exception as exc:  # noqa: BLE001
        OUT["errors"].append(f"make_launcher: {type(exc).__name__}: {str(exc)[:600]}")
    save()

    # ---- empirically bind the extra launcher argument ----
    grid3 = (n // blk, 1, 1)
    tries = {
        "scratch0_then_x": (0, x),
        "x_then_scratch0": (x, 0),
        "scratchNone_then_x": (None, x),
        "x_then_scratchNone": (x, None),
        "scratch_int_then_x_2": (1, x),
        "tensor_scratch": (torch.zeros(8, dtype=torch.int64, device=dev), x),
    }
    good = None
    for label, extra in tries.items():
        r = ck[grid3]
        try:
            r(*extra)
            torch.npu.synchronize()
            err = maxerr(x, lambda a: r(*((extra[0], a) if extra[1] is x else (a, extra[1]))))
            OUT["notes"][f"try_{label}"] = f"OK abs_err={err}"
            if err <= 1e-5 and good is None:
                good = label
        except Exception as exc:  # noqa: BLE001
            OUT["notes"][f"try_{label}"] = f"{type(exc).__name__}: {str(exc)[:250]}"
        save()
    OUT["notes"]["good_form"] = good
    print("GOOD:", good)
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        OUT["errors"].append(traceback.format_exc()[-3000:])
    save()
    print("ERRORS", json.dumps(OUT["errors"], indent=1)[:3000])
    print("NOTES", json.dumps(OUT["notes"], indent=1)[:6000])
