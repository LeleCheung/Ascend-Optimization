"""Inspect the FlagTree/Ascend Triton launch path and micro-time its components."""
import inspect
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
OUT = {"notes": {}, "timings": {}, "errors": []}
TXT = []


def save():
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "probe3.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=str)
    with open(os.path.join(ART, "sources.txt"), "w") as f:
        f.write("\n\n".join(TXT))


@triton.jit
def k_flat(X, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty), mask=mask)


def micro(name, fn, iters=1000):
    fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    torch.npu.synchronize()
    t2 = time.perf_counter()
    OUT["timings"][name] = {"host_us": (t1 - t0) / iters * 1e6, "amortized_us": (t2 - t0) / iters * 1e6}
    print(name, json.dumps(OUT["timings"][name]))


def main():
    from triton.runtime import driver

    # ---- micro-timings of the pieces a launch needs ----
    micro("driver.get_current_device", lambda: driver.active.get_current_device())
    micro("driver.get_current_stream", lambda: driver.active.get_current_stream(0))
    micro("torch.npu.current_stream", lambda: torch.npu.current_stream())
    try:
        micro("stream_obj.npu_stream", lambda: torch.npu.current_stream().npu_stream)
    except Exception as exc:  # noqa: BLE001
        OUT["errors"].append(f"npu_stream: {exc}")

    x = torch.randn(4096 * 4096, generator=torch.Generator().manual_seed(0)).abs().to("npu:0") + 0.05
    n = x.numel()
    blk = 16384
    grid = (n // blk,)

    # ---- JIT-level launch (what we have today) ----
    micro("jit_launch_flat16384", lambda: k_flat[grid](x, n, BLOCK=blk), iters=200)

    # ---- source of the fork's JITFunction.run ----
    try:
        from triton.spec.ascend.runtime import jit as ajit

        src = inspect.getsource(ajit.JITFunction.run)
        TXT.append("=== JITFunction.run ===\n" + src)
        OUT["notes"]["run_len"] = len(src)
    except Exception as exc:  # noqa: BLE001
        OUT["errors"].append(f"getsource JITFunction.run: {type(exc).__name__}: {exc}")

    try:
        from triton.spec.ascend.runtime import jit as ajit

        TXT.append("=== JITFunction.__getitem__ ===\n" + inspect.getsource(ajit.JITFunction.__getitem__))
    except Exception as exc:  # noqa: BLE001
        OUT["errors"].append(f"getitem: {exc}")

    # ---- compiled kernel internals ----
    ck = k_flat.warmup(x, n, BLOCK=blk, grid=(1,))
    OUT["notes"]["ck_type"] = str(type(ck))
    OUT["notes"]["ck_dir"] = [a for a in dir(ck) if not a.startswith("_")]
    for attr in ("run", "launch_metadata", "packed_metadata", "function", "module", "metadata"):
        try:
            v = getattr(ck, attr)
            OUT["notes"][f"attr_{attr}"] = str(v)[:600]
            doc = getattr(v, "__doc__", None)
            if doc:
                OUT["notes"][f"doc_{attr}"] = doc[:1200]
        except Exception as exc:  # noqa: BLE001
            OUT["notes"][f"attr_{attr}"] = f"ERR {type(exc).__name__}: {exc}"
    try:
        TXT.append("=== CompiledKernel.__getitem__ ===\n" + inspect.getsource(type(ck).__getitem__))
    except Exception as exc:  # noqa: BLE001
        OUT["errors"].append(f"ck getitem src: {exc}")
    try:
        TXT.append("=== CompiledKernel class ===\n" + inspect.getsource(type(ck)))
    except Exception as exc:  # noqa: BLE001
        OUT["errors"].append(f"ck class src: {exc}")

    # ---- backend driver source: launcher template & launch() ----
    try:
        import triton.backends.ascend.driver as adrv

        src = inspect.getsource(adrv)
        TXT.append(f"=== ascend.driver ({len(src)} chars) ===\n" + src[:20000])
        OUT["notes"]["driver_src_len"] = len(src)
    except Exception as exc:  # noqa: BLE001
        OUT["errors"].append(f"ascend.driver src: {type(exc).__name__}: {exc}")

    # ---- try a direct compiled-kernel launch with arity discovery ----
    stream = torch.npu.current_stream().npu_stream
    meta = ck.packed_metadata
    func = ck.function

    def attempt(label, args):
        try:
            ck.run(*args)
            torch.npu.synchronize()
            OUT["notes"][f"direct_{label}"] = "OK"
            micro(f"direct_{label}", lambda a=args: ck.run(*a), iters=200)
            return True
        except Exception as exc:  # noqa: BLE001
            OUT["notes"][f"direct_{label}"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            return False

    base = [grid[0], 1, 1, stream, func, meta, None, None, None, x, n]
    for extra in ([0], [], [0, 0], [None, 0]):
        if attempt(f"extra{len(extra)}", base[:-1] + extra + [x, n]):
            break

    # correctness of whichever direct form worked
    for k in list(OUT["notes"]):
        if k.startswith("direct_") and OUT["notes"][k] == "OK":
            try:
                a = x.clone()
                a.npu()
                ck.run(*[grid[0], 1, 1, stream, func, meta, None, None, None, a, a.numel()])
                torch.npu.synchronize()
                ref = torch.log10(x.float())
                OUT["notes"]["direct_err"] = float((a.float() - ref).abs().max())
            except Exception as exc:  # noqa: BLE001
                OUT["notes"]["direct_err"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            break
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        OUT["errors"].append(traceback.format_exc()[-3000:])
    save()
    print("ERRORS", json.dumps(OUT["errors"], indent=1)[:3000])
    print("NOTES", json.dumps(OUT["notes"], indent=1)[:4000])
