"""Pick the cheapest faithful current-stream accessor: measure per-launch cost in a
live launch loop and verify it tracks torch stream switches."""
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
    with open(os.path.join(ART, "probe8.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=str)


@triton.jit
def k_nomask(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    y = tl.log(x.to(tl.float32)) * 0.4342944819032518
    tl.store(X + offs, y.to(X.dtype.element_ty))


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


def main():
    from triton.runtime import driver

    dev = driver.active.get_current_device()
    C = torch_npu._C
    accessors = {}
    for name in ("_npu_getCurrentRawStream", "_npu_getCurrentRawStreamNoWait"):
        fn = getattr(C, name, None)
        if fn is None:
            OUT["notes"][name] = "missing"
            continue
        try:
            accessors[name] = ("dev", fn, int(fn(dev)))
        except TypeError:
            accessors[name] = ("nodev", fn, int(fn()))
    def_drv = driver.active.get_current_stream
    torch_cur = int(torch.npu.current_stream().npu_stream)
    OUT["notes"]["torch_current"] = torch_cur
    OUT["notes"]["driver_current"] = int(def_drv(dev))
    for name, (kind, fn, val) in accessors.items():
        OUT["notes"][name] = {"kind": kind, "value": val, "matches_torch": val == torch_cur}

    # stream fidelity: inside a secondary stream context, do the accessors follow?
    other = torch.npu.Stream()
    fidelity = {}
    with torch.npu.stream(other):
        cur = int(torch.npu.current_stream().npu_stream)
        fidelity["torch"] = cur
        fidelity["driver"] = int(def_drv(dev))
        for name, (kind, fn, _) in accessors.items():
            try:
                fidelity[name] = int(fn(dev) if kind == "dev" else fn())
            except Exception as exc:  # noqa: BLE001
                fidelity[name] = f"ERR {exc}"
        fidelity["other_handle"] = int(other.npu_stream)
    OUT["notes"]["fidelity_secondary_stream"] = fidelity

    # per-launch cost in a live loop
    big = torch.randn(4096 * 4096, generator=torch.Generator().manual_seed(0)).abs().to(
        torch.float16).to(DEV) + 0.05
    small = torch.randn(1024, generator=torch.Generator().manual_seed(0)).abs().to(
        torch.float16).to(DEV) + 0.05
    ck = k_nomask.warmup(big, BLOCK=32768, grid=(1,))
    pyb = ck.run.launch
    grid = big.numel() // 32768
    ck_s = k_nomask.warmup(small, BLOCK=1024, grid=(1,))
    pyb_s = ck_s.run.launch

    def mk(pyb_, grid_, tensor, getter):
        return lambda: pyb_(grid_, 1, 1, getter(), ck.function, ck.packed_metadata,
                            None, None, None, tensor, None)

    st_cached = int(torch.npu.current_stream().npu_stream)
    record("loop_cached_stream::16777216",
           mk(pyb, grid, big, lambda: st_cached))
    for name, (kind, fn, _) in accessors.items():
        g = (lambda fn=fn, dev=dev: fn(dev)) if kind == "dev" else (lambda fn=fn: fn())
        record(f"loop_{name}::16777216", mk(pyb, grid, big, g))
    record("loop_driver_stream::16777216", mk(pyb, grid, big, lambda: def_drv(dev)))
    record("loop_cached_stream::1024", mk(pyb_s, 1, small, lambda: st_cached))
    for name, (kind, fn, _) in accessors.items():
        g = (lambda fn=fn, dev=dev: fn(dev)) if kind == "dev" else (lambda fn=fn: fn())
        record(f"loop_{name}::1024", mk(pyb_s, 1, small, g))
    record("loop_driver_stream::1024", mk(pyb_s, 1, small, lambda: def_drv(dev)))
    record("cand_run::16777216", lambda: cand.run(big))
    record("torch_ref::1024", lambda: torch.log10_(small))
    save()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        OUT["errors"].append(traceback.format_exc()[-3000:])
    save()
    print("ERRORS", json.dumps(OUT["errors"], indent=1)[:2000])
    print("NOTES", json.dumps(OUT["notes"], indent=1)[:3000])
