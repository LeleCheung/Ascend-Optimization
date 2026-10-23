"""Validate + time the submitted candidate: numerics on every dispatch path,
then full-call walltime against torch.scatter_add on the frozen shapes.
"""
import importlib.util
import json
import os
import time

import torch
import torch_npu  # noqa: F401

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu:0"
torch.npu.set_device(0)
RES = {"cases": [], "timing": {}}


def log(tag, **kw):
    print(json.dumps({"tag": tag, **kw}), flush=True)


spec = importlib.util.spec_from_file_location("cand", os.path.join(os.getcwd(), "tmp/main.py"))
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)


def make(shape, dim, dtype, seed=1729, strided=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g).to(dtype=dtype, device=DEV)
    s = torch.randn(shape, generator=g).to(dtype=dtype, device=DEV)
    idx = torch.randint(shape[dim % len(shape)], shape, generator=g).to(DEV)
    if strided:
        xb = torch.empty((*shape, 2), dtype=dtype, device=DEV)
        xb[..., 0] = x
        x = xb[..., 0]
    return x, idx, s


def check(tag, shape, dim, dtype, strided=False, dup=False):
    inp, idx, src = make(shape, dim, dtype, strided=strided)
    if dup:
        idx = torch.zeros_like(idx)
    ref = torch.scatter_add(inp, dim, idx, src)
    si, ssr, sd = inp.clone(), src.clone(), idx.clone()
    got = cand.run(inp, dim, idx, src)
    torch.npu.synchronize()
    tol = 1e-4 if dtype == torch.float32 else 0.015
    d = (got.float() - ref.float()).abs()
    row = {"case": tag, "dtype": str(dtype), "shape": list(shape), "dim": dim,
           "dtype_ok": got.dtype == ref.dtype,
           "shape_ok": tuple(got.shape) == tuple(ref.shape),
           "fresh": got.data_ptr() != inp.data_ptr(),
           "inputs_intact": (torch.equal(inp, si) and torch.equal(src, ssr)
                             and torch.equal(idx, sd)),
           "max_abs": round(float(d.max().item()), 9),
           "matched": round(float((d <= tol + tol * ref.float().abs()).float().mean().item()), 6)}
    row["ok"] = bool(row["dtype_ok"] and row["shape_ok"] and row["fresh"]
                     and row["inputs_intact"] and row["matched"] == 1.0)
    RES["cases"].append(row)
    log("check", **row)


def bench(fn, iters=20, warm=3):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return round((time.perf_counter() - t0) / iters * 1e3, 5)


def main():
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        check(f"owner-{dt}-5x7x9", (5, 7, 9), 2, dt)
        check(f"owner-{dt}-128", (128, 128), -1, dt)
        check(f"owner-{dt}-257", (129, 257), -1, dt)
        check(f"sort-{dt}-1024", (1024, 1024), -1, dt)
        check(f"sort-{dt}-2048", (2048, 2048), -1, dt)
        check(f"sort-{dt}-300", (517, 300), -1, dt)
        check(f"sort-{dt}-17x1023", (17, 1023), -1, dt)
        check(f"sort-{dt}-dup1024", (256, 1024), -1, dt, dup=True)
        check(f"sort-{dt}-dup2048", (64, 2048), -1, dt, dup=True)
        check(f"nd-{dt}-dim0", (5, 7, 9), 0, dt)
        check(f"nd-{dt}-dim1", (5, 7, 9), 1, dt)
        check(f"nd-{dt}-stride", (5, 7, 9), 2, dt, strided=True)
        check(f"nd-{dt}-big-dim0", (64, 33, 2049), 0, dt)
    inp, idx, src = make((3, 4), -1, torch.float16)
    out = cand.run(inp, -1, idx[:0], src[:0])
    torch.npu.synchronize()
    RES["cases"].append({"case": "empty", "ok": bool(torch.equal(out, inp))})
    log("check", **RES["cases"][-1])
    RES["all_ok"] = all(c["ok"] for c in RES["cases"])
    RES["failed"] = [c["case"] for c in RES["cases"] if not c["ok"]]
    log("summary", all_ok=RES["all_ok"], failed=RES["failed"])

    for dt in (torch.float16, torch.float32):
        for shape in ((128, 128), (1024, 1024), (2048, 2048)):
            inp, idx, src = make(shape, -1, dt, seed=1737)
            e = {"ref": bench(lambda: torch.scatter_add(inp, -1, idx, src)),
                 "cand": bench(lambda: cand.run(inp, -1, idx, src))}
            e["speedup"] = round(e["ref"] / e["cand"], 4)
            RES["timing"][f"{dt}-{shape}"] = e
            log("timing", case=f"{dt}-{shape}", **e)
            with open(os.path.join(ART, "validate2.json"), "w") as fh:
                json.dump(RES, fh, indent=1)
    print("DONE", flush=True)


main()
