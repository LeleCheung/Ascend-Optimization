"""Target-side diagnosis #5: fewer programs via statically unrolled multi-tile programs."""

import glob
import json
import os
import sys
import time

import torch
import torch_npu
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
R = {"dev": {}, "src": {}}


def save():
    path = os.path.join(ART, "probe5.json")
    with open(path, "w") as fh:
        json.dump(R, fh, indent=1, default=str)
    print("WROTE", path)


sys.path.insert(0, "tmp")
sys.path.insert(0, ".")
import main as cand  # noqa: E402

dev = torch.npu.current_device()
getter = torch_npu._C._npu_getCurrentRawStreamNoWait


def batch_mean(fn, iters=80, warm=20):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


# ---- where is NPULauncher defined ----
try:
    hits = []
    for pat in ("/usr/local/python3.11.15/lib/python3.11/site-packages/triton/**/*.py",):
        for f in glob.glob(pat, recursive=True):
            try:
                s = open(f, errors="ignore").read()
            except Exception:  # noqa: BLE001
                continue
            if "class NPULauncher" in s:
                i = s.find("class NPULauncher")
                hits.append({"file": f, "body": s[i:i + 3500]})
    R["src"]["npu_launcher"] = hits
except Exception as exc:  # noqa: BLE001
    R["src"]["err"] = repr(exc)
save()


@triton.jit
def _sl(X, BLOCK: tl.constexpr, TILES: tl.constexpr):
    """One program handles TILES consecutive tiles, statically unrolled."""
    pid = tl.program_id(0)
    base = pid * (BLOCK * TILES)
    for i in tl.static_range(TILES):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offs)
        y = tl.log(x.to(tl.float32)) * 0.4342944819032518
        tl.store(X + offs, y.to(X.dtype.element_ty))


def raw_time(kern, tensor, extra, cexpr, gx, iters=80, **opts):
    compiled = kern.warmup(tensor, *extra, grid=(1,), **cexpr, **opts)
    launcher = compiled.run
    call, fn, md = launcher.launch, compiled.function, compiled.packed_metadata
    ptr = tensor.data_ptr()
    params = tuple([ptr] + list(extra) + [None] * len(cexpr))

    def go():
        call(gx, 1, 1, getter(dev), fn, md, None, None, None, *params)
    return batch_mean(go, iters=iters)


N = 4096 * 4096
g = torch.Generator().manual_seed(1729)

PLAN = {
    "float16": [(32768, 1), (32768, 2), (32768, 4), (32768, 8), (16384, 4), (16384, 8)],
    "float32": [(16384, 1), (16384, 2), (16384, 4), (16384, 8), (8192, 8)],
}
for dt, plan in PLAN.items():
    a = (torch.randn(N, generator=g).abs() + 0.05).to(getattr(torch, dt)).npu()
    ent = {"ref": batch_mean(lambda: torch.log10_(a)),
           "seed_run": batch_mean(lambda: cand.run(a))}
    for blk, tiles in plan:
        tag = "b%d_t%d_g%d" % (blk, tiles, N // (blk * tiles))
        try:
            ent[tag] = raw_time(_sl, a, (), {"BLOCK": blk, "TILES": tiles}, N // (blk * tiles))
        except Exception as exc:  # noqa: BLE001
            ent[tag] = "ERR %s" % (str(exc)[:140],)
        save()
    R["dev"][dt] = ent
    save()

print("DONE")
