"""Validate the new counter-based uniform candidate and time it next to aten."""

import importlib.util
import json
import os
import time
import traceback

import torch
import torch_npu  # noqa: F401

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
res = {"bounds": {}, "stats": {}, "bench": {}, "notes": []}


def bench(fn, iters=60, warmup=15, trials=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(trials):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / iters)
    ts.sort()
    return round(ts[len(ts) // 2] * 1e6, 2)


def load():
    spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    m = load()
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        for frm, to in ((0.0, 1.0), (-3.0, 2.0), (0.5, 0.5), (-1e5, 1e5), (0.1, 0.30000001)):
            key = f"{str(dt).split('.')[-1]}_{frm}_{to}"
            try:
                res["bounds"][key] = list(m._bounds(dt, frm, to))
            except Exception:
                res["bounds"][key] = "ERR:" + traceback.format_exc().splitlines()[-1][:160]

    for dt in (torch.float16, torch.bfloat16, torch.float32):
        tag = str(dt).split(".")[-1]
        for n, frm, to in ((65536, 0.0, 1.0), (65536, -3.0, 2.0), (65536, 0.5, 0.5),
                           (65536, -1e5, 1e5), (4096, 0.0, 1.0), (1024, 0.0, 1.0)):
            base = torch.empty(n, dtype=dt, device="npu")
            key = f"{tag}_{n}_{frm}_{to}"
            try:
                a = m.run(base, frm, to)
                b = m.run(base, frm, to)
                af = a.float()
                z = (af - frm) / (to - frm) if to != frm else None
                rec = {
                    "dtype": str(a.dtype),
                    "shape_ok": tuple(a.shape) == tuple(base.shape),
                    "no_alias": not torch._C._is_alias_of(a, base),
                    "base_untouched": bool((base == 0).all().item()) or None,
                    "finite": bool(torch.isfinite(af).all().item()),
                    "min": float(af.min()),
                    "max": float(af.max()),
                    "in_range": bool(((af >= frm) & (af < to)).all().item()) if to != frm else None,
                    "all_eq": bool((af == frm).all().item()) if to == frm else None,
                    "differ_across_calls": bool((a != b).any().item()),
                }
                if z is not None and n >= 4096:
                    rec["mean_off"] = round(float(z.mean()) - 0.5, 5)
                    rec["var_off"] = round(float(z.var(unbiased=False)) - 1 / 12, 6)
                res["stats"][key] = rec
            except Exception:
                res["stats"][key] = "ERR:" + traceback.format_exc().splitlines()[-1][:200]

    # edge shapes
    for label, t in (("empty", torch.empty(0, dtype=torch.float16, device="npu")),
                     ("scalar", torch.empty((), dtype=torch.float32, device="npu")),
                     ("transposed", torch.empty((64, 32), dtype=torch.float16,
                                                device="npu").t())):
        try:
            o = m.run(t, 0.0, 1.0)
            res["notes"].append(f"{label}: ok shape={tuple(o.shape)} "
                                f"contig={o.is_contiguous()} "
                                f"in_range={bool(((o.float() >= 0) & (o.float() < 1)).all().item())}")
        except Exception:
            res["notes"].append(f"{label}: ERR " + traceback.format_exc().splitlines()[-1][:200])

    # timing: candidate vs aten reference, same inputs
    for dt in (torch.float16, torch.float32):
        tag = str(dt).split(".")[-1]
        for n in (1024, 65536, 1048576):
            x = torch.empty(n, dtype=dt, device="npu")
            res["bench"][f"cand_{tag}_{n}"] = bench(lambda x=x: m.run(x, 0.0, 1.0))
            res["bench"][f"aten_{tag}_{n}"] = bench(
                lambda x=x: torch.ops.aten.uniform(x, 0.0, 1.0))

    with open(os.path.join(ART, "validate.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


main()
