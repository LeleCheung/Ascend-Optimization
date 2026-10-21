"""Validate tmp/main.py against the torch reference on all 36 frozen workload
specs (fresh identical inputs per side) and measure per-call latency."""
import json
import os
import sys
import time
import traceback

import torch
import torch_npu  # noqa: F401
import triton  # noqa: F401

sys.path.insert(0, os.path.join(os.getcwd(), "tmp"))
import main as cand  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = {"correctness": {}, "timing": {}, "notes": {}, "errors": []}
DEV = "npu:0"

CORRECTNESS = [
    (0, "float16", [0], False), (1, "float16", [0], True),
    (2, "float16", [1], False), (3, "float16", [1], True),
    (4, "float16", [1025], False), (5, "float16", [1025], True),
    (6, "float16", [7, 9], False), (7, "float16", [7, 9], True),
    (8, "bfloat16", [0], False), (9, "bfloat16", [0], True),
    (10, "bfloat16", [1], False), (11, "bfloat16", [1], True),
    (12, "bfloat16", [1025], False), (13, "bfloat16", [1025], True),
    (14, "bfloat16", [7, 9], False), (15, "bfloat16", [7, 9], True),
    (16, "float32", [0], False), (17, "float32", [0], True),
    (18, "float32", [1], False), (19, "float32", [1], True),
    (20, "float32", [1025], False), (21, "float32", [1025], True),
    (22, "float32", [7, 9], False), (23, "float32", [7, 9], True),
]
TIMING = [
    (0, "float16", [1024]), (1, "float16", [65536]), (2, "float16", [1048576]),
    (3, "float16", [4096, 4096]),
    (4, "bfloat16", [1024]), (5, "bfloat16", [65536]), (6, "bfloat16", [1048576]),
    (7, "bfloat16", [4096, 4096]),
    (8, "float32", [1024]), (9, "float32", [65536]), (10, "float32", [1048576]),
    (11, "float32", [4096, 4096]),
]
SEEDS_CORRECTNESS = {i: 1729 + i for i in range(24)}
SEEDS_TIMING = {0: 1729, 1: 1730, 2: 1731, 3: 1732, 4: 1733, 5: 1734,
                6: 1735, 7: 1736, 8: 1737, 9: 1738, 10: 1739, 11: 1740}
TOL = {"float16": (0.015, 0.015), "bfloat16": (0.015, 0.015), "float32": (1e-4, 1e-4)}


def gen(seed, dtype_name, shape, strided):
    dt = getattr(torch, dtype_name)
    g = torch.Generator().manual_seed(seed)

    def random(shp):
        return torch.randn(shp, generator=g).to(dtype=dt, device=DEV)

    x = random(shape).abs() + 0.05
    if strided:
        base = torch.empty((*x.shape, 2), dtype=dt, device=DEV)
        base[..., 0] = x
        x = base[..., 0]
        return x, base
    return x, None


def check(idx, dtype_name, shape, strided, seed):
    ref, ref_base = gen(seed, dtype_name, shape, strided)
    got, got_base = gen(seed, dtype_name, shape, strided)
    torch.npu.synchronize()
    out_ref = torch.log10_(ref)
    out_got = cand.run(got)
    torch.npu.synchronize()
    rec = {
        "alias_ok": out_got is got,
        "ref_alias_ok": out_ref is ref,
        "shape_ok": tuple(out_got.shape) == tuple(out_ref.shape),
        "dtype_ok": out_got.dtype == out_ref.dtype,
    }
    if out_ref.numel():
        a = out_got.float().double()
        b = out_ref.float().double()
        rtol, atol = TOL[dtype_name]
        diff = (a - b).abs()
        rec["abs_max"] = float(diff.max())
        rec["matched_ratio"] = float((diff <= atol + rtol * b.abs()).float().mean())
        rec["nan_in_out"] = bool(torch.isnan(a).any())
        # mutation of the original storage (and only the view positions)
        if strided:
            ref_changed = not torch.equal(ref_base[..., 0], got_base[..., 0])
            untouched = torch.equal(ref_base[..., 1], got_base[..., 1])
            rec["view_mutated"] = bool(ref_changed)
            rec["outside_untouched"] = bool(untouched)
        else:
            rec["view_mutated"] = True
            rec["outside_untouched"] = True
    OUT["correctness"][f"c{idx}::{dtype_name}::{shape}::strided={strided}"] = rec
    return rec


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


def main():
    # runtime choice made by the candidate
    st = cand._RUNTIME.get("stream")
    if st is None:
        cand._init_runtime()
    OUT["notes"]["runtime"] = {
        "device": int(cand._RUNTIME["device"]),
        "stream_getter": str(cand._RUNTIME["stream"]),
    }
    from triton.runtime import driver

    cur = int(torch.npu.current_stream().npu_stream)
    OUT["notes"]["stream_matches_driver"] = int(cand._RUNTIME["stream"]()) == cur == int(
        driver.active.get_current_stream(0))
    OUT["notes"]["stream_getter_host_us"] = host_time(cand._RUNTIME["stream"])[0]
    OUT["notes"]["driver_stream_host_us"] = host_time(
        lambda: driver.active.get_current_stream(0))[0]
    OUT["notes"]["torch_stream_host_us"] = host_time(
        lambda: torch.npu.current_stream().npu_stream)[0]

    for idx, dt, shape, strided in CORRECTNESS:
        try:
            check(idx, dt, shape, strided, SEEDS_CORRECTNESS[idx])
        except Exception:  # noqa: BLE001
            OUT["errors"].append(f"c{idx}: " + traceback.format_exc()[-800:])
    print("correctness done")
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "validate.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=str)

    # odd shapes through the masked direct path and a misaligned contiguous view
    for label, t in (
        ("masked_65537", torch.randn(65537, generator=torch.Generator().manual_seed(7)).abs().to(DEV) + 0.05),
    ):
        a = t.clone()
        b = t.clone()
        torch.log10_(a)
        cand.run(b)
        torch.npu.synchronize()
        OUT["notes"][label + "_abs_err"] = float((a.float() - b.float()).abs().max())
    base = (torch.randn(4096 + 8, generator=torch.Generator().manual_seed(9)).abs().to(DEV) + 0.05)
    base_before = base.clone()
    view = base[1:4097]
    a = view.clone()
    out = cand.run(view)
    torch.npu.synchronize()
    torch.log10_(a)
    OUT["notes"]["misaligned"] = {
        "ptr_mod16": int(view.data_ptr() % 16),
        "abs_err": float((view.float() - a.float()).abs().max()),
        "alias_ok": out is view,
        "head_untouched": bool(torch.equal(base[:1], base_before[:1])),
        "tail_untouched": bool(torch.equal(base[4097:], base_before[4097:])),
    }
    print("odd shapes done")

    # timing per workload versus the reference
    for idx, dt, shape in TIMING:
        t_c, _ = gen(SEEDS_TIMING[idx], dt, shape, False)
        t_r, _ = gen(SEEDS_TIMING[idx], dt, shape, False)
        h_c, a_c = host_time(lambda: cand.run(t_c), iters=200)
        h_r, a_r = host_time(lambda: torch.log10_(t_r), iters=200)
        OUT["timing"][f"t{idx}::{dt}::{shape}"] = {
            "cand_host_us": h_c, "cand_amortized_us": a_c,
            "ref_host_us": h_r, "ref_amortized_us": a_r,
            "speedup_amortized": a_r / a_c if a_c else None,
        }
        print(f"t{idx} {dt} {shape}: cand {a_c:.2f}us ref {a_r:.2f}us "
              f"speedup {a_r / a_c:.3f}")

    with open(os.path.join(ART, "validate.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=str)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        OUT["errors"].append(traceback.format_exc()[-3000:])
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "validate.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=str)
    print("ERRORS", json.dumps(OUT["errors"], indent=1)[:3000])
