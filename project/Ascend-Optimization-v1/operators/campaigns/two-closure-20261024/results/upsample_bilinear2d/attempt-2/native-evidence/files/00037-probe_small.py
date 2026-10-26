"""Diagnose the small-geometry regression: block kernel alone (QB sweep),
edge kernel alone, both, and the round-1 per-output-row kernel, all timed in
the same harness as the aten reference."""

import json
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

import sys
sys.path.insert(0, "tmp")
import main as cand  # noqa: E402

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
OUT = os.path.join(ART, "probe_small.json")
res = {"rows": [], "errors": []}


def flush():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)


def bench(fn, reps=20, iters=5):
    fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) / reps * 1e3)
    return statistics.median(samples)


CASES = [
    ("t0_fp16", torch.float16, (1, 16, 64, 64), (128, 128)),
    ("t4_fp32", torch.float32, (1, 16, 64, 64), (128, 128)),
    ("t1_fp16", torch.float16, (4, 32, 128, 128), (256, 256)),
    ("t5_fp32", torch.float32, (4, 32, 128, 128), (256, 256)),
]

for name, dtype, shape, size in CASES:
    N, C, IH, IW = shape
    OH, OW = size
    g = torch.Generator().manual_seed(1730)
    x = torch.randn(shape, generator=g).to(dtype=dtype, device="npu")
    ref = torch.ops.aten.upsample_bilinear2d(x, [OH, OW], False, None, None)
    out = torch.empty((N, C, OH, OW), device="npu", dtype=dtype)
    row = {"case": name, "n_prog_ok": True, "variants": []}
    images = N * C

    def add(vname, fn, check=False):
        try:
            fn()
            torch.npu.synchronize()
            ms = bench(fn)
            rec = {"variant": vname, "median_ms": ms}
            if check:
                rec["max_abs_err"] = float((out.float() - ref.float()).abs().max())
            row["variants"].append(rec)
        except Exception as exc:
            row["variants"].append({"variant": vname, "error": repr(exc)[:300]})
        flush()

    add("aten_reference", lambda: torch.ops.aten.upsample_bilinear2d(x, [OH, OW], False, None, None))
    add("candidate_run", lambda: cand.run(x, [OH, OW], False, None, None), check=True)
    add("block_only", lambda: cand._blk2x_kernel[(images * (IH // cand._pick_block_rows(IH, IW, images)),)](
        out, x, IH=IH, IW=IW, OH=OH, OW=OW,
        NB=IH // cand._pick_block_rows(IH, IW, images),
        QB=cand._pick_block_rows(IH, IW, images),
        H16=(dtype is torch.float16)))
    add("edge_only", lambda: cand._edge2x_kernel[(images * 2,)](
        out, x, IH=IH, IW=IW, OH=OH, OW=OW, H16=(dtype is torch.float16)))
    add("block_plain_launch", lambda: cand._ordinary(x, [OH, OW], False, None, None))

    # round-1 kernel, plain launch
    def r1():
        n_rows = images * OH
        n_rows and cand._bilinear_int_kernel[(triton.cdiv(n_rows, cand._pick_rows_per_program(triton.next_power_of_2(IW), n_rows)),)](
            out, x, n_rows, IW * IH, IH * IW, IW, IH / OH,
            C=C, IH=IH, IW=IW, OH=OH, OW=OW,
            BR=cand._pick_rows_per_program(triton.next_power_of_2(IW), n_rows),
            IWP=triton.next_power_of_2(IW))
    add("round1_int_kernel", r1, check=True)

    for qb in (4, 8, 16, 32):
        if IH % qb or IH // qb < 1:
            continue
        nb = IH // qb
        add(f"blk_qb{qb}", lambda qb=qb, nb=nb: cand._blk2x_kernel[(images * nb,)](
            out, x, IH=IH, IW=IW, OH=OH, OW=OW, NB=nb, QB=qb,
            H16=(dtype is torch.float16)))
    res["rows"].append(row)
    flush()

flush()
print(json.dumps(res, indent=1)[:5000])
