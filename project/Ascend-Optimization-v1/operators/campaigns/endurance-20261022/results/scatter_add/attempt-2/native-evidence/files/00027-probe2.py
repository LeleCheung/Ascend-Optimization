"""Diagnostic gate + micro-benchmark for the current candidate (tmp/main.py).

Part 1: semantic self-gate on the frozen contract shapes plus extra coverage
(repeated index, non-last dim, strided input, empty index), compared against the
device reference.  Also compares the device reference against an exact CPU
float64 reference to characterize the reference itself.

Part 2: walltime of candidate ``run`` on the frozen timing shapes, plus a
dense-kernel parameter sweep to place the owner/atomic crossover.

Diagnostic only: not a formal evaluation.
"""

import importlib.util
import json
import os
import time

import numpy as np
import torch
import triton
import triton.language as tl

OUTDIR = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(os.path.dirname(HERE), "main.py")
DEV = "npu"
torch.npu.set_device(0)

spec = importlib.util.spec_from_file_location("cand_main", MAIN)
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

RESULT = {"device": torch.npu.get_device_name(0), "semantics": [], "perf": [], "sweep": []}
RESULT["candidate_sha_note"] = "loaded from workspace tmp/main.py"


def gen(seed, shape, dim, dtype):
    g = torch.Generator().manual_seed(seed)

    def random(sh):
        return torch.randn(sh, generator=g).to(dtype=dtype, device=DEV)

    x = random(shape)
    s = random(shape)
    idx = torch.randint(shape[dim % len(shape)], shape, generator=g).to(DEV)
    return x, dim, idx, s


def cmp(ref, out, atol, rtol):
    r = ref.detach().float().reshape(-1)
    o = out.detach().float().reshape(-1)
    if r.numel() != o.numel():
        return {"error": f"numel {o.numel()} != {r.numel()}"}
    dif = (r - o).abs()
    lim = atol + rtol * r.abs()
    return {
        "mismatched": int((dif > lim).sum()),
        "total": int(r.numel()),
        "max_abs": float(dif.max()) if r.numel() else 0.0,
    }


TOL = {torch.float16: (0.015, 0.015), torch.bfloat16: (0.015, 0.015), torch.float32: (1e-4, 1e-4)}


def semantics():
    cases = []
    for dt, name in ((torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float32, "fp32")):
        for i, dim in enumerate((0, 1, 2)):
            cases.append((f"contract-{name}-3d-dim{dim}", 1729 + 2 * i, [5, 7, 9], dim, dt, "plain"))
        cases.append((f"extra-{name}-17x131-d0", 1800, [17, 131], 0, dt, "plain"))
        cases.append((f"extra-{name}-17x131-d1", 1801, [17, 131], 1, dt, "plain"))
        cases.append((f"dup-{name}-zeros", 1802, [5, 7, 9], 2, dt, "dup"))
        cases.append((f"strided-{name}", 1803, [5, 7, 9], 1, dt, "strided"))
    cases.append(("nonlast-fp16-1024", 1804, [1024, 1024], 0, torch.float16, "plain"))
    cases.append(("nonlast-bf16-256", 1805, [256, 256], 0, torch.bfloat16, "plain"))

    for tag, seed, shape, dim, dt, kind in cases:
        try:
            inp, dm, index, src = gen(seed, shape, dim, dt)
            if kind == "dup":
                idx = torch.zeros(shape, dtype=torch.int64, device=DEV)
                index = idx
            if kind == "strided":
                b = torch.empty((*shape, 2), dtype=dt, device=DEV)
                b[..., 0] = inp
                inp = b[..., 0]
                b2 = torch.empty((*shape, 2), dtype=dt, device=DEV)
                b2[..., 0] = src
                src = b2[..., 0]
            in0, idx0, src0 = inp.clone(), index.clone(), src.clone()
            ref = torch.scatter_add(inp, dm, index, src)
            out = cand.run(inp, dm, index, src)
            atol, rtol = TOL[dt]
            rec = {"tag": tag, "cand_vs_ref": cmp(ref, out, atol, rtol)}
            rec["dtype_ok"] = out.dtype == dt
            rec["shape_ok"] = tuple(out.shape) == tuple(ref.shape)
            rec["input_intact"] = (
                torch.equal(inp, in0) and torch.equal(index, idx0) and torch.equal(src, src0)
            )
            if dim % len(shape) != len(shape) - 1 and int(np.prod(shape)) <= 8192:
                cpu = torch.scatter_add(
                    inp.detach().to("cpu").double(),
                    dm,
                    index.detach().to("cpu"),
                    src.detach().to("cpu").double(),
                )
                rec["ref_vs_cpu64"] = cmp(cpu, ref.detach().to("cpu").double(), 1e-6, 1e-6)
                rec["cand_vs_cpu64"] = cmp(cpu, out.detach().to("cpu").double(), atol, rtol)
            RESULT["semantics"].append(rec)
            print("SEM", json.dumps(rec), flush=True)
        except Exception as exc:  # noqa: BLE001
            RESULT["semantics"].append({"tag": tag, "error": repr(exc)[:500]})
            print("SEM-FAIL", tag, repr(exc)[:400], flush=True)

    # empty index
    try:
        shape = [3, 0, 4]
        inp = torch.randn(shape, device=DEV, dtype=torch.float16)
        src = torch.randn(shape, device=DEV, dtype=torch.float16)
        index = torch.zeros(shape, dtype=torch.int64, device=DEV)
        ref = torch.scatter_add(inp, 1, index, src)
        out = cand.run(inp, 1, index, src)
        rec = {"tag": "empty-index", "cand_vs_ref": cmp(ref, out, 0.015, 0.015),
               "shape_ok": tuple(out.shape) == tuple(ref.shape)}
        RESULT["semantics"].append(rec)
        print("SEM", json.dumps(rec), flush=True)
    except Exception as exc:  # noqa: BLE001
        RESULT["semantics"].append({"tag": "empty-index", "error": repr(exc)[:400]})
        print("SEM-FAIL empty", repr(exc)[:300], flush=True)


# ------------------------------ perf ------------------------------------- #
@triton.jit(do_not_specialize=["ncols"])
def k_owner(inp_ptr, idx_ptr, src_ptr, out_ptr, ncols, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    dst = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dmask = dst < ncols
    base = row * ncols
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s0 in range(0, ncols, CHUNK):
        so = s0 + tl.arange(0, CHUNK)
        smask = so < ncols
        idx = tl.load(idx_ptr + base + so, mask=smask, other=-1).to(tl.int32)
        val = tl.load(src_ptr + base + so, mask=smask, other=0).to(tl.float32)
        hit = (idx[:, None] == dst[None, :]) & smask[:, None]
        acc += tl.sum(tl.where(hit, val[:, None], 0.0), axis=0)
    cur = tl.load(inp_ptr + base + dst, mask=dmask, other=0).to(tl.float32)
    tl.store(out_ptr + base + dst, (cur + acc).to(out_ptr.dtype.element_ty), mask=dmask)


def bench(fn, reps, warmup=2):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    semantics()

    timing = [
        (128, 128, torch.float16, 1729, 30),
        (128, 128, torch.bfloat16, 1732, 30),
        (128, 128, torch.float32, 1735, 30),
        (1024, 1024, torch.float16, 1730, 6),
        (1024, 1024, torch.bfloat16, 1733, 6),
        (1024, 1024, torch.float32, 1736, 6),
        (2048, 2048, torch.float16, 1731, 3),
        (2048, 2048, torch.bfloat16, 1734, 3),
        (2048, 2048, torch.float32, 1737, 3),
    ]
    for rows, ncols, dt, seed, reps in timing:
        try:
            inp, dm, index, src = gen(seed, [rows, ncols], -1, dt)
            ref = torch.scatter_add(inp, dm, index, src)
            out = cand.run(inp, dm, index, src)
            torch.npu.synchronize()
            atol, rtol = TOL[dt]
            rec = cmp(ref, out, atol, rtol)
            ms_cand = bench(lambda: cand.run(inp, dm, index, src), reps)
            ms_ref = bench(lambda: torch.scatter_add(inp, dm, index, src), reps)
            RESULT["perf"].append(
                {"shape": [rows, ncols], "dtype": str(dt), "cand_ms": ms_cand, "ref_ms": ms_ref,
                 "ratio": ms_ref / ms_cand, "check": rec}
            )
            print("PERF", rows, ncols, str(dt), "cand", round(ms_cand, 5), "ref", round(ms_ref, 5),
                  "ratio", round(ms_ref / ms_cand, 3), rec, flush=True)
        except Exception as exc:  # noqa: BLE001
            RESULT["perf"].append({"shape": [rows, ncols], "dtype": str(dt), "error": repr(exc)[:400]})
            print("PERF-FAIL", rows, ncols, repr(exc)[:300], flush=True)

    # dense parameter sweep on the two large fp16 shapes
    for rows, ncols, reps in ((1024, 1024, 5), (2048, 2048, 2)):
        inp, dm, index, src = gen(1730 if ncols == 1024 else 1731, [rows, ncols], -1, torch.float16)
        for block, chunk in ((128, 32), (256, 32), (256, 64), (512, 32), (1024, 32)):
            if block * chunk > 48 * 1024:
                continue
            try:
                out = torch.empty_like(inp)
                fn = lambda: k_owner[(rows, triton.cdiv(ncols, block))](
                    inp, index, src, out, ncols, BLOCK=block, CHUNK=chunk
                )
                fn()
                torch.npu.synchronize()
                ms = bench(fn, reps, warmup=1)
                RESULT["sweep"].append({"shape": [rows, ncols], "block": block, "chunk": chunk, "ms": ms})
                print("SWEEP", rows, ncols, block, chunk, round(ms, 5), flush=True)
            except Exception as exc:  # noqa: BLE001
                RESULT["sweep"].append(
                    {"shape": [rows, ncols], "block": block, "chunk": chunk, "error": repr(exc)[:300]}
                )
                print("SWEEP-FAIL", rows, ncols, block, chunk, repr(exc)[:200], flush=True)

    path = os.path.join(OUTDIR, "probe2.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(RESULT, fh, indent=1)
    print("WROTE", path, flush=True)


main()
