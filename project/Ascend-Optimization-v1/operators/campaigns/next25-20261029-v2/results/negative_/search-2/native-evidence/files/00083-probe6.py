"""Decompose the remaining ~30 us per-call host cost of the round-3 candidate:
pointer-arg marshalling (data_ptr int vs tensor object), stream acquisition
path, launcher arg count, and the fully pre-bound floor.  Diagnostics only.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as cand  # noqa: E402

ART = Path(os.environ.get("KGS_DEBUG_ARTIFACTS", ".")) / "probe6.json"
DEV = "npu:0"
out = {"percall": {}, "stream": {}, "notes": []}


@triton.jit
def _neg_lit1024(X):
    o = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    tl.store(X + o, -tl.load(X + o))


def bench(fn, iters=300, warmup=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    ts = []
    for _ in range(3):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        ts.append((time.perf_counter() - t0) / iters * 1e3)
    ts.sort()
    return round(ts[1], 4)


def main():
    a = torch.zeros(1024, dtype=torch.float16, device=DEV)
    drv = cand._triton_driver
    dev = cand._device_index()

    # ---- stream acquisition paths -------------------------------------
    s_driver = drv.active.get_current_stream(dev)
    out["stream"]["driver_int"] = s_driver
    try:
        from torch_npu._C import _npu_getCurrentRawStreamNoWait as raw_stream
        out["stream"]["npu_c"] = raw_stream(dev)
        out["stream"]["npu_c_matches"] = out["stream"]["npu_c"] == s_driver
    except Exception:
        raw_stream = None
        out["stream"]["npu_c_err"] = traceback.format_exc()[-300:]
    try:
        cs = torch.npu.current_stream()
        out["stream"]["torch_attrs"] = [x for x in dir(cs) if not x.startswith("__")][:40]
        for attr in ("npu_stream", "_as_parameter_", "stream_id", "id"):
            try:
                v = getattr(cs, attr)
                out["stream"]["attr_" + attr] = int(v) if not callable(v) else "callable"
            except Exception:
                out["stream"]["attr_" + attr] = "missing"
    except Exception:
        out["stream"]["torch_err"] = traceback.format_exc()[-300:]

    out["percall"]["stream_driver"] = bench(lambda: drv.active.get_current_stream(dev))
    if raw_stream is not None:
        out["percall"]["stream_npu_c"] = bench(lambda: raw_stream(dev))
    out["percall"]["torch_current_stream"] = bench(lambda: torch.npu.current_stream())
    out["percall"]["data_ptr"] = bench(lambda: a.data_ptr())
    out["percall"]["baseline_cand_run"] = bench(lambda: cand.run(a))

    # ---- pre-bound launch variants ------------------------------------
    ck = cand._neg_exact.warmup(a, grid=(1,), BLOCK=1024)
    launch, fn, packed = ck.run, ck.function, ck.packed_metadata
    S, P = s_driver, a.data_ptr()
    ck_lit = _neg_lit1024.warmup(a, grid=(1,))
    l2, f2, p2 = ck_lit.run, ck_lit.function, ck_lit.packed_metadata

    def v_tensor_driver():
        s = drv.active.get_current_stream(dev)
        launch(1, 1, 1, s, fn, packed, None, None, None, a, 1024)

    def v_ptr_driver():
        s = drv.active.get_current_stream(dev)
        launch(1, 1, 1, s, fn, packed, None, None, None, a.data_ptr(), 1024)

    def v_ptr_cached_stream():
        launch(1, 1, 1, S, fn, packed, None, None, None, a.data_ptr(), 1024)

    def v_ptr_cached_stream_cached_ptr():
        launch(1, 1, 1, S, fn, packed, None, None, None, P, 1024)

    def v_lit_ptr_driver():
        s = drv.active.get_current_stream(dev)
        l2(1, 1, 1, s, f2, p2, None, None, None, a.data_ptr())

    def v_lit_ptr_cached():
        l2(1, 1, 1, S, f2, p2, None, None, None, a.data_ptr())

    if raw_stream is not None:
        def v_ptr_npuc():
            s = raw_stream(dev)
            launch(1, 1, 1, s, fn, packed, None, None, None, a.data_ptr(), 1024)
        out["percall"]["ptr_npuc_stream"] = bench(v_ptr_npuc)
    out["percall"]["tensor_driver"] = bench(v_tensor_driver)
    out["percall"]["ptr_driver"] = bench(v_ptr_driver)
    out["percall"]["ptr_cached_stream"] = bench(v_ptr_cached_stream)
    out["percall"]["ptr_cached_stream_ptr"] = bench(v_ptr_cached_stream_cached_ptr)
    out["percall"]["lit_ptr_driver"] = bench(v_lit_ptr_driver)
    out["percall"]["lit_ptr_cached"] = bench(v_lit_ptr_cached)

    # ---- correctness of the int-pointer launch ------------------------
    try:
        x = torch.arange(0, 1024, dtype=torch.float16, device=DEV) + 1.0
        before = x.clone()
        ck2 = cand._neg_exact.warmup(x, grid=(1,), BLOCK=1024)
        l3, f3, p3 = ck2.run, ck2.function, ck2.packed_metadata
        s = drv.active.get_current_stream(dev)
        l3(1, 1, 1, s, f3, p3, None, None, None, x.data_ptr(), 1024)
        torch.npu.synchronize()
        out["int_ptr_launch_correct"] = bool(torch.equal(x, -before))
    except Exception:
        out["int_ptr_launch_correct"] = "ERR:" + traceback.format_exc()[-400:]


try:
    main()
except Exception:
    out["fatal"] = traceback.format_exc()
finally:
    ART.parent.mkdir(parents=True, exist_ok=True)
    ART.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str)[:6000])
