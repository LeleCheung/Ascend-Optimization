"""Split the residual per-call cost: allocation vs launcher vs Python glue."""
import importlib.util
import json
import os
import time

import torch
import torch_npu  # noqa: F401

MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main.py")
spec = importlib.util.spec_from_file_location("cand3", MAIN)
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

DEV = "npu"
OUT = {}


def wall(fn, iters, warmup=20):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def build_complex(n):
    return torch.complex(torch.randn(n), torch.randn(n)).to(device=DEV)


x = torch.randn(1024, device=DEV)
xc = build_complex(1024)
cand.run(x)
cand.run(xc)
grid = -(-1024 // cand._block_for(1024, cand._COPY_TINY, cand._COPY_MID, cand._COPY_LARGE))
entry = cand._copy_cache[(torch.float32, cand._block_for(1024, cand._COPY_TINY, cand._COPY_MID, cand._COPY_LARGE), x.data_ptr() & 15, 0)]
out = torch.empty(1024, device=DEV)
block = cand._block_for(1024, cand._COPY_TINY, cand._COPY_MID, cand._COPY_LARGE)
launch = entry[0]
fn_handle = entry[1]
meta = entry[2]
sp = x.data_ptr()
dp = out.data_ptr()
stream = cand._cur_stream()

OUT["split_ms"] = {
    "full_run": wall(lambda: cand.run(x), 2000),
    "full_run_complex": wall(lambda: cand.run(xc), 1000),
    "alloc_only": wall(lambda: torch.empty(1024, dtype=torch.float32, device=DEV), 2000),
    "alloc_shape": wall(lambda: torch.empty(x.shape, dtype=x.dtype, device=x.device), 2000),
    "alloc_like": wall(lambda: torch.empty_like(x), 2000),
    "launcher_only": wall(
        lambda: launch(grid, 1, 1, stream, fn_handle, meta, None, None, None, sp, dp, 1024, block),
        2000,
    ),
    "launcher_with_stream_query": wall(
        lambda: launch(grid, 1, 1, cand._cur_stream(), fn_handle, meta, None, None, None, sp, dp, 1024, block),
        2000,
    ),
    "stream_only": wall(lambda: cand._cur_stream(), 2000),
    "launcher_tensor_args": wall(
        lambda: launch(grid, 1, 1, stream, fn_handle, meta, None, None, None, x, out, 1024, block),
        2000,
    ),
}

# does a bigger tile change the device cost at 1M elements?
for n, dt, label in ((1048576, torch.float32, "f32_1m"), (1048576, torch.complex64, "c64_1m")):
    t = build_complex(n) if dt.is_complex else torch.randn(n, device=DEV)
    OUT.setdefault("device_scale", {})[label] = wall(lambda: cand.run(t), 200)

art = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
os.makedirs(art, exist_ok=True)
with open(os.path.join(art, "probe_split.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
