"""Host launch-cost breakdown + captured-graph replay probe.

9 of the 12 timing workloads are launch-bound (~16us/call vs reference ~12us), so
the host path is the dominant term in the geo mean.  This measures where the
~14.7us raw launch goes and whether a captured/replayed NPU graph is cheaper.

Writes $KGS_DEBUG_ARTIFACTS/selu_host.json
"""

import json
import os
import time

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
OUT = {}


@triton.jit
def k_selu(X, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offs)
    xf = x.to(tl.float32)
    y = 1.0507009873554805 * tl.where(xf > 0, xf, 1.6732632423543772 * (tl.exp(xf) - 1.0))
    tl.store(X + offs, y.to(X.dtype.element_ty))


def bench(fn, iters=2000, warmup=50):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    host_us = (time.perf_counter() - t0) / iters * 1e6
    torch.npu.synchronize()
    total_us = (time.perf_counter() - t0) / iters * 1e6
    return {"total_us": round(total_us, 3), "host_us": round(host_us, 3)}


def safe(fn, iters=2000):
    try:
        return bench(fn, iters=iters)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc).replace("\n", " ")[-200:]}


def main():
    OUT["torch_npu_graph_api"] = [n for n in dir(torch.npu) if "raph" in n]

    x = torch.randn(1024, dtype=torch.float32, device="npu:0")
    dev = torch.npu.current_device()
    raw = torch_npu._C._npu_getCurrentRawStreamNoWait
    compiled = k_selu.warmup(x, BLOCK=1024, grid=(1,))
    call = compiled.run.launch
    function = compiled.function
    metadata = compiled.packed_metadata
    stream_val = raw(dev)

    def stream_only():
        raw(dev)

    def launch_fixed_stream():
        call(1, 1, 1, stream_val, function, metadata, None, None, None, x, None)

    def launch_with_query():
        call(1, 1, 1, raw(dev), function, metadata, None, None, None, x, None)

    def jit_wrapper():
        k_selu[(1,)](x, BLOCK=1024)

    def torch_selu():
        torch.selu_(x)

    OUT["host_breakdown_n1024"] = {
        "stream_query_only": safe(stream_only, iters=5000),
        "raw_launch_fixed_stream": safe(launch_fixed_stream),
        "raw_launch_with_stream_query": safe(launch_with_query),
        "jit_wrapper_launch": safe(jit_wrapper),
        "torch_selu_inplace": safe(torch_selu),
    }

    # python-side dispatch cost of a cache-hit style fast path
    key_cache = {}
    key_cache[(torch.float32, 1024, True)] = launch_with_query

    def dict_lookup_path():
        key = (x.dtype, x.numel(), x.data_ptr() & 15 == 0)
        key_cache[key]()

    def identity_fast_path():
        launch_with_query()

    OUT["host_breakdown_n1024"]["dict_lookup_dispatch"] = safe(dict_lookup_path)
    OUT["host_breakdown_n1024"]["identity_fast_path"] = safe(identity_fast_path)

    # ---- captured graph replay
    graph_out = {}
    for api in ("NPUGraph", "CUDAGraph"):
        ctor = getattr(torch.npu, api, None)
        if ctor is None:
            continue
        try:
            g = ctor()
            side = torch.npu.Stream()
            side.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(side):
                for _ in range(3):
                    k_selu[(1,)](x, BLOCK=1024)
            torch.npu.current_stream().wait_stream(side)
            torch.npu.synchronize()
            with torch.npu.graph(g):
                k_selu[(1,)](x, BLOCK=1024)
            torch.npu.synchronize()
            graph_out[f"{api}_replay"] = safe(g.replay)
            # correctness of replay
            probe = torch.randn(1024, dtype=torch.float32, device="npu:0")
            expect = torch.selu(probe.clone())
            got = probe.clone()
            g2 = ctor()
            side2 = torch.npu.Stream()
            side2.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(side2):
                for _ in range(3):
                    k_selu[(1,)](got, BLOCK=1024)
            torch.npu.current_stream().wait_stream(side2)
            torch.npu.synchronize()
            with torch.npu.graph(g2):
                k_selu[(1,)](got, BLOCK=1024)
            g2.replay()
            torch.npu.synchronize()
            graph_out[f"{api}_correct"] = bool(torch.equal(got, expect))
        except Exception as exc:  # noqa: BLE001
            graph_out[f"{api}_error"] = str(exc).replace("\n", " ")[-260:]
    OUT["graph"] = graph_out

    # ---- larger n: confirm replay scales and stays cheap
    x2 = torch.randn(1 << 20, dtype=torch.float32, device="npu:0")
    OUT["large_n_reference_points"] = {
        "torch_selu_inplace_1M": safe(lambda: torch.selu_(x2), iters=1000),
        "jit_1M_b8192": safe(lambda: k_selu[((1 << 20) // 8192,)](x2, BLOCK=8192), iters=1000),
    }

    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "selu_host.json"), "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print(json.dumps(OUT, indent=2, default=str))


if __name__ == "__main__":
    main()
