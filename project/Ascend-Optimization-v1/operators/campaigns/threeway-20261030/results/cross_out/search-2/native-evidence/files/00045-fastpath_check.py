"""Stress the candidate's memoized fast path for correctness and stability.

Covers: repeated identical calls, allocator address reuse after free, interleaved
shapes/dtypes, the general ND path, strided and broadcast inputs, zero-size, and
the aliased-out case. Diagnostic evidence only.
"""
import importlib.util
import json
import os
import traceback

import torch
import torch_npu


def load_candidate():
    spec = importlib.util.spec_from_file_location("cand_main", "tmp/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def maxerr(got, want):
    a = got.float()
    b = want.float()
    if a.numel() == 0:
        return 0.0
    return (a - b).abs().max().item()


def main():
    m = load_candidate()
    dev = "npu:0"
    out = {}
    print("loaded", flush=True)

    # 0. tiny flat tensors first: these are the shapes that faulted
    tiny = []
    for nv in (1, 2, 3, 7, 11, 341, 342, 683):
        for dt in (torch.float16, torch.float32):
            x = torch.randn(nv, 3, dtype=dt, device=dev)
            y = torch.randn(nv, 3, dtype=dt, device=dev)
            z = torch.empty_like(x)
            want = torch.cross(x.clone(), y.clone(), dim=-1)
            m.run(x, y, -1, out=z)
            torch.npu.synchronize()
            tiny.append({"nv": nv, "dt": str(dt), "err": maxerr(z, want)})
    out["tiny"] = tiny
    print("tiny done", flush=True)

    # 1. repeated identical calls (timing-loop shape)
    a = torch.randn(1048576, 3, dtype=torch.float16, device=dev)
    b = torch.randn(1048576, 3, dtype=torch.float16, device=dev)
    o = torch.empty_like(a)
    ref = torch.cross(a, b, dim=-1)
    errs = []
    for _ in range(20):
        m.run(a, b, -1, out=o)
        torch.npu.synchronize()
        errs.append(maxerr(o, ref))
    out["repeat_fp16_1M"] = {"max_err": max(errs)}
    print("repeat done", flush=True)

    # 2. allocator address reuse: free, reallocate, same shape/dtype
    reuse = []
    for _ in range(6):
        a2 = torch.randn(1048576, 3, dtype=torch.float16, device=dev)
        b2 = torch.randn(1048576, 3, dtype=torch.float16, device=dev)
        o2 = torch.empty_like(a2)
        r2 = torch.cross(a2, b2, dim=-1)
        m.run(a2, b2, -1, out=o2)
        torch.npu.synchronize()
        reuse.append({"err": maxerr(o2, r2), "ptrs": [a2.data_ptr(), b2.data_ptr(), o2.data_ptr()]})
        del a2, b2, o2, r2
    out["addr_reuse"] = reuse
    print("addr_reuse done", flush=True)

    # 3. interleaved shapes and dtypes through the same entry point
    inter = []
    for dt in (torch.float16, torch.float32, torch.bfloat16):
        for nv in (1024, 65536, 7):
            x = torch.randn(nv, 3, dtype=dt, device=dev)
            y = torch.randn(nv, 3, dtype=dt, device=dev)
            z = torch.empty_like(x)
            m.run(x, y, -1, out=z)
            m.run(a, b, -1, out=o)  # force memo thrash
            torch.npu.synchronize()
            inter.append({"dtype": str(dt), "nv": nv, "err": maxerr(z, torch.cross(x, y, dim=-1))})
    out["interleaved"] = inter
    print("interleaved done", flush=True)

    # 4. general ND path: middle dim, 3D, strided view, broadcast
    gen = []
    cases = []
    x = torch.randn(2, 3, 9, dtype=torch.float32, device=dev)
    cases.append((x, torch.randn(2, 3, 9, dtype=torch.float32, device=dev), 1, None))
    y = torch.randn(3, 7, dtype=torch.float16, device=dev)
    cases.append((y, torch.randn(3, 7, dtype=torch.float16, device=dev), 0, None))
    base = torch.randn(2, 3, 9, 2, dtype=torch.float32, device=dev)
    p = base[..., 0]
    q = torch.randn(2, 3, 9, dtype=torch.float32, device=dev)
    cases.append((p, q, 1, None))
    bs = torch.randn(1, 3, 5, dtype=torch.float32, device=dev)
    bo = torch.randn(4, 3, 5, dtype=torch.float32, device=dev)
    cases.append((bs, bo, 1, None))
    for (xa, xb, d, _) in cases:
        want = torch.cross(xa, xb, dim=d)
        got = torch.empty_like(want)
        m.run(xa, xb, d, out=got)
        torch.npu.synchronize()
        gen.append({"shape": list(want.shape), "dim": d, "err": maxerr(got, want)})
    out["general"] = gen
    print("general done", flush=True)

    # 5. aliased out (out is the input tensor itself)
    try:
        al = torch.randn(64, 3, dtype=torch.float32, device=dev)
        al2 = torch.randn(64, 3, dtype=torch.float32, device=dev)
        want = torch.cross(al.clone(), al2, dim=-1)
        m.run(al, al2, -1, out=al)
        torch.npu.synchronize()
        out["alias_out"] = {"err_vs_reference_on_fresh_input": maxerr(al, want)}
    except Exception:
        out["alias_out"] = {"error": traceback.format_exc().splitlines()[-1]}

    # 6. zero-size
    try:
        z0 = torch.empty(0, 3, dtype=torch.float32, device=dev)
        z1 = torch.empty(0, 3, dtype=torch.float32, device=dev)
        zo = torch.empty_like(z0)
        r = m.run(z0, z1, -1, out=zo)
        out["zero"] = {"numel": int(r.numel())}
    except Exception:
        out["zero"] = {"error": traceback.format_exc().splitlines()[-1]}

    path = os.path.join(os.environ.get("KGS_DEBUG_ARTIFACTS", "."), "fastpath_check.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
