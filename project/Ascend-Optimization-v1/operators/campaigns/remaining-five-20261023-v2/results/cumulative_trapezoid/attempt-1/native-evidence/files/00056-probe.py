"""Isolate cumulative_trapezoid kernel cost: GM access vs scan vs mask.

Component probe (NOT a formal evaluation). Reports device time for
  copy   : masked 1D loads + add + masked store          (no scan)
  scan   : same + tl.cumsum(axis=0) / tl.cumsum(axis=1)
  nomask : same as copy/scan but with zero masks
Also dumps the generated IR keys/patterns and the available cann extension symbols.
"""
import json
import os
import traceback

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
DEV = "npu"


@triton.jit
def k_copy_masked(y_ptr, o_ptr, L: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BN)
    base = pid * L
    m = cols < (L - 1)
    a = tl.load(y_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
    b = tl.load(y_ptr + base + cols + 1, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + pid * (L - 1) + cols, a + b, mask=m)


@triton.jit
def k_scan_masked(y_ptr, o_ptr, L: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BN)
    base = pid * L
    m = cols < (L - 1)
    a = tl.load(y_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
    b = tl.load(y_ptr + base + cols + 1, mask=m, other=0.0).to(tl.float32)
    t = (a + b) * 0.5
    cs = tl.cumsum(t, axis=0)
    tl.store(o_ptr + pid * (L - 1) + cols, cs, mask=m)


@triton.jit
def k_copy_nomask(y_ptr, o_ptr, L: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BN)
    base = pid * L
    a = tl.load(y_ptr + base + cols).to(tl.float32)
    b = tl.load(y_ptr + base + cols + 1).to(tl.float32)
    tl.store(o_ptr + pid * L + cols, a + b)


@triton.jit
def k_scan_nomask(y_ptr, o_ptr, L: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BN)
    base = pid * L
    a = tl.load(y_ptr + base + cols).to(tl.float32)
    b = tl.load(y_ptr + base + cols + 1).to(tl.float32)
    t = (a + b) * 0.5
    cs = tl.cumsum(t, axis=0)
    tl.store(o_ptr + pid * L + cols, cs)


@triton.jit
def k_scan_2d_nomask(y_ptr, o_ptr, L: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    off = rows[:, None] * L + cols[None, :]
    a = tl.load(y_ptr + off).to(tl.float32)
    b = tl.load(y_ptr + off + 1).to(tl.float32)
    t = (a + b) * 0.5
    cs = tl.cumsum(t, axis=1)
    tl.store(o_ptr + off, cs)


@triton.jit
def k_blocked_scan(y_ptr, o_ptr, M, L: tl.constexpr, BM: tl.constexpr,
                   BN: tl.constexpr, NCHUNK: tl.constexpr):
    """Blocked scan: BM rows per program, NCHUNK sequential chunks of BN."""
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    carry = tl.zeros((BM,), dtype=tl.float32)
    for c in tl.static_range(NCHUNK):
        off = rows[:, None] * L + (c * BN + cols)[None, :]
        a = tl.load(y_ptr + off).to(tl.float32)
        b = tl.load(y_ptr + off + 1).to(tl.float32)
        t = (a + b) * 0.5
        cs = tl.cumsum(t, axis=1) + carry[:, None]
        tl.store(o_ptr + off, cs)
        carry += tl.sum(t, axis=1)


def bench(fn, iters=20):
    fn()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) / iters


def main():
    out = {"timings_ms": {}, "notes": [], "errors": {}}
    M, L = 1024, 4096
    pad = 64
    torch.manual_seed(0)
    yflat = torch.randn(M * L + pad, device=DEV, dtype=torch.float16)
    y = yflat[: M * L].view(M, L)

    o_pad = torch.empty(M * L + pad, device=DEV, dtype=torch.float16)
    o_pack = torch.empty(M * (L - 1), device=DEV, dtype=torch.float16)
    o_same = o_pad[: M * L].view(M, L)

    def t(name, fn):
        try:
            out["timings_ms"][name] = round(bench(fn), 4)
        except Exception as ex:  # noqa: BLE001
            out["errors"][name] = "".join(traceback.format_exception_only(type(ex), ex))

    t("copy_masked", lambda: k_copy_masked[(M,)](y, o_pack, L, L))
    t("scan_masked", lambda: k_scan_masked[(M,)](y, o_pack, L, L))
    t("copy_nomask", lambda: k_copy_nomask[(M,)](y, o_pad, L, L))
    t("scan_nomask", lambda: k_scan_nomask[(M,)](y, o_pad, L, L))
    for bm in (2, 4, 8):
        t(f"scan_2d_bm{bm}", lambda bm=bm: k_scan_2d_nomask[(M // bm,)](y, o_pad, L, bm, L))
    for bm, bn in ((8, 512), (16, 256), (32, 128), (16, 512)):
        nc = L // bn
        t(f"blocked_bm{bm}_bn{bn}",
          lambda bm=bm, bn=bn, nc=nc: k_blocked_scan[(M // bm,)](y, o_same, M, L, bm, bn, nc))

    # Pure reference-side device cost for the same input.
    try:
        def ref():
            torch.cumulative_trapezoid(y, dx=1, dim=-1)
        out["timings_ms"]["torch_ref_full_call"] = round(bench(ref), 4)
    except Exception as ex:  # noqa: BLE001
        out["errors"]["torch_ref_full_call"] = repr(ex)

    # IR introspection for the seed-style scan kernel.
    try:
        k_scan_masked[(M,)](y, o_pack, L, L)
        torch.npu.synchronize()
        dev = torch.npu.current_device()
        caches = getattr(k_scan_masked, "device_caches", None)
        info = {"device_caches_type": str(type(caches))}
        found = {}

        def walk(obj, depth=0):
            if depth > 6 or len(found) > 40:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if hasattr(v, "asm"):
                        found[f"asm_{k}"] = list(v.asm.keys())
                        found[f"asm_{k}_ttgir_head"] = str(v.asm.get("ttgir", ""))[:3000]
                    walk(v, depth + 1)
            elif isinstance(obj, (list, tuple)):
                for v in obj[:8]:
                    walk(v, depth + 1)

        walk(caches)
        info["found"] = found
        if not found:
            try:
                kern = list(k_scan_masked.device_caches[dev][0].values())[0]
                info["asm_keys"] = list(kern.asm.keys())
                info["ttgir_head"] = str(kern.asm.get("ttgir", ""))[:3000]
                info["n_scf_for"] = str(kern.asm.get("ttgir", "")).count("scf.for")
            except Exception as ex:  # noqa: BLE001
                info["fallback_error"] = repr(ex)
        out["ir"] = info
    except Exception as ex:  # noqa: BLE001
        out["errors"]["ir"] = repr(ex)

    # Available native scan/vector helpers.
    try:
        import triton.language.extra as ex_mod
        import triton.language.extra.cann as cann_mod

        syms = {}
        for mod, nm in ((ex_mod, "extra"), (cann_mod, "extra.cann")):
            syms[nm] = [s for s in dir(mod) if not s.startswith("_")]
        try:
            import triton.language.extra.cann.extension as ext
            syms["extra.cann.extension"] = [s for s in dir(ext) if not s.startswith("_")]
        except Exception as ex2:  # noqa: BLE001
            syms["extra.cann.extension"] = repr(ex2)
        out["native_symbols"] = syms
    except Exception as ex:  # noqa: BLE001
        out["errors"]["symbols"] = repr(ex)

    try:
        out["device_props"] = {
            "name": torch.npu.get_device_name(0),
            "num_vectorcore": getattr(
                __import__("torch_npu.utils").utils, "get_device_properties", lambda *_: {}
            )(0) if False else "skipped",
        }
    except Exception:  # noqa: BLE001
        pass

    with open(os.path.join(ART, "probe.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out["timings_ms"], indent=2))
    print("errors:", json.dumps(out["errors"], indent=2))


if __name__ == "__main__":
    main()
