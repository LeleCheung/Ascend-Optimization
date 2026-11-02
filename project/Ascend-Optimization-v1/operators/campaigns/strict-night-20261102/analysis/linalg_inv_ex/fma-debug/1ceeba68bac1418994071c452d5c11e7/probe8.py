"""Probe 8: read the panel width straight off the device.

Runs the candidate's LU kernel on the two singular workloads for BLK in
{1,2,3,4} and compares the produced factorisation bit-exactly with the
reference's LU of A^T (torch.linalg.lu_factor_ex), which the earlier probes
showed reproduces the reference inverse.
"""
import importlib.util
import os

import numpy as np
import torch

np.seterr(all="ignore")
np.set_printoptions(precision=10, linewidth=200)


def load_candidate():
    spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(shape, seed, singular, device="npu:0"):
    gen = torch.Generator().manual_seed(seed)

    def draw(shp):
        return torch.randn(shp, generator=gen).to(device=device, dtype=torch.float32)

    draw(shape)
    x = draw(shape)
    n = shape[-1]
    x = x + torch.eye(n, device=device, dtype=torch.float32) * (n + 1)
    if singular:
        x[..., -1, :] = x[..., 0, :]
    return x


def main():
    cand = load_candidate()
    print("has tl.math.fma:", hasattr(torch, "_dummy"), flush=True)
    import triton

    cases = [("correctness-1", [3, 3], 1730), ("correctness-3", [2, 7, 7], 1732),
             ("correctness-2", [2, 7, 7], 1731)]
    for name, shape, seed in cases:
        n = shape[-1]
        singular = name in ("correctness-1", "correctness-3")
        A = build(shape, seed, singular)
        A_cpu = A.cpu()
        ref = torch.linalg.inv_ex(A_cpu, check_errors=False).inverse.numpy()
        flat = A_cpu.reshape(-1, n, n)
        AT = flat.transpose(-1, -2).contiguous()
        lu_ref, piv_ref, _ = torch.linalg.lu_factor_ex(AT)
        lu_ref = lu_ref.numpy()
        print("=" * 60)
        print(name, "torch LU(AT) last diag:", lu_ref[:, -1, -1], flush=True)
        A_dev = A.reshape(-1, n, n).contiguous()
        batch = flat.shape[0]
        block_n = max(triton.next_power_of_2(n), 16)
        nw = 4 if block_n <= 32 else 8
        for blk in (1, 2, 3, 4):
            lu = torch.empty(batch, n, n, dtype=torch.float32, device="npu:0")
            info = torch.empty(batch, dtype=torch.int32, device="npu:0")
            piv = torch.empty(batch, n, dtype=torch.int32, device="npu:0")
            unit = torch.ones(1, dtype=torch.float32, device="npu:0")
            try:
                cand._linalg_lu_kernel[(batch,)](
                    A_dev, lu, info, piv, unit,
                    N=n, BLOCK_N=block_n, BLK=blk, num_warps=nw, num_stages=1,
                )
                torch.npu.synchronize()
            except Exception as exc:  # noqa: BLE001
                print("  BLK=%d launch failed: %s" % (blk, str(exc)[:300]), flush=True)
                continue
            got = lu.transpose(-1, -2).cpu().numpy()
            exact = np.array_equal(got.view(np.int32), lu_ref.view(np.int32))
            mism = int((got.view(np.int32) != lu_ref.view(np.int32)).sum())
            lastdiag = got[:, -1, -1]
            print("  BLK=%d exact=%s mism=%d lastdiag=%s info=%s"
                  % (blk, exact, mism, [float(x) for x in lastdiag], info.cpu().numpy().tolist()), flush=True)
            if not exact:
                d = np.argwhere(got.view(np.int32) != lu_ref.view(np.int32))
                shown = [(int(i), int(j), int(k), float(got[i, j, k]), float(lu_ref[i, j, k])) for i, j, k in d[:4]]
                print("     first diffs:", shown, flush=True)
        try:
            inv = cand.linalg_inv_ex(A)
            ig = inv.inverse.cpu().numpy()
            rel = np.abs(ig - ref) / np.maximum(np.abs(ref), 1e-300)
            ok = (np.abs(ig - ref) <= 1e-4 + 1e-4 * np.abs(ref))
            print("  wrapper matched=%d/%d maxrel=%.4g"
                  % (int(ok.sum()), ok.size, float(np.nanmax(rel))), flush=True)
        except Exception as exc:  # noqa: BLE001
            print("  wrapper failed:", str(exc)[:300], flush=True)


if __name__ == "__main__":
    main()
