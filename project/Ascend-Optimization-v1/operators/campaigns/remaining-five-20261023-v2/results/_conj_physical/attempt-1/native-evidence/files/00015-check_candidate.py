"""Independent numerics/layout check of the candidate against a CPU reference.

Values are drawn on the CPU with the reference seed, the *layout* (transpose /
conj bit / slicing) is applied identically on the CPU (for the expected value)
and on the device (for the candidate), so device inputs really do carry lazy
conjugate bits and non-contiguous strides.
"""
import importlib.util
import json
import os
import traceback

import torch
import torch_npu  # noqa: F401

MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "main.py")
spec = importlib.util.spec_from_file_location("cand", MAIN)
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

DEV = "npu"
OUT = {}

CORRECTNESS = [
    ("c0", {"dtype": "float32", "shape": [7, 9], "transpose": True, "conj": True}, 1729, 1e-4),
    ("c1", {"dtype": "float32", "shape": [7, 9], "transpose": True, "conj": False}, 1730, 1e-4),
    ("c2", {"dtype": "float32", "shape": [7, 9], "transpose": False, "conj": True}, 1731, 1e-4),
    ("c3", {"dtype": "float32", "shape": [7, 9], "transpose": False, "conj": False}, 1732, 1e-4),
    ("c4", {"dtype": "complex64", "shape": [7, 9], "transpose": True, "conj": True}, 1733, 1.5e-2),
    ("c5", {"dtype": "complex64", "shape": [7, 9], "transpose": True, "conj": False}, 1734, 1.5e-2),
    ("c6", {"dtype": "complex64", "shape": [7, 9], "transpose": False, "conj": True}, 1735, 1.5e-2),
    ("c7", {"dtype": "complex64", "shape": [7, 9], "transpose": False, "conj": False}, 1736, 1.5e-2),
]

TIMING = [
    ("t0", {"dtype": "float32", "shape": [1024]}, 1729, 1e-4),
    ("t1", {"dtype": "float32", "shape": [65536]}, 1730, 1e-4),
    ("t2", {"dtype": "float32", "shape": [1048576]}, 1731, 1e-4),
    ("t3", {"dtype": "complex64", "shape": [1024]}, 1732, 1.5e-2),
    ("t4", {"dtype": "complex64", "shape": [65536]}, 1733, 1.5e-2),
    ("t5", {"dtype": "complex64", "shape": [1048576]}, 1734, 1.5e-2),
]


def make_cpu(s, seed):
    dtype = getattr(torch, s["dtype"])
    g = torch.Generator().manual_seed(seed)
    shape = s["shape"]
    if dtype.is_complex:
        return torch.complex(
            torch.randn(shape, generator=g), torch.randn(shape, generator=g)
        ).to(dtype=dtype)
    return torch.randn(shape, generator=g).to(dtype=dtype)


def std_layout(s):
    def f(x):
        if s.get("transpose"):
            x = x.transpose(-1, -2)
        if s.get("conj"):
            x = x.conj()
        return x

    return f


def check(name, base_cpu, layout, tol, note=""):
    res = {"note": note}
    try:
        expected = torch.ops.aten._conj_physical(layout(base_cpu))
        x_dev = layout(base_cpu.to(DEV))
        res["in_stride"] = list(x_dev.stride())
        res["in_bit"] = bool(x_dev.is_conj())
        snapshot = x_dev.detach().cpu().clone()
        out = cand.run(x_dev)
        got = out.detach().cpu()
        res.update(
            shape_ok=list(out.shape) == list(expected.shape),
            dtype_ok=out.dtype == expected.dtype,
            contiguous=bool(out.is_contiguous()),
            # a 0-element allocation may legally share an address with the input
            no_alias=(out.data_ptr() != x_dev.data_ptr()) or out.numel() == 0,
            input_untouched=bool(torch.equal(x_dev.detach().cpu(), snapshot)),
            bit_free=not out.is_conj(),
            out_stride=list(out.stride()),
        )
        if got.dtype.is_complex:
            res["max_re"] = (got.real - expected.real).abs().max().item() if got.numel() else 0.0
            res["max_im"] = (got.imag - expected.imag).abs().max().item() if got.numel() else 0.0
        else:
            res["max_abs"] = (
                (got.float() - expected.float()).abs().max().item() if got.numel() else 0.0
            )
        res["values_ok"] = bool(
            got.numel() == 0 or torch.allclose(got, expected, rtol=tol, atol=tol)
        )
        res["ok"] = bool(
            res["values_ok"]
            and res["shape_ok"]
            and res["dtype_ok"]
            and res["no_alias"]
            and res["input_untouched"]
            and res["bit_free"]
        )
    except Exception:
        res["error"] = traceback.format_exc()
        res["ok"] = False
    OUT[name] = res
    return res.get("ok", False)


ok_all = True
for name, s, seed, tol in CORRECTNESS + TIMING:
    ok_all &= check(name, make_cpu(s, seed), std_layout(s), tol, f"{s['dtype']}{s['shape']}")

extras = []


def extra(name, base_cpu, layout, tol, note=""):
    extras.append((name, check(name, base_cpu, layout, tol, note)))


ident = lambda x: x

extra("x_3d_complex_transposed", torch.randn(3, 4, 5, dtype=torch.complex64),
      lambda x: x.transpose(0, 2), 1.5e-2, "3d complex transpose(0,2)")
extra("x_gap_slice", torch.randn(7, 18), lambda x: x[:, ::2], 1e-4, "stride-2 columns")
extra("x_offset_view", torch.randn(10, 10), lambda x: x[3:6, 2:9], 1e-4, "narrow offset both dims")
extra("x_complex_gap_conj", torch.randn(5, 8, dtype=torch.complex64),
      lambda x: x[:, ::2].conj(), 1.5e-2, "complex stride2 + conj bit")
extra("x_complex_transposed_offset", torch.randn(6, 9, dtype=torch.complex64),
      lambda x: x[1:6, 2:7].transpose(-1, -2), 1.5e-2, "offset+transpose")
extra("x_complex_channel", torch.randn(4, 6, 8, dtype=torch.complex64),
      lambda x: x.permute(0, 2, 1), 1.5e-2, "complex permute")
extra("x_fp16_stride2", torch.randn(129, 2).to(torch.float16),
      lambda x: x[:, 0], 1e-3, "stride2 fp16")
extra("x_bf16_stride2", torch.randn(129, 2).to(torch.bfloat16),
      lambda x: x[:, 0], 1e-2, "stride2 bf16")
extra("x_single", torch.randn(1), ident, 1e-4, "1 element")
extra("x_scalar0d", torch.randn(()), ident, 1e-4, "0-dim")
extra("x_empty1", torch.randn(0), ident, 1e-4, "empty 1d")
extra("x_empty3", torch.randn(2, 0, 3), ident, 1e-4, "empty 3d")
extra("x_empty_complex", torch.randn(0, 3, dtype=torch.complex64), ident, 1.5e-2, "empty complex")
extra("x_complex_big_transposed", torch.randn(257, 1023, dtype=torch.complex64),
      lambda x: x.transpose(-1, -2), 1.5e-2, "large transposed complex (strided path)")
extra("x_real_big_strided", torch.randn(513, 2049),
      lambda x: x[:, 1:], 1e-4, "large offset real (strided path)")
extra("x_unaligned_offset", torch.randn(4097), lambda x: x[1:], 1e-4, "odd offset real")
extra("x_complex_1m_transposed", torch.randn(1024, 1024, dtype=torch.complex64),
      lambda x: x.transpose(-1, -2), 1.5e-2, "1M complex transposed (strided, UB check)")
extra("x_real_fp16_big_transposed", torch.randn(512, 2048).to(torch.float16),
      lambda x: x.transpose(-1, -2), 1e-3, "1M fp16 transposed (strided, UB check)")

ok_all &= all(ok for _, ok in extras)

seq_ok = True
for name, s, seed, tol in (CORRECTNESS[4], CORRECTNESS[0], TIMING[3], TIMING[2], CORRECTNESS[6]):
    seq_ok &= check("seq_" + name, make_cpu(s, seed), std_layout(s), tol, "sequence reuse")
ok_all &= seq_ok

stream_ok = True
try:
    s2 = torch.npu.Stream()
    base = torch.randn(7, 9, dtype=torch.complex64)
    lay = std_layout({"transpose": True, "conj": True})
    expected = torch.ops.aten._conj_physical(lay(base))
    x_dev = lay(base.to(DEV))
    with torch.npu.stream(s2):
        out = cand.run(x_dev)
        torch.npu.current_stream().synchronize()
    stream_ok = bool(
        list(out.shape) == [9, 7] and torch.allclose(out.cpu(), expected, rtol=1.5e-2, atol=1.5e-2)
    )
    OUT["stream"] = {"ok": stream_ok, "shape": list(out.shape)}
except Exception:
    OUT["stream"] = {"ok": False, "error": traceback.format_exc()}
ok_all &= stream_ok

OUT["_summary"] = {
    "all_ok": bool(ok_all),
    "failures": [k for k, v in OUT.items() if isinstance(v, dict) and v.get("ok") is False],
}
art = os.environ.get("KGS_DEBUG_ARTIFACTS", ".")
os.makedirs(art, exist_ok=True)
with open(os.path.join(art, "check_candidate.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=str)
print(json.dumps(OUT["_summary"], indent=1))
print(json.dumps({k: v for k, v in OUT.items() if isinstance(v, dict) and k != "_summary"},
                 indent=1, default=str)[:9000])
