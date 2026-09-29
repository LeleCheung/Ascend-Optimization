"""Tests for coder workload shape enrichment.

Runs on host (no real torch needed): one test exercises the graceful-degradation
path, another injects a tiny fake torch to exercise the enrichment path.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernelgen.agents.coder.workload_shapes import enrich_workloads_with_shapes

REF = (
    "import torch\n"
    "CASES = {0: ((1024, 1024), torch.float16), 1: ((4096,), torch.float32)}\n"
    "def gen_inputs(ctx, device):\n"
    "    shape, dtype = CASES[ctx['case_id']]\n"
    "    return {'x': torch.rand(shape, dtype=dtype, device=device)}\n"
    "def run(x):\n    return torch.rsqrt(x)\n"
)

DEFN = {
    "name": "flaggems_rsqrt",
    "axes": {"case_id": {"type": "var"}},
    "inputs": {"x": {"shape": "dynamic", "dtype": "dynamic"}},
    "reference": REF,
    "custom_inputs_entrypoint": "gen_inputs",
}
WORKLOADS = [
    {"definition": "flaggems_rsqrt", "workload": {"uuid": "r-0", "axes": {"case_id": 0}, "inputs": {"x": {"type": "custom"}}}},
    {"definition": "flaggems_rsqrt", "workload": {"uuid": "r-1", "axes": {"case_id": 1}, "inputs": {"x": {"type": "custom"}}}},
]


def _install_fake_torch():
    """Minimal fake torch supporting the reference above: float16/float32,
    rand(shape,...) -> object with .shape/.dtype, device()."""
    t = types.ModuleType("torch")

    class _DT:
        def __init__(self, name): self._n = name
        def __repr__(self): return f"torch.{self._n}"
    t.float16 = _DT("float16"); t.float32 = _DT("float32"); t.bfloat16 = _DT("bfloat16")

    class _Tensor:
        def __init__(self, shape, dtype): self.shape = tuple(shape); self.dtype = dtype
    def rand(shape, dtype=None, device=None): return _Tensor(shape, dtype)
    t.rand = rand
    t.randn = rand
    def device(x): return x
    t.device = device
    return t


def test_degrades_without_torch(monkeypatch):
    # Force torch import to fail -> workloads returned unchanged.
    monkeypatch.setitem(sys.modules, "torch", None)
    out = enrich_workloads_with_shapes(DEFN, WORKLOADS)
    assert out == WORKLOADS
    assert "resolved_inputs" not in out[0]["workload"]


def test_enriches_with_fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _install_fake_torch())
    out = enrich_workloads_with_shapes(DEFN, WORKLOADS)
    r0 = out[0]["workload"]["resolved_inputs"]
    r1 = out[1]["workload"]["resolved_inputs"]
    assert r0 == {"x": {"shape": [1024, 1024], "dtype": "float16"}}, r0
    assert r1 == {"x": {"shape": [4096], "dtype": "float32"}}, r1
    # original workloads untouched (copy semantics)
    assert "resolved_inputs" not in WORKLOADS[0]["workload"]


def test_v4_uses_workload_metadata_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    workloads = [{
        "definition": "flaggems_concat",
        "workload": {
            "uuid": "v4-0",
            "axes": {},
            "inputs": {
                "x": {"type": "random", "shape": [64, 32], "dtype": "float16"},
                "parts": {
                    "type": "custom",
                    "shape": [[2, 4], [3, 4]],
                    "dtype": "float32",
                },
                "dim": {"type": "scalar", "value": 1},
            },
        },
    }]
    out = enrich_workloads_with_shapes({"reference": ""}, workloads)
    assert out[0]["workload"]["resolved_inputs"] == {
        "x": {"shape": [64, 32], "dtype": "float16"},
        "parts": {"shape": [[2, 4], [3, 4]], "dtype": "float32"},
    }
    assert "resolved_inputs" not in workloads[0]["workload"]


def test_no_gen_inputs_degrades(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _install_fake_torch())
    d = dict(DEFN, reference="import torch\ndef run(x):\n    return x\n")
    out = enrich_workloads_with_shapes(d, WORKLOADS)
    assert "resolved_inputs" not in out[0]["workload"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
