"""Run the authoritative independent semantic gate on the current candidate and
report per-dtype numerical margins. Diagnostic only."""
import json
import os
import runpy
import sys
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401

ART = Path(os.environ["KGS_DEBUG_ARTIFACTS"])
ART.mkdir(parents=True, exist_ok=True)
out = {}

# 1) authoritative gate, byte-for-byte
try:
    sys.argv = [
        "check_rad2deg.py",
        "--source", "tmp/main.py",
        "--output", str(ART / "semantics.json"),
    ]
    runpy.run_path(".kernelgen/skills/hgkda/scripts/check_rad2deg.py", run_name="__main__")
    out["gate"] = "PASSED"
except Exception:
    out["gate"] = "FAILED"
    out["gate_error"] = traceback.format_exc()[-3000:]

# 2) numerical margins per dtype on the same input families
try:
    import importlib.util

    spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    margins = {}
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        worst_rel = 0.0
        worst_abs = 0.0
        for label, build in [
            ("odd-1025", lambda: torch.linspace(-1, 1, 1025)),
            ("odd-2050", lambda: torch.linspace(-1, 1, 2050)),
            ("randn-65536", lambda: torch.randn(65536)),
        ]:
            cpu = build()
            inp = cpu.to(device="npu:0", dtype=dtype)
            expected = torch.rad2deg(cpu.clone()).float()
            m.run(inp)
            torch.npu.synchronize()
            got = inp.cpu().float()
            d = (got - expected).abs()
            denom = expected.abs().clamp_min(1e-30)
            rel = (d / denom)
            worst_abs = max(worst_abs, float(d.max()))
            worst_rel = max(worst_rel, float(rel.max()))
        margins[str(dtype)] = {"max_abs_err": worst_abs, "max_rel_err": worst_rel,
                               "contract_rtol": 0.015,
                               "gate_tol_used_by_checker": 1e-2 if dtype != torch.float32 else 2e-6}
    out["margins"] = margins
except Exception:
    out["margins_error"] = traceback.format_exc()[-2000:]

(ART / "gate.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
print(json.dumps({"gate": out.get("gate"), "margins": out.get("margins")}, indent=2, default=str))
