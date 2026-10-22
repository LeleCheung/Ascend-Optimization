"""Long-run stability check: 1M repeats of the candidate must stay flat."""

import importlib.util
import json
import os
import time

import torch
import torch_npu  # noqa: F401

ART = os.environ.get("KGS_DEBUG_ARTIFACTS", "/tmp")
res = {"blocks": {}, "samples": {}}


def load():
    spec = importlib.util.spec_from_file_location("cand", "tmp/main.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    m = load()
    for dt, n in ((torch.float16, 1048576), (torch.float32, 1048576),
                  (torch.float16, 1024), (torch.float16, 65536)):
        x = torch.empty(n, dtype=dt, device="npu")
        tag = f"{str(dt).split('.')[-1]}_{n}"
        # warm up and force the int32-boundary case: drive the counter past 2**31
        for _ in range(20):
            m.run(x, 0.0, 1.0)
        torch.npu.synchronize()
        blocks = []
        t_all = time.perf_counter()
        for _ in range(4000):
            m.run(x, 0.0, 1.0)
        torch.npu.synchronize()
        res["blocks"][tag + "_4000_avg_us"] = round(
            (time.perf_counter() - t_all) / 4000 * 1e6, 2)
        # block-by-block to expose a mid-run blowup
        for blk in range(8):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(500):
                m.run(x, 0.0, 1.0)
            torch.npu.synchronize()
            blocks.append(round((time.perf_counter() - t0) / 500 * 1e6, 2))
        res["blocks"][tag + "_per500"] = blocks
        a = m.run(x, 0.0, 1.0)
        b = m.run(x, 0.0, 1.0)
        af = a.float()
        res["samples"][tag] = {
            "in_range": bool(((af >= 0) & (af < 1)).all().item()),
            "differ": bool((a != b).any().item()),
            "mean": round(float(af.mean()), 5),
            "var": round(float(af.var(unbiased=False)), 6),
            "counter": m._RNG[torch.npu.default_generators[0]][1],
        }
    with open(os.path.join(ART, "stability.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


main()
