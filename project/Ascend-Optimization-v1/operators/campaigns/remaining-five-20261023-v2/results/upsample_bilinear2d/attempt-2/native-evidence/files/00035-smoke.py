import json
import os
import time
import traceback

RES = {}


def rec(k, v):
    RES[k] = v
    print("%s = %s" % (k, v), flush=True)


def adir():
    d = os.environ.get("KGS_DEBUG_ARTIFACTS", "")
    return d if d and os.path.isdir(d) else os.getcwd()


rec("phase", "sleep")
time.sleep(40)
rec("phase", "wake")

try:
    import torch
    import torch_npu  # noqa: F401

    rec("torch", torch.__version__)
    rec("torch_npu", torch_npu.__version__)
    rec("dev_count", torch.npu.device_count())
    x = torch.ones(1024, device="npu:0", dtype=torch.float32)
    torch.npu.synchronize()
    rec("ones_sum", float(x.sum().item()))
    y = torch.ones(1 << 20, device="npu:0", dtype=torch.float16)
    z = y + y
    torch.npu.synchronize()
    rec("add_first", float(z[0].item()))
    rec("empty_cache", "start")
    torch.npu.empty_cache()
    rec("empty_cache", "ok")
    rec("verdict", "DEVICE_OK")
except Exception:
    rec("error", traceback.format_exc()[-1500:])
    rec("verdict", "DEVICE_FAIL")

p = os.path.join(adir(), "smoke.json")
with open(p, "w") as fh:
    json.dump(RES, fh, indent=2, default=str)
print("wrote", p, flush=True)
