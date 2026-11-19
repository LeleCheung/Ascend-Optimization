import importlib.util, json, sys
from pathlib import Path
import torch, torch_npu
import flag_gems
p=Path(__file__).parent/'upstream.py'
s=importlib.util.spec_from_file_location('native_upsample_probe',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
print(json.dumps({'flag_gems':flag_gems.__file__,'torch':torch.__version__,'entrypoint':m.upsample_bilinear2d.__name__}),flush=True)
