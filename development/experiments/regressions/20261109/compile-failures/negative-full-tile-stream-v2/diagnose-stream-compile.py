"""只编译候选流式核，显式保留被父版本回退逻辑捕获的编译异常。"""
import importlib.util
import json
from pathlib import Path
import traceback
import torch
import torch_npu

torch.set_num_threads(1)
root=Path('/data/hanle/ascend-optimization/runs/negative-full-tile-stream-20261109-v2')
source=root/'operators/negative_/candidate.py'
spec=importlib.util.spec_from_file_location('negative_compile_diagnostic',source)
candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
tensor=torch.empty(1048576,dtype=torch.float16,device='npu:0')
try:
    compiled=candidate._negative_streaming.warmup(tensor,1048576,BLOCK=32768,grid=(1,))
    value=dict(status='compiled',metadata=str(compiled.metadata),device_kernel_executed=False)
except Exception as error:
    value=dict(status='compilation_failed',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
(root/'stream-compilation-diagnosis.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(value,ensure_ascii=False))
