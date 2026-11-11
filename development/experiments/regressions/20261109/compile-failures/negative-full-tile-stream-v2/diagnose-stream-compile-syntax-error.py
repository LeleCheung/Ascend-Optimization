import torch,torch_npu,importlib.util,json,traceback
from pathlib import Path
root=Path('/data/hanle/ascend-optimization/runs/negative-full-tile-stream-20261109-v2')
p=root/'operators/negative_/candidate.py'
s=importlib.util.spec_from_file_location('negative_diagnostic',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
x=torch.empty(1048576,dtype=torch.float16,device='npu:0')
try:
 c=m._negative_streaming.warmup(x,1048576,BLOCK=32768,grid=(1,));v=dict(status='compiled',metadata=str(c.metadata))
except Exception as e:
 v=dict(status='compilation_failed',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc())
(root/'stream-compilation-diagnosis.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'
',encoding='utf8')
print(json.dumps(v,ensure_ascii=False))
