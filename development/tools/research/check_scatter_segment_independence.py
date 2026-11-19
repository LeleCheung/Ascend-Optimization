"""Check that distinct scatter destinations remain numerically independent."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
import torch
import torch_npu
torch.npu.set_device(0);torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('_checked_scatter',a.source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
cases=[]
for label,first in [('large_finite',1e20),('positive_infinity',float('inf')),('nan',float('nan'))]:
    try:
        inp=torch.zeros((1,1024),dtype=torch.float32,device='npu:0')
        index=torch.arange(1024,dtype=torch.int64,device='npu:0').reshape(1,1024)
        source=torch.ones((1,1024),dtype=torch.float32,device='npu:0');source[0,0]=first
        expected=torch.scatter_add(inp,1,index,source)
        actual=module.run(inp,1,index,source);torch.npu.synchronize()
        torch.testing.assert_close(actual,expected,atol=0,rtol=0,equal_nan=True)
        cases.append(dict(case=label,passed=True))
    except Exception as exc:cases.append(dict(case=label,passed=False,error=str(exc)))
result=dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
    passed=sum(x['passed'] for x in cases),total=len(cases),cases=cases,
    scope='Unique destination indices: each output receives one source; unrelated segments must retain their own finite or special values.')
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
