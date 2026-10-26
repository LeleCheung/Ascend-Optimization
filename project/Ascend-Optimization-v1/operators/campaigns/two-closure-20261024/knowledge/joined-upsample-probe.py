import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json,time
import torch,torch_npu

def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',required=True);a=p.parse_args()
 batch=Path('/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns/two-closure-20261024')
 out=batch/'research/joined-probes'/a.variant
 source=batch/'knowledge'/('upsample-'+a.variant+'.py')
 candidate=load('independent_candidate',source)
 parent=load('independent_parent',batch/'seeds/upsample_bilinear2d.py')
 oracle=load('independent_oracle',batch/'catalog/ops/upsample_bilinear2d/oracle.py')
 cases=[json.loads(l) for l in (batch/'catalog/ops/upsample_bilinear2d/timing.jsonl').read_text().splitlines()][:2]
 rows=[]
 for case in cases:
  args=oracle.gen_inputs(case,'npu:0');expected=oracle.run(**args);torch.npu.synchronize()
  for label,module in [('parent',parent),(a.variant,candidate)]:
   got=module.run(**args);torch.npu.synchronize()
   tol=case['tolerance'];torch.testing.assert_close(got.cpu(),expected.cpu(),atol=tol['atol'],rtol=tol['rtol'])
   for _ in range(3):module.run(**args);torch.npu.synchronize()
   timings=[]
   for _ in range(9):
    t=time.perf_counter();got=module.run(**args);torch.npu.synchronize();timings.append((time.perf_counter()-t)*1000)
   timings.sort();row=dict(case=case['name'],variant=label,correct=True,median_ms=timings[4],samples_ms=timings);rows.append(row);print('TIMING',json.dumps(row),flush=True)
  (out/'timings.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')

if __name__=='__main__':main()
