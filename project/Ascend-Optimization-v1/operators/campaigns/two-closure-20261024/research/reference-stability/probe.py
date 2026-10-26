#!/usr/bin/env python3
"""两项收尾的隔离诊断；通过KGS debug运行，不替代正式评测。"""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
sys.dont_write_bytecode=True

def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def upsample(batch):
 import torch,torch_npu
 oracle=load('closure_upsample_oracle',batch/'catalog/ops/upsample_bilinear2d/oracle.py')
 cases=[json.loads(line) for line in (batch/'catalog/ops/upsample_bilinear2d/timing.jsonl').read_text().splitlines()][:2]
 variants=[dict(name='parent',path=str(batch/'seeds/upsample_bilinear2d.py'))]+json.loads((batch/'knowledge/initial-variants.json').read_text())
 rows=[]
 for entry in variants:
  source=Path(entry['path']);row=dict(name=entry['name'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),cases=[])
  try:
   m=load('closure_'+entry['name'],source)
   for case in cases:
    args=oracle.gen_inputs(case,'npu:0');expected=oracle.run(**args);torch.npu.synchronize()
    got=m.run(**args);torch.npu.synchronize()
    tol=case['tolerance'];torch.testing.assert_close(got.cpu(),expected.cpu(),atol=tol['atol'],rtol=tol['rtol'])
    for _ in range(3):m.run(**args);torch.npu.synchronize()
    durations=[]
    for _ in range(7):
     start=time.perf_counter();got=m.run(**args);torch.npu.synchronize();durations.append((time.perf_counter()-start)*1000)
    durations.sort();item=dict(case=case['name'],correct=True,median_ms=durations[len(durations)//2],samples_ms=durations)
    row['cases'].append(item);print('UPSAMPLE',row['name'],json.dumps(item),flush=True)
   row['status']='passed'
  except Exception as error:
   row.update(status='failed',error=type(error).__name__+': '+str(error)[-6000:]);print('FAILED',row['name'],row['error'][-1500:],flush=True)
  rows.append(row)
  (batch/'research/initial-upsample/results.json').write_text(json.dumps(dict(scope='Codex初步两case诊断，非正式分数',rows=rows),ensure_ascii=False,indent=2)+'\n')

def cumulative(batch):
 import torch,torch_npu
 import numpy as np
 out=batch/'research/reference-stability';out.mkdir(parents=True,exist_ok=True)
 oracle=load('closure_scan_oracle',batch/'catalog/ops/cumulative_trapezoid/oracle.py')
 contract=[json.loads(line) for line in (batch/'catalog/ops/cumulative_trapezoid/timing.jsonl').read_text().splitlines()]
 case=next(c for c in contract if c['inputs']['spec']['dtype']=='float16' and c['inputs']['spec']['dim']==0)
 data=oracle.gen_inputs(case,'npu:0');x=data.get('y',data.get('input'))
 assert x is not None
 original_level=torch.are_deterministic_algorithms_enabled()
 rows=[]
 for mode in ['default','torch_deterministic']:
  torch.use_deterministic_algorithms(mode=='torch_deterministic')
  for shape in [(128,8),(256,8),(512,16),(1024,32),tuple(x.shape)]:
   y=x[:shape[0],:shape[1]].contiguous()
   row=dict(mode=mode,shape=list(y.shape),status='failed',tolerance=case['tolerance'])
   try:
    for kind in ['cumsum','cumulative_trapezoid']:
     values=[]
     for repeat in range(6):
      z=y.cumsum(0) if kind=='cumsum' else torch.cumulative_trapezoid(y,dx=1,dim=0)
      torch.npu.synchronize();values.append(z.cpu().to(torch.float64).numpy())
     stack=np.stack(values);tol=case['tolerance'];radius=tol['atol']+tol['rtol']*np.abs(stack)
     empty=np.max(stack-radius,axis=0)>np.min(stack+radius,axis=0)
     comparisons=[int(np.count_nonzero(np.abs(v-values[0])>tol['atol']+tol['rtol']*np.abs(values[0]))) for v in values[1:]]
     row[kind]=dict(empty_intersections=int(empty.sum()),over_tolerance=comparisons,max_absolute=max(float(np.max(np.abs(v-values[0]))) for v in values[1:]))
     if empty.any():
      identity=mode+'-'+str(shape[0])+'x'+str(shape[1])+'-'+kind
      binary=out/(identity+'-input.bin');y.cpu().numpy().tofile(binary)
      pos=tuple(int(v) for v in np.argwhere(empty)[0])
      row[kind].update(input_file=str(binary),input_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),position=list(pos),six_values=[float(v[pos]) for v in values])
    row['status']='completed'
   except Exception as error:row['error']=type(error).__name__+': '+str(error)[-3000:]
   rows.append(row);print('REFERENCE',json.dumps(row,ensure_ascii=False),flush=True)
   (out/'results.json').write_text(json.dumps(dict(scope='隔离进程诊断官方确定性选项；不改变原合同和验收设置',rows=rows),ensure_ascii=False,indent=2)+'\n')
 torch.use_deterministic_algorithms(original_level)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-dir',type=Path,required=True);p.add_argument('--kind',choices=['upsample','cumulative'],required=True)
 a=p.parse_args();(upsample if a.kind=='upsample' else cumulative)(a.batch_dir)
