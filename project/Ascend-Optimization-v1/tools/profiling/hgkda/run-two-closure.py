#!/usr/bin/env python3
"""新两算子批次串行搜索，扩展上采样特殊门禁，不修改已冻结基础工具。"""
import argparse,fcntl,importlib.util,json
from pathlib import Path

TOOLS=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('two_closure_base',TOOLS/'run-remaining-five.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
base=module.base

class Campaign(module.Campaign):
 def gates(self,op,source,out):
  previous=super().gates(op,source,out)
  if previous['status']!='passed' or op!='upsample_bilinear2d':return previous
  target=out/'special.json'
  if not target.exists():
   try:self.run('two-special-'+out.name,[base.PY,TOOLS/'check-remaining-special.py','--operator',op,'--source',source,'--output',target],1800)
   except RuntimeError:
    if not target.exists():raise
  data=base.read(target)
  assert data['source_sha256']==base.sha(source)
  if data['status']!='passed':return dict(status='semantic_failed',check=str(target),passed=data['passed'],total=data['total'])
  return dict(status='passed',checks=previous['checks']+[str(target)])

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True)
 c=Campaign(p.parse_args().batch)
 with (base.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:c.execute()
  except Exception as error:
   c.state.update(status='needs_attention',error=type(error).__name__+': '+str(error));c.save();raise
