#!/usr/bin/env python3
"""隔离进程复现910B FP16非末轴扫描稳定性；不修改KGS或正式参考。"""
import argparse,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',type=Path,required=True,help='本批已保存的FP16原始输入bin')
 p.add_argument('--shape',type=int,nargs=2,required=True)
 p.add_argument('--kind',choices=['cumsum','cumulative_trapezoid'],default='cumulative_trapezoid')
 p.add_argument('--deterministic',action='store_true',help='单独验证官方确定性执行条件')
 p.add_argument('--repeats',type=int,default=12)
 p.add_argument('--output',type=Path,required=True)
 args=p.parse_args();assert args.repeats>=2 and not args.output.exists()
 import os
 assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7','只用已授权物理卡7'
 import numpy as np
 import torch,torch_npu
 raw=args.input.read_bytes();assert len(raw)==args.shape[0]*args.shape[1]*2
 x=torch.from_numpy(np.frombuffer(raw,dtype=np.float16).copy().reshape(args.shape)).to('npu:0')
 torch.use_deterministic_algorithms(args.deterministic)
 outputs=[]
 for _ in range(args.repeats):
  value=x.cumsum(0) if args.kind=='cumsum' else torch.cumulative_trapezoid(x,dx=1,dim=0)
  torch.npu.synchronize();outputs.append(value.cpu().float().numpy().astype(np.float64))
 stack=np.stack(outputs);radius=.015+.015*np.abs(stack)
 lower=np.max(stack-radius,axis=0);upper=np.min(stack+radius,axis=0);empty=lower>upper
 differences=[int(np.count_nonzero(np.abs(value-outputs[0])>.015+.015*np.abs(outputs[0]))) for value in outputs[1:]]
 result=dict(kind=args.kind,shape=args.shape,dtype='float16',dim=0,dx=1,repeats=args.repeats,
  deterministic=args.deterministic,empty_tolerance_intersections=int(empty.sum()),over_tolerance_vs_first=differences,
  input_sha256=hashlib.sha256(raw).hexdigest(),torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,
  torch_git=torch.version.git_version,physical_device=7,
  scope='同一输入重复执行；atol=rtol=0.015来自原合同；确定性模式单列，不作为默认合同候选成绩')
 if empty.any():
  pos=tuple(int(i) for i in np.argwhere(empty)[0])
  result.update(position=list(pos),values=[float(o[pos]) for o in outputs],max_lower=float(lower[pos]),min_upper=float(upper[pos]))
 args.output.parent.mkdir(parents=True,exist_ok=True)
 args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
