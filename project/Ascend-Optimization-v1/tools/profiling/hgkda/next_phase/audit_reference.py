"""核对新算子的PyTorch参考实际执行路径，防止CPU回退被误写为常规达标。"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='7'
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);trace=a.output/'trace.json'
    result=dict(status='review_required',eligible_default_comparison=False,spec=json.loads(a.spec.read_text(encoding='utf8')))
    with (a.output/'execution.log').open('w') as log:
        r=subprocess.run([sys.executable,str(HERE/'driver.py'),'--spec',str(a.spec),'--side','pytorch','--trace',str(trace)],stdout=log,stderr=subprocess.STDOUT,timeout=300)
    text=(a.output/'execution.log').read_text(encoding='utf8',errors='replace')
    fallback=bool(re.search(r'(fall.?back|falling back|fall back).{0,100}CPU|CPU.{0,100}(fall.?back|falling back|fall back)',text,re.I))
    result.update(returncode=r.returncode,explicit_cpu_fallback_warning=fallback)
    if trace.exists():
        payload=json.loads(trace.read_text(encoding='utf8'));events=payload.get('traceEvents',[]) if isinstance(payload,dict) else payload
        tasks=[e for e in events if e.get('ph')=='X' and ('kernel' in str(e.get('cat','')).lower() or 'Task Type' in e.get('args',{}) or 'memcpy' in e.get('name','').lower())]
        compute=[e for e in tasks if 'memcpy' not in e.get('name','').lower() and 'memset' not in e.get('name','').lower()]
        result.update(device_tasks=tasks,compute_tasks=len(compute),trace=str(trace),
                      interpretation='设备任务及明确回退警告构成初筛；复杂的部分CPU回退仍需原始trace核对，任务存在不证明全部计算都在AIV/Cube')
        if not r.returncode and compute and not fallback:result.update(status='npu_path_observed',eligible_default_comparison=True)
    if fallback:result.update(status='cpu_fallback_observed',eligible_default_comparison=False)
    (a.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','eligible_default_comparison','explicit_cpu_fallback_warning']},ensure_ascii=False))


if __name__=='__main__':main()
