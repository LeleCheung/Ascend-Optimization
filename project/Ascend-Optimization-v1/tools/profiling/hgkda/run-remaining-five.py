#!/usr/bin/env python3
"""复用已验证连续控制器，串行推进会议剩余五算子。"""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
TOOLS=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('remaining_endurance_base',TOOLS/'run-endurance.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

class Campaign(base.Campaign):
    def gates(self,op,source,out):
        checks=[]
        if op in ['scatter_add','cumulative_trapezoid']:
            previous=super().gates(op,source,out)
            if previous['status']!='passed':return previous
            checks.extend(previous['checks'])
        # scan必须先匹配本机NPU参考的舍入行为；CPU另作诊断，不能混用成硬性新容差。
        if op not in ['cumulative_trapezoid','scatter_add']:
            target=out/'remaining-semantics.json'
            if not target.exists():
                try:self.run('remaining-gate-'+op+'-'+out.name,[base.PY,TOOLS/'check-remaining-five.py','--operator',op,'--source',source,'--output',target],2400)
                except RuntimeError:
                    if not target.exists():raise
            data=base.read(target)
            if data['source_sha256']!=base.sha(source):raise ValueError('门禁源码SHA不同')
            if data['status']!='passed':return dict(status='semantic_failed',check=str(target),passed=data['passed'],total=data['total'])
            checks.append(str(target))
        return dict(status='passed',checks=checks)

    def observe(self,op,source,out):
        result=super().observe(op,source,out)
        facts=base.read(out/'facts.json')
        # 仅提示原生参考路径的待核事项；最终是否回退需读原始CPU+NPU事件。
        if op in ['_conj_physical','fractional_max_pool3d']:
            (out/'reference-scope.json').write_text(json.dumps(dict(operator=op,
                status='requires_trace_review',facts=str(out/'facts.json'),
                policy='原始PyTorch完整调用含回退时单列，不当作原生NPU优化收益；不替换参考'),ensure_ascii=False,indent=2)+'\n')
        return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True)
    a=p.parse_args();campaign=Campaign(a.batch)
    with (base.ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:campaign.execute()
        except Exception as error:
            campaign.state.update(status='needs_attention',error=type(error).__name__+': '+str(error));campaign.save();raise
