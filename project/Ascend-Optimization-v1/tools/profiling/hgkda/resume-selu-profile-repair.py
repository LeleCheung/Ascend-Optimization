#!/usr/bin/env python3
"""仅恢复尚未生成候选/提交设备工作的SELU采集修复批次，原错误日志保留。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

TOOLS=Path(__file__).resolve().parent
ROOT=Path('/data/hanle/ascend-optimization')


def read(path):return json.loads(path.read_text())


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-batch',required=True)
    a=p.parse_args()
    if not a.parent_batch.replace('-','').replace('_','').isalnum():raise ValueError('非法批次名')
    parent=TOOLS.parents[2]/'operators/campaigns'/a.parent_batch
    batch=parent.parent/(a.parent_batch+'-selu-profile-repair')
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=read(batch/'endurance-state.json')
        if state['status']!='created' or state.get('jobs') or state.get('operators'):
            raise ValueError('批次已执行设备步骤或状态不同，不能使用本恢复入口')
        target=batch/'candidates/profile-compatible/main.py'
        if target.exists():raise FileExistsError(target)
        record=read(batch/'plan.json')['records'][0]
        seed=batch/record['seed']
        assert hashlib.sha256(seed.read_bytes()).hexdigest()==record['source_sha256']
        spec=importlib.util.spec_from_file_location('resume_selu_campaign',TOOLS/'run-endurance.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        c=m.Campaign(batch.name);c.q.idle()
        row=next(r for r in read(parent/'deliverables/manifest.json')['records'] if r['operator']=='selu_')
        reference=batch/'reference-cc/formal'/Path(row['selected']['result']).name
        assert reference.is_file()
        parent_result=dict(row['selected']);parent_result.update(source=str(seed),result=str(reference))
        d=c.state['operators'].setdefault('selu_',dict(status='恢复实际入选结构的采集兼容修复',parent=parent_result))
        c.module.save(batch/'resume-receipt.json',dict(reason='准备器仅支持kind结构，实际入选源码使用masked布尔结构；失败发生在源码生成与NPU之前',
            seed_sha256=record['source_sha256'],prepare_sha256=hashlib.sha256((TOOLS/'prepare-selu-profile-repair.py').read_bytes()).hexdigest()))
        c.run('prepare-selu-profile-repair-resume',[m.PY,TOOLS/'prepare-selu-profile-repair.py','--parent',seed,'--output',target],600)
        result=c.step('validate:selu_:profile-compatible',lambda:c.candidate('selu_',target,target.parent,'profile-compatible'))
        if result and result.get('status')=='passed':
            d['best']=result
            d['actual_speedup_over_parent']=m.improved(read(reference),read(Path(result['result'])))
            observed=c.step('observe-final:selu_',lambda:c.observe('selu_',target,batch/'research/selu_/final'))
            metrics=c.step('metrics-final:selu_',lambda:c.metrics('selu_',result,batch/'research/selu_/final-metrics'))
            d['status']='同kernel采集修复已验证' if all(x and x.get('status')=='completed' for x in [observed,metrics]) else '源码验证通过，采集尚未闭环'
        else:d['status']='追加修复未通过，原始有效源码与失败保留'
        c.finish()
        for script,extra in [('audit-endurance.py',['--output',batch/'completion-audit.json']),('summarize-endurance.py',[])]:
            subprocess.run(['docker','exec','tle_yy',m.PY,str(TOOLS/script),'--batch-dir',str(batch)]+[str(x) for x in extra],check=True)


if __name__=='__main__':main()
