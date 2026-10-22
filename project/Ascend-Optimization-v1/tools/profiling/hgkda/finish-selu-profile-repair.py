#!/usr/bin/env python3
"""主队列结束后修复SELU采集时切换计算kernel的问题，保持独立来源与评测。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time

TOOLS=Path(__file__).resolve().parent
ROOT=Path('/data/hanle/ascend-optimization')


def read(p):return json.loads(p.read_text())


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-batch',required=True)
    a=p.parse_args()
    if not a.parent_batch.replace('-','').replace('_','').isalnum():raise ValueError('非法父批次名')
    parent=TOOLS.parents[2]/'operators/campaigns'/a.parent_batch
    deadline=time.monotonic()+14*3600
    while True:
        state=parent/'endurance-state.json'
        if state.exists() and read(state)['status']=='completed':break
        if time.monotonic()>deadline:raise TimeoutError('等待主队列超时，不提交设备任务')
        time.sleep(15)
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        row=next((r for r in read(parent/'deliverables/manifest.json')['records'] if r['operator']=='selu_'),None)
        if row is None:raise RuntimeError('主队列无有效SELU源码')
        source=Path(row['selected']['source'])
        if 'def _make_entry(A, n):' not in source.read_text():
            print('最终SELU未采用本次待修复缓存结构，不修改旧成果');return
        name=a.parent_batch+'-selu-profile-repair';batch=parent.parent/name
        if batch.exists():raise FileExistsError('补充批次已存在，先检查，不重复执行')
        plan=read(parent/'plan.json');record=dict(next(r for r in plan['records'] if r['operator']=='selu_'))
        leaf=batch/record['catalog_leaf'];leaf.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(parent/record['catalog_leaf'],leaf,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(parent/'catalog/manifest.json',batch/'catalog/manifest.json')
        seed=batch/record['seed'];seed.parent.mkdir(parents=True);seed.write_bytes(source.read_bytes())
        record.update(source_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),seed_origin=str(source))
        plan.update(records=[record],duration_hours=1,parent_batch=a.parent_batch,
            repair_kind='selu-same-kernel-profile',
            policy='额外Codex同kernel采集兼容修复；不更改计算AST；63项补充及一次完整KGS，来源单列')
        (batch/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
        shutil.copytree(parent/'knowledge',batch/'knowledge')
        spec=importlib.util.spec_from_file_location('selu_profile_campaign',TOOLS/'run-endurance.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        c=m.Campaign(name);c.q.idle()
        ref=batch/'reference-cc/formal';ref.parent.mkdir()
        shutil.copytree(Path(row['selected']['result']).parent,ref)
        parent_result=dict(row['selected']);parent_result.update(source=str(seed),result=str(ref/Path(row['selected']['result']).name))
        d=c.state['operators'].setdefault('selu_',dict(status='同kernel采集兼容修复',parent=parent_result))
        target=batch/'candidates/profile-compatible/main.py'
        c.run('prepare-selu-profile-repair',[m.PY,TOOLS/'prepare-selu-profile-repair.py','--parent',seed,'--output',target],600)
        result=c.step('validate:selu_:profile-compatible',lambda:c.candidate('selu_',target,target.parent,'profile-compatible'))
        if result and result.get('status')=='passed':
            d['best']=result;d['actual_speedup_over_parent']=m.improved(read(Path(parent_result['result'])),read(Path(result['result'])))
            observed=c.step('observe-final:selu_',lambda:c.observe('selu_',target,batch/'research/selu_/final'))
            metrics=c.step('metrics-final:selu_',lambda:c.metrics('selu_',result,batch/'research/selu_/final-metrics'))
            complete=all(x and x.get('status')=='completed' for x in [observed,metrics])
            d['status']='同kernel采集修复已验证' if complete else '源码验证通过，采集尚未闭环'
        else:d['status']='追加修复未通过，原始有效源码与失败保留'
        c.finish()
        for script,args in [('audit-endurance.py',['--batch-dir',batch,'--output',batch/'completion-audit.json']),
                            ('summarize-endurance.py',['--batch-dir',batch])]:
            subprocess.run(['docker','exec','tle_yy',m.PY,str(TOOLS/script)]+[str(v) for v in args],check=True)


if __name__=='__main__':main()
