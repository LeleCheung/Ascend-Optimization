#!/usr/bin/env python3
"""主队列结束后串行验证rad2deg采集修复，独立批次保留，不覆盖主队列成果。"""
import argparse
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
        if time.monotonic()>deadline:raise TimeoutError('等待主队列超时，不重发设备工作')
        time.sleep(15)
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        delivery=read(parent/'deliverables/manifest.json')
        row=next((r for r in delivery['records'] if r['operator']=='rad2deg_'),None)
        if row is None:raise RuntimeError('主队列没有有效rad2deg源码，保留主队列失败，不猜测交付')
        source=Path(row['selected']['source'])
        if '(compiled.run.launch, compiled.function, compiled.packed_metadata, block)' not in source.read_text():
            print('主队列最终源码不使用待修复的裸启动结构；没有修改')
            return
        name=a.parent_batch+'-rad-profile-repair'
        batch=parent.parent/name
        if batch.exists():raise FileExistsError('补充批次已存在，先检查结果，不重复执行')
        plan=read(parent/'plan.json');record=dict(next(r for r in plan['records'] if r['operator']=='rad2deg_'))
        leaf=batch/record['catalog_leaf'];leaf.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(parent/record['catalog_leaf'],leaf,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(parent/'catalog/manifest.json',batch/'catalog/manifest.json')
        seed=batch/record['seed'];seed.parent.mkdir(parents=True);seed.write_bytes(source.read_bytes())
        import hashlib
        record.update(source_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),seed_origin=str(source))
        plan.update(records=[record],duration_hours=1,parent_batch=a.parent_batch,
            policy='额外Codex采集元数据与缓存身份修复；不更改计算kernel，独立语义与一次正式完整评测；不替换原始生成证据')
        (batch/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
        shutil.copytree(parent/'knowledge',batch/'knowledge')
        modspec=importlib.util.spec_from_file_location('rad_repair_campaign',TOOLS/'run-endurance.py')
        m=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(m)
        c=m.Campaign(name);c.q.idle()
        formal=Path(row['selected']['result']).parent
        reference=batch/'reference-cc/formal';reference.parent.mkdir()
        shutil.copytree(formal,reference)
        parent_result=dict(row['selected']);parent_result.update(source=str(seed),result=str(reference/Path(row['selected']['result']).name))
        c.state['operators']['rad2deg_']=dict(status='追加采集兼容修复',parent=parent_result)
        target=batch/'candidates/profile-compatible/main.py'
        c.run('prepare-rad-profile-repair',[m.PY,TOOLS/'prepare-rad-profile-repair.py','--parent',seed,'--output',target],600)
        result=c.step('validate:rad2deg_:profile-compatible',lambda:c.candidate('rad2deg_',target,target.parent,'profile-compatible'))
        if result and result.get('status')=='passed':
            d=c.state['operators']['rad2deg_'];d['best']=result
            d['actual_speedup_over_parent']=m.improved(read(Path(parent_result['result'])),read(Path(result['result'])))
            c.step('observe-final:rad2deg_',lambda:c.observe('rad2deg_',target,batch/'research/rad2deg_/final'))
            c.step('metrics-final:rad2deg_',lambda:c.metrics('rad2deg_',result,batch/'research/rad2deg_/final-metrics'))
            d['status']='有效采集修复已验证'
        else:c.state['operators']['rad2deg_']['status']='追加修复未通过；主批次原件保留'
        c.finish()
        for script,args in [('audit-endurance.py',['--batch-dir',batch,'--output',batch/'completion-audit.json']),
                            ('summarize-endurance.py',['--batch-dir',batch])]:
            subprocess.run(['docker','exec','tle_yy',m.PY,str(TOOLS/script)]+[str(v) for v in args],check=True)


if __name__=='__main__':main()
