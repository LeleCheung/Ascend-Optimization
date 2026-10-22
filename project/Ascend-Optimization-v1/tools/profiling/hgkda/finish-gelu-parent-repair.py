#!/usr/bin/env python3
"""主队列结束后补齐因输入配方适配失败而未执行的GELU父版；保留旧故障。"""
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
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-batch',required=True)
    a=p.parse_args()
    if not a.parent_batch.replace('-','').replace('_','').isalnum():raise ValueError('非法批次名')
    parent=TOOLS.parents[2]/'operators/campaigns'/a.parent_batch
    deadline=time.monotonic()+14*3600
    while True:
        state=parent/'endurance-state.json'
        if state.exists() and read(state)['status']=='completed':break
        if time.monotonic()>deadline:raise TimeoutError('等待主队列超时，不重发设备工作')
        time.sleep(15)
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old=read(parent/'endurance-state.json')['jobs']['parent:gelu_and_mul']
        if old.get('status')!='failed' or "KeyError: 'spec'" not in old.get('error',''):
            raise ValueError('故障身份不同，先核对，不重复评测')
        if list((parent/'results/gelu_and_mul/parent').glob('formal/*.request.json')):
            raise ValueError('父版已有正式请求，先核对，不重复评测')
        name=a.parent_batch+'-gelu-parent-repair';batch=parent.parent/name
        if batch.exists():raise FileExistsError('补充批次已存在，先检查结果')
        plan=read(parent/'plan.json');record=dict(next(r for r in plan['records'] if r['operator']=='gelu_and_mul'))
        leaf=batch/record['catalog_leaf'];leaf.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(parent/record['catalog_leaf'],leaf,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(parent/'catalog/manifest.json',batch/'catalog/manifest.json')
        seed=batch/record['seed'];seed.parent.mkdir(parents=True);seed.write_bytes((parent/record['seed']).read_bytes())
        assert hashlib.sha256(seed.read_bytes()).hexdigest()==record['source_sha256']
        plan.update(records=[record],duration_hours=1,parent_batch=a.parent_batch,
            repair_kind='gelu-parent-input-recipe',
            policy='输入配方门禁适配修复后补齐固定FlagGems-SGLang父版；源码和KGS合同不变；旧故障回执保留；首次正式完整评测，不是重新评测已有成功结果')
        (batch/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
        shutil.copytree(parent/'knowledge',batch/'knowledge')
        (batch/'parent-infrastructure-failure.json').write_text(json.dumps(dict(original_job=old,
            original_log=str(ROOT/'runs'/a.parent_batch/'gate-gelu_and_mul-parent-semantics.log'),
            amendment=str(parent/'native-timing-gate-amendment-04.json')),ensure_ascii=False,indent=2)+'\n')
        spec=importlib.util.spec_from_file_location('gelu_parent_repair',TOOLS/'run-endurance.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        c=m.Campaign(name);c.q.idle()
        d=c.state['operators'].setdefault('gelu_and_mul',dict(status='补齐同批父版'))
        result=c.step('parent:gelu_and_mul',lambda:c.baseline('gelu_and_mul'))
        d['parent']=result
        if result and result.get('status')=='passed':
            d['best']=result;d['actual_speedup_over_parent']=1.0
            c.step('observe-final:gelu_and_mul',lambda:c.observe('gelu_and_mul',seed,batch/'research/gelu_and_mul/final'))
            c.step('metrics-final:gelu_and_mul',lambda:c.metrics('gelu_and_mul',result,batch/'research/gelu_and_mul/final-metrics'))
            d['status']='固定FlagGems-SGLang父版已补齐，无源码优化'
        else:d['status']='父版门禁或完整评测失败，保留真实证据'
        c.finish()
        for script,args in [('audit-endurance.py',['--batch-dir',batch,'--output',batch/'completion-audit.json']),
                            ('summarize-endurance.py',['--batch-dir',batch])]:
            subprocess.run(['docker','exec','tle_yy',m.PY,str(TOOLS/script)]+[str(v) for v in args],check=True)


if __name__=='__main__':main()
