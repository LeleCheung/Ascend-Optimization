#!/usr/bin/env python3
"""根据已测整数反例，对独立uniform副本做一次机制修复与完整验证。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

TOOLS=Path(__file__).resolve().parent
ROOT=Path('/data/hanle/ascend-optimization')


def read(p):return json.loads(p.read_text())


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-batch',required=True)
    p.add_argument('--variant',choices=['v2'],default='')
    a=p.parse_args()
    if not a.parent_batch.replace('-','').replace('_','').isalnum():raise ValueError('非法批次名')
    parent=TOOLS.parents[2]/'operators/campaigns'/a.parent_batch
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        prior=parent.parent/(a.parent_batch+'-uniform-philox-repair')
        assert read(parent/'endurance-state.json')['status']=='completed'
        assert read(prior/'endurance-state.json')['status']=='completed'
        primitive=read(parent/'research/uniform/integer-primitive-probe.json')
        shift=next(r for r in primitive['cases'] if r['operation']=='shr16')
        assert shift['status']=='failed' and shift['mismatches']>0
        source=prior/'candidates/philox-compatible/main.py'
        batch=parent.parent/(a.parent_batch+'-uniform-shift-repair'+('-'+a.variant if a.variant else ''))
        if batch.exists():raise FileExistsError(batch)
        plan=read(parent/'plan.json');record=dict(next(r for r in plan['records'] if r['operator']=='uniform'))
        leaf=batch/record['catalog_leaf'];leaf.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(parent/record['catalog_leaf'],leaf,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(parent/'catalog/manifest.json',batch/'catalog/manifest.json')
        seed=batch/record['seed'];seed.parent.mkdir(parents=True,exist_ok=True);seed.write_bytes(source.read_bytes())
        record.update(source_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),seed_origin=str(source))
        names=['check-uniform-philox.py','prepare-uniform-shift-repair.py','run-uniform-shift-repair.py']
        plan.update(records=[record],duration_hours=1,parent_batch=a.parent_batch,uniform_philox_check_required=True,
                    supplemental_tool_sha256={name:hashlib.sha256((TOOLS/name).read_bytes()).hexdigest() for name in names},
                    policy='Codex根据实际unsigned右移反例追加一次signed shift+mask机制修复；逐位16项、原18+3项语义及一次完整KGS全部通过才交付')
        (batch/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
        shutil.copytree(parent/'knowledge',batch/'knowledge')
        spec=importlib.util.spec_from_file_location('uniform_shift_campaign',TOOLS/'run-endurance.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        c=m.Campaign(batch.name);c.q.idle()
        d=c.state['operators'].setdefault('uniform',dict(status='无符号右移机制修复'))
        target=batch/'candidates/signed-shift/main.py'
        c.run('prepare-uniform-shift',[m.PY,TOOLS/'prepare-uniform-shift-repair.py','--parent',seed,'--output',target],120)
        proof=batch/'research/uniform/fixed-philox.json'
        try:c.run('philox-signed-shift',[m.PY,TOOLS/'check-uniform-philox.py','--source',target,'--output',proof],600)
        except RuntimeError:
            if not proof.exists():raise
        data=read(proof)
        assert data['source_sha256']==hashlib.sha256(target.read_bytes()).hexdigest()
        d['philox_check']=dict(path=str(proof),status=data['status'],passed=data['passed'],total=data['total'])
        if data['status']=='passed':
            result=c.step('validate:uniform:signed-shift',lambda:c.candidate('uniform',target,target.parent,'signed-shift'))
            if result and result.get('status')=='passed':
                d['best']=result
                observed=c.step('observe-final:uniform',lambda:c.observe('uniform',target,batch/'research/uniform/final'))
                metrics=c.step('metrics-final:uniform',lambda:c.metrics('uniform',result,batch/'research/uniform/final-metrics'))
                d['status']='标准Philox与完整语义通过' if all(x and x.get('status')=='completed' for x in [observed,metrics]) else '标准Philox与完整语义通过，采集有缺口'
            else:d['status']='逐位通过，其他语义或完整KGS失败'
        else:d['status']='signed shift机制修复仍未通过逐位检查，不交付标准Philox'
        c.finish()
        for script,extra in [('audit-endurance.py',['--output',batch/'completion-audit.json']),('summarize-endurance.py',[])]:
            subprocess.run(['docker','exec','tle_yy',m.PY,str(TOOLS/script),'--batch-dir',str(batch)]+[str(x) for x in extra],check=True)


if __name__=='__main__':main()
