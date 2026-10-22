#!/usr/bin/env python3
"""主队列结束后串行核验uniform的Philox机制，必要时在独立批次有限修复。"""
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


def read(path):
    return json.loads(path.read_text())


def main():
    import fcntl
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-batch',required=True)
    args=p.parse_args()
    if not args.parent_batch.replace('-','').replace('_','').isalnum():
        raise ValueError('非法父批次名')
    parent=TOOLS.parents[2]/'operators/campaigns'/args.parent_batch
    deadline=time.monotonic()+14*3600
    while True:
        state=parent/'endurance-state.json'
        if state.exists() and read(state)['status']=='completed':
            break
        if time.monotonic()>deadline:
            raise TimeoutError('等待主队列超时，保留原任务')
        time.sleep(15)
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        row=next((r for r in read(parent/'deliverables/manifest.json')['records'] if r['operator']=='uniform'),None)
        if row is None:
            raise RuntimeError('主队列没有有效uniform源码，不猜测交付')
        source=Path(row['selected']['source'])
        if not 'def _philox4(' in source.read_text():
            raise RuntimeError('最终源码不是已审查的手写Philox结构，需另外核查')
        name=args.parent_batch+'-uniform-philox-repair'
        batch=parent.parent/name
        if batch.exists():
            raise FileExistsError('补充批次已存在，不重复设备工作')
        plan=read(parent/'plan.json')
        record=dict(next(r for r in plan['records'] if r['operator']=='uniform'))
        leaf=batch/record['catalog_leaf']
        leaf.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(parent/record['catalog_leaf'],leaf,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(parent/'catalog/manifest.json',batch/'catalog/manifest.json')
        seed=batch/record['seed']
        seed.parent.mkdir(parents=True,exist_ok=True)
        seed.write_bytes(source.read_bytes())
        record.update(source_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),seed_origin=str(source))
        plan.update(records=[record],duration_hours=1,parent_batch=args.parent_batch,uniform_philox_check_required=True,
                    policy='独立Philox逐位核验；已证实的整数机制修复另列贡献；保留原18项语义和3项随机状态及一次完整KGS，不用统计通过代替算法正确')
        (batch/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
        shutil.copytree(parent/'knowledge',batch/'knowledge')
        spec=importlib.util.spec_from_file_location('uniform_repair_campaign',TOOLS/'run-endurance.py')
        m=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        c=m.Campaign(name)
        c.q.idle()
        opstate=c.state['operators'].setdefault('uniform',dict(status='Philox独立核验'))

        def finish():
            c.finish()
            for script,extra in [('audit-endurance.py',['--output',batch/'completion-audit.json']),('summarize-endurance.py',[])]:
                subprocess.run(['docker','exec','tle_yy',m.PY,str(TOOLS/script),'--batch-dir',str(batch)]+[str(x) for x in extra],check=True)

        def bitcheck(candidate,label):
            out=batch/'research/uniform'/(label+'-philox.json')
            try:
                c.run('philox-'+label,[m.PY,TOOLS/'check-uniform-philox.py','--source',candidate,'--output',out],600)
            except RuntimeError:
                if not out.exists():
                    raise
            data=read(out)
            if data['source_sha256']!=hashlib.sha256(candidate.read_bytes()).hexdigest():
                raise ValueError('Philox检查源码身份改变')
            return dict(path=str(out),status=data['status'],passed=data['passed'],total=data['total'])

        parent_check=c.step('philox-parent:uniform',lambda:bitcheck(seed,'parent'))
        opstate['parent_philox_check']=parent_check
        if parent_check and parent_check['status']=='passed':
            opstate['status']='主批Philox逐位核验通过，不重复正式评测'
            finish()
            return
        target=batch/'candidates/philox-compatible/main.py'
        c.step('prepare:uniform:philox-compatible',lambda:c.run('prepare-uniform-philox',
                      [m.PY,TOOLS/'prepare-uniform-philox-repair.py','--parent',seed,'--output',target],120))
        if not target.exists():
            opstate['status']='源码结构待审查，未提交修复评测'
            finish()
            return
        fixed_check=c.step('philox-fixed:uniform',lambda:bitcheck(target,'fixed'))
        opstate['fixed_philox_check']=fixed_check
        if fixed_check and fixed_check['status']=='passed':
            result=c.step('validate:uniform:philox-compatible',lambda:c.candidate('uniform',target,target.parent,'philox-compatible'))
            if result and result.get('status')=='passed':
                opstate['best']=result
                opstate['philox_check']=fixed_check
                c.step('observe-final:uniform',lambda:c.observe('uniform',target,batch/'research/uniform/final'))
                c.step('metrics-final:uniform',lambda:c.metrics('uniform',result,batch/'research/uniform/final-metrics'))
                opstate['status']='Philox修复版通过全部核验'
            else:
                opstate['status']='Philox正确，完整语义或KGS未通过'
        else:
            opstate['status']='修复后Philox设备核验仍失败，不宣称标准算法交付'
        finish()


if __name__=='__main__':
    main()
