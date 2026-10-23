#!/usr/bin/env python3
"""等待五项队列结束后串行补验最终源码，再做离线审计。"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import time

TOOLS=Path(__file__).resolve().parent
ROOT=Path('/data/hanle/ascend-optimization')
PY='/usr/local/python3.11.15/bin/python3.11'


def read(path):return json.loads(path.read_text())
def save(path,data):
    data['updated_at']=datetime.datetime.utcnow().isoformat()+'Z'
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)


def execute(name):
    batch=ROOT/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns'/name
    path=batch/'postflight-state.json'
    state=dict(status='waiting_for_campaign',jobs={})
    save(path,state)
    deadline=time.monotonic()+24*3600
    lock=(ROOT/'runs/physical-card-7-campaign.lock').open('a')
    while True:
        campaign=read(batch/'endurance-state.json')
        if campaign['status']=='completed':
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:pass
        if campaign['status']=='needs_attention':
            state.update(status='waiting_for_campaign_attention',campaign_error=campaign.get('error'));save(path,state)
        if time.monotonic()>deadline:raise TimeoutError('补验等待预算结束；未提交任何设备请求')
        time.sleep(30)
    spec=importlib.util.spec_from_file_location('remaining_postflight_queue',TOOLS/'run-nightly.py')
    queue_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue_module)
    queue=queue_module.Queue(name,ROOT,'tle_yy')
    delivery=read(batch/'deliverables/manifest.json')
    supplemental=read(batch/'knowledge/supplemental-tool-sha256.json')
    script=TOOLS/'check-remaining-special.py'
    if hashlib.sha256(script.read_bytes()).hexdigest()!=supplemental[script.name]:raise ValueError('补验工具身份改变')
    state['status']='running';save(path,state)
    import shlex
    def run(key,parts,target):
        queue.idle()
        state['current_job']=key;state['jobs'][key]=dict(status='running');save(path,state)
        try:
            queue.run('postflight-'+key,' '.join(shlex.quote(str(p)) for p in parts),1800)
            status='passed'
        except RuntimeError:
            if not target.exists():raise
            status='failed'
        state['jobs'][key]=dict(status=status,result=str(target));save(path,state)
        return status
    try:
        for row in delivery['records']:
            op=row['operator'];source=Path(row['delivered'])
            if op in ['upsample_bilinear2d','fractional_max_pool3d']:
                target=batch/'results'/op/'final-special.json'
                run(op+'-special',[PY,script,'--operator',op,'--source',source,'--output',target],target)
            if op=='_conj_physical':
                record=queue.records[op];leaf=batch/record['catalog_leaf']
                cases=[json.loads(line) for line in (leaf/'timing.jsonl').read_text().splitlines() if line.strip()]
                index=next(i for i,r in enumerate(cases) if r['inputs']['spec']['dtype']=='complex64')
                target=batch/'research'/op/'final-complex/facts.json'
                run(op+'-complex-trace',[PY,TOOLS/'observe-endurance.py','--operator',op,'--source',source,
                    '--leaf',leaf,'--index',index,'--output',target.parent],target)
        queue.idle()
        audit=batch/'completion-audit.json'
        run('audit',[PY,TOOLS/'audit-remaining-five.py','--batch-dir',batch,'--output',audit],audit)
        state.update(status='passed' if all(j['status']=='passed' for j in state['jobs'].values()) else 'needs_attention',current_job=None)
        save(path,state)
    except Exception as error:
        state.update(status='needs_attention',error=type(error).__name__+': '+str(error));save(path,state);raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',required=True)
    args=parser.parse_args();execute(args.batch)
