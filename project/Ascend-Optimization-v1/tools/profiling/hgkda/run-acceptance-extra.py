#!/usr/bin/env python3
"""本批追加阶段的宿主串行控制器；未知作业结果时停止，不盲目重发。"""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path('/data/hanle/ascend-optimization')
PROJECT=ROOT/'Ascend-Optimization/project/Ascend-Optimization-v1'
BATCH=PROJECT/'operators/campaigns/acceptance-three-20261026'
TOOLS=PROJECT/'tools/profiling/hgkda'


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+chr(10))
    temp.replace(path)


def execute(stage,script,arguments,timeout):
    out=BATCH/'stages'/stage
    out.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',stage=stage,started_at=datetime.datetime.utcnow().isoformat()+'Z')
    save(out/'state.json',state)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def idle():
        for port in [19655,19657,19659]:
            s=json.load(opener.open('http://127.0.0.1:%d/status'%port,timeout=15))['scheduler']
            assert s['healthy']==1 and not any(s[k] for k in ['active','waiting','broken']),s
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        idle()
        source=(TOOLS/script).read_text()
        request=dict(command=['{python}','run.py']+arguments,
            files=[dict(path='run.py',content=source)],timeout_seconds=timeout)
        save(out/'request.json',request)
        response=json.load(opener.open(urllib.request.Request('http://127.0.0.1:19659/debug/jobs',
            data=json.dumps(request).encode(),headers={'Content-Type':'application/json'},method='POST'),timeout=timeout+45))
        save(out/'response.json',response)
        state.update(status=response['status'],job_id=response['job_id'],script_sha256=hashlib.sha256(source.encode()).hexdigest(),
            finished_at=datetime.datetime.utcnow().isoformat()+'Z')
        save(out/'state.json',state)
        idle()
    if response['status']!='SUCCEEDED':
        raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True)
    p.add_argument('--script',required=True)
    p.add_argument('--timeout',type=int,default=1800)
    p.add_argument('arguments',nargs='*')
    a=p.parse_args()
    assert a.stage.replace('-','').replace('_','').isalnum() and Path(a.script).name==a.script
    execute(a.stage,a.script,a.arguments,a.timeout)
