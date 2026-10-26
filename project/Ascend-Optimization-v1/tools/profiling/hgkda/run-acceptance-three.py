#!/usr/bin/env python3
"""宿主持卡锁与tmux串行执行本批隔离诊断，保存每个已知作业回执。"""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import urllib.request

ROOT = Path('/data/hanle/ascend-optimization')
PROJECT = ROOT/'Ascend-Optimization/project/Ascend-Optimization-v1'
BATCH = PROJECT/'operators/campaigns/acceptance-three-20261026'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+chr(10))
    temporary.replace(path)


def main():
    state_path = BATCH/'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(status='created', jobs={})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def idle():
        for port in [19655,19657,19659]:
            scheduler = json.load(opener.open('http://127.0.0.1:%d/status'%port, timeout=15))['scheduler']
            assert scheduler['healthy']==1 and not any(scheduler[k] for k in ['active','waiting','broken']), scheduler
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state.update(status='running', started_at=datetime.datetime.utcnow().isoformat()+'Z')
        save(state_path,state)
        source = (PROJECT/'tools/profiling/hgkda/probe-acceptance-three.py').read_text()
        for kind in ['cumulative','conjugate']:
            if kind in state['jobs']:
                assert state['jobs'][kind]['status'] in ['SUCCEEDED','FAILED','TIMED_OUT']
                continue
            idle()
            folder = BATCH/'jobs'/kind
            folder.mkdir(parents=True,exist_ok=False)
            request = dict(command=['{python}','probe.py','--kind',kind],
                files=[dict(path='probe.py',content=source)],timeout_seconds=1200)
            save(folder/'request.json',request)
            state['current_job']=kind
            save(state_path,state)
            response = json.load(opener.open(urllib.request.Request('http://127.0.0.1:19659/debug/jobs',
                data=json.dumps(request).encode(),headers={'Content-Type':'application/json'},method='POST'),timeout=1245))
            save(folder/'response.json',response)
            state['jobs'][kind]=dict(status=response['status'],job_id=response['job_id'],
                source_sha256=hashlib.sha256(source.encode()).hexdigest())
            save(state_path,state)
            idle()
        state.update(status='diagnostics_completed',current_job=None,finished_at=datetime.datetime.utcnow().isoformat()+'Z')
        save(state_path,state)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        path=BATCH/'state.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        state.update(status='needs_attention',error=type(error).__name__+': '+str(error))
        save(path,state)
        raise
