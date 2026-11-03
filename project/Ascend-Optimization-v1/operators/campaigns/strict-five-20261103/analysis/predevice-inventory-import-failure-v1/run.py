"""新五项的独立可恢复串行队列；所有设备工作持同一卡锁，未知任务不重发。"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/strict-five-20261103'
PY='/usr/local/python3.11.15/bin/python3.11'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(path)


def main():
    module=importlib.util.spec_from_file_location('strict_five_queue',HERE.parent/'run-nightly.py')
    m=importlib.util.module_from_spec(module);module.loader.exec_module(m)
    sys.path.insert(0,str(HERE.parent/'strict_night'))
    from boundary_upgrade import terminal_workspace
    process=BATCH/'controller-process.json'
    assert not process.exists(),'已有控制器身份；先检查原进程，不重复启动'
    identity=dict(pid=os.getpid(),process_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        status='waiting_for_project_lock',started_at=m.now())
    save(process,identity)
    state_path=BATCH/'state.json'
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        q=m.Queue(BATCH.name,ROOT,'tle_yy')
        previous=PROJECT/'operators/campaigns/strict-night-20261102'
        assert read(previous/'state.json')['status']=='backend_verified'
        assert read(previous/'state.json').get('current_job') is None
        for w in (ROOT/'runs/strict-night-20261102/kg-runs').iterdir():
            if w.is_dir():assert terminal_workspace(w),str(w)
        q.idle()
        identity.update(status='running',acquired_lock_at=m.now());save(process,identity)
        state=read(state_path) if state_path.exists() else dict(status='running',jobs={},frozen_accepted=25,current_job=None)
        assert not state.get('current_job'),'未知任务不能重发'
        end=time.monotonic()+18*3600
        try:
            while time.monotonic()<end:
                state.update(status='waiting_for_tasks',current_job=None,heartbeat=m.now());save(state_path,state)
                for f in sorted((BATCH/'tasks').glob('*.json')):
                    job=read(f);key=job['id'];prior=state['jobs'].get(key)
                    if prior:
                        assert prior['spec_sha256']==sha(f),'任务规范改变，保留现场'
                        assert prior['status'] in {'completed','failed'},'未知任务不能重发'
                        continue
                    assert not job.get('depends_on') or all(state['jobs'].get(k,{}).get('status')=='completed' for k in job['depends_on'])
                    for name,digest in job.get('files_sha256',{}).items():assert sha(name)==digest,name
                    q.idle()
                    state.update(status='running',current_job=key)
                    entry=dict(status='running',spec=str(f),spec_sha256=sha(f),started_at=m.now())
                    state['jobs'][key]=entry;save(state_path,state)
                    try:
                        q.run(key,shlex.join([str(v) for v in job['command']]),job.get('budget_seconds',1800))
                        if job.get('receipt'):assert Path(job['receipt']).exists(),'任务返回成功但无结果'
                        entry.update(status='completed',finished_at=m.now())
                    except RuntimeError as error:
                        # 超时仍可能有进程，用Attention保留现场；仅已退出非零任务可按规范继续。
                        if isinstance(error,m.Attention) or not job.get('allow_failure'):raise
                        entry.update(status='failed',error=str(error),finished_at=m.now())
                    q.idle();state['current_job']=None;save(state_path,state)
                if (BATCH/'finish-request.json').exists():
                    assert all(v['status'] in {'completed','failed'} for v in state['jobs'].values())
                    state.update(status='tasks_finished',current_job=None,finished_at=m.now());save(state_path,state)
                    identity.update(status='completed',finished_at=m.now());save(process,identity);return
                time.sleep(10)
            state.update(status='waiting_window_elapsed',current_job=None);save(state_path,state)
            identity.update(status='completed',finished_at=m.now());save(process,identity)
        except Exception as error:
            state.update(status='requires_review',error=type(error).__name__+': '+str(error));save(state_path,state)
            identity.update(status='failed',error=state['error'],finished_at=m.now());save(process,identity)
            raise


if __name__=='__main__':main()
