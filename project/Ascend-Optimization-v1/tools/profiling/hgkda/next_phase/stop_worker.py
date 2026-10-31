"""停止经过workspace与/proc启动时间核验的本项目KG worker，不操作共享服务。"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request


def proc(pid):
    path=Path('/proc',str(pid))
    try:
        raw=(path/'stat').read_text();f=raw[raw.rindex(')')+2:].split()
        return dict(pid=pid,ppid=int(f[1]),start=f[19],state=f[0],command=(path/'cmdline').read_bytes().split(b'\0'))
    except OSError:return None


def stop(workspace, output, server='http://127.0.0.1:19659'):
    workspace=workspace.resolve();owned_root=Path('/data/hanle/ascend-optimization/runs').resolve()
    workspace.relative_to(owned_root)
    if output.exists():raise FileExistsError(output)
    metadata=workspace/'.kernelgen/run-process.json';d=json.loads(metadata.read_text());pid=d['pid'];current=proc(pid)
    if current and current['state']!='Z':
        if current['start']!=d['process_start'] or b'kernelgen.cli.worker' not in current['command'] or str(workspace).encode() not in current['command']:
            raise RuntimeError('worker身份不匹配，拒绝发信号')
    else:current=None
    processes={}
    for path in Path('/proc').glob('[0-9]*'):
        row=proc(int(path.name))
        if row:processes[row['pid']]=row
    tree={pid};changed=True
    while changed:
        new={p for p,r in processes.items() if r['ppid'] in tree};changed=bool(new-tree);tree|=new
    recorded=[dict(pid=p,start=processes[p]['start'],ppid=processes[p]['ppid']) for p in tree if p in processes]
    if current:
        # 子进程仅由本worker的已核对祖先关系界定；不按用户、镜像名或宽泛命令匹配。
        for target in sorted(tree-{pid},reverse=True)+[pid]:
            before=processes.get(target);live=proc(target)
            if before and live and before['start']==live['start'] and live['state']!='Z':os.kill(target,signal.SIGTERM)
    deadline=time.monotonic()+8
    while time.monotonic()<deadline:
        alive=[r for r in recorded if (p:=proc(r['pid'])) and p['start']==r['start'] and p['state']!='Z']
        if not alive:break
        time.sleep(.2)
    if alive:
        for r in alive:
            p=proc(r['pid'])
            if p and p['start']==r['start'] and p['state']!='Z':os.kill(r['pid'],signal.SIGKILL)
        time.sleep(.5)
    remaining=[r for r in recorded if (p:=proc(r['pid'])) and p['start']==r['start'] and p['state']!='Z']
    if remaining:raise RuntimeError('本worker进程仍存在，继续持锁等待')
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw_metadata=metadata.read_bytes();backup=metadata.with_name('run-process.before-controller-stop.json')
    if not backup.exists():backup.write_bytes(raw_metadata)
    d.update(state='EXITED',finished_at=now,exit_code=None,message='控制器在合作取消未退出后，核验PID/启动时间并终止本worker；实际exit code未由父进程回收')
    metadata.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    progress=workspace/'.kernelgen/run-progress.json'
    if progress.exists():
        backup=progress.with_name('run-progress.before-controller-stop.json')
        if not backup.exists():backup.write_bytes(progress.read_bytes())
        status=json.loads(progress.read_text());status['state']='CANCELLED';progress.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    result=dict(status='own_worker_stopped',workspace=str(workspace),worker_pid=pid,verified_process_start=d['process_start'],processes=recorded,
                finished_at=now,metadata_before_sha256=hashlib.sha256(raw_metadata).hexdigest(),
                scope='只停止本worker和核验的后代；KGS调试作业另按源码身份取消，需确认调度器空闲后才交接卡锁')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def cleanup_debug(workspace, *, server='http://127.0.0.1:19659', container='tle_yy', timeout=180):
    """仅以本worker调试脚本SHA绑定作业；同步API排队时需等目录物化再取消。"""
    scripts=workspace/'stages/optimize/work/tmp/debug'
    wanted={hashlib.sha256(p.read_bytes()).hexdigest():p.name for p in scripts.rglob('*.py')}
    scan="from pathlib import Path;import json,hashlib,sys; wanted=set(json.loads(sys.argv[1]));root=Path('/tmp/kernelgen_server_debug_jobs');print(json.dumps([{'id':p.parent.parent.parent.parent.name,'sha256':h} for p in root.glob('*/workspace/tmp/debug/*.py') if (h:=hashlib.sha256(p.read_bytes()).hexdigest()) in wanted]))"
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));actions=[];deadline=time.monotonic()+timeout
    while True:
        r=subprocess.run(['docker','exec',container,'/usr/local/python3.11.15/bin/python3.11','-c',scan,json.dumps(list(wanted))],capture_output=True,text=True,timeout=30,check=True)
        for row in json.loads(r.stdout):
            url=server+'/debug/jobs/'+row['id']
            try:
                before=json.load(opener.open(url,timeout=15))
                if before['status'] in {'RUNNING','QUEUED'}:
                    after=json.load(opener.open(urllib.request.Request(url,method='DELETE'),timeout=20))
                    actions.append(dict(job_id=row['id'],source_sha256=row['sha256'],file=wanted[row['sha256']],before=before['status'],after=after['status']))
            except urllib.error.HTTPError as error:
                if error.code!=404:raise
        states={port:json.load(opener.open('http://127.0.0.1:'+str(port)+'/status',timeout=15))['scheduler'] for port in [19655,19657,19659]}
        if all(s['healthy']==1 and not any(s[k] for k in ['active','waiting','broken']) for s in states.values()):
            return dict(status='idle',actions=actions,schedulers=states)
        if time.monotonic()>deadline:raise RuntimeError('本worker已停，服务仍非空闲；保留现场，不自动重提')
        time.sleep(3)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(stop(a.workspace,a.output),ensure_ascii=False))
