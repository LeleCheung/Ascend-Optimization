"""等待同一卡锁后探测未测试的原生参考入口；不修改候选、原合同或共享服务。"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/strict-five-20261103'


def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(path)


def main():
    spec=importlib.util.spec_from_file_location('strict_five_reference_queue',HERE.parent/'run-nightly.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    process=BATCH/'reference-stage-process.json';assert not process.exists()
    identity=dict(pid=os.getpid(),process_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],status='waiting_for_project_lock')
    save(process,identity)
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert json.loads((BATCH/'optimizer-process.json').read_text())['status']=='completed'
        q=m.Queue(BATCH.name,ROOT,'tle_yy');q.idle()
        identity.update(status='running',started_at=m.now());save(process,identity)
        try:
            q.run('build-uniform-tensor-v1',shlex.join(['/usr/local/python3.11.15/bin/python3.11',str(HERE/'build_uniform_tensor.py')]),900)
            assert (BATCH/'reference-probes/uniform-tensor-v1/build.json').exists()
            q.idle()
            q.run('probe-uniform-tensor-v1',shlex.join(['/usr/local/python3.11.15/bin/python3.11',str(HERE/'probe_uniform_tensor.py')]),900)
            result=BATCH/'reference-probes/uniform-tensor-v1/numerics.json'
            assert result.exists();q.idle()
            identity.update(status='completed',finished_at=m.now(),result=str(result),
                result_sha256=hashlib.sha256(result.read_bytes()).hexdigest());save(process,identity)
        except Exception as error:
            identity.update(status='failed',error=type(error).__name__+': '+str(error),finished_at=m.now());save(process,identity);raise


if __name__=='__main__':main()
