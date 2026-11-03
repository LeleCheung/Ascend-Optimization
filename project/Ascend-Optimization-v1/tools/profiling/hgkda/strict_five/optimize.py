"""采集队列释放同一卡锁后启动KG搜索；不重启共享服务，不并发用卡。"""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
ROOT=PROJECT.parents[2]
BATCH=PROJECT/'operators/campaigns/strict-five-20261103'


def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8');tmp.replace(path)


def main():
    sys.path.insert(0,str(HERE.parent/'strict_night'))
    from boundary_upgrade import terminal_workspace
    spec=importlib.util.spec_from_file_location('strict_five_optimize_module',HERE/'search.py')
    search=importlib.util.module_from_spec(spec);spec.loader.exec_module(search)
    process=BATCH/'optimizer-process.json';assert not process.exists(),'已有搜索控制器，不重复启动'
    identity=dict(pid=os.getpid(),process_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],status='waiting_for_project_lock')
    save(process,identity)
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert json.loads((BATCH/'state.json').read_text())['status']=='tasks_finished'
        identity.update(status='running',current_job=None);save(process,identity)
        try:
            for op,attempt in [('expm1_',1),('expm1_',2)]:
                identity['current_job']=op+':search-'+str(attempt);save(process,identity)
                sys.argv=[str(HERE/'search.py'),'--operator',op,'--attempt',str(attempt)]
                search.main()
            identity.update(status='completed',current_job=None);save(process,identity)
        except Exception as error:
            identity.update(status='failed',error=type(error).__name__+': '+str(error));save(process,identity);raise


if __name__=='__main__':main()
