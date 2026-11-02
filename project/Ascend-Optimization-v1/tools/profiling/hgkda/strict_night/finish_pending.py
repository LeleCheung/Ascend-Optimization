"""原搜索释放项目锁后，评测已审查提案并核验最佳源码；不重启任何服务。"""
import fcntl
import importlib.util
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from night_common import ROOT,BATCH,read,save
from boundary_upgrade import terminal_workspace


def main():
    spec=importlib.util.spec_from_file_location('strict_night_pending_controller',HERE/'run.py')
    controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
    process=BATCH/'finish-pending-process.json'
    assert not process.exists(), '已有收尾进程记录，先核对身份，不重复启动'
    identity=dict(pid=os.getpid(),process_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                  status='waiting_for_project_lock')
    save(process,identity)
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        runner=controller.Runner()
        try:
            assert runner.state['status'] in {'search_completed_pending_review','backend_verified'}, '搜索尚未正常终态，保留现场'
            assert runner.state.get('current_job') is None, '仍有当前任务，不交接'
            for workspace in (runner.q.runtime/'kg-runs').iterdir():
                if not workspace.is_dir():continue
                assert terminal_workspace(workspace), '工作区终态或进程身份未确认：'+str(workspace)
            runner.wait()
            identity.update(status='running',started_after_lock=runner.module.now())
            save(process,identity)
            runner.state.update(status='finishing_pending_proposals',current_job=None)
            runner.persist()
            runner.process_proposals()
            runner.refresh_existing_scores()
            runner.verify_reviewed_best()
            pending=[op for op,row in runner.state['operators'].items()
                     if row.get('best') and row['best'].get('compiled_status')!='verified']
            runner.wait()
            runner.state.update(status='search_completed_pending_review' if pending else 'backend_verified',
                                current_job=None,finished_at=runner.module.now())
            runner.persist()
            identity.update(status='completed',finished_at=runner.module.now(),pending_best=pending)
            save(process,identity)
            print(controller.quote(['收尾完成',runner.state['status']]),flush=True)
        except Exception as error:
            runner.state.update(status='requires_review',error=type(error).__name__+': '+str(error))
            runner.persist()
            identity.update(status='failed',error=runner.state['error'],finished_at=runner.module.now())
            save(process,identity)
            raise


if __name__=='__main__':main()
