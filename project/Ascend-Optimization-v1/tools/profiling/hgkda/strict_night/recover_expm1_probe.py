"""仅恢复已确认在导入设备库前失败的整块检查；复用正式成绩和63项补测。"""
import fcntl
import importlib.util
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from night_common import ROOT,BATCH,read,save,sha
from boundary_upgrade import terminal_workspace


def alive(value):
    p=Path('/proc')/str(value['pid'])/'stat'
    fields=p.read_text().rsplit(')',1)[1].split() if p.exists() else []
    return bool(fields and fields[19]==str(value['process_start']))


def main():
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        process=BATCH/'finish-pending-process.json'
        old=read(process)
        assert old['status']=='failed' and not alive(old),'原收尾进程仍在运行或非失败终态'
        marker=BATCH/'finish-pending-predevice-recovery.json'
        assert not marker.exists(),'已有恢复身份，先检查，不重复恢复'
        spec=importlib.util.spec_from_file_location('strict_night_recovery_controller',HERE/'run.py')
        controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)
        runner=controller.Runner()
        key='proposal:expm1_:codex-exact-zero-v5'
        assert runner.state['status']=='requires_review' and runner.state['current_job']==key
        assert 'exact-math.log' in runner.state['error']
        out=BATCH/'proposals/expm1_/codex-exact-zero-v5'
        source=out/'main.py'
        assert sha(source)=='ab6d7aeab424f06e2de4860f2aed3b807a457eeb9784c95d4955a81d99b1f56c'
        checker=out/'exact-expm1-checker-9f9cee54be0879466997a3363510945117362a74735a32cc24765cc47f18cec1.py'
        assert sha(checker)=='9f9cee54be0879466997a3363510945117362a74735a32cc24765cc47f18cec1'
        log=ROOT/'runs/strict-night-20261102/proposal-expm1_-codex-exact-zero-v5-exact-math.log'
        error=log.read_text()
        assert "AttributeError: 'str' object has no attribute 'read_bytes'" in error
        assert 'checker_sha256=sha(__file__)' in error
        probe=out/'exact-expm1-probe'
        assert probe.is_dir() and not list(probe.iterdir()),'已有设备检查证据或未知状态，不重发'
        assert checker.read_text().index('checker_sha256=sha(__file__)')<checker.read_text().index('import torch\n')
        formal=out/'pytorch/Codex-strict-PyTorch-1.result.json'
        extra=out/'extra-pytorch.json'
        assert read(formal)['status']=='PASSED' and read(formal)['num_passed']==30
        assert read(extra)['status']=='passed' and read(extra)['passed']==read(extra)['total']==63
        for workspace in (runner.q.runtime/'kg-runs').iterdir():
            if workspace.is_dir():assert terminal_workspace(workspace),str(workspace)
        runner.wait()
        # 保留失败检查器、原日志和空目录；不覆盖任何结果或重新提交正式评测。
        archived=out/'exact-expm1-probe-predevice-failure-v1'
        assert not archived.exists()
        probe.rename(archived)
        backup=BATCH/'finish-pending-process-predevice-failure-v1.json'
        assert not backup.exists();backup.write_bytes(process.read_bytes())
        copied_log=out/'exact-expm1-predevice-failure-v1.log'
        assert not copied_log.exists();copied_log.write_bytes(log.read_bytes())
        receipt=dict(status='known_predevice_failure_recovery',source_sha256=sha(source),
            previous_process=old,old_checker=str(checker),old_checker_sha256=sha(checker),
            error_log=str(copied_log),error_log_sha256=sha(copied_log),
            formal_sha256=sha(formal),supplement_sha256=sha(extra),
            scope='原检查在导入torch前失败，目录为空；只修复检查器路径类型，正式30项及63项补测复用。')
        save(marker,receipt)
        identity=dict(pid=os.getpid(),process_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            status='running',started_after_lock=runner.module.now(),recovery=str(marker))
        save(process,identity)
        try:
            runner.state.pop('error',None)
            runner.state.update(status='finishing_pending_proposals',current_job=None)
            runner.persist();runner.process_proposals();runner.refresh_existing_scores();runner.verify_reviewed_best()
            assert sha(formal)==receipt['formal_sha256'] and sha(extra)==receipt['supplement_sha256']
            runner.wait()
            pending=[op for op,row in runner.state['operators'].items()
                if row.get('best') and row['best'].get('compiled_status')!='verified']
            runner.state.update(status='search_completed_pending_review' if pending else 'backend_verified',
                current_job=None,finished_at=runner.module.now())
            runner.persist()
            identity.update(status='completed',finished_at=runner.module.now(),pending_best=pending)
            save(process,identity)
            print(runner.state['status'],flush=True)
        except Exception as error:
            runner.state.update(status='requires_review',error=type(error).__name__+': '+str(error));runner.persist()
            identity.update(status='failed',error=runner.state['error'],finished_at=runner.module.now());save(process,identity)
            raise


if __name__=='__main__':main()
