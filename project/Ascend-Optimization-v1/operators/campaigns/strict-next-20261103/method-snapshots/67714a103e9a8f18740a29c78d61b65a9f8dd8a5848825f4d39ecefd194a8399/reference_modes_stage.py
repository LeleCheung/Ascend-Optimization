"""连续搜索结束后串行探测官方精度模式；不抢占其他项目的运行设备。"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
ROOT = PROJECT.parents[2]
BATCH = PROJECT / 'operators/campaigns/strict-next-20261103'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    temp.replace(path)


def main():
    path = BATCH / 'reference-modes-process.json'
    assert not path.exists(), '已有参考阶段身份，不能重新启动'
    value = dict(pid=os.getpid(), process_start=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                 status='waiting_for_optimizer')
    save(path, value)
    try:
        optimizer_path = BATCH / 'optimizer-process.json'
        while read(optimizer_path)['status'] in ['waiting_for_proposal_and_profiles', 'running']:
            time.sleep(10)
        optimizer = read(optimizer_path)
        assert optimizer['status'] == 'completed', '先检查优化器异常，不能越过未知设备阶段'
        proc = Path('/proc') / str(optimizer['pid']) / 'stat'
        while proc.exists() and proc.read_text().rsplit(')', 1)[1].split()[19] == str(optimizer['process_start']):
            time.sleep(2)
        value['status'] = 'waiting_for_project_lock'
        save(path, value)
        with (ROOT / 'runs/physical-card-7-campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            spec = importlib.util.spec_from_file_location('strict_next_reference_queue', HERE.parent / 'run-nightly.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            queue = module.Queue(BATCH.name, ROOT, 'tle_yy')
            queue.idle()
            value.update(status='running', started_at=module.now())
            save(path, value)
            checker = HERE / 'probe_expm1_mode.py'
            digest = hashlib.sha256(checker.read_bytes()).hexdigest()
            frozen = BATCH / 'method-snapshots' / digest / checker.name
            frozen.parent.mkdir(parents=True, exist_ok=True)
            if frozen.exists():
                assert frozen.read_bytes() == checker.read_bytes()
            else:
                frozen.write_bytes(checker.read_bytes())
            for mode in ['default', 'high_precision']:
                value.update(current_mode=mode, checker_sha256=digest)
                save(path, value)
                queue.idle()
                queue.run('reference-expm1-mode-' + mode, shlex.join([queue.py, str(checker), '--mode', mode]), 900)
                assert (BATCH / 'reference-probes/legacy-expm1-modes-v1' / mode / 'result.json').exists()
                queue.idle()
            value.update(status='completed', current_mode=None, finished_at=module.now())
            save(path, value)
    except Exception as error:
        value.update(status='requires_review', error=type(error).__name__ + ': ' + str(error))
        save(path, value)
        raise


if __name__ == '__main__':
    main()
