"""等待已登记提案终态后，在同一卡锁内连续运行两轮KG/CC搜索。"""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
ROOT = PROJECT.parents[2]
BATCH = PROJECT / 'operators/campaigns/strict-next-20261103'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    temp.replace(path)


def main():
    process = BATCH / 'optimizer-process.json'
    assert not process.exists(), '已有优化器身份，不重复启动'
    identity = dict(pid=os.getpid(), process_start=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                    status='waiting_for_proposal_and_profiles')
    save(process, identity)
    try:
        for name in ['collector', 'typed-fma-evaluator']:
            path = BATCH / (name + '-process.json')
            while not path.exists() or read(path)['status'] in ['waiting_for_compile_and_profiles', 'waiting_for_project_lock', 'running']:
                time.sleep(10)
            value = read(path)
            if value['status'] != 'completed':
                # 只允许已确认完全发生在设备执行前的离线编译失败；其他异常保留现场。
                assert name == 'typed-fma-evaluator' and value['status'] == 'requires_review'
                assert read(BATCH / 'analysis/expm1_/compile-typed-fma-v2/result.json')['status'] == 'compile_failed'
                assert not (BATCH / 'proposals/expm1_/codex-typed-fma-v2/pytorch').exists()
            stat = Path('/proc') / str(value['pid']) / 'stat'
            while stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[19] == str(value['process_start']):
                time.sleep(2)
        with (ROOT / 'runs/physical-card-7-campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            spec = importlib.util.spec_from_file_location('strict_next_search', HERE / 'search.py')
            search = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(search)
            identity.update(status='running', current_job=None)
            save(process, identity)
            for attempt in [1, 2]:
                identity['current_job'] = 'expm1_:search-' + str(attempt)
                save(process, identity)
                search.run(attempt)
            identity.update(status='completed', current_job=None)
            save(process, identity)
    except Exception as error:
        identity.update(status='requires_review', error=type(error).__name__ + ': ' + str(error))
        save(process, identity)
        raise


if __name__ == '__main__':
    main()
