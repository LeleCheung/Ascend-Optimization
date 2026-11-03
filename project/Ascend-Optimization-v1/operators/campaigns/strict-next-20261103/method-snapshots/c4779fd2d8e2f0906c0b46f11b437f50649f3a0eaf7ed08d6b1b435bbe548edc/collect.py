"""在原物理卡7项目锁内采集最新起点的三类型指标；已有任务身份不重启。"""
import fcntl
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
PY = '/usr/local/python3.11.15/bin/python3.11'


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def main():
    spec = importlib.util.spec_from_file_location('strict_next_queue', HERE.parent / 'run-nightly.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    process = BATCH / 'collector-process.json'
    assert not process.exists(), '已有采集身份，先核对PID和启动时间，不重复提交'
    identity = dict(pid=os.getpid(), process_start=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                    status='waiting_for_project_lock', started_at=module.now())
    save(process, identity)
    try:
        with (ROOT / 'runs/physical-card-7-campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            queue = module.Queue(BATCH.name, ROOT, 'tle_yy')
            queue.idle()
            identity.update(status='running', acquired_lock_at=module.now())
            save(process, identity)
            rows = []
            for dtype in ['float16', 'bfloat16', 'float32']:
                queue.idle()
                output = BATCH / 'analysis/expm1_' / ('paired-best-' + dtype)
                identity['current_job'] = 'paired-best-' + dtype
                save(process, identity)
                save(BATCH / 'progress.json', dict(status='profiling', current_stage=identity['current_job'],
                                                  accepted=25, total=30, finished_profiles=rows, updated_at=module.now()))
                queue.run(identity['current_job'], shlex.join([PY, str(HERE.parent / 'strict_five/paired.py'),
                          '--spec', str(BATCH / 'analysis/expm1_' / (dtype + '-spec.json')),
                          '--output', str(output)]), 1800)
                summary = json.loads((output / 'summary.json').read_text())
                rows.append(dict(dtype=dtype, status=summary['status'], result=str(output / 'summary.json')))
            queue.idle()
            save(BATCH / 'paired-summary.json', dict(status='completed', cases=rows,
                                                    scope='最新最佳Triton与冻结原PyTorch；不赋予CANN参考资格'))
            save(BATCH / 'progress.json', dict(status='profiles_completed', accepted=25, total=30,
                                              current_stage='三类型指标已采集，等待机制诊断及新候选', profiles=rows))
            identity.update(status='completed', current_job=None, finished_at=module.now())
            save(process, identity)
    except Exception as error:
        identity.update(status='requires_review', error=type(error).__name__ + ': ' + str(error), finished_at=module.now())
        save(process, identity)
        raise


if __name__ == '__main__':
    main()
