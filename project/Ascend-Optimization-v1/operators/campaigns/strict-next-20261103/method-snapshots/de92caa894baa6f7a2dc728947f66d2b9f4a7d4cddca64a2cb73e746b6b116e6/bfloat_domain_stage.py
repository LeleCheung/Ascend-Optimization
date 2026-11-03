"""已排队阶段全部结束后核验BF16全位模式；不重测性能或修改既有结果。"""
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
    path = BATCH / 'bfloat-domain-process.json'
    assert not path.exists(), '已有域检查身份，不重复启动'
    value = dict(pid=os.getpid(), process_start=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                 status='waiting_for_existing_stages')
    save(path, value)
    pending = {'waiting_for_compile_and_profiles', 'waiting_for_optimizer', 'waiting_for_project_lock',
               'waiting_for_proposal_and_profiles', 'running'}
    try:
        for name in ['optimizer', 'buffer-evaluator', 'reference-modes']:
            identity = BATCH / (name + '-process.json')
            while not identity.exists() or read(identity)['status'] in pending:
                time.sleep(10)
            previous = read(identity)
            assert previous['status'] == 'completed', '先核对上游阶段异常：' + name
            process = Path('/proc') / str(previous['pid']) / 'stat'
            while process.exists() and process.read_text().rsplit(')', 1)[1].split()[19] == str(previous['process_start']):
                time.sleep(2)
        with (ROOT / 'runs/physical-card-7-campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            spec = importlib.util.spec_from_file_location('strict_next_bfloat_queue', HERE.parent / 'run-nightly.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            queue = module.Queue(BATCH.name, ROOT, 'tle_yy')
            queue.idle()
            value.update(status='running', started_at=module.now())
            save(path, value)
            checker = HERE.parent / 'strict_five/probe_half_domain.py'
            raw = checker.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            frozen = BATCH / 'method-snapshots' / digest / checker.name
            frozen.parent.mkdir(parents=True, exist_ok=True)
            if frozen.exists():
                assert frozen.read_bytes() == raw
            else:
                frozen.write_bytes(raw)
            sources = [('frozen-best', BATCH / 'starting-points/expm1_.py')]
            buffer = BATCH / 'compile-probes/single-buffer-layout-v2'
            result = read(buffer / 'result.json')
            if result.get('qualified_original'):
                assert read(buffer / 'review.json')['all_branches_reviewed']
                sources.append(('buffer-v2', buffer / 'main.py'))
            reports = []
            for label, source in sources:
                output = BATCH / 'analysis/expm1_' / ('bfloat-domain-' + label + '-v1')
                queue.idle()
                queue.run('bfloat-domain-' + label, shlex.join([queue.py, str(checker), '--source', str(source),
                          '--output', str(output), '--dtype', 'bfloat16']), 1200)
                measured = read(output / 'result.json')
                assert measured['checker_sha256'] == digest and measured['dtype'] == 'bfloat16'
                reports.append(dict(label=label, result=str(output / 'result.json'), status=measured['status']))
                queue.idle()
            value.update(status='completed', reports=reports, finished_at=module.now())
            save(path, value)
    except Exception as error:
        value.update(status='requires_review', error=type(error).__name__ + ': ' + str(error))
        save(path, value)
        raise


if __name__ == '__main__':
    main()
