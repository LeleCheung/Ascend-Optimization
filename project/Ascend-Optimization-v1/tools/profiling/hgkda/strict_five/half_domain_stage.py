"""待在途阶段终态后串行补查FP16完整域，不重测正式性能。"""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
ROOT = PROJECT.parents[2]
BATCH = PROJECT/'operators/campaigns/strict-five-20261103'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    temp.replace(path)


def main():
    sys.path.insert(0, str(HERE.parent/'strict_night'))
    from boundary_upgrade import terminal_workspace
    spec = importlib.util.spec_from_file_location('strict_five_domain_queue', HERE.parent/'run-nightly.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    process = BATCH/'half-domain-process.json'
    assert not process.exists(), '已有完整域检查身份，不重复启动'
    identity = dict(pid=os.getpid(), process_start=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                    status='waiting_for_prior_stages')
    save(process, identity)
    try:
        predecessors = ['optimizer', 'half-evaluator', 'half-comparisons', 'reference-stage']
        while True:
            states = {name: read(BATCH/(name+'-process.json'))['status'] for name in predecessors}
            assert not any(v=='failed' for v in states.values()), '先行阶段异常，保留现场待审查'
            if all(v=='completed' for v in states.values()):
                break
            time.sleep(10)
        with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert all(terminal_workspace(w) for w in (ROOT/'runs/strict-five-20261103/kg-runs').iterdir() if w.is_dir())
            queue = module.Queue(BATCH.name, ROOT, 'tle_yy')
            queue.idle()
            identity.update(status='running', started_at=module.now())
            save(process, identity)
            summaries = []
            for name in ['codex-native-half-v1', 'codex-half-comparisons-v2']:
                folder = BATCH/'proposals/expm1_'/name
                previous = read(folder/'result.json')
                if not previous.get('qualified_original'):
                    summaries.append(dict(proposal=name, status='skipped_failed_prior_qualification'))
                    continue
                source = folder/'main.py'
                assert hashlib.sha256(source.read_bytes()).hexdigest() == previous['source_sha256']
                output = folder/'full-fp16-domain'
                assert not output.exists(), '已有补测身份，保留现场'
                checker = HERE/'probe_half_domain.py'
                frozen = folder/('half-domain-checker-'+hashlib.sha256(checker.read_bytes()).hexdigest()+'.py')
                assert not frozen.exists()
                frozen.write_bytes(checker.read_bytes())
                queue.idle()
                queue.run('half-domain-'+name, shlex.join(['/usr/local/python3.11.15/bin/python3.11', str(frozen),
                    '--source', str(source), '--output', str(output)]), 1200)
                result = read(output/'result.json')
                summaries.append(dict(proposal=name, status=result['status'], passed=result['passed'],
                    total=result['total'], result=str(output/'result.json')))
                save(BATCH/'half-domain-summary.json', dict(status='running', proposals=summaries, CANN_accepted=False))
            queue.idle()
            save(BATCH/'half-domain-summary.json', dict(status='completed', proposals=summaries, CANN_accepted=False))
            identity.update(status='completed', finished_at=module.now())
            save(process, identity)
    except Exception as error:
        identity.update(status='failed', error=type(error).__name__+': '+str(error), finished_at=module.now())
        save(process, identity)
        raise


if __name__ == '__main__':
    main()
