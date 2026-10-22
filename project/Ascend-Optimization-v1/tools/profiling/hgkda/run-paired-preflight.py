#!/usr/bin/env python3
"""串行持卡锁完成新批次语义门禁及成对采集，tmux托管。"""
import importlib.util
import json
from pathlib import Path
import shlex


def main():
    root = Path('/data/hanle/ascend-optimization')
    tools = Path(__file__).resolve().parent
    modspec = importlib.util.spec_from_file_location('paired_queue', tools/'run-nightly.py')
    module = importlib.util.module_from_spec(modspec)
    modspec.loader.exec_module(module)
    queue = module.Queue('paired-pointwise-20261022', root, 'tle_yy')
    import fcntl
    with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue.idle()
        queue.state.update(status='running', current_job='preflight')
        module.save(queue.path, queue.state)
        try:
            for op in ['log10_', 'selu_']:
                output = queue.batch/'preflight'/op
                output.mkdir(parents=True, exist_ok=True)
                semantics = output/'semantics-v2.json'
                if not semantics.exists():
                    command = [queue.py, str(tools/'check-pointwise-semantics.py'),
                        '--operator', op, '--source', str(queue.batch/queue.records[op]['seed']),
                        '--output', str(semantics)]
                    try:
                        queue.run('preflight-'+op, ' '.join(shlex.quote(x) for x in command), 1800)
                    except RuntimeError:
                        if not semantics.exists():
                            raise
                # 不把语义失败候选当交付。已有正式通过的起点可采集，用于后续修复诊断。
                queue.state[op+'_semantics'] = json.loads(semantics.read_text())['status']
                module.save(queue.path, queue.state)
                for index in [0, 3, 11]:
                    spec = queue.batch/'paired-specs'/('%s-%d.json' % (op, index))
                    for side in ['reference', 'candidate']:
                        probe = output/('probe-%d-%s.json' % (index, side))
                        if not probe.exists():
                            queue.run('probe-%s-%d-%s' % (op,index,side), ' '.join(shlex.quote(x) for x in [
                                queue.py, str(tools/'paired-driver.py'), '--spec', str(spec),
                                '--side', side, '--probe', str(probe)]), 600)
                    paired = output/('paired-v2-%d' % index)
                    if not paired.exists():
                        queue.state['current_job'] = 'paired-%s-%d' % (op,index)
                        module.save(queue.path, queue.state)
                        queue.run('collect-%s-%d' % (op,index), ' '.join(shlex.quote(x) for x in [
                            queue.py, str(tools/'collect-paired.py'), '--spec', str(spec),
                            '--output', str(paired)]), 2400)
            queue.state.update(status='preflight_completed', current_job=None)
            if queue.state.get('error'):
                queue.state.setdefault('historical_errors', []).append(queue.state.pop('error'))
        except Exception as error:
            queue.state.update(status='needs_attention', error=str(error))
            raise
        finally:
            module.save(queue.path, queue.state)


if __name__ == '__main__':
    main()
