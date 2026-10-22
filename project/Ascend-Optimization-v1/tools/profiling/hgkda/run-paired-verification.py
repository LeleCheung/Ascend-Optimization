#!/usr/bin/env python3
"""KG结束后核验实际交付、指针机制、编译IR及优化后成对计数。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex


def quote(command):
    return ' '.join(shlex.quote(str(x)) for x in command)


def main():
    tools = Path(__file__).resolve().parent
    root = Path('/data/hanle/ascend-optimization')
    s = importlib.util.spec_from_file_location('paired_verify_queue', tools/'run-nightly.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    queue = m.Queue('paired-pointwise-20261022', root, 'tle_yy')
    state_path = queue.batch/'verification-state.json'
    if state_path.exists():
        raise ValueError('验证已有状态，不覆盖旧结果')
    state = dict(status='queued')
    m.save(state_path, state)
    import fcntl
    with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            kg_state = json.loads((queue.batch/'kg-paired-state.json').read_text())
            if kg_state['status'] != 'completed':
                raise RuntimeError('KG队列尚未正常结束，先检查其证据')
            queue.idle()
            state['status'] = 'running'
            m.save(state_path, state)
            delivery = queue.batch/'deliverables'
            queue.run('export-paired-deliverables', quote([queue.py, tools/'export-paired-deliverables.py',
                '--batch-dir', queue.batch, '--output', delivery]), 600)
            records = json.loads((delivery/'manifest.json').read_text())['records']
            research = queue.batch/'research'
            research.mkdir(exist_ok=True)
            queue.run('verify-pointer-mechanism', quote([queue.py, tools/'probe-pointer-launch.py',
                '--source', queue.batch/'candidates/log10_-rawptr-v1.py',
                '--output', research/'same-kernel-pointer-probe.json']), 600)
            for record in records:
                op = record['operator']
                source = Path(record['delivered'])
                for index in [0,3]:
                    original = json.loads((queue.batch/'paired-specs'/('%s-%d.json' % (op,index))).read_text())
                    current = dict(original, source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                    spec = research/('%s-final-%d-spec.json' % (op,index))
                    m.save(spec, current)
                    if index == 3:
                        for label, spec_path in [('seed', queue.batch/'paired-specs'/('%s-%d.json' % (op,index))),
                                                 ('final', spec)]:
                            queue.run('dump-ir-'+op+'-'+label, quote([queue.py, tools/'paired-driver.py',
                                '--spec', spec_path, '--side', 'candidate',
                                '--dump-ir', research/(op+'-'+label+'-ir')]), 900)
                    output = research/('%s-final-paired-%d' % (op,index))
                    state['current_job'] = output.name
                    m.save(state_path, state)
                    queue.run('verify-paired-'+op+'-'+str(index), quote([queue.py, tools/'collect-paired.py',
                        '--spec', spec, '--output', output]), 2400)
                    queue.run('compare-final-'+op+'-'+str(index), quote([queue.py,
                        tools/'skills/hgkda/scripts/compare_opprof.py', '--directory', output,
                        '--output', output/'comparison.json']), 600)
            state.update(status='completed', current_job=None)
        except Exception as error:
            state.update(status='needs_attention', error=str(error))
            raise
        finally:
            m.save(state_path, state)


if __name__ == '__main__':
    main()
