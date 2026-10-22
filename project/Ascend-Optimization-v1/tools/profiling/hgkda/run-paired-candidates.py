#!/usr/bin/env python3
"""成对诊断后的独立Codex候选验证；先语义，再一次正式KGS评测。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suffix', default='rawptr-v1')
    parser.add_argument('--dense', action='store_true')
    parser.add_argument('--batch', default='paired-pointwise-20261022')
    parser.add_argument('--operators', nargs='+', choices=['log10_', 'selu_'], default=['log10_', 'selu_'])
    args = parser.parse_args()
    if not args.suffix.replace('-', '').isalnum():
        raise ValueError('非法候选后缀')
    if not args.batch.replace('-', '').replace('_', '').isalnum():
        raise ValueError('非法批次名')
    root = Path('/data/hanle/ascend-optimization')
    tools = Path(__file__).resolve().parent
    s = importlib.util.spec_from_file_location('paired_candidate_queue', tools/'run-nightly.py')
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    queue = m.Queue(args.batch, root, 'tle_yy')
    state_path = queue.batch/('candidate-state-v1.json' if args.suffix == 'rawptr-v1'
                             else 'candidate-state-'+args.suffix+'.json')
    if state_path.exists():
        raise ValueError('候选批次已存在，不重复执行；保留结果后新建版本')
    state = dict(status='queued', contribution='HG-KDA成对采集＋Codex源码诊断与候选修改', results={})
    m.save(state_path, state)
    import fcntl
    with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            for op in args.operators:
                queue.idle()
                source = queue.batch/'candidates'/(op+'-'+args.suffix+'.py')
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                reused = None
                for previous in sorted((queue.batch/'results'/op).glob('*')):
                    semantics = previous/'semantics.json'
                    dense_check = previous/'dense-semantics.json'
                    if not semantics.exists():continue
                    sem = json.loads(semantics.read_text())
                    if sem.get('source_sha256') != digest or sem.get('status') != 'passed':continue
                    if args.dense:
                        if not dense_check.exists():continue
                        dense_result = json.loads(dense_check.read_text())
                        if (dense_result.get('source_sha256') != digest or dense_result.get('status') != 'passed'):
                            continue
                    results = list((previous/'formal').glob('*.result.json'))
                    if len(results) != 1:continue
                    result = json.loads(results[0].read_text())
                    request = results[0].with_name(results[0].name.replace('.result.json', '.request.json'))
                    if not request.exists():continue
                    submitted = json.loads(request.read_text())['implementation']['sources']
                    if (result.get('status') != 'PASSED' or result.get('num_passed') != 36
                            or result.get('num_workloads') != 36 or len(submitted) != 1
                            or submitted[0]['content'] != source.read_text()):continue
                    reused = dict(status='reused_existing', source_sha256=digest,
                        existing_arm=previous.name, evidence=str(results[0]), scores=[result['geo_mean']],
                        reason='同批同字节源码已有完整门禁与一次正式评测，不重复计时')
                    break
                if reused:
                    state['results'][op] = reused
                    m.save(state_path, state)
                    continue
                output = queue.batch/'results'/op/('codex-'+args.suffix)
                output.mkdir(parents=True, exist_ok=True)
                state.update(status='running', current_operator=op)
                m.save(state_path, state)
                command = [queue.py, str(tools/'check-pointwise-semantics.py'),
                    '--operator', op, '--source', str(source), '--output', str(output/'semantics.json')]
                try:
                    queue.run('candidate-'+args.suffix+'-semantics-'+op, ' '.join(shlex.quote(x) for x in command), 1800)
                except RuntimeError:
                    if not (output/'semantics.json').exists():
                        raise
                semantic = json.loads((output/'semantics.json').read_text())
                if semantic['status'] != 'passed':
                    state['results'][op] = dict(status='semantic_failed', semantic=semantic)
                    m.save(state_path, state)
                    continue
                if args.dense:
                    command = [queue.py, str(tools/'check-dense-pointwise.py'),
                        '--operator', op, '--source', str(source), '--output', str(output/'dense-semantics.json')]
                    try:
                        queue.run('candidate-'+args.suffix+'-dense-'+op,
                            ' '.join(shlex.quote(x) for x in command), 1800)
                    except RuntimeError:
                        if not (output/'dense-semantics.json').exists():raise
                    dense = json.loads((output/'dense-semantics.json').read_text())
                    if dense['status'] != 'passed':
                        state['results'][op]=dict(status='dense_semantic_failed',source=str(source),
                                                  passed=dense['passed'],total=dense['total'])
                        m.save(state_path,state)
                        continue
                result = queue.evaluate(op, source, output/'formal', 'codex-'+args.suffix, 1)
                state['results'][op] = result
                m.save(state_path, state)
            state.update(status='completed', current_operator=None)
        except Exception as error:
            state.update(status='needs_attention', error=str(error))
            raise
        finally:
            m.save(state_path, state)


if __name__ == '__main__':
    main()
