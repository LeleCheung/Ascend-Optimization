#!/usr/bin/env python3
"""等待同一卡锁，单列SELU额外候选的36项语义与两次完整KGS复验。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_semantics(data, source_sha, checker_sha):
    cases = data.get('cases', [])
    return bool(data.get('operator') == 'selu_' and data.get('status') == 'passed'
        and data.get('source_sha256') == source_sha and data.get('checker_sha256') == checker_sha
        and data.get('physical_device') == 7 and data.get('logical_device') == 'npu:0'
        and data.get('passed') == data.get('total') == len(cases) == 36
        and {(r.get('dtype'), r.get('case')) for r in cases} == {
            (dtype, case) for dtype in ['torch.float16', 'torch.bfloat16', 'torch.float32'] for case in range(12)}
        and all(r.get('status') == 'passed' and r.get('repetitions') == 2 for r in cases))


def execute(root, batch, container):
    tools = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('selu_queue', tools/'run-nightly.py')
    queues = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queues)
    queue = queues.Queue(batch, root, container)
    directory = queue.batch
    if queue.plan['execution_mode'] != 'selu_repair':
        raise ValueError('只允许额外SELU批次')
    if queue.state.get('status') in ['completed', 'needs_attention']:
        print('已有终态；不重跑覆盖。检查原始证据后另建批次。', flush=True)
        return
    queue.state.update(status='queued', pid=__import__('os').getpid())
    queues.save(queue.path, queue.state)
    print('等待本项目物理卡7卡锁', flush=True)
    import fcntl
    with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            parent = directory.parent/queue.plan['parent_batch']
            if json.loads((parent/'queue-state.json').read_text())['status'] != 'completed':
                raise RuntimeError('原始主队列尚未正常结束')
            original = next(r for r in json.loads((parent/'plan.json').read_text())['records'] if r['operator'] == 'selu_')
            record = queue.records['selu_']
            source = directory/record['seed']
            if record['contract_sha256'] != original['contract_sha256']:
                raise RuntimeError('不能改变原始合同')
            prefix = directory/'parent/round-0002.py'
            if sha(prefix) != queue.plan['parent_source_sha256'] or not source.read_bytes().startswith(prefix.read_bytes()):
                raise RuntimeError('父源码谱系不一致')
            archive = parent/'results/selu_/HG-KDA/native-evidence'
            entries = json.loads((archive/'archive-manifest.json').read_text())['files']
            matches = [e for e in entries if e['source_path'] == '.kernelgen/evals/round-0002/main.py'
                       and e['sha256'] == sha(prefix)]
            if len(matches) != 1 or sha(archive/matches[0]['archived_path']) != sha(prefix):
                raise RuntimeError('父源码未对应原始组唯一归档')
            checker = tools/'check-selu-repair.py'
            if sha(checker) != queue.plan['checker_sha256']:
                raise RuntimeError('检查器与冻结plan不同')
            output = directory/'results/selu_/codex-launch-repair'
            output.mkdir(parents=True, exist_ok=True)
            semantic_path = output/'supplementary-semantics.json'
            queue.state.update(status='running', current_job='supplementary_semantics',
                source_sha256=record['source_sha256'])
            queues.save(queue.path, queue.state)
            if not semantic_path.exists():
                queue.idle()
                import shlex
                command = [queue.py, str(checker), '--source', str(source),
                    '--expected-sha256', record['source_sha256'], '--output', str(semantic_path)]
                queue.run('selu-supplementary', ' '.join(shlex.quote(x) for x in command), budget=1800)
            semantics = json.loads(semantic_path.read_text())
            if not valid_semantics(semantics, record['source_sha256'], queue.plan['checker_sha256']):
                raise RuntimeError('36项同身份额外语义未通过；不提交性能评测')
            queue.state['current_job'] = 'independent_repeats'
            queues.save(queue.path, queue.state)
            result = queue.evaluate('selu_', source, output/'independent', 'codex-launch-repair', 2)
            queue.state.update(status='completed', current_job=None, result=result,
                scope='额外Codex修复；不改变原始HG-KDA两轮预算和未达标结论')
            queues.save(queue.path, queue.state)
        except Exception as error:
            queue.state.update(status='needs_attention', error=type(error).__name__+': '+str(error))
            queues.save(queue.path, queue.state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/hanle/ascend-optimization'))
    parser.add_argument('--batch', required=True)
    parser.add_argument('--container', default='tle_yy')
    args = parser.parse_args()
    if not args.batch.replace('-', '').replace('_', '').isalnum():
        raise ValueError('非法批次')
    execute(args.root, args.batch, args.container)
