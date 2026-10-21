#!/usr/bin/env python3
"""核验额外SELU候选的父源码、冻结合同、36项语义和两次完整walltime复验。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

TOOLS = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load_module('selu_report', TOOLS/'report-nightly.py')
runner = load_module('selu_runner', TOOLS/'run-selu-repair.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(batch):
    plan = report.read(batch/'plan.json')
    if plan['execution_mode'] != 'selu_repair' or len(plan['records']) != 1:
        raise ValueError('不是单列SELU修复批次')
    parent = batch.parent/plan['parent_batch']
    parent.resolve().relative_to(batch.parent.resolve())
    record = plan['records'][0]
    if record['operator'] != 'selu_' or plan['parent_arm'] != 'HG-KDA' or plan['parent_round'] != 2:
        raise ValueError('父算子或模式不同')
    original = next(r for r in report.read(parent/'plan.json')['records'] if r['operator'] == 'selu_')
    if record['contract_sha256'] != original['contract_sha256'] or record['upstream_commit'] != original['upstream_commit']:
        raise ValueError('修改了固定版本或正式合同')
    source = batch/record['seed']
    source.resolve().relative_to(batch.resolve())
    prefix = batch/'parent/round-0002.py'
    if sha(source) != record['source_sha256'] or sha(prefix) != plan['parent_source_sha256']:
        raise ValueError('候选或父源码字节不符')
    if not source.read_bytes().startswith(prefix.read_bytes()+b'\n\n'):
        raise ValueError('候选未保留父源码前缀')
    if hashlib.sha256(source.read_bytes()[prefix.stat().st_size+2:]).hexdigest() != plan['adapter_sha256']:
        raise ValueError('追加实现与冻结plan不同')
    archive = parent/'results/selu_/HG-KDA/native-evidence'
    matches = [e for e in report.read(archive/'archive-manifest.json')['files']
        if e['source_path'] == '.kernelgen/evals/round-0002/main.py' and e['sha256'] == sha(prefix)]
    if len(matches) != 1:
        raise ValueError('父源码未唯一对应原始归档')
    archived_parent = archive/matches[0]['archived_path']
    archived_parent.resolve().relative_to(archive.resolve())
    if sha(archived_parent) != sha(prefix):
        raise ValueError('父源码归档字节不同')
    leaf = batch/record['catalog_leaf']
    leaf.resolve().relative_to(batch.resolve())
    for name, digest in record['contract_sha256'].items():
        path = leaf/name
        path.resolve().relative_to(leaf.resolve())
        if sha(path) != digest:
            raise ValueError('冻结合同已变化：'+name)
    state_path = batch/'queue-state.json'
    state = report.read(state_path) if state_path.exists() else {}
    if state.get('plan_sha256') and state['plan_sha256'] != sha(batch/'plan.json'):
        raise ValueError('运行时plan不同')
    value = {'batch': batch.name, 'operator': 'selu_', 'status': 'pending',
        'source_sha256': sha(source), 'parent_source_sha256': sha(prefix),
        'same_budget_generated_result': False, 'additional_budget': plan['additional_budget'],
        'scope': '额外Codex落实HG分析与探针的候选；不替换原始两轮实验'}
    output = batch/'results/selu_/codex-launch-repair'
    semantic_path = output/'supplementary-semantics.json'
    if not semantic_path.exists():
        return value
    semantics = report.read(semantic_path)
    value['supplementary_semantics'] = {k:semantics.get(k) for k in ['status','passed','total','source_sha256','checker_sha256']}
    if (sha(TOOLS/'check-selu-repair.py') != plan['checker_sha256'] or
            not runner.valid_semantics(semantics, sha(source), plan['checker_sha256'])):
        value.update(status='invalid', reason='36项同身份补充语义未全部重复通过')
        return value
    value['versions'] = []
    for i in [1, 2]:
        path = output/'independent'/('codex-launch-repair-%d.result.json'%i)
        if not path.exists():
            return value
        if report.read(path).get('status') != 'PASSED':
            value.update(status='invalid', reason='完整KGS复验失败')
            return value
        checked = report.verify(path, record, leaf)
        if checked['source_sha256'] != sha(source) or checked['timing_scope'] != 'walltime':
            raise ValueError('复验候选或计时口径不同')
        value['versions'].append({'repeat': i, 'workloads': len(checked['rows']),
            'relative_pytorch': statistics.geometric_mean(r['reference_latency_ms']/r['latency_ms'] for r in checked['timing'].values()),
            'result_sha256': checked['result_sha256']})
    baseline = report.verify(parent/'results/selu_/FlagGems/FlagGems-1.result.json', original, parent/original['catalog_leaf'])
    for i, row in enumerate(value['versions'], 1):
        checked = report.verify(output/'independent'/('codex-launch-repair-%d.result.json'%i), record, leaf)
        if set(checked['timing']) != set(baseline['timing']):
            raise ValueError('与原版性能case不相同')
        row['relative_flaggems'] = statistics.geometric_mean(
            baseline['timing'][name]['latency_ms']/result['latency_ms'] for name, result in checked['timing'].items())
    value['status'] = 'verified'
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.batch_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())
    print(json.dumps(value, ensure_ascii=False))
    raise SystemExit(int(value['status'] != 'verified'))
