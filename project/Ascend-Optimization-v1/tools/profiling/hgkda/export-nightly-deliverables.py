#!/usr/bin/env python3
"""只从语义及两次完整复验均有效的候选中导出重点算子的交付源码。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
OPERATORS = ['value_selecting_reduction_backward', 'moe_sum_reduce']
ARMS = ['kg-no-profile', 'kg-profile', 'HG-KDA']


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


report = module('delivery_report', TOOLS/'report-nightly.py')
repairs = module('delivery_repairs', TOOLS/'verify-semantic-repairs.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(baseline, repeats):
    """以同case实际候选延迟选择版本，避免PyTorch参考波动改变排名。"""
    expected = set(baseline['timing'])
    if not expected or len(repeats) != 2:
        raise ValueError('需要原版和两次完整复验')
    if any(set(row['timing']) != expected for row in repeats):
        raise ValueError('候选与原版计时case不同')
    if len({r['source_sha256'] for r in repeats}) != 1:
        raise ValueError('两次复验源码不同')
    direct = [statistics.geometric_mean(
        baseline['timing'][key]['latency_ms']/row['timing'][key]['latency_ms']
        for key in expected) for row in repeats]
    return {'relative_flaggems': direct,
        'selection_score': statistics.geometric_mean(direct),
        'relative_pytorch': [statistics.geometric_mean(
            r['reference_latency_ms']/r['latency_ms'] for r in row['timing'].values())
            for row in repeats]}


def choose(candidates):
    eligible = [row for row in candidates if row['status'] == 'verified']
    return max(eligible, key=lambda r: r['selection_score']) if eligible else None


def candidate(batch, record, baseline, arm, repair_batch=None):
    op = record['operator']
    row = {'operator': op, 'mode': arm, 'status': 'pending',
        'additional_budget': repair_batch is not None}
    if repair_batch is None:
        root = batch/'results'/op/arm
        source = root/'export/main.py'
        semantics = root/'supplementary-semantics.json'
        paths = [root/'independent'/('%s-%d.result.json' % (arm, i)) for i in [1, 2]]
        leaf = batch/record['catalog_leaf']
        if not source.exists() or not semantics.exists():
            row['reason'] = '源码或补充语义尚未完成'
            return row
        if not report.valid_semantics(report.read(semantics), op, sha(source), sha(TOOLS/'check-nightly-semantics.py')):
            row.update(status='invalid', reason='补充语义未同身份全部通过')
            return row
    else:
        verified = repairs.verify(repair_batch)
        row.update(repair_batch=repair_batch.name, repair_verification=verified)
        if verified['status'] != 'verified':
            row.update(status=verified['status'], reason=verified.get('reason', '额外修复未完成'))
            return row
        record = report.read(repair_batch/'plan.json')['records'][0]
        source = repair_batch/record['seed']
        leaf = repair_batch/record['catalog_leaf']
        root = repair_batch/'results'/op/'codex-stride-repair'
        paths = [root/'independent'/('codex-stride-repair-%d.result.json' % i) for i in [1, 2]]
    if not all(p.exists() for p in paths):
        row['reason'] = '两次完整复验尚未完成'
        return row
    if not all(report.read(p).get('status') == 'PASSED' for p in paths):
        row.update(status='invalid', reason='正式合同复验失败')
        return row
    loaded = [report.verify(p, record, leaf) for p in paths]
    if any(r['source_sha256'] != sha(source) for r in loaded):
        raise ValueError('交付源码与复验身份不同')
    row.update(scores(baseline, loaded))
    row.update(status='verified', source_sha256=sha(source), source_path=str(source),
        result_paths=[str(p) for p in paths], result_sha256=[r['result_sha256'] for r in loaded],
        workloads=len(loaded[0]['rows']), timing_cases=len(loaded[0]['timing']))
    return row


def immutable(path, data):
    if path.exists() and path.read_bytes() != data:
        raise ValueError('交付源码路径已有不同字节: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def build(batch):
    if report.read(batch/'queue-state.json')['status'] != 'completed':
        raise ValueError('原主队列尚未正常结束')
    records = {r['operator']: r for r in report.read(batch/'plan.json')['records']}
    candidates = []
    selected = {}
    for op in OPERATORS:
        record = records[op]
        baseline = report.verify(batch/'results'/op/'FlagGems/FlagGems-1.result.json', record, batch/record['catalog_leaf'])
        if baseline['source_sha256'] != record['source_sha256']:
            raise ValueError('原版源码与固定plan不同')
        rows = [candidate(batch, record, baseline, arm) for arm in ARMS]
        suffixes = (['value-stride-repair', 'value-profile-repair', 'value-hg-repair', 'value-hg-profile-guided']
            if op == OPERATORS[0] else ['moe-stride-repair', 'moe-hg-layout-repair'])
        for suffix in suffixes:
            extra = batch.parent/(batch.name+'-'+suffix)
            if extra.exists():
                rows.append(candidate(batch, record, baseline, suffix, extra))
        candidates.extend(rows)
        best = choose(rows)
        if best:
            best = dict(best)
            target = batch/'deliverables'/op/best['source_sha256']/'main.py'
            immutable(target, Path(best['source_path']).read_bytes())
            best['delivery_path'] = target.relative_to(batch).as_posix()
            compared = target.with_name('版本对比.md')
            command = [sys.executable, str(report.PROJECT/'tools/evaluation/compare-operator-versions.py'),
                '--version', '固定FlagGems', baseline['path']]
            for i, path in enumerate(best['result_paths'], 1):
                command += ['--version', '有效候选-复验%d' % i, path]
            subprocess.run(command+['--output', str(compared)], check=True, stdout=subprocess.PIPE)
            best['comparison_path'] = compared.relative_to(batch).as_posix()
            selected[op] = best
    resolved = all(row['status'] != 'pending' for row in candidates)
    value = {'batch': batch.name, 'all_candidates_resolved': resolved,
        'all_priority_sources_delivered': len(selected) == len(OPERATORS) and resolved,
        'selection_rule': '同case固定FlagGems延迟/候选延迟，跨两次复验取几何平均；不以跨批次PyTorch参考变化选择版本',
        'scope': '原始组与额外Codex修复分别保留；此交付清单不改变原始四组报告的严格门禁',
        'selected': selected, 'candidates': candidates}
    (batch/'交付清单.json').write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())
    lines = ['# 重点算子交付源码', '',
        ('候选验证已结束，以下为有效最佳源码。' if resolved else '候选仍在验证；以下仅列当前已通过全部检查的源码，最终选择尚未完成。'), '',
        value['selection_rule']+'。', '', value['scope']+'。', '',
        '| 算子 | 选定版本 | 相对PyTorch（两次） | 相对固定FlagGems（两次） | 源码 |',
        '|---|---|---:|---:|---|']
    for op in OPERATORS:
        best = selected.get(op)
        if not best:
            lines.append('| '+op+' | 尚无同时通过补充语义和两次完整复验的候选 | — | — | — |')
            continue
        ratios = [' / '.join('%.3f×' % r for r in best[key]) for key in ['relative_pytorch', 'relative_flaggems']]
        label = best['mode']+('（额外Codex修复）' if best['additional_budget'] else '')
        lines.append('| %s | %s | %s | %s | [main.py](%s) · [逐case](%s) |' %
            (op, label, *ratios, best['delivery_path'], best['comparison_path']))
    lines += ['', '## 各候选有效性', '', '| 算子 | 版本 | 状态 | 原因 |', '|---|---|---|---|']
    for row in candidates:
        lines.append('| %s | %s | %s | %s |' % (row['operator'], row['mode'], row['status'], row.get('reason', '语义和两次正式复验均通过')))
    (batch/'交付清单.md').write_bytes(('\n'.join(lines)+'\n').encode())
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.batch_dir.resolve())
    print(json.dumps({k: result[k] for k in ['batch', 'all_priority_sources_delivered']}, ensure_ascii=False))
    raise SystemExit(int(not result['all_priority_sources_delivered']))
