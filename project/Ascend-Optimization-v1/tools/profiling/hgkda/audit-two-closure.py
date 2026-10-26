#!/usr/bin/env python3
"""核验本批源码、合同及完整门禁，按相同case真实延迟选版。只读原始证据。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(batch):
    plan = read(batch / 'plan.json')
    checks = []

    def check(condition, message):
        if not condition:
            raise ValueError(message)
        checks.append(message)

    for record in plan['records']:
        for name, expected in record['contract_sha256'].items():
            check(digest(batch / record['catalog_leaf'] / name) == expected,
                  record['operator'] + ':原合同SHA:' + name)
    catalogs = {r['operator']: r for r in plan['records']}

    def result_row(op, result_path, request_path, gate_paths, setting):
        data, request = read(result_path), read(request_path)
        n = 24 if op == 'upsample_bilinear2d' else 33
        check(data['status'] == 'PASSED' and data['num_passed'] == data['num_workloads'] == n
              and not data['is_hack'], str(result_path.relative_to(batch)) + ':完整通过')
        sources = request['implementation']['sources']
        check(len(sources) == 1, str(request_path.relative_to(batch)) + ':单源码')
        source = sources[0]['content']
        sha = hashlib.sha256(source.encode()).hexdigest()
        leaf = batch / catalogs[op]['catalog_leaf']
        if 'binding' in request:
            bundle = request_path.parent / 'operator-bundle.tar'
            response = read(request_path.parent / 'bundle-response.json')
            check(request['binding']['bundle_id'] == response['bundle_id'] == 'sha256:' + digest(bundle),
                  str(request_path.relative_to(batch)) + ':实际绑定归档SHA')
            with tarfile.open(str(bundle)) as archive:
                for name, expected in catalogs[op]['contract_sha256'].items():
                    check(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected,
                          str(bundle.relative_to(batch)) + ':原合同字节:' + name)
        else:
            def subset(old, new):
                return all(k in new and subset(v, new[k]) for k, v in old.items()) if isinstance(old, dict) else old == new
            for key, file in [('correctness_workloads', 'correctness.jsonl'), ('timing_workloads', 'timing.jsonl')]:
                expected = [json.loads(line) for line in (leaf / file).read_text().splitlines() if line.strip()]
                actual = request[key]
                check(len(actual) == len(expected), str(request_path.relative_to(batch)) + ':' + key + ':数量')
                for old, new in zip(expected, actual):
                    check(subset(old, new), op + ':' + old['name'] + ':原始合同字段')
        for gate_path in gate_paths:
            gate = read(gate_path)
            check(gate['status'].lower() == 'passed' and gate['total'] == gate['passed']
                  and gate['source_sha256'] == sha and gate['physical_device'] == 7,
                  str(gate_path.relative_to(batch)) + ':同源码语义')
            if op == 'cumulative_trapezoid':
                timing = gate['timing_input_correctness']
                check(timing['total'] == timing['passed'] == 9, op + ':全部性能输入数值')
                check(all(c['stream_switch_checked'] and c['repetitions'] >= 2 for c in timing['cases']),
                      op + ':性能输入重复及换stream')
        rows = [r for r in data['per_workload'] if r['phase'] == 'timing']
        check(len(rows) == (6 if op == 'upsample_bilinear2d' else 9)
              and all(r['latency_ms'] > 0 and r['reference_latency_ms'] > 0 for r in rows),
              str(result_path.relative_to(batch)) + ':全部实际延迟')
        score = math.exp(sum(math.log(r['reference_latency_ms'] / r['latency_ms']) for r in rows) / len(rows))
        check(math.isclose(score, data['geo_mean'], rel_tol=1e-7), op + ':加速比重算')
        return dict(operator=op, source_sha256=sha, result=str(result_path.relative_to(batch)),
                    request=str(request_path.relative_to(batch)), gates=[str(p.relative_to(batch)) for p in gate_paths],
                    conditions=setting, geo_mean=data['geo_mean'],
                    latency_ms=math.exp(sum(math.log(r['latency_ms']) for r in rows) / len(rows)),
                    per_case=rows, num_passed=n)

    upsample = []
    excluded = []
    def chronological_group(path):
        name = path.relative_to(batch / 'results/upsample_bilinear2d').parts[0]
        order = {'parent': 0, 'codex-initial': 1, 'codex-late': 2, 'attempt-1': 3, 'codex-fused': 4, 'attempt-2': 5}
        return order.get(name, 100), str(path)

    for path in sorted((batch / 'results/upsample_bilinear2d').rglob('*.result.json'), key=chronological_group):
        if path.parent.name != 'formal':
            continue
        leaf = path.parent.parent
        gates = [leaf / name for name in ['remaining-semantics.json', 'special.json']]
        if read(path)['status'] != 'PASSED' or not all(p.exists() and read(p)['status'].lower() == 'passed' for p in gates):
            excluded.append(str(path.relative_to(batch)))
            continue
        upsample.append(result_row('upsample_bilinear2d', path,
                                   path.with_name(path.name.replace('.result.json', '.request.json')), gates, 'default'))

    scan = []
    for folder in sorted((batch / 'research').glob('cann-*-deterministic-complete')):
        if not (folder / 'result.json').exists() or read(folder / 'result.json')['status'] != 'PASSED':
            excluded.append(str(folder.relative_to(batch)))
            continue
        conditions = read(folder / 'conditions.json')
        check(conditions['deterministic'] is True and conditions['physical_device'] == 7,
              str(folder.relative_to(batch)) + ':独立确定性条件')
        for name, expected in catalogs['cumulative_trapezoid']['contract_sha256'].items():
            check(conditions['contract_sha256'][name] == expected, str(folder.relative_to(batch)) + ':合同:' + name)
        row = result_row('cumulative_trapezoid', folder / 'result.json', folder / 'request.json',
                         [folder / 'semantics.json'], 'torch.use_deterministic_algorithms(True)')
        check(row['source_sha256'] == conditions['source_sha256'], str(folder.relative_to(batch)) + ':条件源码SHA')
        scan.append(row)

    for path in sorted((batch / 'results/upsample_bilinear2d').rglob('summary.json')):
        for record in read(path).get('records', []):
            if record.get('status') != 'passed':
                excluded.append(dict(summary=str(path.relative_to(batch)), name=record.get('name'),
                                     source_sha256=record.get('source_sha256'), status=record.get('status')))

    selected = {}
    reviews = read(batch / 'review-exclusions.json')['records'] if (batch / 'review-exclusions.json').exists() else []
    excluded_sha = {r['source_sha256'] for r in reviews if r['status'] == 'exclude_from_final_delivery'}
    for op, rows in [('upsample_bilinear2d', upsample), ('cumulative_trapezoid', scan)]:
        check(bool(rows), op + ':至少一份有效完整结果')
        ids = {r['uuid'] for r in rows[0]['per_case']}
        check(all({r['uuid'] for r in row['per_case']} == ids for row in rows), op + ':同case比较')
        # 同一源码重复运行只取最早的正式记录；不按较高分数或较低延迟挑重复。
        by_sha = {}
        for row in rows:
            if row['source_sha256'] in excluded_sha:
                continue
            by_sha.setdefault(row['source_sha256'], row)
        check(bool(by_sha), op + ':源码机制复核后仍有有效候选')
        selected[op] = min(by_sha.values(), key=lambda r: r['latency_ms'])
    first = read(batch / 'research/launch-candidate-first-deterministic/result.json')
    check(first['source_sha256'] == selected['cumulative_trapezoid']['source_sha256']
          and first['candidate_six_bitwise_equal'] and first['empty_tolerance_intersections'] == 0
          and first['mismatch_vs_first_reference'] == [0] * 6, '最终累计积分:候选先运行及六次稳定匹配')
    for name in ['bf16-small', 'fp16-large']:
        facts = read(batch / ('research/scan-final-deterministic-' + name) / 'facts.json')
        check(facts['source_sha256'] == selected['cumulative_trapezoid']['source_sha256']
              and facts['physical_device'] == 7 and facts['status'] == 'completed',
              '最终累计积分:同源码成对路径:' + name)
        for side in facts['sides'].values():
            trace = Path(side['trace'])
            # 允许在Windows控制端从相同批次相对位置核验。
            marker = '/two-closure-20261024/'
            if not trace.exists() and marker in side['trace']:
                trace = batch / side['trace'].split(marker, 1)[1]
            check(trace.exists() and digest(trace) == side['trace_sha256'],
                  '最终累计积分:原始traceSHA:' + name + ':' + side['trace_sha256'])
    ownership = batch / 'research/disjoint-validation-control/write-ownership.json'
    if ownership.exists():
        proof = read(ownership)
        check(proof['status'] == 'passed' and all(r['unique_row_owner'] for r in proof['cases']),
              '修复上采样:分块输出行归属唯一')
        check(proof['source_sha256'] == selected['upsample_bilinear2d']['source_sha256'],
              '最终上采样:采用通过唯一写入检查的源码')
        facts = read(batch / 'research/upsample-final-reviewed/facts.json')
        check(facts['status'] == 'completed' and facts['source_sha256'] == proof['source_sha256']
              and facts['physical_device'] == 7, '最终上采样:同源码成对路径')
    files = sorted({str(p.relative_to(batch)) for p in batch.rglob('*') if p.is_file()
                    and not p.name.startswith('evidence-audit') and 'raw-evidence' not in p.parts})
    return dict(status='passed', checks=len(checks), checked=checks,
                selection='同case实际延迟；同源码重复只用首个正式结果，不挑参考波动的高分',
                default_cumulative_trapezoid='参考重复运行容差不自洽，默认条件仍无稳定交付',
                selected=selected, upsample_candidates=upsample, deterministic_scan_candidates=scan,
                excluded=excluded, source_review_exclusions=reviews,
                files=[dict(path=name, sha256=digest(batch / name)) for name in files])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), '审计原件不覆盖'
    result = audit(args.batch_dir)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], checks=result['checks'],
                         selected={op: {k: r[k] for k in ['source_sha256', 'geo_mean', 'latency_ms']}
                                   for op, r in result['selected'].items()}), ensure_ascii=False))
