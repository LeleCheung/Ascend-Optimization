"""复算两份已有正式记录的候选／参考耗时变化；不执行任何设备或性能测试。"""
import hashlib
import json
import math
from pathlib import Path

BATCH = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geo(values):
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    return math.exp(sum(math.log(v) for v in values)/len(values))


def main():
    output = Path(__file__).with_name('formal-improvement-v1.json')
    assert not output.exists(), '已有派生事实记录不覆盖'
    prior = next(row for row in read(BATCH/'starting-delivery.json')['rows'] if row['operator']=='expm1_')
    best = read(BATCH/'search-state.json')['operators']['expm1_']['best']
    assert best['source_sha256']=='8ec3ef796d4bdbe7c3ef8d246c0ebe259a1563926253145fde82c12d9110a9e3'
    old_path = Path(prior['pytorch']['path'])
    new_path = Path(best['search_result'])
    old = read(old_path)
    new = read(new_path)
    for value in [old, new]:
        assert value['status']=='PASSED' and value['num_passed']==value['num_workloads']==30
        assert not value.get('is_hack')
    assert sha(prior['source'])==prior['source_sha256']
    assert sha(best['source'])==best['source_sha256']==sha(new_path.with_name('main.py'))
    work = new_path.parents[3]
    contract = work/'.kernelgen/evaluation-contract.json'
    binding = read(contract)
    leaf = BATCH/'catalog/ops/expm1_'
    assert binding['definition_name']=='expm1_'
    assert binding['definition']['reference']==(leaf/'oracle.py').read_text(encoding='utf8')
    record = next(row for row in read(BATCH/'plan.json')['records'] if row['operator']=='expm1_')
    assert all(sha(leaf/name)==digest for name, digest in record['contract_sha256'].items())
    cases = {row['name']: row for row in map(json.loads, (leaf/'timing.jsonl').read_text().splitlines())}
    a = {row['uuid']: row for row in old['per_workload'] if row['phase']=='timing'}
    b = {row['uuid']: row for row in new['per_workload'] if row['phase']=='timing'}
    assert a.keys()==b.keys()==cases.keys() and len(a)==12
    rows = []
    for name in a:
        rows.append(dict(case=name, dtype=cases[name]['inputs']['spec']['dtype'],
            shape=cases[name]['inputs']['spec']['shape'],
            old_candidate_ms=a[name]['latency_ms'], new_candidate_ms=b[name]['latency_ms'],
            old_reference_ms=a[name]['reference_latency_ms'], new_reference_ms=b[name]['reference_latency_ms'],
            candidate_actual_speedup=a[name]['latency_ms']/b[name]['latency_ms'],
            reference_time_ratio=b[name]['reference_latency_ms']/a[name]['reference_latency_ms']))
    actual = geo([row['candidate_actual_speedup'] for row in rows])
    drift = geo([row['reference_time_ratio'] for row in rows])
    old_score = geo([row['reference_latency_ms']/row['latency_ms'] for row in a.values()])
    new_score = geo([row['reference_latency_ms']/row['latency_ms'] for row in b.values()])
    assert math.isclose(old_score, old['geo_mean'], rel_tol=1.e-8)
    assert math.isclose(new_score, new['geo_mean'], rel_tol=1.e-8)
    assert math.isclose(actual*drift, new_score/old_score, rel_tol=1.e-8)
    result = dict(status='facts_verified', checker_sha256=sha(__file__),
        old_source_sha256=prior['source_sha256'], new_source_sha256=best['source_sha256'],
        old_result=str(old_path), old_result_sha256=sha(old_path),
        new_result=str(new_path), new_result_sha256=sha(new_path),
        current_contract=str(contract), current_contract_sha256=sha(contract),
        old_score=old_score, new_score=new_score, candidate_actual_speedup=actual,
        reference_time_ratio=drift, score_ratio=new_score/old_score, cases=rows,
        performance_remeasured=False,
        scope='两份已存在的KGS完整调用逐case事实；旧参考与候选分别冻结，不以参考变慢解释实现收益。数值资格和后端交付另查完整补测，不由这份性能分解证明。')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({key:result[key] for key in ['status','old_score','new_score','candidate_actual_speedup','reference_time_ratio','score_ratio']}))


if __name__ == '__main__':
    main()
