"""逐用例比较冻结候选的历史与新回归耗时，区分候选变化和参考变化。"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):return json.loads(Path(path).read_bytes())


def geometric_mean(values):
    assert values and all(math.isfinite(v) and v>0 for v in values)
    return math.exp(sum(math.log(v) for v in values)/len(values))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository',type=Path,required=True)
    parser.add_argument('--run-directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and not args.output.with_suffix('.csv').exists()
    index=read(args.repository/'development/experiments/index.json')
    original={row['operator']:row for row in index['rows']}
    plan=read(args.run_directory/'plan.json')
    assert hashlib.sha256((args.repository/'development/experiments/index.json').read_bytes()).hexdigest()==plan['index_sha256']
    rows=[];cases=[]
    for job in plan['jobs']:
        path=args.run_directory/'operators'/job['operator']/'evaluation/result.json'
        if not path.exists():continue
        baseline=original[job['operator']]
        assert job['source_sha256']==baseline['source']['sha256'],'只能比较同源码的回归'
        previous=read(args.repository/baseline['evidence']['cann']['artifact']['path'])
        current=read(path)
        old={v['uuid']:v for v in previous['per_workload'] if v['phase']=='timing'}
        new={v['uuid']:v for v in current['per_workload'] if v['phase']=='timing'}
        assert set(old)==set(new),'性能用例身份变化'
        changes=[]
        for name,after in new.items():
            before=old[name]
            if before['status']!='PASSED' or after['status']!='PASSED':continue
            candidate=after['latency_ms']/before['latency_ms']
            reference=after['reference_latency_ms']/before['reference_latency_ms']
            value=dict(operator=job['operator'],uuid=name,
                       old_candidate_ms=before['latency_ms'],new_candidate_ms=after['latency_ms'],
                       old_cann_ms=before['reference_latency_ms'],new_cann_ms=after['reference_latency_ms'],
                       candidate_new_over_old=candidate,cann_new_over_old=reference,
                       old_speedup=before['speedup'],new_speedup=after['speedup'])
            cases.append(value);changes.append(value)
        if len(changes)==len(old):
            candidate=geometric_mean([v['candidate_new_over_old'] for v in changes])
            reference=geometric_mean([v['cann_new_over_old'] for v in changes])
            assert math.isclose(current['geo_mean']/previous['geo_mean'],reference/candidate,rel_tol=1e-9)
            rows.append(dict(operator=job['operator'],source_sha256=job['source_sha256'],
                             old_artifact_score=previous['geo_mean'],old_index_score=baseline['evidence']['cann']['score'],
                             historical_measurement=baseline['evidence']['cann'].get('measurement'),new_score=current['geo_mean'],
                             candidate_new_over_old=candidate,cann_new_over_old=reference,timing_cases=len(changes)))
    result=dict(rows=rows,cases=cases,
                scope='相同冻结源码和性能用例；新旧耗时分别比较，不以加速比分数之比推断候选提速或确定原因')
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    if cases:
        with args.output.with_suffix('.csv').open('x',newline='',encoding='utf-8-sig') as output:
            writer=csv.DictWriter(output,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    print(json.dumps(dict(operators=len(rows),timing_cases=len(cases))))


if __name__=='__main__':main()
