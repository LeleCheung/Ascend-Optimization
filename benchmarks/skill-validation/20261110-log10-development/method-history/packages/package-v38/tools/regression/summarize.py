"""汇总独立回归原件；不触发设备评测，不覆盖历史验收索引。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from hgkda.validation import supplement_status


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    outputs = [args.output, args.output.with_suffix('.md'), args.output.with_suffix('.csv')]
    for target in outputs:
        if target.exists():
            raise FileExistsError(target)
    plan = read(run/'plan.json')
    state = read(run/'state.json')
    rows = []
    per_case = []
    integrity_errors = []
    for job in plan['jobs']:
        current = state['operators'].get(job['operator'],{})
        evaluation = current.get('evaluation',{})
        extras = {key:current.get(key) for key in ['supplementary', 'cann_supplementary']}
        for key in ['evaluation', 'supplementary', 'cann_supplementary']:
            record = current.get(key)
            if record:
                path = Path(record['path'])
                if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                    integrity_errors.append(dict(operator=job['operator'], artifact=key,
                                                 error='原件 SHA 与状态记录不一致'))
        result_path = run/'operators'/job['operator']/'evaluation/result.json'
        if result_path.exists():
            result = read(result_path)
            if not evaluation:
                evaluation = {key:result.get(key) for key in
                              ['status','geo_mean','num_passed','num_workloads','is_hack','server_backend']}
            for value in result.get('per_workload', []):
                per_case.append(dict(operator=job['operator'], uuid=value['uuid'],
                         phase=value.get('phase'), status=value.get('status'),
                         candidate_ms=value.get('latency_ms'),
                         cann_ms=value.get('reference_latency_ms'),
                         speedup=value.get('speedup'),
                         abs_err=value.get('abs_err'), rel_err=value.get('rel_err')))
        for key in extras:
            path = run/'operators'/job['operator']/(key+'.json')
            if extras[key] is None and path.exists():
                value = read(path)
                extras[key] = dict(status=supplement_status(job['operator'],value,job['source_sha256']),
                                   passed=value['passed'], total=value['total'], path=str(path),
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        reasons=[]
        if current.get('terminal') and not current.get('accepted'):
            if current.get('error'):
                reasons.append(current['error'])
            elif evaluation:
                if evaluation.get('status')!='PASSED' or evaluation.get('num_passed')!=evaluation.get('num_workloads'):
                    reasons.append('完整评测未通过')
                if evaluation.get('is_hack') is not False:
                    reasons.append('完整评测审查未通过')
                for key in extras:
                    if not extras[key] or extras[key]['status']!='passed':
                        reasons.append(('PyTorch' if key=='supplementary' else 'CANN')+'语义补测未通过')
                if evaluation.get('geo_mean') is not None and evaluation['geo_mean']<=.8:
                    reasons.append('相对 CANN 未超过 0.8×')
            if not reasons:
                reasons.append('验收证据不完整')
        rows.append(dict(operator=job['operator'],group=job['group'],order=job['order'],
             source_sha256=job['source_sha256'],previous_cann_score=job['previous_cann_score'],
             cann_score=evaluation.get('geo_mean'),evaluation=evaluation,
             supplementary=extras['supplementary'],
             cann_supplementary=extras['cann_supplementary'],
             accepted=current.get('accepted',False),status=current.get('status','pending'),
             acceptance_reasons=reasons,
             terminal=current.get('terminal',False),
             error=current.get('error'),reference_kind=job['reference_kind']))
    evidence = []
    for path in sorted(run.rglob('*')):
        if not path.is_file() or any(part.endswith('-cache') for part in path.relative_to(run).parts):
            continue
        if path.suffix in {'.json','.jsonl','.py','.log','.sh','.tar'}:
            evidence.append(dict(path=path.relative_to(run).as_posix(),
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
    result = dict(schema_version='hgkda.regression-summary/v1',run_directory=str(run),
            status=state['status'],expected=plan['expected'],completed=sum(r['terminal'] for r in rows),
            accepted=sum(r['accepted'] for r in rows),rows=rows,per_case=per_case,evidence=evidence,
            integrity_errors=integrity_errors,
            scope='单次新入口回归结果；历史列是拆分前已记录结果，不以两个加速比分数之比宣称实现提速')
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    with args.output.with_suffix('.csv').open('x', newline='', encoding='utf-8-sig') as output:
        columns = ['operator','uuid','phase','status','candidate_ms','cann_ms','speedup','abs_err','rel_err']
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(per_case)
    table = ['# 独立 HG-KDA 固定候选回归', '',
             f"状态：{result['status']}；已处理 {result['completed']}/{result['expected']}；本轮达标 {result['accepted']} 项。", '',
             '候选源码、原输入、容差和计时参数固定；本轮每项一次完整 KGS 正式评测，原补测独立执行。', '',
             '加速比为完整调用的几何平均；CANN 参考类型含原生单接口及已审核的同语义组合，逐项保存在 JSON 的 reference_kind 字段中。', '',
             '| 算子 | 拆分前相对 CANN | 本轮相对 CANN | 完整评测 | PyTorch 语义补测 | CANN 语义补测 | 本轮状态 |',
             '| --- | ---: | ---: | --- | --- | --- | --- |']
    def extra(record):
        return '—' if not record else str(record['passed'])+'/'+str(record['total'])+' '+record['status']
    for row in rows:
        value = row['evaluation']
        score = '—' if row['cann_score'] is None else format(row['cann_score'],'.3f')+'×'
        full = '—' if not value else str(value.get('num_passed'))+'/'+str(value.get('num_workloads'))
        status = '达标' if row['accepted'] else '；'.join(row['acceptance_reasons']) or row['status']
        previous = '—' if row['previous_cann_score'] is None else format(row['previous_cann_score'],'.3f')+'×'
        table.append('| '+ ' | '.join(['`'+row['operator']+'`',previous,
                     score,full,extra(row['supplementary']),extra(row['cann_supplementary']),
                     status.replace('|','／').replace('\n',' ')])+' |')
    args.output.with_suffix('.md').write_text('\n'.join(table)+'\n',encoding='utf8')
    print(json.dumps({k:result[k] for k in ['status','expected','completed','accepted']}))
    if integrity_errors:
        raise RuntimeError('原件身份核验失败，详见 integrity_errors')


if __name__ == '__main__':
    main()
