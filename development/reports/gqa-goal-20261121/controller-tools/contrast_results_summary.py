"""Summarize validated results and actual model usage without hiding failed calls."""
import argparse
import csv
import json
from pathlib import Path


def eligible(row):
    formal = row.get('formal', {})
    checks = row.get('supplementary', {})
    return (formal.get('status') == 'PASSED' and formal.get('num_passed') == 24
            and row.get('source_review', {}).get('status') == 'reviewed_strict_triton'
            and row.get('source_review', {}).get('source_sha256') == row.get('output_sha256')
            and row.get('review_feedback', {}).get('status') != 'repair_required'
            and all(checks.get(k, {}).get('status') == 'passed'
                    and checks[k].get('source_sha256') == row.get('output_sha256')
                    for k in ['supplementary.json', 'cann_supplementary.json']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-root', type=Path)
    args = parser.parse_args()
    data = json.loads(args.inventory.read_bytes())
    result = {'scope': 'One task, one optimization trajectory per arm; not a multi-seed causal estimate',
              'timing': 'KGS profiler, NPU task durations with clear_l2_cache=True; not CPU walltime',
              'arms': {}, 'cost_scope': 'Host-reported model estimates; excludes Codex, hardware and capture costs; not a billing invoice'}
    for arm, rows in data['arms'].items():
        completed = [r for r in rows if r.get('model_result')]
        valid = [r for r in rows if eligible(r)]
        best = max(valid, key=lambda r: r['formal']['geo_mean']) if valid else None
        usage = {key: sum((r['model_result'].get('usage') or {}).get(key, 0) or 0 for r in completed)
                 for key in ['input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens', 'output_tokens']}
        result['arms'][arm] = dict(
            started_sessions=len(rows), completed_sessions=len(completed), validated_candidates=len(valid),
            model_turns=sum(r['model_result'].get('num_turns') or 0 for r in completed),
            model_duration_ms=sum(r['model_result'].get('duration_ms') or 0 for r in completed),
            estimated_usd=sum(r['model_result'].get('total_cost_usd') or 0 for r in completed),
            usage=usage,
            best=dict(round=best['round'], source_sha256=best['output_sha256'],
                      geo_mean=best['formal']['geo_mean'], timings=best['timings']) if best else None)
    control = result['arms']['no-simulator']['best']
    treatment = result['arms']['with-simulator']['best']
    if control and treatment:
        before = {r['uuid']: r for r in control['timings']}
        result['comparison'] = {
            'selection': 'Each arm best validated six-case geometric mean',
            'same_budget_complete': all(result['arms'][a]['completed_sessions'] == 8
                                        for a in ['no-simulator', 'with-simulator']),
            'cases': [dict(case=r['uuid'], no_simulator_ms=before[r['uuid']]['latency_ms'],
                           with_simulator_ms=r['latency_ms'],
                           candidate_time_ratio=before[r['uuid']]['latency_ms']/r['latency_ms'],
                           no_simulator_cann_speedup=before[r['uuid']]['speedup'],
                           with_simulator_cann_speedup=r['speedup'],
                           reference_time_ratio=before[r['uuid']]['reference_latency_ms']/r['reference_latency_ms'])
                      for r in treatment['timings']]}
    if args.replay_root:
        result['final_replay']={}
        for arm in ['no-simulator','with-simulator']:
            folder=args.replay_root/arm
            selection=json.loads((folder/'selection.json').read_bytes())
            sha=selection['source_sha256']
            assert sha==result['arms'][arm]['best']['source_sha256'],'replay differs from selected best'
            formal=folder/('formal-'+sha[:12])
            replay=json.loads((formal/'evaluation/result.json').read_bytes())
            checks={}
            for name in ['supplementary.json','cann_supplementary.json']:
                check=json.loads((formal/name).read_bytes())
                assert check['source_sha256']==sha
                checks[name]={k:check[k] for k in ['status','passed','total']}
            result['final_replay'][arm]=dict(source_sha256=sha,status=replay['status'],
                num_passed=replay['num_passed'],num_workloads=replay['num_workloads'],geo_mean=replay.get('geo_mean'),
                checks=checks,timings=[{k:x.get(k) for k in ['uuid','status','latency_ms','reference_latency_ms','speedup']}
                                     for x in replay['per_workload'] if x['phase']=='timing'])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    with args.output.with_name('trace-rounds.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=['arm','round','input_sha256','output_sha256','eligible','formal_status','formal_passed','preflight_status',
                'six_case_geo_mean','target_latency_ms','target_cann_speedup','model_turns','model_usd','output_tokens','simulator_reads']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for arm in ['no-simulator','with-simulator']:
            for row in data['arms'][arm]:
                valid=eligible(row);f=row.get('formal',{});m=row.get('model_result',{})
                t=next((x for x in row.get('timings',[]) if x['uuid']=='upsample_bilinear2d-timing-1'),{})
                writer.writerow(dict(arm=arm,round=row['round'],input_sha256=row['input_sha256'],output_sha256=row.get('output_sha256'),
                    eligible=valid,formal_status=f.get('status'),formal_passed=f.get('num_passed'),preflight_status=row.get('preflight',{}).get('status'),
                    six_case_geo_mean=f.get('geo_mean') if valid else None,target_latency_ms=t.get('latency_ms') if valid else None,
                    target_cann_speedup=t.get('speedup') if valid else None,model_turns=m.get('num_turns'),model_usd=m.get('total_cost_usd'),
                    output_tokens=(m.get('usage') or {}).get('output_tokens'),simulator_reads=len(row.get('simulator_reads',[]))))
    print(json.dumps({a: {k: v for k, v in x.items() if k not in ['best', 'usage']}
                      for a, x in result['arms'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
