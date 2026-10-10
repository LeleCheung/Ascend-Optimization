#!/usr/bin/env python3
"""对完整通过候选的指定 case 采集真机指标，并生成中文体检报告。"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operator')
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--case-id', action='append', required=True)
    parser.add_argument('--kernel-prefix', required=True)
    parser.add_argument('--server', default='http://127.0.0.1:19655')
    args = parser.parse_args()
    result = json.loads(args.evaluation.read_text(encoding='utf-8'))
    assert result.get('status') == 'PASSED' and not result.get('is_hack')
    assert result['num_passed'] == result['num_workloads']
    valid = {r['uuid'] for r in result['per_workload'] if r.get('phase') == 'timing' and r.get('status') == 'PASSED'}
    assert len(set(args.case_id)) == len(args.case_id) and set(args.case_id) <= valid
    request = args.evaluation.with_name(args.evaluation.name.replace('.result.json', '.request.json'))
    label = args.evaluation.name.removesuffix('.result.json').rsplit('-', 1)[0]
    source = args.evaluation.with_name(label + '.py')
    provenance = args.evaluation.with_name(label + '.provenance.json')
    inspect = args.evaluation.parent / 'inspect.json'
    assert all(p.is_file() for p in (request, source, provenance, inspect))
    args.output.mkdir(parents=True, exist_ok=False)
    tools = Path(__file__).resolve().parent
    analyze = [sys.executable, str(tools.parent / 'profiling/ascend_profiling_workflow.py'),
               args.operator, str(source), str(args.evaluation), str(request), str(inspect),
               str(args.output / '体检报告.md'), '--timing-scope', 'device_kernel',
               '--evaluation-provenance', str(provenance), '--kernel-prefix', args.kernel_prefix]
    for index, case in enumerate(args.case_id):
        output = args.output / ('case-' + str(index))
        subprocess.run([sys.executable, str(tools / 'profile-operator.py'),
                        '--evaluation', str(args.evaluation), '--request', str(request),
                        '--inspect', str(inspect), '--case-id', case, '--output', str(output),
                        '--server', args.server, '--level', 'metrics',
                        '--aic-metrics', 'PipeUtilization', '--timeout', '900'], check=True)
        analyze += ['--profile', str(output / 'response.json'), str(output / 'request.json'), str(output / 'artifacts')]
    subprocess.run(analyze, check=True)
    print('OPTIMIZED_PROFILE_REPORT_READY', args.operator, flush=True)


if __name__ == '__main__':
    main()
