"""Extract measured instruction and profiling facts from preserved IR replays."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def replay(observation, name):
    directory = observation / 'simulator'
    summary = read(directory / 'ir-summary.json')
    listing = read(directory / 'listing.json')
    pair_path = observation / 'paired/comparison.json'
    pair = read(pair_path)
    candidate = pair.get('result', pair)['candidate']
    if candidate.get('kernel_profiles'):
        matches = [row for row in candidate['kernel_profiles']
            if row['basic']['Op Name'] == summary['kernel']]
        assert len(matches) == 1, 'Select the replayed kernel explicitly within multi-kernel profiling'
        profile = matches[0]
    else:
        profile = candidate
        assert profile['basic']['Op Name'] == summary['kernel']
    selected = {'aiv_time(us)', 'aiv_mte2_instructions', 'aiv_mte3_instructions',
        'read_main_memory_datas(KB)', 'GM_to_UB_datas(KB)', 'UB_to_GM_datas(KB)',
        'aiv_vec_time(us)', 'aiv_mte2_time(us)', 'aiv_mte3_time(us)',
        'aiv_scalar_time(us)', 'aiv_scalar_wait_time(us)'}
    metrics = {}
    for table_name, table in profile['tables'].items():
        for key, value in table['fields'].items():
            if key in selected:
                if key in metrics:
                    assert all(metrics[key].get(field) == item for field, item in value.items()), 'Conflicting metric tables'
                    metrics[key]['tables'].append(table_name)
                else:
                    metrics[key] = dict(value, tables=[table_name])
    cores = {core['name']: core for kernel in listing['kernels'] for core in kernel['cores']}
    reports = []
    for report in summary['reports']:
        instructions = cores[report['core']]['instructions']
        counts = Counter()
        for instruction in instructions:
            counts[instruction['pipeline'] + '/' + instruction['instruction']] += instruction.get('calls') or 0
        reports.append(dict(core=report['core'], clock_validation=report['absolute_unit_validation'],
            raw_window=report['timeline'].get('raw_clock_window'),
            capture_scope=report['timeline']['capture_scope'], instruction_calls=dict(counts),
            mte_copies=[row for row in instructions if row['instruction'] == 'MOV_SRC_TO_DST_ALIGN'],
            timeline_pipelines=report['timeline']['pipelines']))
    return dict(variant=name, observation=str(observation),
        application_completed=summary['application_completed'], execution=read(directory / 'execution.json'),
        reports=reports, profiling=dict(path=str(pair_path), basic=profile['basic'], metrics=metrics,
            scope='Only the replayed kernel; whole-call combined counters remain in the original comparison',
            whole_call_basic=candidate['basic']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--prior', type=Path, action='append', default=[])
    args = parser.parse_args()
    observations = [(path, path.name) for path in args.prior]
    observations.append((args.run / 'observations' / args.invocation, args.invocation))
    facts = dict(status='measured_facts_for_host_review', invocation=args.invocation,
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        replays=[replay(path, name) for path, name in observations],
        scope='Per-core instruction counts and per-program profiling from bound captures. '
              'Partial windows, execution exits and independent clocks preserved. Formal latency remains separate.')
    output = args.run / ('post-mechanism-' + args.invocation.replace('-', '') + '-facts.json')
    with output.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(facts, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=facts['status'], output=str(output))))


if __name__ == '__main__':
    main()
