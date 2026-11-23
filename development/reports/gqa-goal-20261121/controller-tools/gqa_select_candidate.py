"""Record retention only after unchanged GQA cases and bound post-observations complete."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--measurement-result', type=Path,
                        help='Explicit justified same-source four-case measurement; preserve original result')
    args = parser.parse_args()
    run = args.run.resolve()
    iteration = run / 'iterations' / args.invocation
    parent = read(run / 'selection.json') if (run / 'selection.json').exists() else read(run / 'parent-selection.json')
    original_result_path = iteration / 'parent-validation/results.json'
    result_path = args.measurement_result.resolve() if args.measurement_result else original_result_path
    result = read(result_path)
    if args.measurement_result:
        assert result_path.is_relative_to(run)
        original = read(original_result_path)
        assert result['source_sha256'] == original['source_sha256']
        protocol = read(result_path.parent / 'protocol.json')
        assert protocol['original_result_sha256'] == sha(original_result_path)
        assert protocol['source_sha256'] == result['source_sha256']
        assert protocol['contract_sha256'] == result['contract_sha256']
        assert protocol['reason']
    assert result['contract_sha256'] == sha(run / 'contract.json')
    assert {r['case_id'] for r in result['cases']} == {1,2,4,7}
    full_pass = all(r.get('status') == 'measured' and all(
        r.get(side + '_correctness', {}).get('official_checker', {}).get('passed') for side in ['candidate','native'])
        for r in result['cases'])
    call = run / 'agent-calls' / args.invocation
    assert sha(call / 'output-source.py') == result['source_sha256']
    host = read(call / 'host-review.json')
    assert host['status'] == 'preedit_reads_and_semantics_approved'
    target_row = next(r for r in result['cases'] if r['case_id'] == 4)
    target = target_row.get('candidate_timing', {}).get('full_call_us')
    retained = full_pass and target < parent['target_latency_us']
    stability = run / 'conditional-stability' / args.invocation
    stability_result = None
    if (stability / 'protocol.json').exists():
        stability_result = read(stability / 'summary.json')
        assert stability_result['source_identities']['B']['sha256'] == result['source_sha256']
        retained = retained and stability_result['both_candidate_blocks_faster']
    binding = read(iteration / 'binary-binding-and-facts.json') if full_pass else None
    measurement_binary_binding = None
    if full_pass:
        assert binding['source_sha256'] == result['source_sha256']
        assert binding['status'] == 'formal_profile_simulator_binary_verified'
        if args.measurement_result:
            from gqa_binary_execution import execution_signature
            original_binary = Path(binding['simulator_cache_binaries'][0]['path'])
            assert sha(original_binary) == binding['target_binary_sha256']
            signature = execution_signature(original_binary)
            matches = [dict(path=str(path.relative_to(run)), sha256=sha(path))
                       for path in (result_path.parent / 'cache').rglob('*.npubin')
                       if execution_signature(path) == signature]
            assert matches, 'Remeasured execution differs from bound post-observation binary'
            measurement_binary_binding = dict(original_binary_sha256=sha(original_binary),
                matches=matches, execution_signature=signature,
                scope='Allocated ELF sections, Ascend stack/argument metadata and public symbols match. '
                      'Debug metadata and whole-file identities remain separate.')
    initial = read(run / 'parent-validation/results.json')
    cases = []
    for row in result['cases']:
        old = next(r for r in initial['cases'] if r['case_id'] == row['case_id'])
        original = old['candidate_timing']['full_call_us']
        new = row.get('candidate_timing', {}).get('full_call_us')
        cases.append(dict(case_id=row['case_id'], original_us=original, candidate_us=new,
            latency_reduction_pct=(1-new/original)*100 if new else None,
            native_us=row.get('native_timing', {}).get('full_call_us'),
            official_passed=row.get('candidate_correctness', {}).get('official_checker', {}).get('passed')))
    value = dict(status='retained_complete_four_case_subset' if retained else (
        ('qualified_gain_not_confirmed_keep_best' if target < parent['target_latency_us'] else
         'qualified_no_new_target_gain_keep_best') if full_pass else 'incomplete_candidate_keep_best'),
        invocation=args.invocation, source_sha256=result['source_sha256'],
        immediate_source_parent_sha256=sha(call / 'input-source.py'),
        retained_parent_source_sha256=parent['source_sha256'], historical_retained_us=parent['target_latency_us'],
        initial_retained_us=next(r['candidate_timing']['full_call_us'] for r in initial['cases'] if r['case_id']==4),
        target_latency_us=target, target_case=4, cases=cases, qualified=full_pass, retained=retained,
        target_cann_us=target_row.get('native_timing', {}).get('full_call_us'),
        target_cann_speedup=target_row['native_timing']['full_call_us']/target if target else None,
        target_reduction_vs_historical_pct=(1-target/parent['target_latency_us'])*100 if target else None,
        four_case_latency_geomean_reduction_pct=(1-math.prod(r['candidate_us']/r['original_us'] for r in cases)**.25)*100
            if full_pass else None,
        contract_sha256=sha(run / 'contract.json'), formal_result_sha256=sha(result_path),
        measurement_result=str(result_path.relative_to(run)),
        original_formal_result_sha256=sha(original_result_path),
        measurement_binary_binding=measurement_binary_binding,
        post_binding='iterations/' + args.invocation + '/binary-binding-and-facts.json' if binding else None,
        selected_at=datetime.now(timezone.utc).isoformat(),
        conditional_stability=stability_result,
        measurement_policy=read(run / ('measurement-policy-' + args.invocation + '.json'))
            if (run / ('measurement-policy-' + args.invocation + '.json')).exists() else None,
        scope='Fixed official four-case subset; full20-case qualification remains future work. Partial replay windows '
              'and application completion are recorded separately. No additional tests or unconditional ABBA.')
    save(run / ('retention-' + args.invocation + '.json'), value)
    if retained:
        save(run / 'selection.json', value)
        save(run / 'retained-current.json', dict(source_sha256=result['source_sha256'],
            source='agent-calls/' + args.invocation + '/output-source.py', selection='selection.json', target_us=target))
    print({key:value[key] for key in ['status','target_latency_us','historical_retained_us','retained']})


if __name__ == '__main__':
    main()
