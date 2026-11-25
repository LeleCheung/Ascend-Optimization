"""Append accepted same-source IR improvements with explicit measurement events."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .cann_trace import rebase
from .research_progress import write_report


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_contract(path):
    return {key: value for key, value in read(path).items() if key != 'implementation'}


def bind_historical_measurements(data, reports, review_path, target):
    """Bind existing events to hashed original sources, requests and results."""
    review = read(review_path)
    rows = review['history']
    indices = [row['event_index'] for row in rows]
    assert len(set(indices)) == len(indices), 'Duplicate historical measurement event'
    assert set(indices) == {i for i, event in enumerate(data['events']) if event['qualified']}, \
        'Historical review must cover every qualified event'
    for row in rows:
        event = data['events'][row['event_index']]
        assert event['source_sha256'] == row['source_sha256'], 'Historical event source identity changed'
        paths = {key: (reports / row[key]).resolve() for key in ['source', 'result', 'actual_request']}
        assert all(path.is_relative_to(reports.resolve()) for path in paths.values())
        assert sha(paths['source']) == row['source_sha256'], 'Historical source bytes changed'
        assert sha(paths['result']) == row['result_sha256'], 'Historical result bytes changed'
        assert sha(paths['actual_request']) == row['request_sha256'], 'Historical request bytes changed'
        request = read(paths['actual_request'])
        assert any(hashlib.sha256(source['content'].encode()).hexdigest() == row['source_sha256']
                   for source in request['implementation']['sources']), 'Actual request lacks historical source'
        result = read(paths['result'])
        assert result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads'] > 0
        case = next(case for case in result['per_workload']
                    if case['uuid'] == target and case['phase'] == 'timing')
        assert case['status'] == 'PASSED'
        event.update(latency_ms=case['latency_ms'], raw_geo_mean=result['geo_mean'],
            raw_target_case_speedup=case['speedup'],
            measurement_event_id=paths['result'].parent.relative_to(reports.resolve()).as_posix(),
            replay=dict(result=row['result'], result_sha256=row['result_sha256'],
                request=row['actual_request'], request_sha256=row['request_sha256'],
                source=row['source'], num_passed=result['num_passed'], num_workloads=result['num_workloads']))
    data['historical_measurements_review'] = dict(
        path=review_path.relative_to(reports).as_posix(), sha256=sha(review_path))


def check_measurement_decision(selection, run):
    stability = selection.get('stability', {})
    if stability.get('candidate_below_every_original'):
        return
    assert stability.get('status') == 'not_requested_conditional_policy', 'Resolve the recorded measurement concern before import'
    assert stability.get('reason') and not stability.get('unresolved_measurement_concern', False)
    assert read(run / 'validation-policy.json')['abba'] == 'conditional'


def check_ir_lineage(selection, run, original_sha, candidate_sha):
    if not selection.get('implementation_parent_ir_sha256'):
        return
    chain = selection['ir_lineage']
    previous = original_sha
    for link in chain:
        assert link['parent_ir_sha256'] == previous, 'Disconnected IR repair ancestry'
        decision = read(run / link['decision'])
        submission = read(run / link['submission'])
        assert decision['input_files'][link['path']] == previous
        assert submission['output_files'][link['path']] == link['ir_sha256']
        previous = link['ir_sha256']
    assert previous == candidate_sha
    assert chain[-1]['parent_ir_sha256'] == selection['implementation_parent_ir_sha256']


def import_case(reports, operator, folder, method, story, attribution, historical_roots=(),
                phase='ir-performance-20261119'):
    run = reports / phase / operator
    continuation = phase != 'ir-performance-20261119'
    selection = read(run / 'selection.json')
    assert selection['status'] == 'retained_qualified_ir_improvement'
    check_measurement_decision(selection, run)
    invocation = selection['invocation']
    evaluation = run / 'evaluations' / invocation
    latest = read(evaluation / 'result.json')
    assert sha(evaluation / 'result.json') == selection['formal_result_sha256']
    checks = read(evaluation / 'supplementary-summary.json')
    assert all(row['status'] == 'passed' for row in checks.values())
    source = selection['source_sha256']
    target = read(run / 'config.json')['experiment_goal']['target_case']
    manifest = read(evaluation / 'implementation/ir-manifest.json')
    changed = [row for row in manifest['overrides'] if row['sha256'] != row['original_sha256']]
    assert len(changed) == 1, 'Add multi-specialization identity support before importing this case'
    original_sha, candidate_sha = changed[0]['original_sha256'], changed[0]['sha256']
    check_ir_lineage(selection, run, original_sha, candidate_sha)
    output = reports / folder
    current = output / 'mainline-presentation.json'
    backup = output / ('presentation-before-' + phase + '.json' if continuation
                       else 'presentation-before-ir-20261119.json')
    if not backup.exists():
        backup.write_bytes(current.read_bytes())
    data = copy.deepcopy(read(backup))
    assert data['events'][-1]['source_sha256'] == source
    if selection.get('historical_measurements'):
        binding = selection['historical_measurements']
        review_path = run / binding['path']
        assert review_path.resolve().is_relative_to(run.resolve())
        assert sha(review_path) == binding['sha256'], 'Historical measurement review changed'
        bind_historical_measurements(data, reports, review_path, target)
    if continuation:
        displayed_ir = data['events'][-1].get('ir_sha256')
        if displayed_ir:
            assert displayed_ir == original_sha, 'Continuation must start at the displayed retained IR'
        else:
            assert original_sha == changed[0]['compiler_original_sha256'], 'A source-only parent requires compiler-original IR'
        verification = read(run / 'retained-baseline-verification.json')
        assert verification['status'] == 'verified'
        assert verification['expected_binaries'] == verification['observed_binaries']
        assert verification['result_sha256'] == sha(run / 'evaluations/unchanged-ir/result.json')
    contract = request_contract(evaluation / 'request.json')
    for event in data['events']:
        if not event['qualified']:
            continue
        if not event.get('replay'):
            candidates = [reports / root / ('formal-' + event['source_sha256'][:12]) /
                          'evaluation/result.json' for root in historical_roots]
            found = [path for path in candidates if path.exists()]
            assert len(found) == 1, 'A unique complete historical measurement is required'
            path = found[0]
            result = read(path)
            assert result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
            request = read(path.with_name('request.json'))
            sources = request['implementation']['sources']
            assert any(hashlib.sha256(row['content'].encode()).hexdigest() == event['source_sha256']
                       for row in sources), 'Historical request lacks the declared source bytes'
            case = next(row for row in result['per_workload'] if row['uuid'] == target)
            event.update(latency_ms=case['latency_ms'], raw_geo_mean=result['geo_mean'],
                raw_target_case_speedup=case['speedup'],
                measurement_event_id=path.parent.relative_to(reports).as_posix(),
                replay=dict(result=path.relative_to(reports).as_posix(), result_sha256=sha(path),
                    num_passed=result['num_passed'], num_workloads=result['num_workloads']))
        result_path = reports / event['replay']['result']
        request_path = (reports / event['replay']['request'] if event['replay'].get('request')
                        else result_path.with_name('request.json'))
        assert request_contract(request_path) == contract

    parent_id = data['events'][-1]['measurement_event_id']
    for name, ir_sha, transition in [('unchanged-ir', original_sha, True),
                                     (invocation, candidate_sha, False)]:
        measured = run / 'evaluations' / name
        result = read(measured / 'result.json')
        assert result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
        assert request_contract(measured / 'request.json') == contract
        event_id = phase + '/' + operator + '/' + name
        case = next(row for row in result['per_workload'] if row['uuid'] == target)
        event = dict(label=('Same-source retained-IR repeat' if continuation else 'Same-source original-IR repeat') if transition else 'IR performance continuation ' + invocation,
            source_sha256=source, parent_source_sha256=source, ir_sha256=ir_sha,
            parent_ir_sha256=original_sha, selected_layer='remeasurement' if transition else 'single_operator_ir',
            qualified=True, measurement_event_id=event_id, parent_measurement_event_id=parent_id,
            latency_ms=case['latency_ms'], raw_geo_mean=result['geo_mean'],
            raw_target_case_speedup=case['speedup'], location='device', category='ir',
            timing='profiler; exact source and compiler receipts; L2 cache cleared',
            replay=dict(result=(measured / 'result.json').relative_to(reports).as_posix(),
                result_sha256=sha(measured / 'result.json'), num_passed=result['num_passed'],
                num_workloads=result['num_workloads'], supplementary=None if transition else checks))
        if transition:
            event.update(measurement_transition=True, same_protocol_remeasurement=True,
                measurement_transition_label='Same-source repeat',
                measurement_note='Full same-source original-IR reevaluation; record same-contract timing variation separately.')
        else:
            event.update(method=method, attribution=attribution, category='sim', location='ir',
                original_evidence_category='ir',
                presentation_attribution_policy='Confirmed 2026-11-20: simulator-guided IR changes are orange with a distinct IR shape; other contributions remain in details.',
                measurement_bridge_event_id=parent_id,
                selection=(run / 'selection.json').relative_to(reports).as_posix())
            if selection.get('simulator_replay'):
                event['simulator_replay'] = selection['simulator_replay']
            if selection.get('implementation_parent_ir_sha256'):
                event.update(parent_ir_sha256=selection['implementation_parent_ir_sha256'],
                    comparison_ir_sha256=original_sha, ir_lineage=selection['ir_lineage'])
        data['events'].append(event)
        data.setdefault('lineage_connections', []).append(dict(start=parent_id, end=event_id,
            kind='same_source_remeasurement' if transition else ('registered_ir_path' if event.get('ir_lineage') else 'registered_ir_edit'),
            source_sha256=source, parent_ir_sha256=event['parent_ir_sha256'], ir_sha256=ir_sha,
            **(dict(ir_lineage=event['ir_lineage'], comparison_ir_sha256=original_sha) if event.get('ir_lineage') else {}),
            proof=(run / (('retained-baseline-verification.json' if continuation else 'original-ir-verification.json')
                          if transition else 'agent-calls/' + invocation + '/before-edit-decision.json')).relative_to(reports).as_posix()))
        parent_id = event_id

    anchor = {row['uuid']: row['reference_latency_ms'] for row in latest['per_workload'] if row['phase'] == 'timing'}
    for index, event in enumerate(data['events']):
        if not event['qualified']:
            continue
        measured = read(reports / event['replay']['result'])
        conversion = rebase(latest, measured)
        conversion['scope'] = 'Recalculate measured per-case candidate times using a common CANN reference; retain original paired scores separately.'
        event.update(score=conversion['score'], geo_mean=conversion['score'], conversion=conversion,
            target_case_speedup=anchor[target] / event['latency_ms'],
            cann_anchor=dict(kind='accepted IR evaluation per-case reference', cases=anchor),
            score_is_rebased=index != len(data['events']) - 1)
    data.update(ir_case_story=story, scope=data['scope'] +
        ' IR continuation keeps Triton source fixed; repeats stay in details. IR nodes preserve full contracts, checks, actual ancestry, and measurements.')
    data.setdefault('history_links', []).append(dict(label='IR acceptance and measurement records',
        href='../' + phase + '/' + operator + '/selection.json'))
    write_report(data, output)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    import_case(args.root / 'development/reports', 'batchnorm', 'batchnorm-simulator-20261118',
        'Widen IR output tiles\n8x128 -> 8x256',
        'Source prefetch/tiling -> fixed-source output IR -> 42.955 to 39.261 us',
        'Pre-change simulator windows suggest synchronization timing; Codex combines actual IR, lowering, and UB failures to propose wider output. CC/DeepSeek registers/edits IR. Full hardware evaluation and ABBA confirm gains; before/after window scopes remain separate.')
    reports = args.root / 'development/reports'
    if (reports / 'ir-performance-20261119/upsample/selection.json').exists():
        import_case(reports, 'upsample', 'native-upsample-contrast-20261118',
            'Select boundaries before expansion\nFP16 select then extend',
            'Packed source writeback -> narrower fixed-source IR boundary select -> hardware gain',
            'Codex suggests packing/layout investigation. The model uses profiling, hotspot regions, and actual lowering to move boundary select before FP16 extension. BCMLIR confirms narrower select/slices; full evaluation and interleaving confirm gain. Preserve window scopes.',
            historical_roots=('native-upsample-contrast-20261118/evidence/with-simulator',
                              'upsample-performance-20261118/evidence'))
    scatter_selection = reports / 'ir-performance-20261119/scatter/selection.json'
    if scatter_selection.exists():
        presentation = read(scatter_selection)['presentation']
        import_case(reports, 'scatter', 'scatter-simulator-20261118',
            presentation['method'], presentation['story'], presentation['attribution'])
    for operator, folder in [('batchnorm', 'batchnorm-simulator-20261118'),
                             ('upsample', 'native-upsample-contrast-20261118'),
                             ('qkv', 'qkv-simulator-20261118'),
                             ('scatter', 'scatter-simulator-20261118')]:
        selection_path = reports / 'midday-whitebox-20261120' / operator / 'selection.json'
        if selection_path.exists():
            presentation = read(selection_path)['presentation']
            import_case(reports, operator, folder, presentation['method'], presentation['story'],
                        presentation['attribution'], phase='midday-whitebox-20261120')
    # Simulator iteration and edit layer are separate presentation dimensions.
    # Keep the recorded judgment-source category and frozen acceptance receipts.
    notes = read(reports / 'group-meeting-20261119/case-evidence-notes.json')
    for name, folder in [('batchnorm', 'batchnorm-simulator-20261118'),
                         ('upsample', 'native-upsample-contrast-20261118'),
                         ('qkv', 'qkv-simulator-20261118'),
                         ('scatter', 'scatter-simulator-20261118')]:
        output = reports / folder
        data = read(output / 'mainline-presentation.json')
        data['optimization_loop'] = notes['optimization_loop']
        for event in data['events']:
            if event.get('selected_layer') == 'single_operator_ir':
                if not event.get('simulator_replay'):
                    phase = event['selection'].split('/')[0]
                    invocation = event['measurement_event_id'].split('/')[-1]
                    matching = [step for step in notes['operators'][name]['steps']
                        if phase + '/' in step.get('before_decision', '')
                        and '/agent-calls/' + invocation + '/' in step.get('before_decision', '')]
                    assert len(matching) == 1, 'Bind each IR event to its own replay step'
                    step = matching[0]
                    event['simulator_replay'] = {key: step[key] for key in
                        ['replay', 'before_decision', 'after_replay']}
        write_report(data, output)


if __name__ == '__main__':
    main()
