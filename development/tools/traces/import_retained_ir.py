"""Extend a preserved delivery lineage with a qualified IR case and fixed input view."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from .cann_trace import rebase
from .research_progress import write_report
from .stitched import method_label
from .import_ir_progress import check_ir_lineage


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_case(root, run):
    reports = root / 'development/reports'
    config = read(run / 'config.json')
    selection = read(run / 'selection.json')
    assert selection['status'] == 'retained_qualified_ir_improvement'
    assert selection['stability']['candidate_below_every_original']
    op = config['operator']
    invocation = selection['invocation']
    lineage = run / ('retention-lineage-' + invocation.replace('-', '') + '.json')
    if lineage.exists():
        record = read(lineage)
        assert record['selection_sha256'] == sha(run / 'selection.json')
        selection = dict(selection, **{k: record[k] for k in ['implementation_parent_ir_sha256', 'ir_lineage']})
        check_ir_lineage(selection, run, selection['original_ir_sha256'], selection['ir_sha256'])
    target = config['experiment_goal']['target_case']
    candidate_dir = run / 'evaluations' / invocation
    baseline_dir = run / 'evaluations/unchanged-ir'
    if not baseline_dir.exists():
        baseline_dir = run.parent / Path(config['comparator_run']).name / 'evaluations/unchanged-ir'
    latest = read(candidate_dir / 'result.json')
    assert sha(candidate_dir / 'result.json') == selection['formal_result_sha256']
    assert latest['status'] == 'PASSED' and latest['num_passed'] == latest['num_workloads']
    assert all(row['status'] == 'passed' for row in read(candidate_dir / 'supplementary-summary.json').values())
    assert read(run / ('observation-binding-' + invocation + '.json'))['status'] == 'formal_observation_binaries_verified'
    gallery = reports / 'operator-traces-stitched-20261117'
    row = next(row for row in read(gallery / 'cann-view.json')['operators'] if row['operator'] == op)
    assert row['display_events'][-1]['source_sha256'] == selection['source_sha256'] == row['delivery']['sha256']
    anchors = {r['uuid']: r['reference_latency_ms'] for r in latest['per_workload'] if r['phase'] == 'timing'}
    events = []
    for old in row['display_events']:
        if old['is_delivery']:
            result_path = root / row['delivery']['result']
        else:
            result_path = root / 'development/experiments/lineage/20261117/latency-results' / (old['record']['result']['sha256'] + '.json')
        assert sha(result_path) == old['record']['result']['sha256']
        measured = read(result_path)
        qualified = old['score'] is not None and old['status'] == 'PASSED'
        point = next((r for r in measured['per_workload'] if r['uuid'] == target and r['phase'] == 'timing'), None)
        method = method_label(old, {})
        event = dict(label='FlagGems' if old['is_baseline'] else 'Historical delivery',
            qualified=qualified, score=rebase(latest, measured)['score'] if qualified else None,
            source_sha256=old['source_sha256'], measurement_event_id=old['id'],
            raw_geo_mean=measured.get('geo_mean'), score_is_rebased=True,
            latency_ms=point.get('latency_ms') if point and point['status'] == 'PASSED' else None,
            target_case_speedup=anchors[target] / point['latency_ms'] if point and point['status'] == 'PASSED' else None,
            replay=dict(result=os.path.relpath(result_path, reports).replace('\\', '/'),
                result_sha256=sha(result_path), num_passed=measured['num_passed'], num_workloads=measured['num_workloads']),
            original_event=old)
        if method:
            event.update(method=method['method'], category=method['color'], location=method['location'])
        events.append(event)
    parent_id = events[-1]['measurement_event_id']
    for name, directory in [('unchanged-ir', baseline_dir), (invocation, candidate_dir)]:
        measured = read(directory / 'result.json')
        point = next(r for r in measured['per_workload'] if r['uuid'] == target and r['phase'] == 'timing')
        transition = name == 'unchanged-ir'
        event_id = str(run.relative_to(reports)) + '/' + name
        event = dict(label='Same-implementation measurement' if transition else invocation, qualified=True,
            score=rebase(latest, measured)['score'], raw_geo_mean=measured['geo_mean'],
            score_is_rebased=transition, source_sha256=selection['source_sha256'],
            parent_source_sha256=selection['source_sha256'],
            ir_sha256=selection['original_ir_sha256'] if transition else selection['ir_sha256'],
            parent_ir_sha256=selection['original_ir_sha256'], measurement_event_id=event_id,
            parent_measurement_event_id=parent_id, latency_ms=point['latency_ms'],
            target_case_speedup=anchors[target] / point['latency_ms'],
            selected_layer='remeasurement' if transition else 'single_operator_ir',
            measurement_transition=transition, category='sim', location='ir',
            replay=dict(result=(directory / 'result.json').relative_to(reports).as_posix(),
                result_sha256=sha(directory / 'result.json'), num_passed=measured['num_passed'], num_workloads=measured['num_workloads']))
        if not transition:
            event.update(**selection['presentation'], selection=(run / 'selection.json').relative_to(reports).as_posix(),
                simulator_replay=selection['simulator_replay'])
            event['recorded_method'] = event['method']
            event['method'] = event['method'].replace('Target-shape time', 'Fixed shape ')
            if selection.get('ir_lineage'):
                event.update(parent_ir_sha256=selection['implementation_parent_ir_sha256'],
                    comparison_ir_sha256=selection['original_ir_sha256'], ir_lineage=selection['ir_lineage'])
        events.append(event)
        parent_id = event_id
    scope = ('Recalculate full contracts with latest per-case CANN anchors; historical delivery scores remain frozen in details.'
        "Hide same-implementation measurements; retain temporal/timer variation. Actual IR increments use this run's fixed contract."
        'selection.json separately records target-shape gains and full-contract geometric-mean timing changes.')
    data = dict(operator=op, metric='kgs_full_contract_geo_mean', events=events, scope=scope,
        delivery_frozen=row['delivery'], original_lineage=row,
        cann_anchor=dict(cases=anchors, result=str(candidate_dir / 'result.json')),
        optimization_loop='Simulator regions/timing -> layer selection -> hardware acceptance -> continuous replay',
        history_links=[dict(label='Native-to-delivery history', href='../operator-traces-stitched-20261117/' + op + '/delivery-history.html'),
            dict(label='Current IR acceptance', href='../' + (run / 'selection.json').relative_to(reports).as_posix())])
    output = reports / (op + '-simulator-20261120')
    write_report(data, output)
    fixed = copy.deepcopy(data)
    fixed.update(metric='fixed_case_cann_speedup', case=target, shape=selection['fixed_input'], compact=True)
    for event in fixed['events']:
        event['score'] = event['target_case_speedup']
        event['qualified'] = event['score'] is not None
    write_report(fixed, reports / (op + '-shape-20261120'))
    overrides = read(gallery / 'presentation-overrides.json')
    overrides[op] = dict(directory=output.relative_to(root).as_posix(), scope=scope)
    (gallery / 'presentation-overrides.json').write_text(json.dumps(overrides, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    print(import_case(args.root.resolve(), args.run.resolve()))


if __name__ == '__main__':
    main()
