"""Bind frozen-source profiler replays to figures without changing old evidence."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .cann_trace import rebase
from .research_progress import write_report
from .stitched import method_label


def read(path):
    return json.loads(path.read_bytes())


def import_replay(reports, replay, folder):
    output = reports / folder
    original_path = output / 'mainline-presentation.json'
    original = read(original_path)
    saved = output / 'presentation-before-profiler-replay-20261119.json'
    if not saved.exists():
        saved.write_bytes(original_path.read_bytes())
    original = read(saved)
    metadata = {e.get('source_sha256'): e for e in original['events']}
    history = read(reports / 'operator-traces-stitched-20261117/cann-view.json')
    annotations = {}
    label_path = reports / 'group-meeting-20261119/presentation-labels.json'
    labels = read(label_path).get('methods', {}) if label_path.exists() else {}
    for row in history['operators']:
        if row['operator'] == original['operator']:
            annotations = {e['source_sha256']: method_label(e, {}) for e in row['display_events']}
    manifest = read(replay / 'manifest.json')
    summaries = read(replay / 'summary.json')
    assert len(summaries) == len(manifest['measurements'])
    latest_item = manifest['measurements'][-1]
    latest = read(replay / latest_item['name'] / 'result.json')
    assert latest['status'] == 'PASSED' and latest['num_passed'] == latest['num_workloads']
    assert abs(rebase(latest, latest)['score'] - latest['geo_mean']) < 1e-8
    anchor = {row['uuid']: row['reference_latency_ms'] for row in latest['per_workload'] if row['phase'] == 'timing'}
    assert len(anchor) == sum(row['phase'] == 'timing' for row in latest['per_workload'])
    events = []
    for index, (item, summary) in enumerate(zip(manifest['measurements'], summaries)):
        result_path = replay / item['name'] / 'result.json'
        corrected = reports / 'presentation-scatter-native-adapter-20261119/scatter-native-corrected/result.json'
        corrected_source = corrected.parent / 'source.py'
        if index == 0 and folder == 'scatter-simulator-20261118' and corrected.exists():
            result_path = corrected
        result = read(result_path)
        source = (corrected_source if result_path == corrected else replay / item['name'] / 'source.py').read_bytes()
        assert hashlib.sha256(source).hexdigest() == item['sha256']
        event = copy.deepcopy(metadata.get(item['sha256'], {}))
        annotation = annotations.get(item['sha256'])
        if annotation and not event.get('method'):
            event.update(method=annotation['method'], category=annotation['color'],
                location={'kernel': 'device', 'device': 'device', 'host': 'host', 'mixed': 'mixed'}[annotation['location']],
                attribution=annotation.get('attribution'))
        if item['sha256'] in labels:
            event['method'] = labels[item['sha256']]
        event.pop('historical', None)
        event.pop('measurement_transition', None)
        if 'record' in event:
            event['historical_record'] = event.pop('record')
        qualified = result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
        supplements = None
        if '-history-' in item['name']:
            supplements = read(replay / item['name'] / 'supplementary-summary.json')
            qualified = qualified and all(check['status'] == 'passed' for check in supplements.values())
        timings = [row for row in result.get('per_workload', []) if row['phase'] == 'timing']
        if qualified:
            assert len(timings) == len(anchor)
            assert {row['uuid'] for row in timings} == set(anchor)
            assert all(row['status'] == 'PASSED' and row['latency_ms'] > 0 for row in timings)
            conversion = rebase(latest, result)
            conversion['scope'] = 'Convert measured candidate times using a common CANN reference; keep original paired scores separately.'
            normalized = conversion['score']
            target = next(row for row in timings if row['uuid'] == item['target_case'])
        else:
            normalized = None
            target = None
            conversion = None
        event.update(label='FlagGems' if index == 0 else 'Repeat V' + str(index),
            measurement_event_id='/'.join(result_path.relative_to(reports).parts[:2]),
            axis_label='FlagGems' if index == 0 else 'V' + str(index),
            source_sha256=item['sha256'], qualified=qualified, score=normalized,
            geo_mean=normalized, raw_geo_mean=result.get('geo_mean'),
            latency_ms=target['latency_ms'] if target else None,
            target_case_speedup=anchor[item['target_case']] / target['latency_ms'] if target else None,
            raw_target_case_speedup=target.get('speedup') if target else None,
            score_is_rebased=index != len(manifest['measurements']) - 1,
            timing='profiler; frozen-source replay 2026-11-19',
            replay=dict(result=result_path.relative_to(reports).as_posix(),
                result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                num_passed=result['num_passed'], num_workloads=result['num_workloads'],
                supplementary=supplements),
            cann_anchor=dict(kind='latest replay per-case reference', cases=anchor))
        event['conversion'] = conversion
        events.append(event)
    data = copy.deepcopy(original)
    if data.get('target_case_scope'):
        data['historical_target_case_scope'] = data.pop('target_case_scope')
    data.update(events=events, metric='full_contract_cann_speedup',
        scope='Unified-profiler reevaluation from native FlagGems to current best, with fixed sources. Use per-case CANN times from the final qualified source measurement as common anchors; '
              'recalculate full-contract geometric-mean performance from actual candidate timings. Keep original scores, ancestry, transfers, failures, and regressions in details.',
        replay_manifest=(replay / 'manifest.json').relative_to(reports).as_posix(),
        historical_presentation=saved.name)
    data['history_links'] = [dict(label='Full continuation: '+path.parent.name,
        href=path.relative_to(output).as_posix()) for path in sorted(output.glob('validated-*/continuation.json'))]
    write_report(data, output)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    reports = args.root / 'development/reports'
    for replay, folder in [('presentation-replay-20261119', 'batchnorm-simulator-20261118'),
            ('presentation-scatter-replay-20261119', 'scatter-simulator-20261118')]:
        import_replay(reports, reports / replay, folder)


if __name__ == '__main__':
    main()
