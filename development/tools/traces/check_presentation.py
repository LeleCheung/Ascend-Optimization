"""Check native roots, continuous best-progress lines, and fixed-input identity."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from .selection import remeasurement_indices


def read(path):
    return json.loads(path.read_bytes())


def geometry(path):
    root = ET.fromstring(path.read_text(encoding='utf8'))
    kinds = {'connection', 'method-switch', 'source-lineage', 'retained-best', 'measurement-transition'}
    segments = []
    for node in root.iter():
        assert node.attrib.get('class') != 'measurement-transition', 'Remeasurement segment on trace'
        if node.tag.rsplit('}', 1)[-1] == 'text':
            assert not any(word in ''.join(node.itertext()) for word in
                ('复测', '换计时', '换参考')), 'Remeasurement label on trace'
        if node.attrib.get('class') in kinds:
            numbers = [float(n) for n in re.findall(r'-?\d+(?:\.\d+)?', node.attrib['d'])]
            assert len(numbers) == 4
            segments.append(numbers)
    assert segments, 'A native-to-result curve needs at least one connection'
    assert all(line[3] <= line[1] + .02 for line in segments), 'Descending main line'
    assert all(abs(a[2] - b[0]) < .02 and abs(a[3] - b[1]) < .02
        for a, b in zip(segments, segments[1:])), 'Disconnected main line'
    return len(segments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    reports = args.root / 'development/reports'
    overview = reports / 'operator-traces-stitched-20261117'
    raw = {r['operator']: r for r in read(overview / 'cann-view.json')['operators']}
    records = read(overview / 'progress-presentation.json')['operators']
    rows = []
    for record in records:
        operator = record['operator']
        folder = overview / operator
        if record.get('replacement'):
            data = read(folder / 'mainline-presentation.json')
            native = data['events'][0]
            assert native['label'] == 'FlagGems' and native['source_sha256']
            indices = data['presentation_selection']['displayed_indices']
            assert not set(indices) & remeasurement_indices(data['events'])
            scores = [data['events'][i]['score'] for i in indices if data['events'][i]['score'] is not None]
            for event in data['events']:
                if event.get('selected_layer') == 'single_operator_ir':
                    assert event['category'] == 'sim' and event['location'] == 'ir', 'IR presentation needs orange simulator guidance and IR-layer shape'
        else:
            baseline = raw[operator]['display_events'][0]
            assert baseline['source_key'] == 'native:' + operator
            assert record['displayed_ids'][0] == baseline['id']
            replays = {raw[operator]['display_events'][i]['id']
                for i in remeasurement_indices(raw[operator]['display_events'])}
            assert not set(record['displayed_ids']) & replays
            scores = [v for v in record['scores'] if v is not None]
            assert all(c['kind'] in {'progress_summary', 'documented_method_switch'} for c in record['connections'])
        assert all(a <= b for a, b in zip(scores, scores[1:]))
        count = geometry(folder / 'mainline.svg')
        rows.append(dict(operator=operator, native_root=True, continuous=True, monotonic=True,
            segments=count, source_ancestry_complete=raw[operator]['full_lineage_verified']))
    assert len(rows) == 26
    curves = read(reports / 'group-meeting-20261119/cases.json')
    fixed = []
    for name, curve in zip(['batchnorm', 'upsample', 'qkv', 'scatter'], curves):
        assert curve['metric'] == 'fixed_case_cann_speedup'
        assert curve['case'] and curve['shape'] and curve['events'][0]['source_sha256']
        assert len({event['case'] for event in curve['events']}) == 1
        assert not set(curve['presentation_selection']['displayed_indices']) & remeasurement_indices(curve['events'])
        fixed.append(dict(operator=curve['operator'], case=curve['case'], shape=curve['shape'],
            segments=geometry(reports / 'group-meeting-20261119' / (name + '.svg')),
            continuous=True, monotonic=True))
    assert len(fixed) == 4
    eight_path = reports / 'group-meeting-20261119/eight-figures.json'
    if eight_path.exists():
        eight = read(eight_path)
        figures = eight['figures']
        assert len(figures) == 8 and eight['aggregate_count'] == eight['fixed_shape_count'] == 4
        assert len({row['svg'] for row in figures}) == 8
        for figure in figures:
            path = eight_path.parent / figure['svg']
            geometry(path)
            if figure['group'] == 'aggregate':
                source = reports / figure['source']
                assert hashlib.sha256(source.read_bytes()).hexdigest() == figure['source_sha256']
                assert path.read_bytes() == source.with_name('mainline.svg').read_bytes()
                assert path.read_bytes() == (overview / figure['operator'] / 'mainline.svg').read_bytes()
                assert '全用例几何平均' in path.read_text(encoding='utf8')
            else:
                assert figure['metric'] == 'fixed_case_cann_speedup' and figure['case']
                assert '固定shape' in path.read_text(encoding='utf8')
    result = dict(schema='hgkda.presentation-acceptance/v1', operator_figures=rows, fixed_case_figures=fixed,
        scope='Numerical, source-root, and SVG geometry checks; screenshot inspection is separately recorded.',
        historical_source_branches=[row['operator'] for row in rows if not row['source_ancestry_complete']])
    (reports / 'group-meeting-20261119/acceptance.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(operator_figures=len(rows), fixed_case_figures=len(fixed),
        meeting_figures=8 if eight_path.exists() else 4, status='passed')))


if __name__ == '__main__':
    main()
