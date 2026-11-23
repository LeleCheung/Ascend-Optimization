"""Review short-instruction CSV quantization before independent long-PC clock calibration."""
import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys

from ir_campaign import read, save, sha


def review(run):
    directory = run / 'simulator-parent'
    summary = read(directory / 'summary.json')
    listing = read(directory / 'listing.json')
    assert summary['source_sha256'] == listing['source_sha256']
    config = read(run / 'config.json')
    sys.path.insert(0, str(Path(config['package']) / 'src'))
    from hgkda.simulator import validate_time_unit, pc
    reviews = []
    updates = []
    for report in summary['reports']:
        if report['absolute_unit_validation']['status'] == 'verified':
            continue
        core = next(c for c in listing['kernels'][0]['cores'] if c['name'] == report['core'])
        paths = list((directory / 'raw').glob('OPPROF_*/simulator/' + core['name'] + '/trace.json'))
        assert len(paths) == 1
        raw = read(paths[0])
        events = raw.get('traceEvents', []) if isinstance(raw, dict) else raw
        names = {e.get('pid'): e.get('args', {}).get('name') for e in events
                 if e.get('ph') == 'M' and e.get('name') == 'process_name'}
        spans = defaultdict(list)
        for event in events:
            args = event.get('args', {})
            if (event.get('ph') == 'X' and 'pc_addr' in args and event.get('dur', 0) > 0
                    and names.get(event.get('pid')) != 'CACHEMISS'):
                spans[pc(args['pc_addr'])].append((event['ts'], event['ts'] + event['dur']))
        rows = []
        for instruction in core['instructions']:
            reference = instruction.get('running_time_us')
            if (not reference or reference < .1 or instruction['pc'] not in spans
                    or instruction['pipeline'].upper() in {'SCALAR', 'SCALARLDST', 'FLOWCTRL'}
                    or re.fullmatch(r'BAR|SET_FLAG|WAIT_FLAG', instruction['instruction'], re.I)):
                continue
            merged = []
            for start, end in sorted(spans[instruction['pc']]):
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(end, merged[-1][1])
                else:
                    merged.append([start, end])
            duration = sum(end-start for start, end in merged)
            rows.append(dict(pc=instruction['pc'], instruction=instruction['instruction'],
                csv_us=reference, trace_union_us=duration,
                csv_calls=instruction['calls'], trace_events=len(spans[instruction['pc']]),
                relative_error=abs(duration/reference-1), absolute_error_us=abs(duration-reference)))
        mismatches = [row for row in rows if row['relative_error'] >= .05]
        assert mismatches and all(row['csv_us'] < 1 and row['absolute_error_us'] <= .011
                                  and row['csv_calls'] == row['trace_events'] for row in mismatches), (
                                      'Requires separate diagnosis; discrepancies exceed short-PC quantization', mismatches)
        calibration = validate_time_unit(raw, [i for i in core['instructions']
                                               if (i.get('running_time_us') or 0) >= 1], 'us')
        assert calibration['matching_pcs'] == calibration['compared_pcs']
        reviews.append(dict(core=core['name'], raw_trace_sha256=sha(paths[0]), rows=rows,
                            short_pc_mismatches=mismatches, long_pc_calibration=calibration))
        calibration.update(original_validation=report['absolute_unit_validation'],
                           calibration_policy='Independent CSV>=1us PCs at original 5% tolerance; '
                           'short-PC errors <=0.011us with equal event counts retained in resolution review.')
        updates.append((report, calibration))
    assert updates, 'No unresolved clock failure'
    original = directory / 'summary-initial-clock-failure.json'
    assert not original.exists(), 'Preserve existing review before rerunning'
    save(original, summary)
    for report, calibration in updates:
        report['absolute_unit_validation'] = calibration
    save(directory / 'clock-calibration-resolution.json', dict(
        status='long_pc_calibration_passed_short_rounding_discrepancies_retained', reports=reviews))
    save(directory / 'summary.json', summary)
    print([(r['core'], len(r['short_pc_mismatches']), r['long_pc_calibration']['compared_pcs'])
           for r in reviews])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    review(parser.parse_args().run.resolve())
