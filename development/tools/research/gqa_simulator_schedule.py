"""Save source-mapped instruction intervals from a verified GQA replay window."""
import argparse
import re
import sys
from pathlib import Path

from ir_campaign import read, save, sha


def collect(run, source_pattern):
    binding = read(run / 'binary-binding-and-facts.json')
    assert binding['status'] == 'formal_profile_simulator_binary_verified'
    source = run / 'workspace/candidate.py'
    assert sha(source) == binding['source_sha256']
    config = read(run / 'config.json')
    source_paths = {str(source)}
    if config.get('bcmlir_manifest'):
        manifest = read(Path(config['bcmlir_manifest']))
        assert manifest['source_sha256'] == binding['source_sha256']
        original_location = Path(manifest['original_source_location'])
        assert sha(original_location) == binding['source_sha256']
        source_paths.add(str(original_location))
    source_lines = {path: Path(path).read_text().splitlines() for path in source_paths}

    def code_for_frame(frame):
        path = frame.get('file')
        if path not in source_paths:
            return ''
        if frame.get('code'):
            return frame['code']
        line = frame.get('line')
        lines = source_lines[path]
        return lines[line-1] if isinstance(line, int) and 0 < line <= len(lines) else ''
    sys.path.insert(0, str(Path(config['package']) / 'src'))
    from hgkda.simulator import pc, number
    directory = run / 'simulator-parent'
    listing = read(directory / 'listing.json')
    summary = read(directory / 'summary.json')
    assert listing['source_sha256'] == summary['source_sha256'] == binding['source_sha256']
    pattern = re.compile(source_pattern)
    reports = []
    for core in listing['kernels'][0]['cores']:
        qualified = next(r for r in summary['reports'] if r['core'] == core['name'])
        assert qualified['absolute_unit_validation']['status'] == 'verified'
        paths = list((directory / 'raw').glob('OPPROF_*/simulator/' + core['name'] + '/trace.json'))
        assert len(paths) == 1
        timeline = read(directory / (core['name'] + '-relative-timeline.json'))['timeline']
        assert sha(paths[0]) == timeline['raw_trace_sha256']
        window = timeline['raw_clock_window']
        origin, end = window['start'], window['end']
        span = end - origin
        assert span > 0
        selected = {row['pc']: row for row in core['instructions'] if any(
            frame.get('file') in source_paths and pattern.search(code_for_frame(frame))
            for frame in row.get('source_stack', []))}
        raw = read(paths[0])
        events = raw.get('traceEvents', []) if isinstance(raw, dict) else raw
        names = {e.get('pid'): e.get('args', {}).get('name') for e in events
                 if e.get('ph') == 'M' and e.get('name') == 'process_name'}
        intervals = {key: [] for key in selected}
        for event in events:
            args = event.get('args', {})
            if event.get('ph') != 'X' or 'pc_addr' not in args or names.get(event.get('pid')) == 'CACHEMISS':
                continue
            key = pc(args['pc_addr'])
            if key not in intervals:
                continue
            start, duration = number(event.get('ts')), number(event.get('dur'))
            if start is None or not duration:
                continue
            intervals[key].append(dict(start_us=start, end_us=start + duration,
                start_fraction=(start - origin) / span, end_fraction=(start + duration - origin) / span,
                lane=names.get(event.get('pid')), name=event.get('name')))
        rows = []
        for key, instruction in selected.items():
            ordered = sorted(intervals[key], key=lambda r: r['start_us'])
            if ordered:
                rows.append(dict(pc=key, instruction=instruction['instruction'], pipeline=instruction['pipeline'],
                    csv_calls=instruction.get('calls'), csv_duration_us=instruction.get('running_time_us'),
                    source_stack=instruction.get('source_stack'), intervals=ordered,
                    matched_source_frames=[dict(frame, resolved_code=code_for_frame(frame))
                        for frame in instruction.get('source_stack', [])
                        if frame.get('file') in source_paths and pattern.search(code_for_frame(frame))]))
        rows.sort(key=lambda r: r['intervals'][0]['start_us'])
        reports.append(dict(core=core['name'], raw_trace_sha256=sha(paths[0]),
            clock_validation=qualified['absolute_unit_validation'], window=window, instructions=rows))
    return dict(source_sha256=binding['source_sha256'], target_binary_sha256=binding['target_binary_sha256'],
        source_pattern=source_pattern, source_mapping_paths=sorted(source_paths), reports=reports,
        scope='Instruction intervals within each verified partial core window. Order comparisons stay within '
              'the same core; independent core origins, overlapping instructions and repeated PCs remain '
              'explicit. CSV instruction sums and trace intervals remain separate. Full-call timing is separate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source-pattern', default=r'\b(?:qk|v|v16_pf|acc)\s*=')
    args = parser.parse_args()
    run = args.run.resolve()
    result = collect(run, args.source_pattern)
    save(run / 'simulator-schedule.json', result)
    print(dict(status='source_mapped_schedule_saved', cores=len(result['reports']),
               pcs=sum(len(r['instructions']) for r in result['reports'])))
