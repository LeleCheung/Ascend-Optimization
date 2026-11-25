"""Inventory unresolved lineage and declining points without changing history."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = Path('development/experiments/lineage/20261117')
    source_map = {}
    for directory in ('source-evidence', 'links-v3'):
        for path in (evidence / directory).rglob('*.py'):
            source_map[hashlib.sha256(path.read_bytes()).hexdigest()] = path
    for path in Path('operators').rglob('*.py'):
        source_map[hashlib.sha256(path.read_bytes()).hexdigest()] = path
    rows = []
    for row in read(args.view)['operators']:
        events = row['display_events']
        byid = {event['id']: event for event in events}
        gaps = []
        for connection in row['connections']:
            if connection['kind'] != 'unresolved':
                continue
            start, end = byid[connection['start']], byid[connection['end']]
            gaps.append(dict(start=start, end=end))
        declines = []
        for start, end in zip(events, events[1:]):
            if start['score'] is None or end['score'] >= start['score']:
                continue
            paths = [source_map.get(e['source_sha256']) for e in (start, end)]
            item = dict(start=start, end=end, historical_score_drop=1-end['score']/start['score'],
                        source_paths=[str(p) if p else None for p in paths], cause='pending')
            if all(paths):
                before, after = [p.read_text(encoding='utf-8-sig') for p in paths]
                item['ast_equal'] = ast.dump(ast.parse(before)) == ast.dump(ast.parse(after))
                diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                   fromfile=str(paths[0]), tofile=str(paths[1])))
                dest = args.output / (row['operator'] + '-decline.diff')
                dest.write_text(diff, encoding='utf8')
                item['diff'] = dest.name
            declines.append(item)
        rows.append(dict(operator=row['operator'], gaps=gaps, declines=declines))
    (args.output/'inventory.json').write_text(json.dumps(dict(operators=rows), ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    text = ['# Trace continuation and performance recovery', '',
            'Declines indicate fixed-reference historical timing differences, not established same-environment code regressions. Keep fresh experiments separate.', '',
            '| Operator | Missing connections | Declining segment |', '| --- | ---: | --- |']
    for row in rows:
        values = '；'.join(f"{d['start']['score']:.4f} → {d['end']['score']:.4f}×" for d in row['declines']) or 'None'
        text.append(f"| {row['operator']} | {len(row['gaps'])} | {values} |")
    (args.output/'README.md').write_text('\n'.join(text)+'\n', encoding='utf8')
    print(json.dumps(dict(operators=len(rows), unresolved_connections=sum(len(r['gaps']) for r in rows),
                          declining_operators=sum(bool(r['declines']) for r in rows))))


if __name__ == '__main__':
    main()
