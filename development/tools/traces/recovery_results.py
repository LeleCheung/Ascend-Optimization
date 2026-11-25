"""Render completed recovery measurements with a fixed within-run CANN anchor."""
import html
import json
from pathlib import Path

from .cann_trace import rebase


def main():
    root = Path.cwd()
    evidence = root / 'development/experiments/lineage/20261117/recovery-results-v4'
    out = root / 'development/reports/operator-traces-stitched-20261117'
    records = json.loads((evidence / 'exported-results.json').read_bytes())['rows']
    groups = {}
    for row in records:
        op = row['operator']
        groups.setdefault(op, []).append(row)
    summaries = []
    for op, rows in groups.items():
        manifest = json.loads((root / 'operators' / op / 'manifest.json').read_bytes())
        baseline = next(r for r in rows if r['source_sha256'] == manifest['selected']['sha256'])
        anchor = json.loads((evidence / baseline['directory'] / 'evaluation/result.json').read_bytes())
        values = []
        for row in rows:
            result = evidence / row['directory'] / 'evaluation/result.json'
            conversion = rebase(anchor, json.loads(result.read_bytes())) if row['full_passed'] and result.exists() else None
            values.append(dict(**row, fixed_cann=conversion,
                               candidate_gain=conversion['score'] / baseline['geo_mean'] if conversion else None,
                               baseline=row is baseline))
        summaries.append(dict(operator=op, baseline_sha256=baseline['source_sha256'], measurements=values))
    (out / 'recovery-results.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf8')
    sections = []
    for group in summaries:
        rows = []
        for r in group['measurements']:
            fixed = r['fixed_cann']
            score = f"{fixed['score']:.3f}×" if fixed else 'Failed'
            gain = f"{r['candidate_gain']:.3f}×" if fixed else '—'
            original = f"{r['geo_mean']:.3f}×" if r.get('geo_mean') else '—'
            label = 'Delivered-source repeat' if r['baseline'] else 'Later candidate'
            rows.append(f'<tr><td>{label}</td><td><code>{r["source_sha256"][:12]}</code></td><td>{original}</td><td>{score}</td><td>{gain}</td><td>{'All passed' if r["full_passed"] else 'Failed'}</td></tr>')
        sections.append('<h2>'+html.escape(group['operator'])+'</h2><table><tr><th>Version</th><th>Source</th><th>Paired CANN</th><th>Fixed CANN</th><th>Relative to run start</th><th>Full checks</th></tr>'+''.join(rows)+'</table>')
    page="""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Current recovery/optimization</title>
<style>body{max-width:1200px;margin:32px auto;padding:0 20px;background:#fff7e8;color:#1b2434;font:20px/1.7 "Microsoft YaHei",sans-serif}h1{font-size:34px}h2{font-size:26px;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ded7c9;text-align:left}a{color:#276980}code{font-size:17px}table{display:block;overflow:auto}</style>
<p><a href="index.html">Back to 26-operator traces</a> · <a href="recovery-results.json">Per-case data</a></p><h1>Current recovery/optimization</h1>
<p>Same full contract, 、walltime timing。Fixed CANN column uses starting per-case reference timings；The relative-to-start column compares candidate times directly。This removes reference-denominator changes，while retaining cross-time measurement variation。</p>
<p>Prior delivery remains intact。This page retains all completed formal evaluations，including invalid/negative candidates，same-source repeats are separate from new optimization。</p>"""+''.join(sections)+'</html>'
    (out / 'recovery.html').write_text(page, encoding='utf8')
    print(json.dumps(dict(operators=len(summaries), measurements=sum(len(g['measurements']) for g in summaries))))


if __name__ == '__main__':
    main()
