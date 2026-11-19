"""Snapshot validated best candidates without changing exploratory workspaces."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def inspect(run, source):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    folder = run / ('formal-' + digest[:12])
    reasons = []
    result_path = folder / 'evaluation/result.json'
    result = json.loads(result_path.read_bytes()) if result_path.exists() else {}
    if result.get('status') != 'PASSED' or result.get('num_passed') != 24:
        reasons.append('complete formal contract not passed')
    for name in ['supplementary.json', 'cann_supplementary.json']:
        p = folder / name
        v = json.loads(p.read_bytes()) if p.exists() else {}
        if v.get('source_sha256') != digest or v.get('status') != 'passed':
            reasons.append(name + ' missing, mismatched, or failed')
    review = run / 'source-reviews' / (digest + '.json')
    v = json.loads(review.read_bytes()) if review.exists() else {}
    if v.get('source_sha256') != digest or v.get('status') != 'reviewed_strict_triton':
        reasons.append('source path review missing')
    qualification = source.parent / 'source-review-feedback.json'
    if not qualification.exists():
        qualification = run / 'workspace/preflight-qualification.json'
    if qualification.exists():
        q = json.loads(qualification.read_bytes())
        if q.get('source_sha256') == digest and q.get('status') == 'repair_required':
            reasons.append('unresolved source safety review')
    target = next((x for x in result.get('per_workload', [])
                   if x.get('uuid') == 'upsample_bilinear2d-timing-1'), {})
    return dict(source_sha256=digest, source=str(source), eligible=not reasons,
                exclusion_reasons=reasons, geo_mean=result.get('geo_mean'),
                target_speedup=target.get('speedup'), result=str(result_path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    a = p.parse_args()
    report = dict(recorded_at=datetime.now(timezone.utc).isoformat(),
                  scope='Retrospective selection snapshot; does not rewrite earlier controller decisions or exploratory source.',
                  arms={})
    for arm in ['no-simulator', 'with-simulator']:
        run = a.study / arm
        rows = [inspect(run, s) for s in sorted((run / 'agent-calls').glob('*/output-source.py'))]
        eligible = [r for r in rows if r['eligible'] and (r['geo_mean'] or 0) > 0]
        report['arms'][arm] = dict(candidates=rows,
            best_full_contract=max(eligible, key=lambda r: r['geo_mean']) if eligible else None,
            best_target=max(eligible, key=lambda r: r['target_speedup'] or 0) if eligible else None)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest = a.study / 'retention' / (stamp + '.json')
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print(dest)


if __name__ == '__main__':
    main()
