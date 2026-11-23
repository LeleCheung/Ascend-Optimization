"""Remeasure two already qualified sources with the unchanged full KGS contract."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from hgkda.adapters.kgs import evaluate, upload_bundle


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    read = lambda path: json.loads(path.read_bytes())
    sha = lambda data: hashlib.sha256(data).hexdigest()
    config = read(a.run / 'config.json')
    a.output.mkdir(parents=True, exist_ok=False)
    records = []
    comparison_contract = None
    for label, digest in [('before', a.before), ('after', a.after)]:
        formal = a.run / ('formal-' + digest[:12])
        audit = read(formal / 'audit-final.json')['rows'][0]
        assert audit['source_sha256'] == digest and audit['full_passed']
        assert all(c['status'] == 'passed' for c in audit['checks'].values())
        source = (formal / 'candidate.py').read_bytes()
        request = read(formal / 'request.json')
        contract = (request['binding'], request['settings'])
        if comparison_contract is None:
            comparison_contract = contract
        else:
            assert contract == comparison_contract, 'Remeasurement contracts differ'
        assert sha(source) == digest
        assert sha(request['implementation']['sources'][0]['content'].encode()) == digest
        archive = Path(config['job']['archive'])
        assert request['binding']['bundle_id'] == 'sha256:' + sha(archive.read_bytes())
        with Path(config['lock']).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            upload_bundle(config['kgs_server'], archive)
            result = evaluate(config['kgs_server'], formal / 'request.json',
                              formal / 'review.json', a.output / label)['evaluation']
        original = read(formal / 'evaluation/result.json')
        expected = {(x['uuid'], x['phase']) for x in original['per_workload']}
        observed = [(x['uuid'], x['phase']) for x in result['per_workload']]
        assert len(observed) == len(expected) and set(observed) == expected
        assert result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
        records.append(dict(label=label, source_sha256=digest,
                            settings=request['settings'], result=result))
        (a.output / 'progress.json').write_text(json.dumps(records, indent=2) + '\n')
    summary = dict(scope='One fresh full-contract remeasurement per unchanged qualified source, '
                   'same configured KGS service and settings. Original supplementary checks '
                   'remain bound to the exact sources. Results preserve timing variability; '
                   'this check does not automatically change retained selection.', records=records)
    (a.output / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(status='completed', output=str(a.output))))


if __name__ == '__main__':
    main()
