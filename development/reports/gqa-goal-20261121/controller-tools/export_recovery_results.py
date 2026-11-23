"""Export completed recovery evidence, excluding credentials and runtime caches."""
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    root = Path('/data/hanle/ascend-optimization/runs/midday-recovery-walltime-20261117')
    target = root.parent / 'midday-recovery-results-20261117.tar.gz'
    selected = set()
    rows = []
    for audit in root.glob('*/formal-*/audit-final.json'):
        folder = audit.parent
        row = json.loads(audit.read_bytes())['rows'][0]
        row = dict(row, run=folder.parent.name, directory=str(folder.relative_to(root)))
        rows.append(row)
        for name in ('candidate.py', 'audit-final.json', 'review.json', 'request.json', 'supplementary.json',
                     'cann_supplementary.json', 'extra.json', 'evaluation/result.json'):
            path = folder / name
            if path.is_file():
                selected.add(path)
    for name in {row['run'] for row in rows}:
        run = root / name
        for pattern in ('trace/decisions/*.json', 'trace/events/*.json'):
            selected.update(path for path in run.glob(pattern) if path.is_file())
        for call in (run/'agent-calls').glob('*'):
            output=call/'output-source.py';provenance=call/'source-provenance.json'
            if not output.is_file() or not provenance.is_file():
                continue
            identity=json.loads(provenance.read_bytes())
            if identity.get('output_sha256')!=hashlib.sha256(output.read_bytes()).hexdigest():
                continue
            selected.update(p for p in (call/'input-source.py',output,provenance) if p.is_file())
    files = [dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(selected)]
    manifest = root / 'exported-results.json'
    manifest.write_text(json.dumps(dict(rows=rows, files=files), indent=2) + '\n')
    with tarfile.open(target, 'w:gz') as archive:
        for path in sorted(selected):
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        archive.add(manifest, arcname=manifest.name, recursive=False)
    print(json.dumps(dict(archive=str(target), files=len(files), completed=len(rows))))


if __name__ == '__main__':
    main()
