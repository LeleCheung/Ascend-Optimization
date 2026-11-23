"""Export completed replay evidence with a per-file hash receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert json.loads((run / 'progress.json').read_bytes())['status'] == 'completed'
    manifest = json.loads((run / 'manifest.json').read_bytes())
    paths = [run / name for name in ['manifest.json', 'summary.json', 'service-status.json',
        'service.json', 'controller-process.json', 'progress.json'] if (run / name).exists()]
    for item in manifest['measurements']:
        paths += [run / item['name'] / name for name in ['request.json', 'result.json', 'source.py', 'provenance.json']]
        paths += list((run / item['name']).glob('replay-*.json'))
        paths += list((run / item['name']).glob('replay-*.log'))
        extra = run / item['name'] / 'supplementary-summary.json'
        if extra.exists():
            paths.append(extra)
    receipt = dict(schema='hgkda.presentation-replay-export/v1', run=str(run), files=[
        dict(path=str(path.relative_to(run)), bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths])
    receipt_path = run / 'export-receipt.json'
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    with tarfile.open(args.output, 'w:gz') as archive:
        for path in paths + [receipt_path]:
            archive.add(path, arcname=str(Path(run.name) / path.relative_to(run)), recursive=False)
    print(json.dumps(dict(archive=str(args.output), files=len(paths),
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
