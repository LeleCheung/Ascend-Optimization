"""Run unchanged published checkers for previously unreviewed replay sources."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.study / 'config.json').read_bytes())
    sys.path.insert(0, str(Path(config['package']) / 'src'))
    from hgkda.runtime.evaluate import check
    manifest = json.loads((args.run / 'manifest.json').read_bytes())
    for item in manifest['measurements']:
        if '-history-' not in item['name']:
            continue
        directory = args.run / item['name']
        directory.mkdir(exist_ok=True)
        source = Path(item['source'])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
        results = {}
        with Path(config['lock']).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            specs = dict(config['job']['checks'], **config.get('special_checks', {}))
            for name, spec in specs.items():
                output = directory / ('replay-' + name + '.json')
                if output.exists():
                    result = json.loads(output.read_bytes())
                    assert result['source_sha256'] == item['sha256']
                    results[name] = dict(status='passed' if result['passed'] == result['total'] else 'failed',
                        passed=result['passed'], total=result['total'])
                else:
                    results[name] = check(config, name, spec, source, output)
        (directory / 'supplementary-summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(dict(name=item['name'], checks=results)), flush=True)


if __name__ == '__main__':
    main()
