"""Export immutable IR case evidence, with hashes and remote artifact locations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--stability', type=Path)
    parser.add_argument('--additional-stability', type=Path, action='append', default=[])
    parser.add_argument('--compilation-only', action='store_true',
                        help='Freeze model and CPU evidence while device evaluations continue')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.compilation_only or not (args.stability or args.additional_stability)
    run = args.run.resolve()
    campaign = run.parent
    selected = set()
    external_methods = {}

    def add(path):
        path = path.resolve()
        assert path.is_relative_to(campaign), path
        if path.is_file():
            selected.add(path)

    add(run / 'config.json')
    for name in ['original-ir-verification.json', 'selection.json',
                 'continuation-provenance.json', 'retained-baseline-verification.json', 'validation-policy.json']:
        add(run / name)
    if (run / 'selection.json').exists():
        selection = json.loads((run / 'selection.json').read_bytes())
        for link in selection.get('ir_lineage', []):
            for field in ['decision', 'submission']:
                path = (run / link[field]).resolve()
                assert path.is_relative_to(run)
                add(path)
        if selection.get('historical_measurements'):
            binding = selection['historical_measurements']
            review = (run / binding['path']).resolve()
            assert review.is_relative_to(run) and digest(review) == binding['sha256']
            add(review)
    for pattern in ['*-binary-comparison.json', 'provisional-gain-*.json', 'post-mechanism-*.json',
                    'observation-binding-*.json', '*-binary-comparison-scope.json',
                    'retention-*.json', 'completion-*.json', 'stability-screen-*.json',
                    'stability-confirmation*.json', 'workspace-restore-before-*.json',
                    'presentation-metadata*.json', 'exploration-parent-*.json',
                    'numeric-repair-preparation-*.json']:
        for path in run.glob(pattern):
            add(path)
    for path in (run / 'parent-inputs').rglob('*'):
        add(path)
    call = run / 'agent-calls' / args.invocation
    input_manifest = json.loads((call / 'input-manifest.json').read_bytes())
    core_inputs = {'frozen.py', 'context.json', 'profiling.json', 'hotspots.json',
                   'timeline.json', 'ir-inventory.json', 'lowered-inventory.json'}
    verified_inputs = []
    for name, expected in input_manifest.items():
        if name not in core_inputs and not name.startswith('compiled-readonly/'):
            continue
        snapshot = call / 'input-workspace' / name
        path = snapshot if snapshot.is_file() else run / 'workspace' / name
        assert digest(path) == expected, 'Logged pre-edit evidence changed: ' + name
        add(path)
        verified_inputs.append(dict(path=name, sha256=expected))
    for path in call.rglob('*'):
        if path.is_file() and 'cache' not in path.relative_to(call).parts:
            add(path)
    for name in ([] if args.compilation_only else ['frozen-baseline', 'unchanged-ir', args.invocation]):
        evaluation = run / 'evaluations' / name
        for path in evaluation.glob('*.json'):
            add(path)
        for path in (evaluation / 'implementation').rglob('*'):
            add(path)
        for path in (evaluation / 'compiler-receipts').glob('*.jsonl'):
            add(path)
    for name in ([] if args.compilation_only else ['unchanged-ir', args.invocation]):
        observation = run / 'observations' / name
        for pattern in ['*.json', 'paired/*.json', 'simulator/*.json',
                        'paired/implementation/ir-manifest.json',
                        'paired/implementation/compiler-receipts/*.jsonl',
                        'simulator/launch-replay/manifest.json',
                        'simulator/implementation/ir-manifest.json',
                        'simulator/implementation/compiler-receipts/*.jsonl',
                        'resumes/*/*.json', 'resumes/*/*.log',
                        'simulator-retries/*/*.json', 'simulator-retries/*/*.log',
                        'isolated-replays/*/*.json', 'isolated-replays/*/*.log',
                        'isolated-replays/*/launch-snapshot/manifest.json']:
            for path in observation.glob(pattern):
                add(path)
        config_path = observation / 'config.json'
        if config_path.exists():
            observation_config = json.loads(config_path.read_bytes())
            package = Path(observation_config['package'])
            package_manifest = package / 'method-manifest.json'
            if not package_manifest.exists():
                package_manifest = Path(observation_config['method_manifest'])
            package_id = digest(package_manifest)
            support_files = [package_manifest, *(package / 'src').rglob('*.py')]
            for path in support_files:
                if '__pycache__' in path.parts:
                    continue
                if path.resolve().is_relative_to(campaign):
                    add(path)
                else:
                    relative = path.relative_to(package).as_posix()
                    external_methods['observation-packages/' + package_id + '/' + relative] = path
    stability_roots = ([args.stability] if args.stability else []) + args.additional_stability
    for stability_root in stability_roots:
        stability = stability_root.resolve()
        assert stability.is_relative_to(run)
        for pattern in ['*.json', '*/*.json', '*/process.log', '*/compiler-receipts/*.jsonl']:
            for path in stability.glob(pattern):
                add(path)

    # Export exact artifacts referenced by compiler receipts under content hashes.
    artifacts = {}
    for path in sorted(selected):
        if path.suffix != '.jsonl' or path.parent.name != 'compiler-receipts':
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            for item in json.loads(line).get('artifacts', []):
                original = Path(item['path'])
                assert original.is_relative_to(campaign) and digest(original) == item['sha256']
                name = 'compiler-artifacts/' + item['sha256'] + '.' + item['kind']
                artifacts[name] = dict(path=original, sha256=item['sha256'], kind=item['kind'])
    entries = [(path, path.relative_to(campaign).as_posix()) for path in sorted(selected)]
    entries += [(row['path'], name) for name, row in sorted(artifacts.items())]
    entries += [(path, name) for name, path in sorted(external_methods.items())]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'x:gz') as archive:
        for path, name in entries:
            archive.add(path, arcname=name, recursive=False)
    receipt = dict(remote_root=str(campaign), timestamp=datetime.now(timezone.utc).isoformat(),
        exporter_sha256=digest(__file__),
        compilation_only=args.compilation_only,
        stability_roots=[str(path.resolve()) for path in stability_roots],
        archive=args.output.name, archive_sha256=digest(args.output),
        preedit_core_inputs_verified_against_logged_manifest=verified_inputs,
        files=[dict(path=name, sha256=digest(path), bytes=path.stat().st_size,
                    remote_original=str(path)) for path, name in entries],
        scope=('Registered model call, frozen input snapshots and CPU compiler artifacts; '
               'live device evaluations and observations are excluded.' if args.compilation_only else
               'Registered model call, formal evaluations, exact compiler artifacts and observation summaries; '
               'raw profiling, simulator traces and large diagnostic replay tensors remain at the recorded remote paths.'))
    receipt_path = args.output.with_name(args.output.name + '.receipt.json')
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(archive=str(args.output), receipt=str(receipt_path), files=len(entries),
                         bytes=args.output.stat().st_size)))


if __name__ == '__main__':
    main()
