"""Export GQA decisions, evaluations and bound observations, leaving raw traces remote."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--through', help='Only completed invocations up to this name')
    parser.add_argument('--after', help='Incremental export after this invocation; existing history stays local')
    parser.add_argument('--ir-branch', action='append', default=[],
                        help='Include this named IR branch during incremental export')
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    run = campaign / 'gqa'
    paths = set()
    additional = {}
    read_parent = run / 'continuation-provenance.json'
    allowed_parent = (Path(json.loads(read_parent.read_bytes())['parent_observation']).resolve()
                      if read_parent.exists() else None)

    def add(path):
        if path.is_file() and not any('settings' in part.lower() for part in path.parts):
            assert path.is_relative_to(campaign)
            assert path.resolve().is_relative_to(campaign) or (
                allowed_parent and path.resolve().is_relative_to(allowed_parent))
            paths.add(path)

    for path in (campaign / 'controller-tools').glob('*.py'):
        add(path)
    for path in campaign.glob('observation-method-v*/src/hgkda/runtime/launch_replay.py'):
        add(path)

    for name in ['contract.json', 'retained-parent.json', 'prior-continuation.json', 'continuation-provenance.json',
                 'selection.json', 'retained-current.json', 'retained-implementation.json',
                 'parent-selection.json', 'host-analysis.json',
                 'binary-binding-and-facts.json', 'observation-status.json']:
        add(run / name)
    for folder in ['trace', 'source-reviews']:
        for path in (run / folder).rglob('*'):
            add(path)
    for path in (run / 'host-timing-diagnostic').rglob('*'):
        if 'cache' not in path.relative_to(run / 'host-timing-diagnostic').parts:
            add(path)
    # Export the successful direct pass capture and selected unchanged-stage probes.
    # Failed wrapper exceptions can contain megabytes of repeated diagnostics.
    for directory in (run / 'lowered-ir-analysis').glob('*-cli'):
        for path in directory.rglob('*'):
            if path.suffix in {'.json', '.mlir', '.txt', '.o', '.so'}:
                add(path)
    for call in (run / 'agent-calls').iterdir():
        if args.through and call.name > args.through:
            continue
        if args.after and call.name <= args.after:
            continue
        for name in ['input-source.py', 'output-source.py', 'source.diff', 'source-provenance.json',
                     'method-manifest.json', 'prompt.md', 'agent-events.jsonl', 'model-usage.json',
                     'common-review.json', 'host-review.json', 'input-diagnosis.json']:
            add(call / name)
    for observation in [run, *(run / 'iterations').iterdir()]:
        if args.through and observation != run and observation.name > args.through:
            continue
        if args.after and observation != run and observation.name <= args.after:
            continue
        for name in ['contract.json', 'feedback.json', 'retained-parent.json', 'driver-execution.json',
                     'observation-status.json', 'binary-binding-and-facts.json', 'simulator-schedule.json',
                     'auxiliary-profiling.json', 'later-stage-replay.json']:
            add(observation / name)
        for path in (observation / 'tools').glob('*.py'):
            add(path)
        for path in observation.glob('simulator-*-schedule.json'):
            add(path)
        for path in observation.glob('*-mechanism.json'):
            add(path)
        for folder in ['parent-validation', 'capture-parent/candidate', 'capture-parent/reference',
                       'paired-parent', 'simulator-parent']:
            for path in (observation / folder).glob('*.json'):
                add(path)
        for path in (observation / 'capture-parent').glob('*/actual-ir/*'):
            add(path)
        for path in (observation / 'paired-parent').glob('*/profile/*.csv'):
            add(path)
        for path in (observation / 'capture-auxiliary').glob('*/actual-ir/*'):
            add(path)
        add(observation / 'target-launch-snapshot/actual-ir/launch-snapshot/manifest.json')
        # Failed compilations still carry the actual source-to-IR lowering.
        for path in (observation / 'parent-validation/cache').rglob('*'):
            if path.suffix in {'.ttir', '.bcmlir', '.ttadapter'}:
                add(path)
        binding = observation / 'binary-binding-and-facts.json'
        if binding.exists():
            facts = json.loads(binding.read_bytes())
            for binary_row in facts.get('all_profile_binaries', []):
                add(observation / 'capture-parent/candidate/actual-ir' / binary_row['profile_path'])
            binary = Path(facts['simulator_cache_binaries'][0]['path'])
            if binary.is_relative_to(campaign):
                add(binary)
            else:
                assert allowed_parent and binary.resolve().is_relative_to(allowed_parent)
                additional['parent-observation-binaries/' + facts['target_binary_sha256'] + '.npubin'] = binary
    for name in ['response.md', 'model-usage.json', 'agent-events.jsonl']:
        add(campaign / 'model-result-review' / name)
    for folder in ['conditional-stability', 'conditional-remeasurement', 'evidence-history']:
        for invocation in (run / folder).glob('*'):
            if args.through and invocation.name > args.through:
                continue
            if args.after and invocation.name <= args.after:
                continue
            for path in invocation.rglob('*'):
                if 'cache' not in path.relative_to(invocation).parts:
                    add(path)
    for pattern in ['retention-*.json', 'restore-before-*.json', 'restoration-*.json',
                    'host-analysis-*.json', 'measurement-policy-*.json',
                    'compiler-option-*.json', 'unmeasured-*.json']:
        for path in run.glob(pattern):
            add(path)
    for path in (run / 'normalization-range-diagnostic').glob('*.json'):
        add(path)
    for path in (run / 'native-reciprocal-research').glob('*'):
        if path.suffix in {'.json', '.txt'}:
            add(path)
    for path in (run / 'workspace/failed-ir').rglob('*'):
        add(path)
    # Independent manual-IR compile-repair research preserves its own identity.
    # Keep source, edits, receipts and observations; omit settings, raw dumps and
    # tensor snapshots, just as in the source campaign above.
    ir_extensions = {'.py', '.json', '.jsonl', '.md', '.txt', '.ttir', '.ttadapter', '.bcmlir', '.npubin', '.mlir', '.o',
                     '.cpp', '.h', '.td'}
    for ir_branch in campaign.glob('gqa-ir-*'):
        if args.after and ir_branch.name not in args.ir_branch:
            continue
        if not ir_branch.is_dir():
            continue
        for path in ir_branch.rglob('*'):
            relative = path.relative_to(ir_branch)
            if (path.is_file() and path.suffix in ir_extensions
                    and not any(part in {'raw', '__pycache__', 'target-launch-snapshot'} for part in relative.parts)
                    and not any('settings' in part.lower() for part in relative.parts)):
                add(path)
    for path in run.glob('host-execution-review-*.json'):
        add(path)
    for pattern in ['host-analysis-ir-*.json', '*language-evidence.txt']:
        for path in campaign.glob(pattern):
            add(path)
    receipt = dict(remote_root=str(campaign), files=[str(p.relative_to(campaign)) for p in sorted(paths)],
                   incremental_after=args.after,
                   additional_external_parent_artifacts={k: str(v) for k, v in additional.items()},
                   scope='Decisions, model traces, source snapshots, actual IR and target binary, four-case '
                         'evaluations, profile tables and simulator summaries. Large raw simulator traces '
                         'remain in their original remote run directories; model settings are excluded.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'x:gz') as archive:
        for path in sorted(paths):
            archive.add(path, arcname=path.relative_to(campaign).as_posix(), recursive=False)
        for name, path in additional.items():
            archive.add(path, arcname=name, recursive=False)
    # Hash the exact archived bytes, including snapshots of active trace indexes.
    # This also permits verification after transfer without rereading live files.
    with tarfile.open(args.output, 'r:gz') as archive:
        receipt['archived_file_sha256'] = {
            member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            for member in archive.getmembers() if member.isfile()}
    receipt['archive_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.manifest.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(files=len(paths), archive=str(args.output), bytes=args.output.stat().st_size)))


if __name__ == '__main__':
    main()
