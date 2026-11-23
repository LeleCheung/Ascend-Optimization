"""Freeze an accepted, already-observed GQA version for another independent continuation."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-observation', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    parent, run = args.parent_observation.resolve(), args.run.resolve()
    result = read(parent / 'parent-validation/results.json')
    binding = read(parent / 'binary-binding-and-facts.json')
    digest = sha(args.source)
    assert result['source_sha256'] == binding['source_sha256'] == digest
    assert binding['status'] == 'formal_profile_simulator_binary_verified'
    selected = read(args.selection)
    assert selected['source_sha256'] == digest and selected['target_latency_us'] > 0
    run.mkdir(parents=True, exist_ok=False)
    (run / 'workspace').mkdir()
    shutil.copyfile(args.source, run / 'workspace/candidate.py')
    for name in ['contract.json', 'observation-status.json', 'binary-binding-and-facts.json']:
        shutil.copyfile(parent / name, run / name)
    shutil.copyfile(args.selection, run / 'parent-selection.json')
    shutil.copyfile(parent / 'parent-validation/results.json', run / 'parent-results.json')
    for name in ['parent-validation', 'capture-parent', 'paired-parent', 'simulator-parent']:
        (run / name).symlink_to(parent / name, target_is_directory=True)
    shutil.copytree(parent / 'tools', run / 'tools')
    config = read(args.template / 'config.json')
    for key in ['required_simulator_evidence', 'source_sha256', 'frozen_source']:
        config.pop(key, None)
    config.update(run_directory=str(run), trace_round_id='parent',
                  paired_directory=str(run / 'paired-parent'), agent_tool_path=str(run / 'decision_bridge.py'),
                  active_source='candidate.py', source_frozen=False)
    save(run / 'config.json', config)
    save(run / 'retained-parent.json', dict(status='frozen_accepted_observed_parent', source_sha256=digest,
        original_source=str(args.source.resolve()), selection=str(args.selection.resolve()),
        observation=str(parent), selection_sha256=sha(args.selection), target_latency_us=selected['target_latency_us'],
        created_at=datetime.now(timezone.utc).isoformat()))
    save(run / 'continuation-provenance.json', dict(source_sha256=digest,
        scope='Existing immutable source-bound four-case benchmark and paired/simulator observations reused; '
              'new candidates, evaluations and traces use a separate run. No repeated baseline capture.',
        parent_observation=str(parent), result_sha256=sha(parent / 'parent-validation/results.json'),
        contract_sha256=sha(run / 'contract.json')))
    print(dict(run=str(run), source_sha256=digest, target_us=selected['target_latency_us']))


if __name__ == '__main__':
    main()
