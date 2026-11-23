"""Author an isolated hypothesis while the identical-source no-op is evaluated."""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys

from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', default='ir-001')
    args = parser.parse_args()
    parent, run = args.parent.resolve(), args.run.resolve()
    assert not (parent / 'agent-calls').exists()
    assert not run.exists()
    config = copy.deepcopy(read(parent / 'config.json'))
    accepted = read(parent / 'retained-parent.json')
    original = read(Path(accepted['evaluation']))
    assert original['status'] == 'PASSED' and original['num_passed'] == original['num_workloads']
    assert sha(parent / 'workspace/frozen.py') == sha(Path(accepted['original_source'])) == config['source_sha256']
    assert read(parent / 'baseline-capture/manifest.json')['status'] == 'captured'
    assert (parent / 'evaluations/unchanged-ir/request.json').is_file()
    shutil.copytree(parent / 'workspace', run / 'workspace')
    for name in ['model-settings.json', 'validation-policy.json', 'retained-parent.json']:
        shutil.copyfile(parent / name, run / name)
    config.update(run_directory=str(run), frozen_source=str(run / 'workspace/frozen.py'),
        comparator_run=str(parent), branch_kind='isolated_hypothesis_during_original_ir_validation')
    save(run / 'config.json', config)
    (run / 'evaluations').mkdir()
    (run / 'evaluations/unchanged-ir').symlink_to(parent / 'evaluations/unchanged-ir', target_is_directory=True)
    context = read(run / 'workspace/context.json')
    context.update(original_ir_validation_run=str(parent), original_ir_validation_pending=True,
        comparator='Accepted identical retained source; current original-IR full evaluation must pass before candidate evaluation')
    save(run / 'workspace/context.json', context)
    save(run / 'workspace/feedback.json', dict(original_acceptance=original,
        current_original_ir_validation='RUNNING in the linked comparator run; no candidate device evaluation until verified',
        current_source_profiling=str(run / 'workspace/profiling.json')))
    save(parent / ('generation-handoff-' + args.invocation + '.json'), dict(run=str(run), invocation=args.invocation,
        reason='Separate immutable model workspace from the live comparator evaluation and its later feedback'))
    with (run / ('generation-' + args.invocation + '.log')).open('x') as stream:
        result = subprocess.run([sys.executable, str(Path(config['tools']) / 'ir_model.py'), '--run', str(run),
            '--invocation', args.invocation, 'generate'], stdout=stream, stderr=subprocess.STDOUT)
    save(run / 'driver-handoff.json', dict(status='host_review_required', invocation=args.invocation,
        returncode=result.returncode, original_ir_validation=str(parent), model_status=read(run / 'status.json')))


if __name__ == '__main__':
    main()
