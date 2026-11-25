"""Freeze actual compilation/original-IR requests before parallel model IR diagnosis."""
import argparse
from pathlib import Path
import subprocess
import sys

from ir_campaign import capture, implementation, read, save, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', default='ir-001')
    args = parser.parse_args()
    run = args.run.resolve()
    assert not (run / 'agent-calls' / args.invocation).exists()
    save(run / 'bootstrap-status.json', dict(status='capturing_actual_best_ir'))
    manifest = run / 'baseline-capture/manifest.json'
    if not manifest.exists():
        capture(run)
    assert read(manifest)['status'] == 'captured'
    if not (run / 'workspace/ir-inventory.json').exists():
        seed(run)
    if not (run / 'evaluations/unchanged-ir/request.json').exists():
        implementation(run, 'unchanged-ir', original_ir=True)
    assert read(run / 'evaluations/unchanged-ir/provenance.json')['changed_ir_count'] == 0
    brief = run / 'workspace/codex-brief.json'
    value = read(brief)
    value['authoring_schedule'] = dict(actual_ir_captured=True, original_ir_request_frozen=True,
        current_frozen_source_formal_status='pending', current_original_ir_formal_status='pending',
        source_start='Preserved current fully qualified best; historical full validation remains in context',
        policy='Authoring may overlap other operators; reviewed modified formal evaluation requires verified current baselines')
    save(brief, value)
    save(run / 'bootstrap-status.json', dict(status='authoring_ir', invocation=args.invocation))
    command = [sys.executable, str(Path(__file__).with_name('ir_model.py')), '--run', str(run),
        '--invocation', args.invocation, 'generate']
    with (run / ('generation-' + args.invocation + '.log')).open('x') as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    save(run / 'bootstrap-status.json', dict(status='authoring_finished', returncode=process.returncode,
        invocation=args.invocation, model_status=read(run / 'status.json')))


if __name__ == '__main__':
    main()
