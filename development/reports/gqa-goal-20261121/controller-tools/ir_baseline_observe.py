"""Bind fresh retained-baseline observations before the first model call."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    assert not (run / 'agent-calls').exists(), 'Refresh initial evidence before authoring starts'
    verification = run / 'retained-baseline-verification.json'
    while not verification.exists():
        time.sleep(10)
    record = read(verification)
    assert record['status'] == 'verified'
    assert sha(run / 'evaluations/unchanged-ir/result.json') == record['result_sha256']
    collector = Path(__file__).with_name('ir_collect_v3.py')
    command = [sys.executable, str(collector), '--run', str(run), '--invocation', 'unchanged-ir',
               '--package', str(args.package)]
    with (run / 'baseline-collection.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    observation = run / 'observations/unchanged-ir'
    assert read(observation / 'status.json')['status'] == 'ir_evidence_collected'
    summary = read(observation / 'simulator/ir-summary.json')
    config = read(run / 'config.json')
    assert summary['source_sha256'] == config['source_sha256']
    workspace = run / 'workspace'
    shutil.copyfile(observation / 'paired/comparison.json', workspace / 'profiling.json')
    shutil.copyfile(observation / 'simulator/ir-summary.json', workspace / 'hotspots.json')
    save(workspace / 'timeline.json', dict(source_sha256=config['source_sha256'],
         reports=[dict(core=r['core'], timeline=r['timeline'], absolute_unit_validation=r['absolute_unit_validation'])
                  for r in summary['reports']],
         original=str(observation / 'simulator/ir-summary.json'),
         original_sha256=sha(observation / 'simulator/ir-summary.json')))
    context = read(workspace / 'context.json')
    context.update(evidence_ready=True, baseline=record,
                   evidence_inputs=[dict(path=name, sha256=sha(workspace / name)) for name in
                                    ['profiling.json', 'hotspots.json', 'timeline.json']])
    save(workspace / 'context.json', context)
    previous = Path(config['continuation_parent'])
    history = []
    for call in sorted((previous / 'agent-calls').iterdir()):
        diagnosis = call / 'output-diagnosis.json'
        if not diagnosis.exists():
            continue
        result = previous / 'evaluations' / call.name / 'result.json'
        history.append(dict(invocation=call.name, diagnosis=read(diagnosis),
                            result=read(result) if result.exists() else None))
    save(workspace / 'prior-explorations.json', history)
    with (run / 'generation-ir-001.log').open('x') as log:
        subprocess.run([sys.executable, str(Path(__file__).with_name('ir_model.py')),
            '--run', str(run), '--invocation', 'ir-001', 'generate'],
            stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
