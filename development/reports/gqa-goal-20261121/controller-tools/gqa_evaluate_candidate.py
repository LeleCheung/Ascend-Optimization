"""Run the unchanged four-case GQA contract and replay a host-reviewed candidate."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--simulator-minutes', type=int, default=5,
                        help='Per-capture window; independent of optimization search budget')
    args = parser.parse_args()
    run = args.run.resolve()
    call = run / 'agent-calls' / args.invocation
    assert (call / 'model-usage.json').exists(), 'Wait for model handoff'
    source = call / 'output-source.py'
    assert sha(source) == sha(run / 'workspace/candidate.py')
    config = read(run / 'config.json')
    sys.path.insert(0, str(Path(config['package']) / 'src'))
    from hgkda.gate import admit
    review = run / 'source-reviews' / (sha(source) + '.json')
    admit(source.read_text(encoding='utf8'), read(review))
    assert read(call / 'host-review.json')['status'] == 'preedit_reads_and_semantics_approved'
    output = run / 'iterations' / args.invocation
    output.mkdir(parents=True, exist_ok=False)
    (output / 'workspace').mkdir()
    shutil.copyfile(source, output / 'workspace/candidate.py')
    shutil.copyfile(run / 'contract.json', output / 'contract.json')
    shutil.copytree(run / 'tools', output / 'tools')
    config.update(run_directory=str(output), trace_round_id=args.invocation)
    save(output / 'config.json', config)
    save(output / 'retained-parent.json', dict(source_sha256=sha(source),
        parent_source_sha256=sha(call / 'input-source.py'), author=str(call),
        source_review_sha256=sha(review), status='host_reviewed_candidate_pending_device_validation'))
    command = [sys.executable, str(Path(__file__).with_name('gqa_continue_observe.py')),
        '--run', str(output), '--simulator-minutes', str(args.simulator_minutes)]
    with (output / 'driver.log').open('x') as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    save(output / 'driver-execution.json', dict(command=command, returncode=process.returncode))
    result_path = output / 'parent-validation/results.json'
    result = read(result_path) if result_path.exists() else {}
    rows = result.get('cases', [])
    qualified = len(rows) == 4 and {r['case_id'] for r in rows} == {1, 2, 4, 7} and all(
        r['status'] == 'measured' and r.get('candidate_correctness', {}).get('official_checker', {}).get('passed')
        and r.get('native_correctness', {}).get('official_checker', {}).get('passed') for r in rows)
    parent = read(run / 'parent-validation/results.json')
    feedback = dict(source_sha256=sha(source), parent_source_sha256=sha(call / 'input-source.py'),
        qualified=qualified, result=str(result_path), postcapture_returncode=process.returncode,
        scope='Fixed official four-case subset; original contract and historical measurements preserved.', cases=[])
    for row in rows:
        original = next(r for r in parent['cases'] if r['case_id'] == row['case_id'])
        latency = row.get('candidate_timing', {}).get('full_call_us')
        old = original['candidate_timing']['full_call_us']
        feedback['cases'].append(dict(case_id=row['case_id'], parent_us=old, candidate_us=latency,
            parent_over_candidate=old / latency if latency else None,
            official_checker=row.get('candidate_correctness', {}).get('official_checker'),
            error=row.get('candidate_error', row.get('error'))))
    save(output / 'feedback.json', feedback)
    print(feedback)


if __name__ == '__main__':
    main()
