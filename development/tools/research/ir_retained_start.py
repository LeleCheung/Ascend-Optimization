"""Prepare an IR study from retained-source observations, then hand off to CC."""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys

from ir_campaign import capture, evaluate, read, save, seed, sha
from ir_initialize import verify_noop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--model-settings', type=Path, required=True)
    args = p.parse_args()
    parent, run = args.parent.resolve(), args.run.resolve()
    assert not run.exists()
    observation = read(parent / 'observation-status.json')
    assert observation['status'] == 'parent_dual_evidence_ready'
    digest = observation['source_sha256']
    workspace = run / 'workspace'
    workspace.mkdir(parents=True)
    config = copy.deepcopy(read(parent / 'config.json'))
    config.update(run_directory=str(run), source_sha256=digest, frozen_source=str(workspace / 'frozen.py'),
        selected_layer='single_operator_ir', active_source='frozen.py', source_frozen=True,
        policy='performance_first', observation_parent=str(parent), parent_directory=str(parent), tools=str(run.parent / 'tools'))
    save(run / 'config.json', config)
    shutil.copyfile(parent / 'workspace/candidate.py', workspace / 'frozen.py')
    assert sha(workspace / 'frozen.py') == digest
    shutil.copyfile(args.model_settings, run / 'model-settings.json')
    shutil.copyfile(parent / 'validation-policy.json', run / 'validation-policy.json')
    shutil.copyfile(parent / 'validation-policy.json', workspace / 'validation-policy.json')
    shutil.copyfile(parent / 'workspace/contract.json', workspace / 'contract.json')
    for name, original in [('profiling.json', parent / 'paired-parent/comparison.json'),
            ('hotspots.json', parent / 'simulator-parent/ir-summary.json')]:
        shutil.copyfile(original, workspace / name)
    summary = read(workspace / 'hotspots.json')
    assert summary['source_sha256'] == digest
    save(workspace / 'timeline.json', dict(source_sha256=digest,
        reports=[dict(core=row['core'], timeline=row['timeline'],
            absolute_unit_validation=row['absolute_unit_validation']) for row in summary['reports']],
        original=str(parent / 'simulator-parent/ir-summary.json')))
    save(run / 'retained-parent.json', read(parent / 'retained-parent.json'))
    capture(run)
    seed(run)
    save(workspace / 'context.json', dict(operator=config['operator'], source_sha256=digest,
        target_case=config['experiment_goal']['target_case'], candidate_layer='TTIR', source_edit_allowed=False,
        comparator='Compiler-original IR of the exact retained source; current complete baseline follows',
        candidate_author='CC/DeepSeek', diagnosis_author='Codex',
        evidence_scope=summary['scope'], application_completed=summary['application_completed']))
    save(workspace / 'codex-brief.json', dict(role='Evidence analysis; model chooses and authors every candidate edit',
        findings=['Selected current-source profiling and complete-core simulator observations are supplied.',
            'Inspect the relation between MTE2/MTE3 synchronization, vector waits and loop/copy organization.',
            'Use actual backend artifacts to distinguish work already optimized from a remaining IR opportunity.'],
        direction='Investigate a concrete loop,buffer,layout or synchronization transformation; preserve arithmetic order where feasible. Choose after reading the evidence.',
        constraints=['Read the whole source and actual TTIR/backend mapping before choosing a transformation.',
            'Preserve ABI,strict Triton semantics,special values and the frozen full contract and supplements.',
            'Treat bandwidth,wave and critical-path claims as hypotheses unless independently supported.',
            'Source may express the same intent;this study isolates a fixed-source compiler transformation.',
            'Explicit no-change is valid if the backend has already removed the suspected work.'],
        validation='Fixed complete contract and existing supplements;conditional ABBA;post simulator replay'))
    evaluate(run, 'unchanged-ir')
    result = read(run / 'evaluations/unchanged-ir/result.json')
    assert result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
    verify_noop(run)
    target = next(row for row in result['per_workload']
        if row['uuid'] == config['experiment_goal']['target_case'] and row['phase'] == 'timing')
    save(workspace / 'feedback.json', dict(current=result, target=target,
        original_acceptance=read(run / 'retained-parent.json'),
        scope='Same-source current complete baseline is raw detail;no plotted remeasurement node'))
    invocation = 'ir-001'
    command = [sys.executable, str(Path(config['tools']) / 'ir_model.py'),
        '--run', str(run), '--invocation', invocation, 'generate']
    with (run / 'generation-ir-001.log').open('x') as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    save(run / 'driver-handoff.json', dict(status='host_review_required', invocation=invocation,
        model_returncode=process.returncode, model_status=read(run / 'status.json')))


if __name__ == '__main__':
    main()
