"""Provide frozen GQA evidence to the existing unbounded model runner."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ir_campaign import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--host-analysis', type=Path)
    parser.add_argument('--observation', type=Path, help='Immutable qualified observation for a later call')
    parser.add_argument('--compact-evidence', action='store_true')
    parser.add_argument('--evidence-mode', choices=['simulator', 'profiling-source-ir'], default='simulator')
    args = parser.parse_args()
    run, template = args.run.resolve(), args.template.resolve()
    observation = args.observation.resolve() if args.observation else run
    workspace = run / 'workspace'
    source = workspace / 'candidate.py'
    digest = sha(source)
    state = read(observation / 'observation-status.json')
    include_simulator = args.evidence_mode == 'simulator'
    if state['status'] == 'parent_dual_evidence_ready':
        assert state['source_sha256'] == digest
    else:
        # Generation using complete hardware/IR evidence may overlap its parent's
        # immutable post replay. Retention still requires the replay binding.
        assert not include_simulator
        assert state['status'] in {'collecting_pair', 'collecting_simulator'}
        assert sha(observation / 'workspace/candidate.py') == digest
        formal = read(observation / 'parent-validation/results.json')
        assert formal['source_sha256'] == digest
        assert formal['contract_sha256'] == sha(run / 'contract.json')
        assert {row['case_id'] for row in formal['cases']} == {1, 2, 4, 7}
        assert all(row['status'] == 'measured' and all(
            row[side + '_correctness']['official_checker']['passed']
            for side in ['candidate', 'native']) for row in formal['cases'])
        assert (observation / 'paired-parent/comparison.json').is_file()
        manifest = read(observation / 'capture-parent/candidate/actual-ir/manifest.json')
        assert manifest['source_sha256'] == digest
        assert manifest['contract_sha256'] == sha(run / 'contract.json')
    assert not (run / 'agent-calls' / args.invocation).exists()
    assert not (run / ('process-' + args.invocation + '.json')).exists()
    config = read(run / 'config.json')
    template_config = read(template / 'config.json')
    package = Path(template_config['package'])
    for name in ['simulator-evidence.json', 'simulator-evidence-full.json',
                 'current-profiling.json', 'current-profiling-full.json',
                 'previous-result.json', 'context-brief.json', 'diagnosis.json',
                 'repair-feedback.json', 'host-repair-analysis.json']:
        if (workspace / name).exists():
            history = run / 'evidence-history' / args.invocation
            history.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(workspace / name, history / name)
            if name.startswith('simulator-evidence') or name in {
                    'diagnosis.json', 'repair-feedback.json', 'host-repair-analysis.json'}:
                (workspace / name).unlink()
    if not (workspace / 'skills').exists():
        shutil.copytree(package / 'skills', workspace / 'skills')
    shutil.copyfile(source, workspace / 'starter.py')
    shutil.copyfile(run / 'contract.json', workspace / 'contract.json')
    shutil.copyfile(template / 'model-settings.json', run / 'model-settings.json')
    shutil.copyfile(Path(__file__).with_name('contrast_decision_bridge.py'), run / 'decision_bridge.py')
    selections = [read(path) for path in [run / 'parent-selection.json',
                  *sorted(run.glob('retention-*.json'))] if path.exists()]
    prior_retained = any(row.get('source_sha256') == digest and row.get('retained', True)
                         for row in selections)
    save(workspace / 'starter-provenance.json', dict(
        kind='frozen_retained_gqa' if prior_retained else 'qualified_gqa_exploration',
        source_sha256=digest, prior_best_source_provided=prior_retained,
        origin=str(observation / 'retained-parent.json'), observation=str(observation)))
    save(workspace / 'knowledge.json', dict(records=[], scope='Current source and measured evidence; preserve the recorded historical two-part residual precision structure.'))
    if include_simulator:
        original = read(observation / 'simulator-parent/summary.json')
        assert original['source_sha256'] == digest and original['reports']
        assert all(r['absolute_unit_validation']['status'] == 'verified' for r in original['reports'])
        simulator = dict(original, reported_status=original['status'], status='qualified_observation',
            capture_scope="Normalize each core's common-clock window independently. application_completed records completion; full-call hardware timing is separate.")
        save(workspace / 'simulator-evidence.json', simulator)
    save(workspace / 'current-profiling.json', dict(source_sha256=digest,
        comparison=read(observation / 'paired-parent/comparison.json')))
    if args.compact_evidence:
        from gqa_evidence_brief import compact
        compact(workspace, include_simulator=include_simulator)
    save(workspace / 'previous-result.json', dict(source_sha256=digest,
        current=read(observation / 'parent-validation/results.json'), historical=read(run / 'parent-results.json'),
        previous_iterations={path.parent.parent.name: read(path) for path in sorted(
            (run / 'iterations').glob('*/parent-validation/results.json'))}))
    contract = read(run / 'contract.json')
    result = read(observation / 'parent-validation/results.json')
    target_us = next(r['candidate_timing']['full_call_us'] for r in result['cases'] if r['case_id'] == 4)
    selection_path = run / 'selection.json'
    best_us = read(selection_path)['target_latency_us'] if selection_path.exists() else target_us
    host_analysis = ('Use current hotspot regions, pipeline windows, and actual IR to choose one concrete mechanism; '
                     'assess handoff, movement, computation, and buffer costs separately.')
    requested_mechanism = None
    if args.host_analysis:
        analysis = read(args.host_analysis)
        assert analysis['source_sha256'] == digest
        host_analysis = analysis['analysis']
        requested_mechanism = analysis.get('requested_mechanism')
    shutil.copyfile(Path(contract['golden']), workspace / 'official-golden.py')
    ir = workspace / 'actual-ir'
    if ir.exists():
        prior_ir = run / 'evidence-history' / args.invocation / 'previous-actual-ir'
        prior_ir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ir), str(prior_ir))
    ir.mkdir()
    artifacts = []
    for path in sorted((observation / 'capture-parent/candidate/actual-ir').rglob('*')):
        if path.is_file() and path.suffix in {'.ttir', '.bcmlir', '.ttadapter'}:
            target = ir / (str(len(artifacts)) + '-' + path.name)
            shutil.copyfile(path, target)
            artifacts.append(dict(path=target.relative_to(workspace).as_posix(), sha256=sha(path)))
    assert artifacts
    save(workspace / 'context-brief.json', dict(source_sha256=digest, actual_ir=artifacts,
        current_profiling='current-profiling.json',
        current_simulator='simulator-evidence.json' if include_simulator else None,
        evidence_mode=args.evidence_mode,
        prior_host_simulator_knowledge=True,
        previous_result='previous-result.json', current_target_us=target_us,
        retained_best_target_us=best_us, observation_parent_was_retained=prior_retained,
        parent_post_observation_status=state['status'],
        scope='Fix case1/2/4/7 and original checker/golden identities; keep initial comparison and continuation separate.',
        host_analysis=host_analysis))
    config.update(package=str(package), paired_directory=str(observation / 'paired-parent'),
        active_source='candidate.py', context_kind='performance_continuation',
        require_preflight_review=True, formal_cases=4, optimization_policy='performance_first',
        method_manifest=template_config['method_manifest'], agent_tool_path=str(run / 'decision_bridge.py'),
        job=dict(operator='gqa', checks=dict(supplementary=dict(previous_total=4))),
        evidence_mode=args.evidence_mode)
    config.pop('required_simulator_evidence', None)
    if include_simulator:
        config['required_simulator_evidence'] = dict(path='simulator-evidence.json',
            sha256=sha(workspace / 'simulator-evidence.json'))
    save(run / 'config.json', config)
    save(run / 'controller-state.json', dict(status='prepared'))
    prompt = f'''Continue optimizing CANNBench Level4 GQA. Observed parent: {target_us:.5f} us; retained best: {best_us:.5f} us; see starter-provenance.json for parent identity. First read skills/hgkda/SKILL.md, task.json, contract.json, context-brief.json, previous-result.json, current-profiling.json, simulator-evidence.json, candidate.py, and relevant actual IR. Fix case1/2/4/7, checker, golden, I/O dtypes, and timing contract; target case4 below retained best {best_us:.5f} us, recording all other case changes. Preserve the parent's probability structure. Before changing intermediate dtype/order, state numerical risks and input scope; use the original four-case checker and unchanged tolerances.
Simulation covers partial windows on three cores; Cube/Vector normalize independently. State incremental hotspot/pipeline findings and retain missing full-call completion. Choose one mechanism. Prefer Triton for expressible tiling, layout, and computation; hand off concrete compiler defects as IR suggestions in diagnosis. Edit only candidate.py and diagnosis.json.
Before source edits, use Edit to fill diagnosis.json: source_sha256, evidence(path/category/role/finding), hypothesis, alternatives, discriminating_observation, change, expected, validation, selected_layer, layer_reason, simulator_incremental_judgment. Evidence paths use workspace/<filename>; categories are paired_profiling/simulator/source_ir/evaluation. After successfully reading current simulation, register through bridge decision --record diagnosis.json before source edits. Use Edit for candidate edits; Bash only invokes the bridge. All device computation uses strict Triton; host only allocates, handles metadata, and launches. No case_id/input-value dispatch, output caching, or external compute. Preserve the public entry point.
After one evaluable mechanism, call preflight --source candidate.py. At awaiting_source_review preserve bytes and hand off host read-only review/full four-case hardware evaluation. Full results and post-change replay follow. No total round/tool/time budget; this handoff is an acceptance boundary. Write new diagnoses/comments in concise English, recording observed benefits and costs.
'''
    if not include_simulator:
        prompt = prompt.replace(', simulator-evidence.json', '')
        prompt = prompt.replace(
            'Simulation covers partial windows on three cores; Cube/Vector normalize independently. State incremental hotspot/pipeline findings and retain missing full-call completion.',
            'This invocation supplies current paired profiling and actual IR; hypotheses use these and source analysis. The host retains prior simulator knowledge as continuation background. Attribute incremental tools only to actual inputs; collect post-change simulation as well.')
        prompt = prompt.replace(
            'After successfully reading current simulation, register through bridge decision --record diagnosis.json before source edits.',
            'After successfully reading current profiling and related TTIR/BCMLIR, register through bridge decision --record diagnosis.json before source edits. simulator_incremental_judgment states remaining ambiguity and why no fresh simulator input is needed; cite only supplied/read evidence.')
    if requested_mechanism:
        prompt += '\nHost-specified independent experiment for this invocation: ' + requested_mechanism + '\n'
    prompt_path = run / (args.invocation + '-prompt.md')
    prompt_path.write_text(prompt, encoding='utf8')
    command = [sys.executable, '-m', 'hgkda.runtime.agent_runner', '--config', str(run / 'config.json'),
        '--model-settings', str(run / 'model-settings.json'), '--invocation', args.invocation,
        '--prompt-file', str(prompt_path)]
    with (run / ('generation-' + args.invocation + '.log')).open('x') as stream:
        process = subprocess.Popen(command, env=dict(os.environ, PYTHONPATH=str(package / 'src')),
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    record = dict(pid=process.pid, command=command, starting_source_sha256=digest,
        started_at=datetime.now(timezone.utc).isoformat())
    save(run / ('process-' + args.invocation + '.json'), record)
    print(record)


if __name__ == '__main__':
    main()
