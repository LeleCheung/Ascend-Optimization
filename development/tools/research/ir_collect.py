"""复采冻结源码与IR组合的profiling及模拟器双分析。"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ir_campaign import read, save, sha


def summarize(config, directory):
    sys.path.insert(0, str(Path(config['package']) / 'src'))
    from hgkda.regions import analyze
    from hgkda.runtime.transfer_context import compact_regions
    from hgkda.simulator import parse, validate_time_unit
    from hgkda.timeline import relative_timeline
    spec = read(directory / 'spec.json')
    request = read(directory / 'request.json')
    source = Path(spec['source'])
    assert sha(source) == spec['source_sha256']
    debug_source = Path(spec.get('debug_source', source))
    assert sha(debug_source) == spec['source_sha256']
    verified_sources = {str(path): dict(text=path.read_text(), sha256=sha(path))
        for path in [source, debug_source]}
    listing = parse(directory / 'raw/kernel-000', request['kernel'], debug_source,
        spec['source_sha256'], spec['case'])
    save(directory / 'listing.json', listing)
    reports = []
    for core in listing['kernels'][0]['cores']:
        instructions = analyze(listing, kernel=request['kernel'], core=core['name'], sources=verified_sources)
        paths = list((directory / 'raw').glob('**/' + core['name'] + '/trace.json'))
        assert len(paths) == 1
        raw = paths[0].read_bytes()
        binding = dict(source_sha256=sha(source), case=spec['case'], kernel=request['kernel'],
            core=core['name'], raw_trace_sha256=sha(paths[0]))
        relative = relative_timeline(listing, raw, binding=binding, sources=verified_sources)
        save(directory / (core['name'] + '-relative-timeline.json'), relative)
        try:
            absolute = validate_time_unit(json.loads(raw), core['instructions'], request['trace_time_unit'])
        except ValueError as error:
            absolute = dict(status='failed', proposed_unit=request['trace_time_unit'], error=str(error))
        regions = compact_regions(instructions)
        regions['instruction_summary_timeline'] = regions.pop('timeline')
        reports.append(dict(core=core['name'], **regions,
            top_instructions=sorted(core['instructions'], key=lambda item: item.get('running_time_us') or 0,
                reverse=True)[:16],
            instruction_calls_by_pipeline=dict(Counter({
                lane: sum(item.get('calls') or 0 for item in core['instructions'] if item['pipeline'] == lane)
                for lane in {item['pipeline'] for item in core['instructions']}})),
            timeline=relative['timeline'], absolute_unit_validation=absolute))
    assert reports
    summary = dict(status='ir_bound_observation' if spec.get('ir_input') else 'source_bound_observation',
        source_sha256=sha(source), case=spec['case'],
        kernel=request['kernel'], ir_input=spec.get('ir_input'), reports=reports,
        application_completed=(Path(spec['ir_directory']) / 'manifest.json').exists(),
        absolute_time_qualified=all(report['absolute_unit_validation']['status'] == 'verified'
            for report in reports),
        scope='Instruction hotspots and normalized same-core trace windows; original exit and clock checks preserved')
    save(directory / 'ir-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--kernel-name')
    parser.add_argument('--isolate-target', action='store_true')
    parser.add_argument('--timeout-minutes', type=int)
    args = parser.parse_args()
    if args.timeout_minutes is not None and args.timeout_minutes < 1:
        parser.error('timeout-minutes must be positive')
    pair_module = args.package / 'src/hgkda/runtime/collect_pair.py'
    assert pair_module.is_file(), 'Observation package does not contain collect_pair.py'
    assert '--ir-manifest' in pair_module.read_text(encoding='utf8'), 'Observation package lacks IR capture support'
    run = args.run.resolve()
    evaluation = run / 'evaluations' / args.invocation
    config = read(run / 'config.json')
    provenance = read(evaluation / 'provenance.json')
    implementation = evaluation / 'implementation'
    assert sha(implementation / 'frozen.py') == config['source_sha256'] == provenance['source_sha256']
    assert sha(implementation / 'ir-manifest.json') == provenance['manifest_sha256']
    for item in provenance['source_files']:
        assert sha(implementation / item['path']) == item['sha256']
    result = read(evaluation / 'result.json')
    assert result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
    if provenance['changed_ir_count']:
        checks = read(evaluation / 'supplementary-summary.json')
        assert checks and all(row['status'] == 'passed' for row in checks.values())
        review = read(run / 'agent-calls' / args.invocation / 'ir-review.json')
        changed = {item['path']: item['sha256'] for item in read(implementation / 'ir-manifest.json')['overrides']
            if item['sha256'] != item['original_sha256']}
        assert review['decision'] == 'approved' and review['changed_ir'] == changed
    out = run / 'observations' / args.invocation
    out.mkdir(parents=True, exist_ok=False)
    (out / 'workspace').mkdir()
    shutil.copyfile(implementation / 'frozen.py', out / 'workspace/candidate.py')
    parent = Path(config.get('parent_directory') or config['observation_parent'])
    shutil.copytree(parent / 'reference-catalog', out / 'reference-catalog')
    (out / 'source-reviews').mkdir()
    shutil.copyfile(parent / 'source-reviews' / (config['source_sha256'] + '.json'),
        out / 'source-reviews' / (config['source_sha256'] + '.json'))
    config.update(run_directory=str(out), package=str(args.package.resolve()),
        capture_debug_source=config['frozen_source'],
        ir_capture_runtime=str(Path(config['tools']) / 'ir_override_runtime.py'),
        simulator_target_alias=dict(expected_arch='Ascend910B4-1',observed_arch='Ascend910B4'))
    save(out / 'config.json', config)
    save(out / 'input-provenance.json', dict(provenance, evaluation=str(evaluation),
        result_sha256=sha(evaluation / 'result.json'), collection_tool_sha256=sha(__file__),
        package_files=[dict(path=str(path.relative_to(args.package)), sha256=sha(path))
            for path in sorted(args.package.rglob('*.py')) if '__pycache__' not in path.parts]))
    environment = dict(os.environ, PYTHONPATH=str(args.package / 'src'))

    def execute(name, command):
        with (out / (name + '.log')).open('x') as stream:
            process = subprocess.run([sys.executable, *command], env=environment,
                stdout=stream, stderr=subprocess.STDOUT)
        save(out / (name + '-process.json'), dict(command=command, returncode=process.returncode))
        return process.returncode

    pair = out / 'paired'
    assert execute('paired', ['-m', 'hgkda.runtime.collect_pair', '--config', str(out / 'config.json'),
        '--source', str(implementation / 'frozen.py'), '--ir-manifest', str(implementation / 'ir-manifest.json'),
        '--case', config['experiment_goal']['target_case'], '--output', str(pair)]) == 0
    discovery = read(pair / 'candidate-discovery.json')
    durations = {}
    for task in discovery['tasks']:
        if task['task_type'].startswith('KERNEL_AI'):
            durations[task['name']] = durations.get(task['name'], 0) + task['duration_us']
    kernel = args.kernel_name or max(durations, key=durations.get)
    assert kernel in durations
    save(out / 'simulator-selection.json', dict(kernel=kernel, observed_task_duration_us=durations,
        selection='explicit host choice' if args.kernel_name else 'largest accumulated AI-core task duration'))
    if args.isolate_target:
        assert args.kernel_name, 'Isolated collection requires an explicit target'
        assert execute('isolated-target', [str(Path(__file__).with_name('ir_isolate_collection.py')),
            '--run', str(run), '--invocation', args.invocation, '--package', str(args.package),
            '--isolation-id', 'post-formal-target']) == 0
        return
    simulator = out / 'simulator'
    execute('simulator', ['-m', 'hgkda.runtime.collect_simulator', '--config', str(out / 'config.json'),
        '--paired-directory', str(pair), '--output', str(simulator), '--kernel-name', kernel,
        '--soc', 'Ascend910B4', '--simulator-library', config['sdk'] + '/aarch64-linux/simulator/Ascend910B4/lib',
        '--timeout-minutes', str(args.timeout_minutes or config.get('simulator_timeout_minutes', 1)),
        '--trace-unit', 'us', '--allow-postcapture-exit'])
    summary = summarize(config, simulator)
    save(out / 'status.json', dict(status='ir_evidence_collected', source_sha256=config['source_sha256'],
        ir_manifest_sha256=provenance['manifest_sha256'], reports=len(summary['reports'])))


if __name__ == '__main__':
    main()
