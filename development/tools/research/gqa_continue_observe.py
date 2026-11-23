"""Observe a frozen GQA parent through its unchanged official-subset harness."""
import argparse
from collections import Counter
import csv
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from ir_campaign import read, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--simulator-minutes', type=int, default=5)
    p.add_argument('--resume-simulator', action='store_true', help='Reuse completed formal/profile/snapshot artifacts')
    args = p.parse_args()
    run = args.run.resolve()
    config = read(run / 'config.json')
    source = run / 'workspace/candidate.py'
    parent = read(run / 'retained-parent.json')
    assert sha(source) == parent['source_sha256']
    package = Path(config['package'])
    sys.path.insert(0, str(package / 'src'))
    from hgkda.runtime.trace import action_trace
    from hgkda.compare import compare
    from hgkda.regions import analyze
    from hgkda.runtime.transfer_context import compact_regions
    from hgkda.simulator import parse, validate_time_unit
    from hgkda.timeline import relative_timeline

    def execute(name, command, simulator=False, timeout=1200):
        out = run / name
        out.mkdir(parents=True, exist_ok=False)
        save(out / 'request.json', dict(command=command, source_sha256=sha(source),
            contract_sha256=sha(run / 'contract.json'), simulator=simulator))
        values = dict(ASCEND_RT_VISIBLE_DEVICES=str(config['device']), OMP_NUM_THREADS='1',
            PYTHONDONTWRITEBYTECODE='1', TRITON_DISABLE_LINE_INFO='false',
            PYTHONPATH=str(Path(config.get('observation_package', config['package'])) / 'src'),
            TRITON_CACHE_DIR=str(out / 'cache'))
        if config.get('bcmlir_manifest'):
            values['GQA_BCMLIR_MANIFEST'] = config['bcmlir_manifest']
        shell = 'source ' + shlex.quote(config['sdk'] + '/set_env.sh') + '\n'
        shell += '\n'.join('export ' + key + '=' + shlex.quote(value) for key, value in values.items()) + '\n'
        if simulator:
            shell += 'export LD_LIBRARY_PATH=' + shlex.quote(config['sdk'] + '/aarch64-linux/simulator/Ascend910B4/lib') + ':${LD_LIBRARY_PATH:-}\n'
        shell += 'exec ' + shlex.join(command)
        with action_trace(config, 'simulate' if simulator else 'pair', name.replace('/', '-'),
                [out / 'request.json', out / 'execution.json']):
            with Path(config['lock']).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                with (out / 'collector.log').open('x') as log:
                    process = subprocess.run(['docker', 'exec', '-w', str(out), config['container'],
                        'bash', '-c', shell], stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
                save(out / 'execution.json', dict(returncode=process.returncode))
        return out, process.returncode

    if args.resume_simulator:
        result = read(run / 'parent-validation/results.json')
        assert result['source_sha256'] == sha(source)
        assert result['contract_sha256'] == sha(run / 'contract.json')
        assert {row['case_id'] for row in result['cases']} == {1, 2, 4, 7}
        assert all(row['status'] == 'measured' and all(
            row[side + '_correctness']['official_checker']['passed']
            for side in ['candidate', 'native']) for row in result['cases'])
        manifest = read(run / 'capture-parent/candidate/actual-ir/manifest.json')
        assert manifest['source_sha256'] == sha(source)
        assert manifest['contract_sha256'] == sha(run / 'contract.json')
        for row in manifest['files']:
            assert sha(run / 'capture-parent/candidate/actual-ir' / row['path']) == row['sha256']
        assert (run / 'paired-parent/comparison.json').is_file()
        assert not (run / 'simulator-parent').exists(), 'Preserve the failed collection before resuming'
        collect_simulator(run, source, config, args.simulator_minutes, execute)
        return

    save(run / 'observation-status.json', dict(status='checking_frozen_parent'))
    if config.get('existing_validation'):
        existing = config['existing_validation']
        assert sha(existing['result']) == existing['result_sha256']
        destination = run / 'parent-validation'
        destination.mkdir(exist_ok=False)
        shutil.copyfile(existing['result'], destination / 'results.json')
        (destination / 'cache').symlink_to(existing['cache'], target_is_directory=True)
        save(destination / 'reuse.json', existing)
    else:
        command = [config['device_python'], str(run / 'tools/attention_bench.py')]
        if config.get('bcmlir_manifest'):
            command = [config['device_python'], str(run / 'tools/gqa_ir_bench_driver.py'),
                '--manifest', config['bcmlir_manifest'], '--harness', str(run / 'tools/attention_bench.py')]
        _, rc = execute('parent-validation', [*command,
            '--contract', str(run / 'contract.json'), '--candidate', str(source),
            '--output', str(run / 'parent-validation/results.json')])
        assert rc == 0
    result = read(run / 'parent-validation/results.json')
    assert result['source_sha256'] == sha(source)
    assert {r['case_id'] for r in result['cases']} == {1, 2, 4, 7}
    assert all(r['status'] == 'measured' and all(r[s + '_correctness']['official_checker']['passed']
        for s in ['candidate', 'native']) for r in result['cases'])
    save(run / 'observation-status.json', dict(status='collecting_pair', parent_cases_passed=4))
    pair = run / 'paired-parent'
    pair.mkdir()
    case = next(c for c in read(run / 'contract.json')['cases'] if c['case_id'] == 4)
    save(pair / 'spec.json', dict(source=str(source), source_sha256=sha(source), case=case,
        seed=1733, contract_sha256=sha(run / 'contract.json')))
    for side in ['candidate', 'reference']:
        out = run / ('capture-parent/' + side)
        app = shlex.join([config['device_python'], str(run / 'tools/attention_profile_driver.py'),
            '--contract', str(run / 'contract.json'), '--source', str(source), '--case', '4',
            '--side', side, '--output', str(out / 'actual-ir')])
        cmd = ['msprof', 'op', '--application=' + app, '--launch-count=1', '--warm-up=2',
            '--aic-metrics=Roofline', '--output=' + str(out / 'raw')]
        if side == 'candidate':
            cmd.append('--kernel-name=_gqa_fwd_kernel')
        _, rc = execute('capture-parent/' + side, cmd)
        assert rc == 0
        dest = pair / side / 'profile'
        dest.mkdir(parents=True)
        for path in (out / 'raw').glob('OPPROF_*/*.csv'):
            shutil.copyfile(path, dest / path.name)
        with (dest / 'OpBasicInfo.csv').open() as stream:
            basic = list(csv.DictReader(stream))
        assert len(basic) == 1
        save(dest.parent / 'collection.json', dict(status='completed', side=side,
            spec_sha256=sha(pair / 'spec.json'), expected_kernel=basic[0]['Op Name'], groups='Roofline'))
    save(pair / 'comparison.json', compare(pair))
    compiled = read(run / 'capture-parent/candidate/actual-ir/manifest.json')
    extra_kernels = sorted({Path(row['path']).stem for row in compiled['files']
                            if row['path'].endswith('.npubin')
                            and Path(row['path']).stem != '_gqa_fwd_kernel'})
    auxiliary = []
    for kernel_name in extra_kernels:
        out = run / 'capture-auxiliary' / kernel_name
        app = shlex.join([config['device_python'], str(run / 'tools/attention_profile_driver.py'),
            '--contract', str(run / 'contract.json'), '--source', str(source), '--case', '4',
            '--side', 'candidate', '--output', str(out / 'actual-ir')])
        _, rc = execute('capture-auxiliary/' + kernel_name, ['msprof', 'op', '--application=' + app,
            '--launch-count=1', '--warm-up=2', '--kernel-name=' + kernel_name,
            '--aic-metrics=Roofline', '--output=' + str(out / 'raw')])
        assert rc == 0
        tables = {}
        for path in (out / 'raw').glob('OPPROF_*/*.csv'):
            with path.open() as stream:
                tables[path.name] = list(csv.DictReader(stream))
        assert len(tables.get('OpBasicInfo.csv', [])) == 1
        auxiliary.append(dict(kernel=kernel_name, source_sha256=sha(source), tables=tables,
            manifest=str(out / 'actual-ir/manifest.json'),
            manifest_sha256=sha(out / 'actual-ir/manifest.json')))
    if auxiliary:
        save(run / 'auxiliary-profiling.json', dict(kernels=auxiliary,
            scope='Separate hardware captures of auxiliary stages; full-call benchmark includes all stages.'))
        snapshot_out = run / 'target-launch-snapshot'
        _, rc = execute('target-launch-snapshot', [config['device_python'],
            str(run / 'tools/attention_profile_driver.py'), '--contract', str(run / 'contract.json'),
            '--source', str(source), '--case', '4', '--side', 'candidate',
            '--snapshot-kernel', '_gqa_fwd_kernel', '--output', str(snapshot_out / 'actual-ir')])
        assert rc == 0
        manifest = snapshot_out / 'actual-ir/launch-snapshot/manifest.json'
        snapshot = read(manifest)
        assert snapshot['source_sha256'] == sha(source) and snapshot['case'] == case
        assert set(snapshot['preceding_real_launches']) == set(extra_kernels)
        save(run / 'later-stage-replay.json', dict(manifest=str(manifest), manifest_sha256=sha(manifest),
            preceding_real_launches=snapshot['preceding_real_launches'],
            scope='Real-device auxiliary stages produced snapshot inputs. Simulator replays only the selected attention launch; cache warmth from preceding stages is not reproduced.'))
    collect_simulator(run, source, config, args.simulator_minutes, execute)


def collect_simulator(run, source, config, simulator_minutes, execute):
    package = Path(config['package'])
    sys.path.insert(0, str(package / 'src'))
    from hgkda.regions import analyze
    from hgkda.runtime.transfer_context import compact_regions
    from hgkda.simulator import parse, validate_time_unit
    from hgkda.timeline import relative_timeline

    case = next(c for c in read(run / 'contract.json')['cases'] if c['case_id'] == 4)
    auxiliary = read(run / 'auxiliary-profiling.json')['kernels'] if (run / 'auxiliary-profiling.json').exists() else []
    if auxiliary:
        replay = read(run / 'later-stage-replay.json')
        manifest = Path(replay['manifest'])
        assert sha(manifest) == replay['manifest_sha256']
        snapshot = read(manifest)
        assert snapshot['source_sha256'] == sha(source) and snapshot['case'] == case
        assert snapshot['contract_sha256'] == sha(run / 'contract.json')
        assert sha(manifest.parent / snapshot['tensors']) == snapshot['tensors_sha256']
    save(run / 'observation-status.json', dict(status='collecting_simulator', parent_cases_passed=4))
    out = run / 'simulator-parent'
    app = shlex.join([config['device_python'], str(run / 'tools/attention_profile_driver.py'),
        '--contract', str(run / 'contract.json'), '--source', str(source), '--case', '4',
        '--side', 'candidate', '--output', str(out / 'actual-ir'),
        *(['--kernel-replay', str(manifest)] if auxiliary else [])])
    _, rc = execute('simulator-parent', ['msprof', 'op', 'simulator', '--application=' + app,
        '--kernel-name=_gqa_fwd_kernel', '--soc-version=Ascend910B4', '--core-id=0',
        '--launch-count=1', '--dump=on', '--timeout=' + str(simulator_minutes),
        '--output=' + str(out / 'raw')], simulator=True, timeout=(simulator_minutes + 3) * 60)
    comparison = read(run / 'paired-parent/comparison.json')
    kernel = comparison.get('result', comparison)['candidate']['basic']['Op Name']
    assert kernel.startswith('_gqa_fwd_kernel'), 'Unexpected profiled target kernel'
    listing = parse(out / 'raw', kernel, source, sha(source), case)
    save(out / 'listing.json', listing)
    reports = []
    for core in listing['kernels'][0]['cores']:
        traces = list((out / 'raw').glob('OPPROF_*/simulator/' + core['name'] + '/trace.json'))
        assert len(traces) == 1
        raw = traces[0].read_bytes()
        binding = dict(source_sha256=sha(source), case=case, kernel=kernel,
            core=core['name'], raw_trace_sha256=sha(traces[0]))
        relative = relative_timeline(listing, raw, binding=binding,
            sources={str(source): dict(text=source.read_text(), sha256=sha(source))})
        save(out / (core['name'] + '-relative-timeline.json'), relative)
        try:
            clock = validate_time_unit(json.loads(raw), core['instructions'], 'us')
        except ValueError as error:
            clock = dict(status='failed', error=str(error))
        regions = compact_regions(analyze(listing, kernel=kernel, core=core['name'],
            sources={str(source): dict(text=source.read_text(), sha256=sha(source))}))
        regions['instruction_summary_timeline'] = regions.pop('timeline')
        counts = Counter()
        for row in core['instructions']:
            counts[row['pipeline'] + '/' + row['instruction']] += row.get('calls') or 0
        reports.append(dict(core=core['name'], **regions, timeline=relative['timeline'],
            absolute_unit_validation=clock, instruction_calls=dict(counts),
            top_instructions=sorted(core['instructions'], key=lambda r:r.get('running_time_us') or 0,
                reverse=True)[:16]))
    assert reports
    complete = (out / 'actual-ir/manifest.json').exists()
    save(out / 'summary.json', dict(status='source_bound_observation', source_sha256=sha(source),
        case=case, kernel=kernel, reports=reports, application_completed=complete,
        execution_returncode=rc, scope='Observed per-core windows; full-case measurements and checker results remain separate'))
    save(run / 'observation-status.json', dict(status='parent_dual_evidence_ready',
        source_sha256=sha(source), parent_cases_passed=4, reports=len(reports), application_completed=complete))


if __name__ == '__main__':
    main()
