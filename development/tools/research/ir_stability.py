"""Use original KGS timing for interleaved fixed-case comparisons; keep full-contract scores separate."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys

from ir_campaign import read, save, sha


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def binaries(directory):
    result = {}
    for path in sorted(Path(directory).glob('*.jsonl')):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get('override_applied'):
                continue
            artifact = next(item for item in row['artifacts'] if item['kind'] == 'npubin')
            assert sha(artifact['path']) == artifact['sha256']
            key = row['ast_sha256']
            assert key not in result or result[key] == artifact['sha256']
            result[key] = artifact['sha256']
    assert result
    return result


def timing_backend(root):
    timing_root = Path(root).resolve()
    sys.path[:0] = [str(timing_root), str(timing_root / 'client')]
    from kernelgen_server.backends.ascend.device import NpuDevice
    from kernelgen_server.backends.ascend.profiler import profiler_npu_core
    timing_files = [Path(sys.modules[NpuDevice.__module__].__file__).resolve(),
        Path(sys.modules[profiler_npu_core.__module__].__file__).resolve()]
    assert all(path.is_relative_to(timing_root) for path in timing_files)
    return NpuDevice, profiler_npu_core, timing_files


def device(spec_path):
    spec = read(spec_path)
    import torch
    import torch_npu
    from ir_override_runtime import verify_receipts
    NpuDevice, profiler_npu_core, timing_files = timing_backend(spec['kgs_root'])
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    runtime_path = Path(spec['manifest']).parent / 'ir_override_runtime.py'
    runtime = imported('_stability_override', runtime_path)
    source = runtime.install(spec['manifest'])
    assert sha(source) == spec['source_sha256']
    candidate = imported('_stability_candidate', source)
    oracle = imported('_stability_inputs', Path(spec['leaf']) / 'oracle.py')
    inputs = oracle.gen_inputs(spec['case'], 'npu:0')
    fn = lambda: candidate.run(**inputs)
    fn()
    torch.npu.synchronize()
    manifest = read(spec['manifest'])
    verification = verify_receipts(Path(manifest['receipt_directory']), manifest)
    actual_binaries = binaries(manifest['receipt_directory'])
    assert actual_binaries == spec['expected_binaries'], 'Stability binary differs from formal evaluation'
    backend = NpuDevice(perf_mode='profiler', num_trials=spec['settings']['num_trials'])
    samples = []
    for trial in range(spec['settings']['num_trials']):
        warmup, active = backend.budget_to_counts(fn, (), spec['settings']['warmup_ms'],
            spec['settings']['benchmark_ms'], 'npu:0')
        elapsed, directory = profiler_npu_core(fn, warmup=warmup, active=active,
            prof_dir_name='trial-' + str(trial), clear_l2_cache_flag=True, dsl='triton_ascend')
        assert math.isfinite(elapsed) and elapsed > 0
        samples.append(dict(trial=trial, latency_us=elapsed, warmup_count=warmup,
            active_count=active, profile_directory=directory))
    result = dict(status='measured', source_sha256=sha(source), manifest_sha256=sha(spec['manifest']),
        case=spec['case'], settings=spec['settings'], samples=samples,
        median_us=statistics.median(row['latency_us'] for row in samples), compiler_verification=verification,
        override_runtime_sha256=sha(runtime_path),
        receipt_verifier_sha256=sha(Path(__file__).with_name('ir_override_runtime.py')),
        verified_binaries=actual_binaries,
        protocol='Installed KGS NpuDevice.budget_to_counts and profiler_npu_core; L2 clear enabled, raw profiles retained',
        implementation_files=[dict(path=str(p), sha256=sha(p)) for p in timing_files])
    save(Path(spec['output']), result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-spec', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--invocation')
    parser.add_argument('--kgs-root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-timing-root', type=Path)
    args = parser.parse_args()
    if args.verify_timing_root:
        import torch
        import torch_npu
        _, _, paths = timing_backend(args.verify_timing_root)
        assert not torch.npu.is_initialized()
        print(json.dumps(dict(status='timing_imports_verified', npu_initialized=False,
            files=[dict(path=str(path), sha256=sha(path)) for path in paths])))
        return
    if args.device_spec:
        device(args.device_spec)
        return
    assert args.run and args.invocation
    assert args.kgs_root and (args.kgs_root / 'kernelgen_server/backends/ascend/profiler.py').is_file()
    run = args.run.resolve()
    config = read(run / 'config.json')
    evaluation = run / 'evaluations' / args.invocation
    result = read(evaluation / 'result.json')
    assert result['status'] == 'PASSED' and result['num_passed'] == result['num_workloads']
    checks = read(evaluation / 'supplementary-summary.json')
    assert checks and all(row['status'] == 'passed' for row in checks.values())
    output = (args.output or run / 'stability' / args.invocation).resolve()
    assert output.is_relative_to(run)
    output.mkdir(parents=True, exist_ok=False)
    source_evaluations = {'A': run / 'evaluations/unchanged-ir', 'B': evaluation}
    case = read(run / 'workspace/context.json')['target_case']
    leaf = Path(config['job']['checks']['supplementary']['leaf'])
    cases = [json.loads(line) for line in (leaf / 'timing.jsonl').read_text().splitlines() if line.strip()]
    target_case = next(row for row in cases if row['name'] == case)
    settings = read(evaluation / 'request.json')['settings']
    save(output / 'protocol.json', dict(source_sha256=config['source_sha256'], case=target_case,
        settings=settings, sequence='ABBA', variants={key: str(path) for key, path in source_evaluations.items()},
        kgs_root=str(args.kgs_root.resolve()),
        timestamp=datetime.now(timezone.utc).isoformat(), scope='Fixed-case stability; original complete formal contract remains authoritative'))
    records = []
    with Path(config['lock']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index, variant in enumerate('ABBA'):
            block = output / (str(index) + '-' + variant)
            block.mkdir()
            parent = source_evaluations[variant]
            provenance = read(parent / 'provenance.json')
            for item in provenance['source_files']:
                assert sha(parent / 'implementation' / item['path']) == item['sha256']
            implementation = block / 'implementation'
            shutil.copytree(parent / 'implementation', implementation)
            manifest_path = implementation / 'ir-manifest.json'
            manifest = read(manifest_path)
            assert manifest['source_sha256'] == config['source_sha256']
            assert not manifest.get('analysis_target_alias')
            manifest.update(cache_root=str(block / 'compiler-state'), receipt_directory=str(block / 'compiler-receipts'))
            save(manifest_path, manifest)
            spec_path = block / 'spec.json'
            save(spec_path, dict(manifest=str(manifest_path), source_sha256=config['source_sha256'],
                leaf=str(leaf), case=target_case, settings=settings, output=str(block / 'result.json'),
                parent_manifest_sha256=provenance['manifest_sha256'],
                kgs_root=str(args.kgs_root.resolve()),
                expected_binaries=binaries(parent / 'compiler-receipts')))
            command = [config['device_python'], str(Path(__file__).resolve()), '--device-spec', str(spec_path)]
            environment = dict(ASCEND_RT_VISIBLE_DEVICES=str(config['device']), OMP_NUM_THREADS='1',
                PYTHONPATH=config.get('device_pythonpath', ''), PYTHONDONTWRITEBYTECODE='1')
            shell = 'source ' + shlex.quote(config['sdk'] + '/set_env.sh') + '\n'
            shell += '\n'.join('export ' + key + '=' + shlex.quote(value) for key, value in environment.items())
            shell += '\ncd ' + shlex.quote(str(block)) + '\nexec ' + shlex.join(command)
            with (block / 'process.log').open('x') as stream:
                subprocess.run(['docker', 'exec', config['container'], 'bash', '-c', shell],
                    stdout=stream, stderr=subprocess.STDOUT, timeout=1800, check=True)
            measured = read(block / 'result.json')
            records.append(dict(index=index, variant=variant, median_us=measured['median_us'],
                result_sha256=sha(block / 'result.json')))
            save(output / 'progress.json', dict(status='measuring', records=records))
    a = [row['median_us'] for row in records if row['variant'] == 'A']
    b = [row['median_us'] for row in records if row['variant'] == 'B']
    save(output / 'summary.json', dict(status='stability_measured', records=records,
        original_median_us=statistics.median(a), candidate_median_us=statistics.median(b),
        ratio=statistics.median(a) / statistics.median(b), candidate_below_every_original=max(b) < min(a),
        automatic_retention=False, scope='Four alternating blocks, three trials per block, one frozen case; formal all-case qualification is separate'))


if __name__ == '__main__':
    main()
