"""Run a source-bound statistical grouping supplement after full qualification."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from ir_campaign import read, save, sha
from ir_stability import binaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    config = read(run / 'config.json')
    call = run / 'agent-calls' / args.invocation
    output = call / 'supplemental-grouping'
    output.mkdir(exist_ok=False)
    protocol = dict(seed=2716, spatial_sizes=[128, 256, 384, 512, 4096],
                    recipes=['normal', 'constant', 'offset128', 'offset4096', 'alternating'],
                    momentum=0.1, eps=1e-5, atol=1e-4, rtol=1e-4,
                    preregistered_at=datetime.now(timezone.utc).isoformat(),
                    driver_sha256=sha(__file__),
                    probe_sha256=sha(Path(__file__).with_name('batchnorm_grouping_probe.py')),
                    scope='Diagnostic supplement for changed Welford grouping;both variants '
                          'use identical inputs and FP64 population moments.Published contract '
                          'and formal performance remain separate.Inherited failures retained.')
    save(output / 'protocol.json', protocol)
    evaluation = run / 'evaluations' / args.invocation
    while not (evaluation / 'supplementary-summary.json').exists():
        time.sleep(10)
    formal = read(evaluation / 'result.json')
    checks = read(evaluation / 'supplementary-summary.json')
    if not (formal['status'] == 'PASSED' and formal['num_passed'] == formal['num_workloads']
            and checks and all(row['status'] == 'passed' for row in checks.values())):
        save(output / 'summary.json', dict(status='skipped_formal_not_qualified'))
        return
    parents = {'original': run / 'evaluations/unchanged-ir', 'candidate': evaluation}
    results = {}
    with Path(config['lock']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for variant, parent in parents.items():
            block = output / variant
            block.mkdir()
            provenance = read(parent / 'provenance.json')
            for item in provenance['source_files']:
                assert sha(parent / 'implementation' / item['path']) == item['sha256']
            implementation = block / 'implementation'
            shutil.copytree(parent / 'implementation', implementation)
            manifest_path = implementation / 'ir-manifest.json'
            manifest = read(manifest_path)
            assert manifest['source_sha256'] == config['source_sha256']
            manifest.update(cache_root=str(block / 'cache'),
                            receipt_directory=str(block / 'compiler-receipts'))
            save(manifest_path, manifest)
            spec = dict(protocol, variant=variant, manifest=str(manifest_path),
                        source_sha256=config['source_sha256'], output=str(block / 'result.json'),
                        parent_manifest_sha256=provenance['manifest_sha256'],
                        expected_binaries=binaries(parent / 'compiler-receipts'))
            save(block / 'spec.json', spec)
            environment = dict(ASCEND_RT_VISIBLE_DEVICES=str(config['device']),
                               OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                               PYTHONPATH=config.get('device_pythonpath', ''))
            shell = 'source ' + shlex.quote(config['sdk'] + '/set_env.sh') + '\n'
            shell += '\n'.join('export ' + key + '=' + shlex.quote(value)
                               for key, value in environment.items())
            shell += '\nexec ' + shlex.join([config['device_python'],
                str(Path(__file__).with_name('batchnorm_grouping_probe.py')),
                '--spec', str(block / 'spec.json')])
            with (block / 'process.log').open('x') as stream:
                process = subprocess.run(['docker', 'exec', config['container'], 'bash', '-c', shell],
                                         stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
            save(block / 'process.json', dict(returncode=process.returncode))
            assert process.returncode == 0, 'Probe execution failed;inspect preserved process.log'
            results[variant] = read(block / 'result.json')
    indexed = {variant: {(row['spatial'], row['recipe']): row for row in result['rows']}
               for variant, result in results.items()}
    assert set(indexed['original']) == set(indexed['candidate'])
    comparison = []
    for key, original in indexed['original'].items():
        candidate = indexed['candidate'][key]
        assert original['input_sha256'] == candidate['input_sha256']
        comparison.append(dict(spatial=key[0], recipe=key[1],
                               original_passed=original['passed'],
                               candidate_passed=candidate['passed']))
    summary = dict(status='completed', records=comparison,
                   original_passed=results['original']['passed'],
                   candidate_passed=results['candidate']['passed'],
                   total=len(comparison),
                   inherited_failures=[row for row in comparison if not row['original_passed']],
                   new_regressions=[row for row in comparison
                                    if row['original_passed'] and not row['candidate_passed']],
                   result_sha256={key: sha(output / key / 'result.json') for key in parents},
                   protocol_sha256=sha(output / 'protocol.json'), automatic_retention=False,
                   scope=protocol['scope'])
    save(output / 'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ['status', 'original_passed',
                                                  'candidate_passed', 'total']}))


if __name__ == '__main__':
    main()
