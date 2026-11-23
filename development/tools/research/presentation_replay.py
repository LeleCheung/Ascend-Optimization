"""Remeasure frozen implementations under one explicit presentation contract."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_bytes())
    run = args.manifest.parent
    sys.path.insert(0, str(Path(manifest['package']) / 'src'))
    from hgkda.adapters.kgs import call, upload_bundle

    service = run / 'service.json'
    if not service.exists() and manifest.get('service_ownership', 'project-owned') == 'project-owned':
        with (run / 'kgs.log').open('x') as log:
            process = subprocess.Popen(manifest['service_command'], stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        save(service, dict(pid=process.pid, command=manifest['service_command'],
            server=manifest['server'], scope='project-owned', timing='profiler'))
    for _ in range(90):
        try:
            status = call(manifest['server'], '/status')
            save(run / 'service-status.json', status)
            break
        except Exception:
            time.sleep(2)
    else:
        raise RuntimeError('Presentation replay service did not become ready')
    outcomes = []
    for index, item in enumerate(manifest['measurements']):
        output = run / item['name']
        output.mkdir(exist_ok=True)
        if (output / 'result.json').exists():
            result = json.loads((output / 'result.json').read_bytes())
        else:
            request = copy.deepcopy(json.loads(Path(item['request']).read_bytes()))
            raw = Path(item['source']).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            assert digest == item['sha256'], 'Frozen source identity changed'
            request['implementation']['name'] = 'presentation_' + item['name']
            request['implementation']['sources'] = [dict(path='main.py', content=raw.decode('utf8'))]
            if item.get('package_context'):
                # Preserve package-relative imports in a separate name-only loader.
                module_name = item['package_context'] + '._hgkda_frozen_native'
                loader = ('import importlib.util\nimport pathlib\nimport sys\n'
                    'import ' + item['package_context'] + '\n'
                    '_spec = importlib.util.spec_from_file_location(' + repr(module_name) +
                    ', pathlib.Path(__file__).with_name("upstream_native.py"))\n'
                    '_module = importlib.util.module_from_spec(_spec)\n'
                    'sys.modules[_spec.name] = _module\n'
                    '_spec.loader.exec_module(_module)\n'
                    'run = getattr(_module, ' + repr(item['native_callable']) + ')\n')
                request['implementation']['sources'] = [dict(path='main.py', content=loader),
                    dict(path='upstream_native.py', content=raw.decode('utf8'))]
            request['implementation']['entrypoint'] = item.get('entrypoint', 'main.py::run')
            # Baselines retain their original source, including original dependency paths.
            save(output / 'request.json', request)
            (output / 'source.py').write_bytes(raw)
            save(output / 'provenance.json', dict(item, source_sha256=digest,
                purpose='Frozen-source replay for a common presentation timing contract',
                settings=request['settings'], binding=request['binding']))
            archive = Path(item['archive'])
            assert 'sha256:' + hashlib.sha256(archive.read_bytes()).hexdigest() == request['binding']['bundle_id']
            save(run / 'progress.json', dict(status='evaluating', index=index, measurement=item['name']))
            with Path(manifest['lock']).open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                upload_bundle(manifest['server'], archive)
                result = call(manifest['server'], '/evaluate', request, timeout=2400,
                    operation_id='presentation-replay-20261119-' + item['name'])
            save(output / 'result.json', result)
        target = next((row for row in result.get('per_workload', [])
            if row['uuid'] == item['target_case'] and row['phase'] == 'timing'), None)
        outcomes.append(dict(name=item['name'], source_sha256=item['sha256'],
            status=result.get('status'), num_passed=result.get('num_passed'),
            num_workloads=result.get('num_workloads'), geo_mean=result.get('geo_mean'), target=target))
        save(run / 'summary.json', outcomes)
    save(run / 'progress.json', dict(status='completed', measurements=len(outcomes)))


if __name__ == '__main__':
    main()
