"""Prepare source-frozen IR studies and run generic capture/evaluation stages."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


STUDIES = [('batchnorm', 'batchnorm-simulator-20261118'),
    ('upsample', 'upsample-performance-20261118'),
    ('qkv', 'qkv-simulator-20261118'), ('scatter', 'scatter-simulator-20261118')]


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup(root):
    root.mkdir(parents=True, exist_ok=True)
    tools = root / 'tools'
    tools.mkdir(exist_ok=True)
    for name in ['ir_capture.py', 'ir_override_runtime.py', 'ir_campaign.py']:
        shutil.copyfile(Path(__file__).with_name(name), tools / name)
    for short, parent_name in STUDIES:
        parent = root.parent / parent_name
        best = read(parent / 'best/selection.json')
        digest = best['source_sha256']
        source = parent / 'best/candidate.py'
        assert sha(source) == digest
        run = root / short
        workspace = run / 'workspace'
        workspace.mkdir(parents=True, exist_ok=True)
        assert not (run / 'agent-calls').exists(), 'Prepared study already entered candidate generation'
        if (workspace / 'frozen.py').exists():
            assert sha(workspace / 'frozen.py') == digest
        else:
            shutil.copyfile(source, workspace / 'frozen.py')
        config = copy.deepcopy(read(parent / 'config.json'))
        config.update(run_directory=str(run), parent_directory=str(parent),
            source_sha256=digest, frozen_source=str(workspace / 'frozen.py'),
            tools=str(tools), kgs_server='http://127.0.0.1:19730',
            selected_layer='single_operator_ir', active_source='frozen.py',
            source_frozen=True, policy='performance_first',
            comparator=dict(selection=best, original_validation=best['validation']))
        save(run / 'config.json', config)
        shutil.copyfile(parent / 'model-settings.json', run / 'model-settings.json')
        assert read(parent / ('paired-' + digest[:8]) / 'spec.json')['source_sha256'] == digest
        evidence = []
        for label, path in [
            ('profiling', parent / ('paired-' + digest[:8]) / 'comparison.json'),
            ('hotspots', parent / ('simulator-' + digest[:8]) / 'summary.json'),
            ('timeline', parent / ('simulator-' + digest[:8]) / 'relative-timeline-summary.json')]:
            embedded = label == 'timeline' and not path.exists()
            if embedded:
                path = parent / ('simulator-' + digest[:8]) / 'summary.json'
                summary = read(path)
                assert summary['source_sha256'] == digest
                reports = summary['reports']
                assert reports and all(report['timeline']['status'] == 'completed' for report in reports)
                assert all(report['unit_validation']['status'] == 'verified' for report in reports)
                document = dict(source_sha256=digest, reports=[dict(core=report['core'],
                    timeline=report['timeline'], unit_validation=report['unit_validation']) for report in reports],
                    absolute_time_qualified=True, scope='Existing source-bound independently unit-checked windows',
                    original_summary=str(path), original_summary_sha256=sha(path))
            else:
                document = read(path)
            if 'source_sha256' in document:
                assert document['source_sha256'] == digest
            destination = workspace / (label + '.json')
            if embedded:
                save(destination, document)
            else:
                shutil.copyfile(path, destination)
            evidence.append(dict(category=label, path=destination.name, sha256=sha(destination),
                original=str(path), source_sha256=digest))
        save(workspace / 'context.json', dict(operator=config['operator'], source_sha256=digest,
            target_case=config['experiment_goal']['target_case'], evidence=evidence,
            comparator=best, candidate_author='CC/DeepSeek', diagnosis_author='Codex',
            source_edit_allowed=False, candidate_layer='TTIR',
            goal='Improve the current fully qualified best through actual IR changes'))
        save(run / 'status.json', dict(status='prepared', source_sha256=digest))
    save(root / 'campaign.json', dict(schema='hgkda.ir-campaign/v1', minimum_distinct_gains=3,
        studies=[short for short, _ in STUDIES], source_frozen=True,
        scope='Current-best IR improvements; regression repair and compiler-option trials are separate'))


def capture(run):
    config = read(run / 'config.json')
    source = Path(config['frozen_source'])
    assert sha(source) == config['source_sha256']
    output = run / 'baseline-capture'
    command = [config['device_python'], str(Path(config['tools']) / 'ir_capture.py'),
        '--source', str(source), '--leaf', config['job']['checks']['supplementary']['leaf'],
        '--case', config['experiment_goal']['target_case'], '--output', str(output)]
    environment = dict(ASCEND_RT_VISIBLE_DEVICES=str(config['device']),
        PYTHONPATH=config.get('device_pythonpath', ''), OMP_NUM_THREADS='1',
        PYTHONDONTWRITEBYTECODE='1')
    shell = 'source ' + shlex.quote(config['sdk'] + '/set_env.sh') + '\n'
    shell += '\n'.join('export ' + key + '=' + shlex.quote(value) for key, value in environment.items())
    shell += '\nexec ' + shlex.join(command)
    save(run / 'status.json', dict(status='waiting_for_capture_lock', source_sha256=sha(source)))
    with Path(config['lock']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        save(run / 'status.json', dict(status='capturing_target_ir', source_sha256=sha(source)))
        with (run / 'baseline-capture.log').open('x') as log:
            subprocess.run(['docker', 'exec', config['container'], 'bash', '-c', shell],
                stdout=log, stderr=subprocess.STDOUT, timeout=2400, check=True)
    manifest = read(output / 'manifest.json')
    assert manifest['status'] == 'captured' and manifest['source_sha256'] == sha(source)
    save(run / 'status.json', dict(status='baseline_ir_captured', specializations=len(manifest['entries']),
        source_sha256=sha(source)))


def supply_lowered(run):
    manifest = read(run / 'baseline-capture/manifest.json')
    workspace = run / 'workspace'
    entries = []
    for entry in manifest['entries']:
        for item in entry['artifacts']:
            if item['kind'] not in {'ttadapter', 'bcmlir', 'npubin'}:
                continue
            original = run / 'baseline-capture' / item['path']
            assert sha(original) == item['sha256']
            record = dict(item, ast_sha256=entry['ast_sha256'], symbol=entry['symbol'], original=str(original))
            if item['kind'] != 'npubin':
                relative = Path('compiled-readonly') / entry['ast_sha256'] / entry['cache_hash'] / original.name
                target = workspace / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    assert sha(target) == item['sha256']
                else:
                    target.write_bytes(original.read_bytes())
                record['path'] = relative.as_posix()
            entries.append(record)
    save(workspace / 'lowered-inventory.json', dict(source_sha256=manifest['source_sha256'], entries=entries,
        scope='Read-only actual lowering and binary identities from the frozen-source capture'))


def seed(run):
    config = read(run / 'config.json')
    manifest = read(run / 'baseline-capture/manifest.json')
    workspace = run / 'workspace'
    inventory = []
    ast_options = {}
    for entry in manifest['entries']:
        ir = next(item for item in entry['artifacts'] if item['kind'] == 'ttir')
        original = run / 'baseline-capture' / ir['path']
        assert sha(original) == ir['sha256']
        key = entry['ast_sha256']
        options = entry['compiler_options']
        if key in ast_options:
            from ir_override_runtime import option_identity
            assert option_identity(ast_options[key]) == option_identity(options), 'AST reused with different options; isolate binding before IR edits'
            continue
        ast_options[key] = options
        folder = workspace / 'ir' / key
        folder.mkdir(parents=True, exist_ok=False)
        candidate = folder / original.name
        candidate.write_bytes(original.read_bytes())
        inventory.append(dict(ast_sha256=key, symbol=entry['symbol'], filename=original.name,
            path=candidate.relative_to(workspace).as_posix(), original=str(original),
            original_sha256=ir['sha256'], compiler_options=options, target=entry['target'],
            signature=entry['signature'], constants=entry['constants']))
    save(workspace / 'ir-inventory.json', dict(source_sha256=config['source_sha256'],
        compiler_identity=manifest['compiler_identity'], entries=inventory))
    save(workspace / 'diagnosis.json', dict(source_sha256=config['source_sha256'], selected_layer='single_operator_ir',
        evidence=[], hypothesis='', change='', expected='', validation=''))
    supply_lowered(run)
    save(run / 'status.json', dict(status='original_ir_ready', source_sha256=config['source_sha256'],
        specializations=len(inventory)))


def implementation(run, invocation, include_overrides=True, original_ir=False):
    config = read(run / 'config.json')
    workspace = run / 'workspace'
    assert sha(workspace / 'frozen.py') == config['source_sha256']
    inventory = read(workspace / 'ir-inventory.json')['entries']
    output = run / 'evaluations' / invocation
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / 'implementation'
    source_dir.mkdir()
    shutil.copyfile(workspace / 'frozen.py', source_dir / 'frozen.py')
    shutil.copyfile(Path(config['tools']) / 'ir_override_runtime.py', source_dir / 'ir_override_runtime.py')
    overrides = []
    if include_overrides:
        for item in inventory:
            raw = Path(item['original']) if original_ir else workspace / item['path']
            assert sha(Path(item['original'])) == item['original_sha256']
            destination = source_dir / item['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw, destination)
            overrides.append(dict(item, sha256=sha(destination)))
    manifest = dict(schema='hgkda.ir-override-input/v1', source='frozen.py',
        source_sha256=config['source_sha256'], overrides=overrides,
        compiler_identity=read(workspace / 'ir-inventory.json')['compiler_identity'],
        cache_root=str(output / 'compiler-state'), receipt_directory=str(output / 'compiler-receipts'))
    save(source_dir / 'ir-manifest.json', manifest)
    loader = ('import importlib.util\nimport pathlib\nimport sys\n'
        '_root = pathlib.Path(__file__).parent\n'
        '_helper_spec = importlib.util.spec_from_file_location("_ir_override_runtime", _root / "ir_override_runtime.py")\n'
        '_helper = importlib.util.module_from_spec(_helper_spec)\n'
        '_helper_spec.loader.exec_module(_helper)\n'
        '_source = _helper.install(_root / "ir-manifest.json")\n'
        '_spec = importlib.util.spec_from_file_location("_ir_frozen_candidate", _source)\n'
        '_module = importlib.util.module_from_spec(_spec)\n'
        'sys.modules[_spec.name] = _module\n'
        '_spec.loader.exec_module(_module)\n'
        'run = _module.run\n')
    (source_dir / 'main.py').write_text(loader, encoding='utf8')
    request = copy.deepcopy(read(config['job']['request']))
    request['implementation'].update(name='ir_' + run.name + '_' + invocation, entrypoint='main.py::run',
        sources=[dict(path=path.relative_to(source_dir).as_posix(), content=path.read_text(encoding='utf8'))
            for path in sorted(source_dir.rglob('*')) if path.is_file()])
    save(output / 'request.json', request)
    save(output / 'provenance.json', dict(source_sha256=config['source_sha256'],
        manifest_sha256=sha(source_dir / 'ir-manifest.json'), override_count=len(overrides),
        changed_ir_count=sum(item['sha256'] != item['original_sha256'] for item in overrides),
        source_files=[dict(path=p.relative_to(source_dir).as_posix(), sha256=sha(p))
            for p in sorted(source_dir.rglob('*')) if p.is_file()]))
    return output, request


def evaluate(run, invocation, baseline=False, resume=False):
    config = read(run / 'config.json')
    inventory = read(run / 'workspace/ir-inventory.json')['entries']
    changed = {item['path']: sha(run / 'workspace' / item['path']) for item in inventory
        if sha(run / 'workspace' / item['path']) != item['original_sha256']}
    original_ir = invocation == 'unchanged-ir'
    if original_ir:
        changed = {}
    if changed and not baseline:
        review = read(run / 'agent-calls' / invocation / 'ir-review.json')
        assert review['decision'] == 'approved' and review['source_sha256'] == config['source_sha256']
        assert review['changed_ir'] == changed, 'Review must bind every actual changed IR file'
    if resume:
        output = run / 'evaluations' / invocation
        assert not (output / 'result.json').exists(), 'Completed evaluation requires a new measurement event'
        request = read(output / 'request.json')
        provenance = read(output / 'provenance.json')
        assert provenance['source_sha256'] == config['source_sha256']
        for item in provenance['source_files']:
            assert sha(output / 'implementation' / item['path']) == item['sha256']
    else:
        output, request = implementation(run, invocation, include_overrides=not baseline, original_ir=original_ir)
    assert not read(output / 'implementation/ir-manifest.json').get('analysis_target_alias'), 'Formal evaluation keeps exact device target binding'
    sys.path.insert(0, str(Path(config['package']) / 'src'))
    from hgkda.adapters.kgs import call, upload_bundle
    from hgkda.runtime.evaluate import check
    archive = Path(config['job']['archive'])
    assert 'sha256:' + sha(archive) == request['binding']['bundle_id']
    operation = 'ir-campaign-' + run.name + '-' + invocation + '-' + sha(output / 'request.json')[:12]
    save(output / 'operation.json', dict(operation_id=operation, request_sha256=sha(output / 'request.json')))
    save(run / 'status.json', dict(status='waiting_for_formal_lock', invocation=invocation,
        source_sha256=config['source_sha256'], operation_id=operation))
    with Path(config['lock']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        save(run / 'status.json', dict(status='evaluating', invocation=invocation,
            source_sha256=config['source_sha256'], operation_id=operation))
        upload_bundle(config['kgs_server'], archive)
        result = call(config['kgs_server'], '/evaluate', request, timeout=2400,
            operation_id=operation)
        save(output / 'result.json', result)
        checks = {}
        if result.get('status') == 'PASSED':
            from ir_override_runtime import verify_receipts
            verification = verify_receipts(output / 'compiler-receipts',
                read(output / 'implementation/ir-manifest.json'))
            save(output / 'compiler-verification.json', verification)
            save(output / 'formal-compiler-receipts.json', dict(
                source_sha256=config['source_sha256'],
                scope='Closed formal evaluation receipts, captured before supplementary processes',
                files=[dict(path=path.relative_to(output).as_posix(), sha256=sha(path))
                       for path in sorted((output / 'compiler-receipts').glob('*.jsonl'))]))
        if changed and not baseline and result.get('status') == 'PASSED':
            for name, spec in dict(config['job']['checks'], **config.get('special_checks', {})).items():
                checks[name] = check(config, name, spec, output / 'implementation/main.py', output / (name + '.json'))
        save(output / 'supplementary-summary.json', checks)
    target = next((row for row in result.get('per_workload', [])
        if row['uuid'] == config['experiment_goal']['target_case'] and row['phase'] == 'timing'), None)
    save(run / 'status.json', dict(status='formal_result', invocation=invocation,
        source_sha256=config['source_sha256'], result_status=result.get('status'), target=target))
    print(json.dumps(dict(operator=config['operator'], invocation=invocation,
        status=result.get('status'), target=target)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'capture', 'seed', 'evaluate'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', default='unchanged-ir')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume a preserved request without a formal result')
    args = parser.parse_args()
    if args.action == 'setup':
        setup(args.run)
    elif args.action == 'capture':
        capture(args.run)
    elif args.action == 'seed':
        seed(args.run)
    else:
        evaluate(args.run, args.invocation, args.baseline, args.resume)


if __name__ == '__main__':
    main()
