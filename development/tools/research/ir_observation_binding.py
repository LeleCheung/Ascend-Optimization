"""Verify post-edit profiling and simulator identities against formal binaries."""
import argparse
import json
from pathlib import Path

if __package__:
    from .ir_binary_compare import binaries, read, sha
else:
    from ir_binary_compare import binaries, read, sha


def observed(spec, formal_manifest, formal_binaries, required):
    binding = spec['ir_input']
    path = Path(binding['manifest'])
    assert sha(path) == binding['manifest_sha256']
    manifest = read(path)
    source = formal_manifest['source_sha256']
    assert source == spec['source_sha256'] == binding['source_sha256'] == sha(spec['source'])
    for name, expected_sha in binding['resources'].items():
        resource = (path.parent / name).resolve()
        assert resource.is_relative_to(path.parent.resolve())
        assert sha(resource) == expected_sha, 'Observation resource bytes changed'
    assert manifest['overrides'] == formal_manifest['overrides'], 'Observation IR differs from formal evaluation'
    expected = {row['ast_sha256']: row for row in manifest['overrides']}
    actual = {}
    receipts = sorted(Path(manifest['receipt_directory']).glob('*.jsonl'))
    assert receipts, 'Observation has no compiler receipts'
    for receipt in receipts:
        for line in receipt.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            assert row['source_sha256'] == source and row['manifest_sha256'] == sha(path)
            if not row['override_applied']:
                continue
            key = row['ast_sha256']
            assert row['ir_sha256'] == expected[key]['sha256']
            artifact = next(item for item in row['artifacts'] if item['kind'] == 'npubin')
            assert sha(artifact['path']) == artifact['sha256'] == formal_binaries[key], 'Observation binary differs from formal evaluation'
            assert key not in actual or actual[key] == artifact['sha256']
            actual[key] = artifact['sha256']
    assert required <= set(actual), 'Target binary absent from observation receipts'
    return dict(spec_source_sha256=source, manifest_sha256=sha(path), binaries=actual,
                receipt_files=[dict(path=str(path), sha256=sha(path)) for path in receipts])


def verify(run, invocation):
    evaluation = run / 'evaluations' / invocation
    observation = run / 'observations' / invocation
    formal_manifest_path = evaluation / 'implementation/ir-manifest.json'
    formal = read(formal_manifest_path)
    formal_bins = binaries(evaluation, formal['source_sha256'])
    status = read(observation / 'status.json')
    assert status['status'] == 'ir_evidence_collected'
    assert status['ir_manifest_sha256'] == sha(formal_manifest_path)
    pair_spec = read(observation / 'paired/spec.json')
    assert pair_spec['ir_input']['original_manifest_sha256'] == sha(formal_manifest_path)
    config = read(run / 'config.json')
    assert pair_spec['case']['name'] == config['experiment_goal']['target_case']
    required_pair = {item['ast_sha256'] for item in formal['overrides']}
    pair = observed(pair_spec, formal, formal_bins, required_pair)
    simulator = observation / 'simulator'
    spec = read(simulator / 'spec.json')
    request = read(simulator / 'request.json')
    summary = read(simulator / 'ir-summary.json')
    assert spec['case'] == pair_spec['case'] == request['case'] == summary['case']
    assert request['kernel'] == summary['kernel']
    target = {item['ast_sha256'] for item in formal['overrides'] if item['symbol'] == request['kernel']}
    assert target, 'Simulator kernel absent from formal IR manifest'
    replay = observed(spec, formal, formal_bins, target)
    if config['operator'] == 'scatter_add':
        launch = spec['kernel_replay']
        assert launch['kernel'] == request['kernel']
        assert sha(launch['manifest']) == launch['manifest_sha256']
        snapshot = read(Path(launch['manifest']))
        assert snapshot['source_sha256'] == formal['source_sha256']
        assert snapshot['case'] == spec['case'] and snapshot['kernel'] == request['kernel']
        snapshot_root = Path(launch['manifest']).resolve().parent
        tensors = (snapshot_root / snapshot['tensors']).resolve()
        assert tensors.is_relative_to(snapshot_root)
        assert sha(tensors) == snapshot['tensors_sha256'], 'Replay tensor bytes changed'
    return dict(status='formal_observation_binaries_verified', invocation=invocation,
        source_sha256=formal['source_sha256'], formal_manifest_sha256=sha(formal_manifest_path),
        formal_result_sha256=sha(evaluation / 'result.json'), pair=pair, simulator=replay,
        kernel=request['kernel'], case=spec['case'],
        application_completed=summary['application_completed'], execution=read(simulator / 'execution.json'),
        clocks=[dict(core=row['core'], validation=row['absolute_unit_validation']) for row in summary['reports']],
        summary_sha256=sha(simulator / 'ir-summary.json'), verifier_sha256=sha(__file__),
        scope='Exact source, IR, case and formal binary bindings for paired profiling and simulator replay. Capture windows, exits, clock coverage and mechanism conclusions require separate interpretation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--invocation', required=True)
    args = parser.parse_args()
    result = verify(args.run.resolve(), args.invocation)
    output = args.run / ('observation-binding-' + args.invocation + '.json')
    with output.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['status', 'invocation', 'kernel', 'application_completed']}))


if __name__ == '__main__':
    main()
