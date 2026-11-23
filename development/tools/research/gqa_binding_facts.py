"""Bind GQA full-case, hardware profiling and partial simulator artifacts."""
import argparse
from pathlib import Path

from ir_campaign import read, save, sha


def facts(run, source=None):
    source = source or run / 'workspace/candidate.py'
    digest = sha(source)
    result = read(run / 'parent-validation/results.json')
    assert result['source_sha256'] == digest
    assert result['contract_sha256'] == sha(run / 'contract.json')
    assert {r['case_id'] for r in result['cases']} == {1, 2, 4, 7}
    assert all(r['status'] == 'measured' and all(r[side + '_correctness']['official_checker']['passed']
        for side in ['candidate', 'native']) for r in result['cases'])
    manifest_path = run / 'capture-parent/candidate/actual-ir/manifest.json'
    manifest = read(manifest_path)
    assert manifest['source_sha256'] == digest and manifest['contract_sha256'] == sha(run / 'contract.json')
    target = next(r for r in read(run / 'contract.json')['cases'] if r['case_id'] == 4)
    assert manifest['case'] == target
    for row in manifest['files']:
        assert sha(manifest_path.parent / row['path']) == row['sha256']
    binaries = [row for row in manifest['files'] if row['path'].endswith('.npubin')]
    binary = next(row for row in binaries if Path(row['path']).name == '_gqa_fwd_kernel.npubin')['sha256']
    formal = [dict(path=str(p), sha256=sha(p)) for p in (run / 'parent-validation/cache').rglob('*.npubin')]
    matches = [r for r in formal if r['sha256'] == binary]
    assert matches, 'Profile binary absent from formal candidate cache'
    all_compiled = []
    for row in binaries:
        matching = [item for item in formal if item['sha256'] == row['sha256']]
        debug_identity = None
        if not matching and Path(row['path']).name != '_gqa_fwd_kernel.npubin':
            from elf_debug_identity import verify
            for item in formal:
                if Path(item['path']).name != Path(row['path']).name:
                    continue
                try:
                    debug_identity = verify(item['path'], manifest_path.parent / row['path'])
                except AssertionError:
                    continue
                matching = [dict(item, identity='same_executable_and_data_debug_locations_differ')]
                break
        assert matching, 'Auxiliary profile binary absent from formal candidate cache'
        all_compiled.append(dict(profile_path=row['path'], sha256=row['sha256'], formal_matches=matching,
                                 debug_identity=debug_identity))
    replay = read(run / 'later-stage-replay.json') if (run / 'later-stage-replay.json').exists() else None
    auxiliary = read(run / 'auxiliary-profiling.json') if (run / 'auxiliary-profiling.json').exists() else None
    if len(binaries) > 1:
        assert replay and auxiliary
        assert sha(replay['manifest']) == replay['manifest_sha256']
        snapshot = read(replay['manifest'])
        assert snapshot['source_sha256'] == digest and snapshot['case'] == target
        assert snapshot['contract_sha256'] == sha(run / 'contract.json')
        assert sha(Path(replay['manifest']).parent / snapshot['tensors']) == snapshot['tensors_sha256']
        expected = {Path(row['path']).stem for row in binaries} - {'_gqa_fwd_kernel'}
        assert set(snapshot['preceding_real_launches']) == expected
        assert {row['kernel'] for row in auxiliary['kernels']} == expected
        for capture in auxiliary['kernels']:
            assert sha(capture['manifest']) == capture['manifest_sha256']
            actual = read(capture['manifest'])
            assert actual['source_sha256'] == digest
            assert actual['case'] == target and actual['contract_sha256'] == sha(run / 'contract.json')
            assert {row['sha256'] for row in actual['files'] if row['path'].endswith('.npubin')} == {
                row['sha256'] for row in binaries}
    simulator = run / 'simulator-parent'
    sim_cache = simulator / 'cache'
    simulator_compiler_binding = None
    config = read(run / 'config.json')
    if config.get('bcmlir_manifest'):
        import json
        ir_manifest_path = Path(config['bcmlir_manifest'])
        ir_manifest = read(ir_manifest_path)
        assert ir_manifest['source_sha256'] == digest
        assert ir_manifest['contract_sha256'] == sha(run / 'contract.json')
        receipts = [json.loads(line) for line in Path(ir_manifest['receipts']).read_text().splitlines()]
        assert any(row['manifest_sha256'] == sha(ir_manifest_path) and row['override_applied']
                   and row['compiled'] and row['binary_sha256'] == binary for row in receipts)
        sim_cache = Path(ir_manifest['cache']) / sha(ir_manifest_path)
        simulator_compiler_binding = dict(manifest=str(ir_manifest_path),
            manifest_sha256=sha(ir_manifest_path), changed_ir_sha256=ir_manifest['changed_ir_sha256'],
            cache=str(sim_cache), mode='source_bound_bcmlir_private_cache')
    sim_binaries = [dict(path=str(p), sha256=sha(p)) for p in sim_cache.rglob('*.npubin')]
    assert sim_binaries and {r['sha256'] for r in sim_binaries} == {binary}
    summary = read(simulator / 'summary.json')
    assert summary['source_sha256'] == digest and summary['case'] == target
    assert read(run / 'observation-status.json')['status'] == 'parent_dual_evidence_ready'
    assert all(r['absolute_unit_validation']['status'] == 'verified' for r in summary['reports'])
    comparison = read(run / 'paired-parent/comparison.json')
    candidate = comparison.get('result', comparison)['candidate']
    metrics = {}
    for table_name, table in candidate['tables'].items():
        for key, value in table['fields'].items():
            if value.get('valid', 0) and (key.startswith(('aic_', 'aiv_')) or key.endswith('_datas(KB)')):
                metrics.setdefault(key, dict(value, table=table_name))
    reports = [dict(core=r['core'], clock=r['absolute_unit_validation'],
        raw_window=r['timeline'].get('raw_clock_window'), instruction_calls=r['instruction_calls'],
        pipelines=[{key: lane.get(key) for key in
            ['pipeline', 'event_count', 'productive_ratio', 'wait_only_ratio', 'idle_ratio']}
            for lane in r['timeline']['pipelines']], top_instructions=r['top_instructions'])
        for r in summary['reports']]
    return dict(status='formal_profile_simulator_binary_verified', source_sha256=digest,
        source_snapshot=str(source),
        contract_sha256=sha(run / 'contract.json'), result_sha256=sha(run / 'parent-validation/results.json'),
        target_case=4, target_binary_sha256=binary, matching_formal_binaries=matches,
        all_profile_binaries=all_compiled, auxiliary_profiling=auxiliary, later_stage_replay=replay,
        simulator_cache_binaries=sim_binaries, profile_manifest_sha256=sha(manifest_path),
        simulator_compiler_binding=simulator_compiler_binding,
        simulator_summary_sha256=sha(simulator / 'summary.json'),
        application_completed=summary['application_completed'], execution=read(simulator / 'execution.json'),
        full_cases=result['cases'], profile_basic=candidate['basic'], metrics=metrics, reports=reports,
        scope='Exact target binary shared by formal benchmark, hardware profiling and isolated simulator cache. '
            'Simulator invocation completion and per-core capture boundaries remain separate; partial-window counts '
            'and fractions are interpreted in each window. Four official cases qualify this research subset.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, help='Immutable source snapshot for an earlier observation')
    args = parser.parse_args()
    value = facts(args.run.resolve(), args.source.resolve() if args.source else None)
    save(args.run / 'binary-binding-and-facts.json', value)
    print({k: value[k] for k in ['status', 'target_binary_sha256', 'application_completed']})
