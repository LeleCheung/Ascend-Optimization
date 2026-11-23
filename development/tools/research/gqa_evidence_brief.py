"""Compact source-bound GQA evidence while preserving its full immutable reports."""
from ir_campaign import read, save, sha


def compact(workspace, include_simulator=True):
    profile_path = workspace / 'current-profiling.json'
    source_profile = read(profile_path)
    save(workspace / 'current-profiling-full.json', source_profile)
    full_profile_sha = sha(workspace / 'current-profiling-full.json')
    comparison = source_profile['comparison']
    actual = comparison.get('result', comparison)
    metrics = {'aic_cube_time(us)', 'aic_fixpipe_time(us)', 'aic_mte1_time(us)', 'aic_mte2_time(us)',
        'aic_mte1_instructions', 'aic_mte2_instructions', 'aic_scalar_wait_id1_time(us)',
        'aic_cube_total_instr_number', 'aiv_vec_time(us)', 'aiv_mte2_time(us)', 'aiv_mte3_time(us)',
        'aiv_mte2_instructions', 'aiv_mte3_instructions', 'aiv_scalar_wait_time(us)',
        'GM_to_L1_datas(KB)', 'L0C_to_GM_datas(KB)', 'GM_to_UB_datas(KB)', 'UB_to_GM_datas(KB)'}
    sides = {}
    for side in ['candidate', 'reference']:
        record = actual[side]
        sides[side] = dict(basic=record['basic'], tables={name: dict(fields={key: value for key, value in
            table['fields'].items() if key in metrics and value.get('valid')})
            for name, table in record['tables'].items() if any(key in metrics and value.get('valid')
                for key, value in table['fields'].items())})
    save(profile_path, dict(source_sha256=source_profile['source_sha256'], comparison=sides,
        full_report='current-profiling-full.json', full_report_sha256=full_profile_sha,
        scope='Exact selected fields from the full paired report. Per-program service and full-call timing remain separate.'))

    if not include_simulator:
        return
    sim_path = workspace / 'simulator-evidence.json'
    simulator = read(sim_path)
    save(workspace / 'simulator-evidence-full.json', simulator)
    reports = []
    for row in simulator['reports']:
        timeline = row['timeline']
        reports.append(dict(core=row['core'], source_sha256=row['source_sha256'],
            absolute_unit_validation=row['absolute_unit_validation'], limitations=row['limitations'],
            instruction_calls=row['instruction_calls'],
            timeline={**{key: value for key, value in timeline.items() if key != 'pipelines'},
                      'pipelines': [{key: lane.get(key) for key in ['pipeline', 'event_count', 'productive_ratio',
                          'wait_only_ratio', 'idle_ratio']} for lane in timeline['pipelines']]},
            top_instructions=[{key: value for key, value in instruction.items() if key != 'source_stack'}
                              for instruction in row['top_instructions'][:8]]))
    save(sim_path, dict(**{key: value for key, value in simulator.items() if key != 'reports'}, reports=reports,
             full_report='simulator-evidence-full.json', full_report_sha256=sha(workspace / 'simulator-evidence-full.json')))
