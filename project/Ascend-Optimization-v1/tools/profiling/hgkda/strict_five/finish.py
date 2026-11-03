"""离线核验剩余五项的终态，合并原30项总表；不提交设备任务或重复计时。"""
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
BATCH = PROJECT / 'operators/campaigns/strict-five-20261103'
PREVIOUS = BATCH.parent / 'strict-night-20261102'
REMOTE = '/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/'


def physical(path):
    path = Path(path)
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' else path


def read(path):
    return json.loads(physical(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(physical(path).read_bytes()).hexdigest()


def save(path, value):
    physical(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8', newline='\n')


def main():
    sys.path.insert(0, str(HERE.parent / 'strict_night'))
    from measurements import score
    from expm1_exact_probe import qualified
    evidence = {}

    def track(name, expected=None):
        raw = str(name)
        path = PROJECT / raw[len(REMOTE):] if raw.startswith(REMOTE) else Path(name)
        path = path.resolve()
        relative = path.relative_to(PROJECT).as_posix()
        digest = sha(path)
        if expected is not None:
            assert digest == expected, relative
        evidence[REMOTE + relative] = digest
        return path

    def load(name):
        return read(track(name))

    previous = load(BATCH / 'starting-delivery.json')
    plan = load(BATCH / 'plan.json')
    track(PREVIOUS / 'delivery.json', plan['starting_delivery_sha256'])
    assert previous == read(PREVIOUS / 'delivery.json') and previous['accepted'] == 25
    old_audit = load(PREVIOUS / 'analysis/final-goal-audit.json')
    assert old_audit['status'] == 'verified'
    # 继承原审计的实体身份，不能只继承25这个计数。
    for name, digest in previous['evidence_sha256'].items():
        track(name, digest)
    for row in previous['rows']:
        track(row['source'], row['source_sha256'])
    for record in plan['records']:
        for name, digest in record['contract_sha256'].items():
            track(BATCH / record['catalog_leaf'] / name, digest)
    runtime = load(BATCH / 'analysis/terminal-runtime-proof.json')
    assert runtime['status'] == 'verified' and runtime['physical_card'] == 7
    assert runtime['service_idle'] and runtime['project_card_lock_available']
    assert runtime['all_workspaces_terminal'] and all(runtime['workspaces'].values())
    assert runtime['service']['timing'] == 'walltime'
    for name, item in runtime['processes'].items():
        value = load(track(BATCH / name, item['sha256']))
        assert value['status'] == 'completed' and not item['identity_live']
    state = load(BATCH / 'search-state.json')
    assert state['status'] == 'search_completed_pending_review' and state['current_job'] is None
    assert len(state['jobs']) == 2 and all(v['status'] == 'completed' for v in state['jobs'].values())
    for archive in sorted((BATCH / 'search-archives/expm1_').glob('*.tar.gz')):
        manifest = load(archive.with_name(archive.name + '.manifest.json'))
        track(archive, manifest['archive_sha256'])
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            assert {m.name for m in members} == set(manifest['files'])
            assert len(members) == len(manifest['files'])
            for member in members:
                assert member.isfile() and not member.issym()
                content = stream.extractfile(member).read()
                item = manifest['files'][member.name]
                assert len(content) == item['bytes']
                assert hashlib.sha256(content).hexdigest() == item['sha256']

    selected = load(BATCH / 'reviewed-best-candidate.json')
    completion = load(BATCH / 'reviewed-best-result.json')
    assert completion['status'] == 'verified_original_PyTorch_only' and not completion['CANN_accepted']
    source = track(selected['source'], selected['source_sha256'])
    digest = selected['source_sha256']
    assert digest == completion['source_sha256']
    assert selected['qualification'] == 'passed' and selected['compiled_status'] == 'verified'
    assert selected['review_status'] == 'reviewed_strict_triton'
    review = load(source.with_name('review.json'))
    assert review['source_sha256'] == digest and review['all_branches_reviewed'] and review['dependencies_reviewed']
    proof = load(source.with_name('backend-verification.json'))
    assert proof['status'] == 'verified' and proof['source_sha256'] == digest
    for key in ['review', 'runtime', 'trace']:
        track(proof[key], proof[key + '_sha256'])
    assert proof['tasks'] and all('_expm1_' in task['name'] for task in proof['tasks'])
    compiled = load(proof['compiled'])
    manifest = load(track(compiled['manifest'], compiled['manifest_sha256']))
    assert manifest['spec']['source_sha256'] == digest and manifest['entries']
    for kernel in manifest['entries']:
        assert {'ttir', 'npubin'} <= {v['kind'] for v in kernel['artifacts']}
        for artifact in kernel['artifacts']:
            track(artifact['path'], artifact['sha256'])

    formal_dir = source.parent / 'formal-reference'
    formal_path = track(formal_dir / 'result.json')
    formal = read(formal_path)
    current_score = score(formal)
    assert math.isclose(current_score, selected['search_score'], rel_tol=1e-10)
    solution = load(formal_dir / 'solution.json')
    assert len(solution['sources']) == 1 and solution['sources'][0]['content'] == physical(source).read_text(encoding='utf8')
    track(formal_dir / 'main.py', digest)
    load(formal_dir / 'identity.json')
    assert load(formal_dir / 'run-process.json')['finished_at']
    contract = load(formal_dir / 'evaluation-contract.json')
    leaf = BATCH / 'catalog/ops/expm1_'
    assert contract['definition']['reference'] == (leaf / 'oracle.py').read_text(encoding='utf8')
    expected = set()
    for filename, phase in [('correctness.jsonl', 'correctness'), ('timing.jsonl', 'timing')]:
        records = [json.loads(line) for line in (leaf / filename).read_text(encoding='utf8').splitlines() if line.strip()]
        assert contract[phase + '_workloads'] == records
        expected.update((v['name'], phase) for v in records)
    assert expected == {(v['uuid'], v['phase']) for v in formal['per_workload']}
    supplement = load(completion['supplement'])
    assert supplement['source_sha256'] == digest and supplement['status'] == 'passed'
    assert supplement['passed'] == supplement['total'] == 63
    exact = load(completion['exact'])
    assert qualified(exact, digest)
    track(source.parent / ('exact-expm1-checker-' + exact['checker_sha256'] + '.py'), exact['checker_sha256'])
    domain = load(completion['full_fp16_domain'])
    assert domain['source_sha256'] == digest and domain['status'] == 'passed'
    assert domain['passed'] == domain['total'] == 4
    track(source.parent / ('half-domain-checker-' + domain['checker_sha256'] + '.py'), domain['checker_sha256'])
    assert all(r['passed'] and r['unique_input_bits'] == 65536 and r['storage_guards_passed']
               and r['zero_sign_passed'] and r['returned_input_object'] for r in domain['cases'])
    runtime_check = load(proof['runtime'])
    assert runtime_check['status'] == 'passed' and runtime_check['source_sha256'] == digest
    followups = []
    domain_summary = load(BATCH / 'half-domain-summary.json')
    assert domain_summary['status'] == 'completed' and len(domain_summary['proposals']) == 2
    for proposal in domain_summary['proposals']:
        folder = BATCH / 'proposals/expm1_' / proposal['proposal']
        result = load(folder / 'result.json')
        assert result['status'] == 'evaluated' and result['qualified_original'] and not result['CANN_accepted']
        proposal_source = track(folder / 'main.py', result['source_sha256'])
        value = load(result['formal'])
        assert math.isclose(score(value), result['PyTorch'], rel_tol=1e-10)
        assert result['formal'].endswith('.result.json')
        request_path = result['formal'].removesuffix('.result.json') + '.request.json'
        payload = load(request_path)
        assert payload['implementation']['sources'][0]['content'] == physical(proposal_source).read_text(encoding='utf8')
        manual = load(folder / 'review.json')
        assert manual['source_sha256'] == result['source_sha256'] and manual['status'] == 'reviewed_strict_triton'
        assert manual['all_branches_reviewed'] and manual['dependencies_reviewed']
        assert result['PyTorch'] < current_score and not result['improved_over_frozen']
        full = load(proposal['result'])
        assert full['status'] == 'passed' and full['source_sha256'] == result['source_sha256']
        assert full['passed'] == full['total'] == 4
        for name in [result['supplement'], result['exact']]:
            checked = load(name)
            assert checked['status'] == 'passed' and checked['source_sha256'] == result['source_sha256']
        followups.append(dict(proposal=proposal['proposal'], score=result['PyTorch'], selected=False))

    uniform = load(BATCH / 'reference-probes/uniform-tensor-v1/numerics.json')
    assert uniform['passed'] == 6 and uniform['total'] == 12 and not uniform['qualified_CANN']
    assert sum(not r['passed'] for r in uniform['cases']) == 6
    build = load(BATCH / 'reference-probes/uniform-tensor-v1/build.json')
    track(build['binary'], build['binary_sha256'])
    track(build['source'], build['source_sha256'])
    track(HERE / 'probe_uniform_tensor.py', uniform['checker_sha256'])
    for name in ['sdk-reference-symbols-v1/result.json', 'legacy-kernel-symbols-v2/filtered-summary.json',
                 'installed-expm1-impl-v1/result.json', 'expm1_/paired-start-v1/summary.json',
                 'expm1_/paired-start-v1/comparison.json', 'expm1_/formal-improvement-v1.json']:
        load(BATCH / 'analysis' / name)
    drift_path = PROJECT / 'operators/campaigns/reference-closure-20261031/analysis/cumulative_trapezoid/reference-stability-replay-v1/default.json'
    drift = load(drift_path)
    assert not drift['deterministic'] and drift['empty_tolerance_intersections'] > 0
    assert drift['max_lower'] > drift['min_upper']
    native_expm1 = load(PROJECT / 'operators/campaigns/cann-priority-20261102/references/expm1_/extra-checks.json')
    assert native_expm1['status'] == 'failed' and native_expm1['passed'] == 60 and native_expm1['total'] == 63

    delivery = copy.deepcopy(previous)
    rows = delivery['rows']
    row = next(v for v in rows if v['operator'] == 'expm1_')
    old_score = row['pytorch']['score']
    assert not row['accepted'] and current_score > old_score
    row.update(source=selected['source'], source_sha256=digest, selection='new_verified_original_PyTorch_candidate',
               status='原PyTorch正确；性能不足且CANN参考未合格', review=review,
               pytorch=dict(status='PASSED', score=current_score, path=REMOTE + formal_path.relative_to(PROJECT).as_posix(),
                            passed=formal['num_passed'], total=formal['num_workloads'], measurement='existing_once'),
               supplementary=dict(status='passed', path=completion['supplement'], source_sha256=digest, passed=63, total=63),
               runtime=dict(status='passed', path=proof['runtime'], passed=runtime_check['passed'], total=runtime_check['total']),
               compiled_path=dict(status='verified', path=proof['compiled'], source_sha256=digest),
               device_path=dict(status='captured', trace=proof['trace'], trace_sha256=proof['trace_sha256']),
               exact_expm1_probe=dict(path=completion['exact'], passed=12, total=12),
               full_fp16_domain=dict(path=completion['full_fp16_domain'], passed=4, total=4),
               provenance=selected['provenance'])
    assert row['cann'] == {} and not row['accepted']
    for frozen, result in zip(previous['rows'], rows):
        if frozen['accepted']:
            assert frozen == result, '已达标25项不能被本轮改写'
    assert len(rows) == 30 and len({r['operator'] for r in rows}) == 30
    assert sum(r['accepted'] for r in rows) == 25
    delivery.update(schema='hgkda-strict-five-delivery-v1', queue_status='completed_with_documented_gaps',
                    current_job=None, frozen_accepted=25, newly_accepted=0, accepted=25,
                    evidence_sha256=evidence, followups=followups, terminal_proof=runtime,
                    scope='本轮选中源码及参考缺口的终态交付；25项沿用冻结证据。未宣称30/30达标。')
    save(BATCH / 'delivery.json', delivery)
    lines = ['# 严格Triton30项总表', '', '当前25/30：1015为7/10，1029为18/20。验收要求为相对合格CANN／Ascend C参考超过0.8×，并通过原正确性、补测、人工审查及实际Triton后端核验。',
             '', '所有成绩均来自KGS完整调用walltime。已达标25项沿用冻结源码和原测量；本轮仅更新expm1_的合格候选，未新增CANN达标项。', '']
    fmt = lambda v: f'{v:.3f}×' if v is not None else '—'
    for group in ['1015', '1029']:
        lines += [f'## {group}', '', '|算子|相对CANN／Ascend C|相对PyTorch|完整评测|状态|', '|---|---:|---:|---|---|']
        for r in rows:
            if r['group'] != group:
                continue
            pt = r['pytorch']
            lines.append(f'|`{r["operator"]}`|{fmt(r["cann"].get("score"))}|{fmt(pt.get("score"))}|{pt.get("passed", "—")}/{pt.get("total", "—")}|{r["status"]}|')
        lines.append('')
    lines += ['`fractional_max_pool3d`和`linalg_inv_ex`的PyTorch参考包含CPU回退，数值不能作为相对原生CANN的性能结论。累计积分仍有FP16补测失败。源码SHA、参考类型及原始证据路径见delivery.json。', '']
    (BATCH / '严格Triton30项总表.md').write_text('\n'.join(lines), encoding='utf8', newline='\n')
    old_row = next(r for r in previous['rows'] if r['operator'] == 'expm1_')
    old = load(old_row['pytorch']['path'])
    old_timing = {r['uuid']: r for r in old['per_workload'] if r['phase'] == 'timing'}
    new_timing = [r for r in formal['per_workload'] if r['phase'] == 'timing']
    gain = math.exp(sum(math.log(old_timing[v['uuid']]['latency_ms'] / v['latency_ms']) for v in new_timing) / len(new_timing))
    comparison = load(BATCH / 'analysis/expm1_/formal-improvement-v1.json')
    assert comparison['new_source_sha256'] == digest
    assert math.isclose(comparison['candidate_actual_speedup'], gain, rel_tol=1e-10)
    with (BATCH / 'expm1逐case对照.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['用例', '旧候选毫秒', '新候选毫秒', '旧PyTorch毫秒', '新PyTorch毫秒', '候选实际加速', '新相对PyTorch'])
        for value in formal['per_workload']:
            if value['phase'] != 'timing':
                continue
            before = old_timing[value['uuid']]
            writer.writerow([value['uuid'], before['latency_ms'], value['latency_ms'], before['reference_latency_ms'],
                             value['reference_latency_ms'], before['latency_ms'] / value['latency_ms'],
                             value['reference_latency_ms'] / value['latency_ms']])
    # 报告引用的新增旧成绩也纳入最终实体审计；不改变冻结测量。
    delivery['evidence_sha256'] = evidence
    save(BATCH / 'delivery.json', delivery)
    audit = dict(status='verified', operators=30, frozen_accepted=25, accepted=25, newly_accepted=0,
                 evidence_files=len(evidence), selected_source_sha256=digest, original_contract_files=20,
                 original_PyTorch_formal='30/30', supplement='63/63', exact_blocks='12/12', full_fp16_paths='4/4',
                 rejected_slower_proposals=followups, uniform_tensor_reference='6/12，未取得参考资格',
                 delivery_sha256=sha(BATCH / 'delivery.json'),
                 scope='核验原文件身份、逐case成绩、原合同、全部在途进程终态与选中稿门禁；不代表30项性能达标。')
    save(BATCH / 'final-audit.json', audit)
    print(json.dumps(dict(status='verified', accepted=25, evidence_files=len(evidence), old_PyTorch=old_score,
                          new_PyTorch=current_score, score_improvement=current_score / old_score - 1), ensure_ascii=False))


if __name__ == '__main__':
    main()
