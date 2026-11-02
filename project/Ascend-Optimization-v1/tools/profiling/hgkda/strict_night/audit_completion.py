"""离线核验本轮真实搜索、最终选择和终态证据；不调用设备、不读取模型思考。"""
import hashlib
import json
import os
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[4]
BATCH=PROJECT/'operators/campaigns/strict-night-20261102'
REMOTE='/data/hanle/ascend-optimization/Ascend-Optimization/project/Ascend-Optimization-v1/'
PRIORITY={'hardswish','fix','hardshrink_backward','rad2deg_','round_out',
          'special_round_out','hardswish_out','log2_','fmod_tensor_'}
REMAINDER={'cumulative_trapezoid','linalg_inv_ex','expm1_','uniform','fractional_max_pool3d'}


def main():
    evidence={}
    def file(path):
        if str(path).startswith(REMOTE):path=PROJECT/str(path)[len(REMOTE):]
        path=Path(path).resolve();path.relative_to(PROJECT.resolve())
        relative=path.relative_to(PROJECT).as_posix()
        if os.name=='nt':path=Path('\\\\?\\'+str(path))
        evidence[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def read(path):return json.loads(file(path).read_text(encoding='utf8'))
    state=read(BATCH/'state.json');delivery=read(BATCH/'delivery.json')
    assert state['status']=='backend_verified' and state.get('current_job') is None and not state.get('error')
    assert all(job['status']=='completed' for job in state['jobs'].values()),'仍有未结束任务'
    finish=read(BATCH/'finish-pending-process.json')
    assert finish['status']=='completed' and finish['pending_best']==[]
    runtime=read(BATCH/'analysis/terminal-runtime-proof.json')
    assert runtime['status']=='verified' and runtime['project_card_lock_available']
    assert runtime['service_idle'] and runtime['all_workspaces_terminal'] and runtime['finisher_exited']
    assert runtime['finish_process_sha256']==evidence['operators/campaigns/strict-night-20261102/finish-pending-process.json']
    assert delivery['accepted']==25 and len(delivery['rows'])==30
    assert delivery['queue_status']=='backend_verified' and delivery['current_job'] is None
    assert delivery['groups']=={'1015':{'accepted':7,'total':10},'1029':{'accepted':18,'total':20}}
    previous=read(BATCH/'starting-delivery.json')
    assert previous['accepted']==16
    old={row['operator']:row for row in previous['rows']}
    rows={row['operator']:row for row in delivery['rows']}
    newly_accepted={op for op,row in rows.items() if row['accepted'] and not old[op]['accepted']}
    assert newly_accepted==PRIORITY and {op for op,row in rows.items() if not row['accepted']}==REMAINDER
    for op in PRIORITY:
        row=rows[op];best=state['operators'][op]['best']
        assert row['source_sha256']!=old[op]['source_sha256']
        assert best['reference']=='qualified_CANN' and best['qualification']=='passed'
        assert best['compiled_status']=='verified' and best['review_status']=='reviewed_strict_triton'
        assert row['cann']['score']>.8
    for op in REMAINDER:
        attempts=state['operators'][op]['attempts']
        assert attempts and all(item['status'] in {'candidates_evaluated','no_new_valid_candidate'} for item in attempts)
        assert all((item.get('candidates') or item['status']=='no_new_valid_candidate') for item in attempts)
    for name,digest in delivery['evidence_sha256'].items():
        file(name);assert evidence[str(name)[len(REMOTE):]]==digest
    contracts=read(BATCH/'analysis/frozen-contract-audit.json')
    assert contracts['status']=='verified' and contracts['operators']==30 and contracts['contract_file_comparisons']==240
    assert contracts['input_tolerance_reference_and_timing_files_unchanged']
    for name,digest in contracts['evidence_sha256'].items():
        file(PROJECT/name);assert evidence[name]==digest
    archives=[]
    for archive in sorted((BATCH/'search-archives').glob('*/*.tar.gz')):
        manifest=read(archive.with_name(archive.name+'.manifest.json'))
        file(archive);assert evidence[archive.relative_to(PROJECT).as_posix()]==manifest['archive_sha256']
        archives.append(dict(path=archive.relative_to(PROJECT).as_posix(),sha256=manifest['archive_sha256'],
            files=len(manifest['files'])))
    assert len(archives)==sum(len(row.get('attempts',[])) for row in state['operators'].values())
    tests=read(BATCH/'analysis/controller-tests-probe-recovery-v2.json')
    assert tests['returncode']==0 and 'Ran 54 tests' in tests['stderr'] and not tests['device_executed']
    out=BATCH/'proposals/expm1_/codex-exact-zero-v5'
    formal=read(out/'pytorch/Codex-strict-PyTorch-1.result.json')
    extra=read(out/'extra-pytorch.json');exact=read(out/'exact-expm1-probe/result.json')
    runtime_v5=read(out/'runtime-audit.json')
    assert formal['status']=='PASSED' and formal['num_passed']==formal['num_workloads']==30
    assert extra['status']=='passed' and extra['passed']==extra['total']==63
    assert runtime_v5['status']=='passed' and runtime_v5['passed']==runtime_v5['total']==19
    from expm1_exact_probe import qualified
    digest=hashlib.sha256(file(out/'main.py').read_bytes()).hexdigest()
    assert qualified(exact,digest)
    checker=file(out/('exact-expm1-checker-'+exact['checker_sha256']+'.py'))
    assert hashlib.sha256(checker.read_bytes()).hexdigest()==exact['checker_sha256']
    recovery=read(BATCH/'finish-pending-predevice-recovery.json')
    assert recovery['formal_sha256']==hashlib.sha256(file(out/'pytorch/Codex-strict-PyTorch-1.result.json').read_bytes()).hexdigest()
    assert recovery['supplement_sha256']==hashlib.sha256(file(out/'extra-pytorch.json').read_bytes()).hexdigest()
    for key in ['old_checker','error_log']:
        assert hashlib.sha256(file(recovery[key]).read_bytes()).hexdigest()==recovery[key+'_sha256']
    assert formal['geo_mean']<rows['expm1_']['pytorch']['score'] and rows['expm1_']['source_sha256']!=digest
    result=dict(status='verified',search_and_followup_finished=True,operators=30,frozen_accepted=16,
        accepted=25,newly_accepted=sorted(newly_accepted),groups=delivery['groups'],
        remaining=sorted(REMAINDER),completed_jobs=len(state['jobs']),search_archives=archives,
        tool_tests=54,original_contract_comparisons=240,
        expm1_v5=dict(formal='30/30',supplement='63/63',exact_blocks='12/12',runtime='19/19',
            PyTorch=formal['geo_mean'],selected=False,reason='低于已核验最佳'),
        delivery_sha256=hashlib.sha256(file(BATCH/'delivery.json').read_bytes()).hexdigest(),
        evidence_sha256=evidence,
        scope='核验冻结合同、真实搜索归档、选中最佳完整交付证据和终态回执；不执行NPU，不读取模型思考，不将五项缺口计达标。Git三端同步另由提交与同步回执证明。')
    target=BATCH/'analysis/final-goal-audit.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'evidence_sha256','search_archives','scope'}},ensure_ascii=False))


if __name__=='__main__':main()
