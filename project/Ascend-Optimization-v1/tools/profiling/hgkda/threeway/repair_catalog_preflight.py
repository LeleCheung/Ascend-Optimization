"""仅修复已证实在KG启动前失败的catalog元数据；不重发设备请求。"""
import datetime
import hashlib
import json
from pathlib import Path


def main():
    project=Path(__file__).resolve().parents[4]
    root=project.parents[2]
    batch=project/'operators/campaigns/threeway-20261030'
    runtime=root/'runs/threeway-20261030'
    path=batch/'next-state.json';state=json.loads(path.read_text())
    key='search:cross_out:1';job=state['jobs'][key]
    log=runtime/'cross_out-attempt-1.log'
    expected='catalog manifest not found: '+str(batch/'catalog-snapshots/cross_out-attempt-1/manifest.json')
    assert log.read_text().strip()==expected
    workspace=runtime/'kg-runs/cross_out-threeway-20261030-attempt-1'
    assert not workspace.exists(), '工作区已存在，不能推定尚未启动KG'
    assert job['status']=='running' and 'KG退出但完成元数据未知' in state.get('error','')
    plan=json.loads((batch/'plan.json').read_text())
    for record in plan['records']:
        for name,sha in record['contract_sha256'].items():
            assert hashlib.sha256((batch/record['catalog_leaf']/name).read_bytes()).hexdigest()==sha
    manifest=batch/'catalog/manifest.json'
    assert not manifest.exists()
    manifest.write_text(json.dumps(dict(api_version='v6.2',name=batch.name,evaluator='native',layout='per-operator'),indent=2)+'\n')
    resolution=dict(status='confirmed_preflight_failure_before_kg',log=str(log),
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),missing_workspace=str(workspace),
        added_manifest=str(manifest),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        contract_leaf_bytes_unchanged=True,recovery_attempt=2,
        reason='启动脚本在调用kg run前检查manifest并退出；无工作区、worker或已提交KGS请求可重发')
    (batch/'catalog-preflight-resolution.json').write_text(json.dumps(resolution,ensure_ascii=False,indent=2)+'\n')
    job.update(status='failed',result=resolution,error=state['error'],finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    state.setdefault('search_recovery_attempts',{})['cross_out']=2
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)
    print(json.dumps(resolution,ensure_ascii=True))


if __name__=='__main__':main()
