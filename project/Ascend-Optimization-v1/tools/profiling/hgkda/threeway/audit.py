"""收尾只读核验：合同/源码/结果、结束的worker、空闲服务、无未知请求及卡锁释放。"""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
from report import build


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    project=Path(__file__).resolve().parents[4]
    root=project.parents[2];batch=project/'operators/campaigns/threeway-20261030'
    runtime=root/'runs/threeway-20261030';state=read(batch/'next-state.json')
    running=[key for key,value in state['jobs'].items() if value['status']=='running']
    if running:raise ValueError('任务尚未结束 '+str(running))
    sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout.splitlines()
    active=[name for name in sessions if name.startswith('hanle-threeway-20261030-')]
    if active:raise ValueError('本批控制器仍运行 '+str(active))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));services={}
    for port in [19655,19657,19659]:
        response=json.load(opener.open('http://127.0.0.1:%s/status'%port,timeout=15))
        value=response['scheduler']
        if value['healthy']!=1 or any(value[key] for key in ['active','waiting','broken']):
            raise ValueError('服务尚未健康空闲 '+str(port))
        services[str(port)]=value
    with (root/'runs/physical-card-7-campaign.lock').open('a') as file:
        fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
    workers=[]
    for directory in sorted((runtime/'kg-runs').glob('*')):
        path=directory/'.kernelgen/run-process.json'
        if not path.exists():raise ValueError('KG工作区完成身份未知 '+str(directory))
        worker=read(path)
        if not worker.get('finished_at'):raise ValueError('KG worker尚未确认退出 '+str(path))
        operations=directory/'.kernelgen/active-server-operations.json'
        if operations.exists() and read(operations).get('operations'):raise ValueError('工作区仍登记活动请求 '+str(directory))
        workers.append(dict(path=str(path),sha256=digest(path),state=worker['state'],finished_at=worker['finished_at']))
    known_failures={}
    for path in batch.glob('**/transport-failure.json'):
        value=read(path)
        assert value['status']=='http_failure_observed' and value['http_status']==500 and value['no_automatic_retry']
        assert digest(value['request'])==value['request_sha256']
        assert digest(value['evidence_log'])==value['evidence_log_sha256']
        known_failures[str(Path(value['request']).resolve())]=str(path)
    unresolved=[];failed_requests=[];requests=0
    for path in batch.glob('**/*.request.json'):
        payload=read(path)
        if payload.get('api_version')!='v6.2' or not payload.get('implementation'):continue
        requests+=1
        if path.with_name(path.name.replace('.request.json','.result.json')).exists():continue
        if str(path.resolve()) in known_failures:failed_requests.append(known_failures[str(path.resolve())])
        else:unresolved.append(str(path))
    if unresolved:raise ValueError('请求结果未知，不能收尾 '+str(unresolved))
    index=build(batch)
    for row in index['rows']:
        best=row['best']
        if best.get('status')!='qualified':raise ValueError('最终源码尚未合格 '+row['operator'])
        if row['operator']=='cross_out' and best.get('source_sha256')!=row['initial'].get('source_sha256'):
            if not best['source_record'].get('cross_safety'):raise ValueError('叉积缺少新语义门禁')
    for path in (batch/'source-review').glob('*/cross-safety-negative-control.json'):
        value=read(path)
        failures=[row for row in value['cases'] if not row['passed']]
        if value['status']!='failed' or not any(row['case'].startswith('same-addresses-changing-dim-') for row in failures):
            raise ValueError('负对照未暴露预期动态dim问题 '+str(path))
    knowledge=batch/'knowledge/index.json'
    if knowledge.exists():
        denied=read(batch/'disqualified-sources.json')
        for record in read(knowledge).get('active',[]):
            if record.get('status')=='validated' and record.get('evidence',{}).get('source_sha256') in denied:
                raise ValueError('被拒绝的源码仍进入成功知识检索')
    baseline=read(batch/'baseline-verified-final.json')
    assert baseline['status']=='verified' and baseline['service_software_checked']
    paths=read(batch/'path-index-v2.json')
    for row in paths:
        assert digest(row['trace'])==row['trace_sha256']
        assert row['status']=='observed'
    ir=[]
    for manifest in (batch/'actual-ir').glob('*/*/manifest.json'):
        value=read(manifest);assert value['entries']
        for entry in value['entries']:
            for artifact in entry['artifacts']:assert digest(artifact['path'])==artifact['sha256']
        ir.append(dict(path=str(manifest),sha256=digest(manifest),source_sha256=value['spec']['source_sha256']))
    result=dict(status='verified',time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        queue=state['status'],operators=len(index['rows']),evidence_files=len(index['files']),formal_requests=requests,
        unresolved_requests=[],recorded_terminal_http_failures=failed_requests,workers=workers,
        services=services,card_lock='available',active_batch_controllers=[],observed_device_paths=len(paths),actual_ir=ir,
        baseline=baseline,scope='核验终态与证据身份；历史失败和未达标结果保留，不改写为全部达标')
    (batch/'final-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({key:result[key] for key in ['status','queue','operators','evidence_files','formal_requests','observed_device_paths','card_lock']}))


if __name__=='__main__':main()
