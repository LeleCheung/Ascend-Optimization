"""收尾核验合同、附件、请求终态、共享服务空闲及本批worker退出。"""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
from report_remaining import build,BATCH,PROJECT,read,sha


def main():
    root=PROJECT.parents[2]
    state=read(BATCH/'next-state.json')
    baseline=read(BATCH/'baseline-verified.json')
    assert baseline['status']=='verified' and baseline['service_software_checked']
    if state['status']!='all_completed':raise RuntimeError('主队列尚未完成')
    running=[key for key,value in state['jobs'].items() if value['status']=='running']
    if running:raise RuntimeError('存在未结束任务 '+str(running))
    sessions=subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout.splitlines()
    own=[name for name in sessions if name.startswith('hanle-remaining-nine-')]
    if own:raise RuntimeError('本批控制器仍运行 '+str(own))
    with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    services={};opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for port in [19655,19657,19659]:
        value=json.load(opener.open('http://127.0.0.1:%d/status'%port,timeout=15))
        scheduler=value['scheduler']
        assert scheduler['healthy']==1 and not any(scheduler[k] for k in ['active','waiting','broken'])
        services[str(port)]=scheduler
    identity={};requests=0
    for record in read(BATCH/'plan.json')['records']:
        for name,digest in record['contract_sha256'].items():
            path=BATCH/record['catalog_leaf']/name;assert sha(path)==digest;identity[str(path)]=digest
        seed=BATCH/record['seed'];assert sha(seed)==record['source_sha256'];identity[str(seed)]=sha(seed)
    for path in BATCH.rglob('*.request.json'):
        request=read(path)
        if request.get('api_version')!='v6.2' or not request.get('implementation'):continue
        result=path.with_name(path.name.replace('.request.json','.result.json'))
        assert result.exists(),'请求结果未知 '+str(path)
        requests+=1;identity[str(path)]=sha(path);identity[str(result)]=sha(result)
    for path in (BATCH/'proposals').glob('*/*/manifest.json'):
        manifest=read(path)
        for name,digest in manifest['files'].items():assert sha(name)==digest;identity[name]=digest
    workers=[]
    for workspace in (root/'runs'/BATCH.name/'kg-runs').glob('*'):
        metadata=workspace/'.kernelgen/run-process.json';worker=read(metadata)
        assert worker.get('finished_at'),'worker尚未结束'
        operations=workspace/'.kernelgen/active-server-operations.json'
        assert not operations.exists() or not read(operations).get('operations'),'仍有登记操作'
        workers.append(dict(metadata=str(metadata),sha256=sha(metadata),state=worker['state']))
    for manifest in (BATCH/'actual-ir').glob('*/*/manifest.json'):
        value=read(manifest);assert value['entries']
        for entry in value['entries']:
            for item in entry['artifacts']:assert sha(item['path'])==item['sha256'];identity[item['path']]=item['sha256']
    paths=[]
    for summary in (BATCH/'paths').glob('*/*/*/summary.json'):
        value=read(summary);assert value['status']=='observed';assert sha(value['trace'])==value['trace_sha256']
        identity[value['trace']]=value['trace_sha256'];paths.append(str(summary))
    report=build()
    for row in report['rows']:
        if row['best']:
            assert row['verified']['status']=='verified' and row['verified']['extra_status']=='passed'
            raw=read(row['best']['result']);assert not raw.get('is_hack')
        if row['stable']:
            assert row['confirmation']['status']=='verified' and row['confirmation']['score']>.8
    output=dict(status='verified',time_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        formal_requests=requests,unresolved_requests=[],evidence_files=identity,services=services,workers=workers,
        active_batch_controllers=[],card_lock='available',device_paths=len(paths),newly_stable=report['newly_stable'],
        fixed_software_verification=baseline,
        unaccepted=[row['operator'] for row in report['rows'] if not row['stable']],
        scope='失败和未达标原件保留；终态核验不把失败改写为达标')
    (BATCH/'final-audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:output[k] for k in ['status','formal_requests','device_paths','newly_stable','unaccepted']}))


if __name__=='__main__':main()
