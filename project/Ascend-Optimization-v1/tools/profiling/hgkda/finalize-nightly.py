#!/usr/bin/env python3
"""等待本批串行队列释放卡锁，补测最终源码并核验四组真实证据。"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import urllib.request


def existing_semantics(path,source_sha,checker_sha):
    """已有失败也是证据；同身份直接读取，不同身份须另开尝试目录。"""
    if not path.exists():return None
    value=json.loads(path.read_text(encoding='utf-8'))
    if value.get('source_sha256')!=source_sha or value.get('checker_sha256')!=checker_sha:
        raise RuntimeError('已有补测身份不同，保留证据并另开尝试目录: '+str(path))
    return value


def finalize(root,batch,container):
    project=root/'Ascend-Optimization/project/Ascend-Optimization-v1'
    directory=project/'operators/campaigns'/batch
    output=directory/'finalization.json'
    if output.exists():
        previous=output.read_bytes()
        history=directory/'finalization-history'/(hashlib.sha256(previous).hexdigest()+'.json')
        history.parent.mkdir(exist_ok=True)
        if history.exists() and history.read_bytes()!=previous:raise RuntimeError('核验历史内容冲突')
        history.write_bytes(previous)
    record={'status':'running','started_at':datetime.datetime.utcnow().isoformat()+'Z','checks':[]}
    def save():output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    def run(args,name):
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for port in [19655,19657,19659]:
            scheduler=json.load(opener.open('http://127.0.0.1:%d/status'%port,timeout=15))['scheduler']
            if scheduler['healthy']!=1 or any(scheduler[k] for k in ['active','waiting','broken']):
                raise RuntimeError('补测前服务忙或异常: '+str(port))
        command='source /usr/local/Ascend/cann-9.0.0/set_env.sh\nexport ASCEND_RT_VISIBLE_DEVICES=7\nexport PYTHONPATH='+shlex.quote(str(root/'Ascend-Optimization/third_party/FlagGems/src'))+'\n'
        command+=' '.join(shlex.quote(str(x)) for x in args)
        log=root/'runs'/batch/('finalize-'+name+'.log')
        with log.open('ab') as stream:
            result=subprocess.run(['docker','exec','-i',container,'bash','-s'],input=command.encode(),stdout=stream,stderr=subprocess.STDOUT,timeout=3600)
        return result.returncode
    state=json.loads((directory/'queue-state.json').read_text())
    if state['status']!='completed':
        record.update(status='needs_attention',error='优化队列尚未正常完成，未开始补测');save();return False
    save();py='/usr/local/python3.11.15/bin/python3.11'
    tools=project/'tools/profiling/hgkda';seen={}
    for op in ['value_selecting_reduction_backward','moe_sum_reduce']:
        for arm in ['kg-no-profile','kg-profile','HG-KDA']:
            key='kg:'+op+':'+arm
            if state['jobs'].get(key,{}).get('status')!='passed':
                record['checks'].append({'operator':op,'arm':arm,'status':'missing_complete_result'});save();continue
            source=directory/'results'/op/arm/'export/main.py'
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            previous=seen.get((op,digest))
            target=directory/'results'/op/arm/'supplementary-semantics.json'
            existing=existing_semantics(target,digest,hashlib.sha256((tools/'check-nightly-semantics.py').read_bytes()).hexdigest())
            if existing is not None:
                pass
            elif previous is not None:
                data=dict(previous);data['reused_identical_source']=True
                target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
            else:
                run([py,tools/'check-nightly-semantics.py','--operator',op,'--source',source,'--output',target],op+'-'+arm)
            data=json.loads(target.read_text()) if target.exists() else {'status':'failed','error':'补测没有返回结果'}
            seen[(op,digest)]=data
            record['checks'].append({'operator':op,'arm':arm,'source_sha256':digest,
                                    'status':data['status'],'evidence':str(target.relative_to(directory))});save()
    report_exit=run([py,tools/'report-nightly.py','--batch-dir',directory,'--require-complete'],'report')
    record.update(status='completed' if report_exit==0 and all(c['status']=='passed' for c in record['checks']) else 'needs_attention',
                  finished_at=datetime.datetime.utcnow().isoformat()+'Z',report_exit_code=report_exit)
    save();return record['status']=='completed'


if __name__=='__main__':
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/data/hanle/ascend-optimization'))
    parser.add_argument('--batch',required=True)
    parser.add_argument('--container',default='tle_yy')
    args=parser.parse_args()
    if not args.batch.replace('-','').replace('_','').isalnum():raise ValueError('非法批次名')
    print('等待本项目卡锁，队列结束后补测',flush=True)
    with (args.root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not finalize(args.root,args.batch,args.container):raise SystemExit(1)
