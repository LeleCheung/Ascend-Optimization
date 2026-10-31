"""910B宿主统一入口：恢复已完成采集、模型选工具、生成、完整验证与新证据。"""
import argparse,fcntl,hashlib,json,os,shlex,subprocess,sys,time,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];REPO=PROJECT.parents[1];ROOT=REPO.parent

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def idle():
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));result=[]
    for port in [19655,19657,19659]:
        d=json.load(opener.open('http://127.0.0.1:'+str(port)+'/status',timeout=10))['scheduler']
        if d['active'] or d['waiting'] or d['broken'] or d['healthy']!=1:raise RuntimeError('项目服务非健康空闲：'+str(port))
        result.append(dict(port=port,scheduler=d))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();b=a.batch.resolve();b.mkdir(parents=True,exist_ok=True)
    state_path=b/'workflow-state.json'
    state=json.loads(state_path.read_bytes()) if state_path.exists() else dict(status='running',steps=[],entry='HG-KDA单算子三步演示')
    if state.get('status')=='completed':print(json.dumps(state,ensure_ascii=False));return
    def step(name,args,marker,npu=False,condition=None):
        old=next((r for r in state['steps'] if r['step']==name),None)
        if marker.exists():
            obj=json.loads(marker.read_bytes())
            if condition and not condition(obj):raise RuntimeError('已有产物未满足门禁：'+name)
            if old is None:
                state['steps'].append(dict(step=name,status='completed',origin='本批前序真实执行，恢复采用原件',marker=str(marker),sha256=hashlib.sha256(marker.read_bytes()).hexdigest()));save(state_path,state)
            return
        if old is not None:raise RuntimeError('步骤已尝试但结果未完整，禁止自动重发：'+name)
        row=dict(step=name,status='running',command=args,start=time.time());state['steps'].append(row);save(state_path,state)
        try:
            lock=None
            if npu:
                lock=(ROOT/'runs/physical-card-7-campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX);idle()
                env='source /usr/local/Ascend/cann-9.0.0/set_env.sh; export ASCEND_RT_VISIBLE_DEVICES=7; export PYTHONPATH='+str(REPO)+'/third_party/FlagGems/src:'+str(REPO)+'/sources/kernelgen_server/client:'+str(REPO)+'/sources/kernelgen_server:'+str(REPO)+'/sources; export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost; unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; cd '+shlex.quote(str(PROJECT))+'; '
                args=['docker','exec','tle_yy','bash','-c','set -euo pipefail; '+env+shlex.join(args)]
            with (b/('workflow-'+name+'.log')).open('w') as stream:r=subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,timeout=3600)
            if lock:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
            if r.returncode or not marker.exists():raise RuntimeError(name+'未完成，查看独立日志')
            obj=json.loads(marker.read_bytes())
            if condition and not condition(obj):raise RuntimeError(name+'产物门禁未通过')
            row.update(status='completed',returncode=r.returncode,finish=time.time(),marker=str(marker),sha256=hashlib.sha256(marker.read_bytes()).hexdigest());save(state_path,state)
        except Exception as e:
            row.update(status='failed',error=type(e).__name__+': '+str(e));state.update(status='failed');save(state_path,state);raise
    py='/usr/local/python3.11.15/bin/python3.11';host=str(ROOT/'runtime/kg-controller/venv/bin/python')
    step('collect-seed',[py,str(HERE/'collect.py'),'--batch',str(b)],b/'seed-profile/state.json',True,lambda d:d['status']=='completed')
    ins=next((d for d in [b/'seed-instruction-v3',b/'seed-instruction'] if (d/'summary.json').exists()),b/'seed-instruction')
    step('instruction-seed',[py,str(HERE/'instruction.py'),'--batch',str(b)],ins/'summary.json',True,lambda d:d['instructions']>0)
    marker=b/'evidence-qualified.json' if (b/'evidence-qualified.json').exists() else b/'evidence.json'
    step('evidence',[py,str(HERE/'evidence.py'),'--batch',str(b)],marker,True,lambda d:d['status']=='completed')
    step('agent',[host,str(HERE/'agent.py'),'--batch',str(b)],b/'agent/proposal.json')
    proposal=json.loads((b/'agent/proposal.json').read_bytes())
    step('search',[py,str(HERE.parent/'matmul_pipeline/search.py'),'--parent',str(b/'seed.py'),'--output',str(b/'search'),'--configs',*proposal['config_ids']],b/'search/state.json',True,lambda d:d['status']=='completed')
    step('validation',[py,str(HERE/'validate.py'),'--batch',str(b)],b/'validation/comparison.json',True,lambda d:d['status']=='completed')
    step('collect-candidate',[py,str(HERE/'collect.py'),'--batch',str(b),'--side','candidate'],b/'candidate-profile/state.json',True,lambda d:d['status']=='completed')
    step('instruction-candidate',[py,str(HERE/'instruction.py'),'--batch',str(b),'--side','candidate'],b/'candidate-instruction/summary.json',True,lambda d:d['instructions']>0)
    state.update(status='completed',finished=time.time(),services=idle());save(state_path,state);print(json.dumps(state,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
