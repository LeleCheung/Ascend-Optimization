#!/usr/bin/env python3
"""五项主队列与散射修复退出后，核验独立有效候选并生成最终交付。"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shlex
import shutil
import time
import urllib.request

TOOLS=Path(__file__).resolve().parent
ROOT=Path('/data/hanle/ascend-optimization')
PROJECT=ROOT/'Ascend-Optimization/project/Ascend-Optimization-v1'
PY='/usr/local/python3.11.15/bin/python3.11'
MAIN='remaining-five-20261023-v2'
SCATTER='scatter-segmented-20261023'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def execute():
    batch=PROJECT/'operators/campaigns'/MAIN
    scatter=PROJECT/'operators/campaigns'/SCATTER
    path=batch/'finalization-state.json'
    assert not path.exists(),'不要覆盖已执行的收尾；先核对原始结果'
    state=dict(status='waiting_for_predecessors',jobs={},started_at=datetime.datetime.utcnow().isoformat()+'Z')
    save(path,state)
    with (ROOT/'runs/physical-card-7-campaign.lock').open('a') as lock:
        deadline=time.monotonic()+4*3600
        while True:
            main=read(batch/'endurance-state.json')
            post=read(batch/'postflight-state.json') if (batch/'postflight-state.json').exists() else {}
            repair=read(scatter/'endurance-state.json')
            wait=read(scatter/'wait-state.json')
            if repair['status']=='needs_attention':raise RuntimeError('散射收尾需处理；不争用设备')
            if main['status']=='completed' and post.get('status') in ['passed','needs_attention'] and repair['status']=='completed' and wait['status']=='passed':
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:pass
            if time.monotonic()>deadline:raise TimeoutError('等待前序队列预算耗尽；未提交设备请求')
            time.sleep(15)
        module=load('five_delivery_campaign',TOOLS/'run-remaining-five.py')
        queue_module=load('five_delivery_queue',TOOLS/'run-nightly.py')
        q=queue_module.Queue(MAIN,ROOT,'tle_yy')
        # 复用冻结门禁和评测方法，不实例化会写入主队列状态的控制器。
        campaign=object.__new__(module.Campaign)
        campaign.batch=batch;campaign.q=q
        def run(key,parts,budget=1800):
            q.idle()
            state.update(status='running',current_job=key);save(path,state)
            q.run('finalize-'+key,'export PYTHONDONTWRITEBYTECODE=1\n'+' '.join(shlex.quote(str(x)) for x in parts),budget)
        campaign.run=run
        def step(key,fn):
            state.update(status='running',current_job=key);save(path,state)
            try:
                result=fn()
            except RuntimeError as error:
                if not key.startswith(('metrics-','observe-')):raise
                # 已知采集失败单列；服务状态未知时仍停止，不能继续发设备请求。
                q.idle()
                result=dict(status='failed',error=str(error))
                state['jobs'][key]=dict(status='failed',result=result);save(path,state);return result
            state['jobs'][key]=dict(status='completed',result=result);save(path,state);return result
        try:
            q.idle()
            pool=batch/'results/fractional_max_pool3d/codex-bitround/candidates/4228f95c27e2dedb1edad3d8c37441a96e2c6384c84d423533be7413b30a0a66'
            for name,n in [('remaining-semantics.json',27),('special.json',30)]:
                data=read(pool/name)
                assert data['status']=='passed' and data['passed']==data['total']==n and data['source_sha256']==sha(pool/'main.py')
            selected={r['operator']:dict(r['selected']) for r in read(batch/'deliverables/manifest.json')['records']}
            selected['fractional_max_pool3d']=step('pool-reuse',lambda:campaign.candidate('fractional_max_pool3d',pool/'main.py',pool,'final-reuse'))
            assert selected['fractional_max_pool3d']['status']=='passed'
            selected['fractional_max_pool3d']['special']=str(pool/'special.json')
            assert read(scatter/'completion-audit.json')['status']=='passed'
            selected['scatter_add']=read(scatter/'deliverables/manifest.json')['records'][0]['selected']

            def packed_probe():
                folder=batch/'research/upsample_bilinear2d/packed-store-probe'
                folder.mkdir(parents=True,exist_ok=False)
                source='import sys\nsys.dont_write_bytecode=True\n'+(batch/'knowledge/upsample-packed-store-probe.py').read_text()
                (folder/'probe.py').write_text(source)
                request=dict(command=['{python}','probe.py'],files=[dict(path='probe.py',content=source)],timeout_seconds=300)
                save(folder/'request.json',request)
                q.idle()
                response=json.load(q.opener.open(urllib.request.Request(q.server+'/debug/jobs',data=json.dumps(request).encode(),headers={'Content-Type':'application/json'},method='POST'),timeout=330))
                save(folder/'response.json',response)
                if response['status']!='SUCCEEDED' or response.get('exit_code')!=0:return dict(status='probe_failed',job_id=response['job_id'])
                data=read(folder/'timings.json')
                assert data['source_sha256']==sha(batch/'knowledge/upsample-packed-store.py')
                rows=data['rows'];names={r['case'] for r in rows}
                assert len(names)==2 and all(r['correct'] for r in rows)
                ratios=[]
                for name in names:
                    values={r['variant']:r['walltime_ms'] for r in rows if r['case']==name}
                    assert len(values)==2 and all(v>0 for v in values.values())
                    ratios.append(values['parent']/values['packed_store'])
                gain=math.exp(sum(math.log(v) for v in ratios)/len(ratios))
                return dict(status='passed',job_id=response['job_id'],diagnostic_gain=gain)

            probe=step('packed-store-probe',packed_probe)
            if probe['status']=='passed' and probe['diagnostic_gain']>1.05:
                source=batch/'knowledge/upsample-packed-store.py'
                out=batch/'results/upsample_bilinear2d/codex-packed/candidates'/sha(source)
                out.mkdir(parents=True,exist_ok=False);shutil.copyfile(source,out/'main.py')
                candidate=step('packed-store-candidate',lambda:campaign.candidate('upsample_bilinear2d',out/'main.py',out,'codex-packed'))
                if candidate['status']=='passed':
                    special=out/'special.json'
                    try:run('packed-special',[PY,TOOLS/'check-remaining-special.py','--operator','upsample_bilinear2d','--source',out/'main.py','--output',special],1800)
                    except RuntimeError:
                        if not special.exists():raise
                    extra=read(special)
                    if extra['status']=='passed' and candidate['latency_ms']<selected['upsample_bilinear2d']['latency_ms']:
                        candidate['special']=str(special)
                        selected['upsample_bilinear2d']=candidate

            if 'special' not in selected['upsample_bilinear2d']:
                selected['upsample_bilinear2d']['special']=str(batch/'results/upsample_bilinear2d/final-special.json')
            for op in ['fractional_max_pool3d','upsample_bilinear2d']:
                value=selected[op]
                if op=='upsample_bilinear2d' and value['source_sha256']==main['operators'][op]['best']['source_sha256']:
                    value['facts']=str(batch/'research'/op/'final/facts.json');continue
                out=batch/'research'/op/'reviewed-final'
                step('observe-'+op,lambda op=op,value=value,out=out:campaign.observe(op,Path(value['source']),out))
                value['facts']=str(out/'facts.json')
                step('metrics-'+op,lambda op=op,value=value:campaign.metrics(op,value,batch/'research'/op/'reviewed-final-metrics'))
            selected['_conj_physical']['facts']=str(batch/'research/_conj_physical/final/facts.json')
            selected['_conj_physical']['complex_facts']=str(batch/'research/_conj_physical/final-complex/facts.json')
            selected['scatter_add']['facts']=str(scatter/'research/scatter_add/final/facts.json')
            audit=load('reviewed_formal_audit',TOOLS/'audit-endurance.py')
            records={r['operator']:r for r in read(batch/'plan.json')['records']}
            rows=[]
            for op in ['scatter_add','upsample_bilinear2d','fractional_max_pool3d','_conj_physical']:
                value=selected[op];source=Path(value['source']);digest=sha(source)
                assert digest==value['source_sha256']
                result=read(Path(value['result']));record=records[op]
                audit.verify_formal_contract(record,Path(value['result']).parent,result)
                assert result['status']=='PASSED' and result['num_passed']==result['num_workloads']==record['correctness_cases']+record['timing_cases']
                request=read(Path(value['result']).with_name(Path(value['result']).name.replace('.result.json','.request.json')))
                assert len(request['implementation']['sources'])==1 and request['implementation']['sources'][0]['content']==source.read_text()
                gates=list(value['gates'])+([value['special']] if value.get('special') else [])
                expected=sorted({'scatter_add':[18,18],'upsample_bilinear2d':[48,36],'fractional_max_pool3d':[27,30],'_conj_physical':[34]}[op])
                checked=[read(Path(p)) for p in gates]
                assert sorted(d['total'] for d in checked)==expected
                for d in checked:
                    assert d['status']=='passed' and d['source_sha256']==digest and d['passed']==d['total'] and len(d['cases'])==d['total']
                    assert all(r['status']=='passed' for r in d['cases'])
                target=batch/'deliverables/reviewed'/op/'main.py';target.parent.mkdir(parents=True,exist_ok=True)
                assert not target.exists();shutil.copyfile(source,target)
                assert sha(target)==digest
                rows.append(dict(operator=op,source=str(target),selected=value,score=result['geo_mean'],formal=result['num_workloads'],gates=gates,
                    reference_scope='参考含CPU回退，单列' if op in ['fractional_max_pool3d','_conj_physical'] else '原样PyTorch/NPU完整调用',
                    meets_numerical_speed_target=result['geo_mean']>.8))
            manifest=dict(status='reviewed',records=rows,unclosed=['cumulative_trapezoid'],under_target=[r['operator'] for r in rows if r['score']<=.8],
                policy='保留原批次原始状态和错误候选；本清单以新增动态范围门禁、独立有效源码及正式合同核验为准',
                cumulative_reference_counterexample=str(batch/'research/cumulative_trapezoid/reference-stability-probe/stability-models.json'))
            save(batch/'final-delivery.json',manifest)
            state.update(status='completed',current_job=None,delivered=4,unclosed=manifest['unclosed'],under_target=manifest['under_target']);save(path,state)
        except Exception as error:
            state.update(status='needs_attention',error=type(error).__name__+': '+str(error));save(path,state);raise


if __name__=='__main__':execute()
