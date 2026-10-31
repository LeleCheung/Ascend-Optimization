"""串行完成追加搜索、开发选版、补充门禁、完整KGS和成对硬件采集。"""
import argparse,hashlib,json,math,shutil,statistics,subprocess,sys,time,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
def save(path,d):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();b=a.batch.resolve();out=b/'validation';out.mkdir()
    parent=PROJECT/'operators/campaigns/roofline-20261027/validation/delivered.py'
    state=dict(status='running',steps=[]);save(out/'state.json',state)
    def run(label,args):
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for port in [19655,19657,19659]:
            d=json.load(opener.open('http://127.0.0.1:'+str(port)+'/status',timeout=10))['scheduler']
            if d['active'] or d['waiting'] or d['broken'] or d['healthy']!=1:raise ValueError('项目队列非空闲')
        row=dict(step=label,status='running',command=args);state['steps'].append(row);save(out/'state.json',state)
        with (out/(label+'.log')).open('w') as stream:r=subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,timeout=3000)
        row.update(status='completed' if r.returncode==0 else 'failed',returncode=r.returncode);save(out/'state.json',state)
        if r.returncode:raise RuntimeError(label+'失败')
    try:
        assert json.loads((b/'search-v2/state.json').read_bytes())['status']=='completed'
        proposal=json.loads((b/'agent-review/proposal.json').read_bytes())
        run('search-v3',[sys.executable,str(HERE/'search.py'),'--parent',str(parent),'--output',str(b/'search-v3'),'--configs',*proposal['config_ids']])
        rows=[]
        for label in ['search-v2','search-v3']:
            for row in json.loads((b/label/'state.json').read_bytes())['cases']:
                if row['status']=='passed':rows.append(dict(**row,source=str(b/label/row['case']/'candidate.py')))
        if not rows:raise ValueError('没有完整开发集通过候选')
        best=max(rows,key=lambda r:r['speedup']);candidate=Path(best['source']);save(out/'selection-before-kgs.json',dict(selected=best,candidates=rows,
            policy='仅由预登记开发集1024／2048／4096 FP16与BF16选版；不依据正式结果再调参'))
        for label,source in [('parent',parent),('candidate',candidate)]:
            run(label+'-semantics',[sys.executable,str(HERE.parent/'device_mechanism/check.py'),'--operator','matmul_bias_activation',
                '--source',str(source),'--seed',str(parent),'--output',str(out/(label+'-semantics')),'--semantic-only'])
            run(label+'-large',[sys.executable,str(HERE/'check_large.py'),'--source',str(source),'--output',str(out/(label+'-large'))])
            run(label+'-kgs',[sys.executable,str(PROJECT/'tools/evaluation/evaluate-operator.py'),'--operator','matmul_bias_activation',
                '--source',str(source),'--output',str(out/label),'--label',label,'--server','http://127.0.0.1:19659','--repeats','1','--timing-scope','device_kernel'])
        results={label:json.loads((out/label/(label+'-1.result.json')).read_bytes()) for label in ['parent','candidate']}
        maps={label:{r['uuid']:r for r in value['per_workload'] if r['phase']=='timing'} for label,value in results.items()}
        if set(maps['parent'])!=set(maps['candidate']) or len(maps['parent'])!=15:raise ValueError('正式合同不匹配')
        cases=[]
        for key,pvalue in maps['parent'].items():
            cvalue=maps['candidate'][key]
            if pvalue['axes']!=cvalue['axes']:raise ValueError('输入轴不匹配')
            for r in [pvalue,cvalue]:
                if r['status']!='PASSED' or not math.isfinite(r['latency_ms']) or r['latency_ms']<=0:raise ValueError('无效正式延迟')
            cases.append(dict(case_id=key,axes=pvalue['axes'],parent_ms=pvalue['latency_ms'],candidate_ms=cvalue['latency_ms'],speedup=pvalue['latency_ms']/cvalue['latency_ms']))
        score=statistics.geometric_mean(r['speedup'] for r in cases)
        chosen=candidate if score>1.01 else parent;shutil.copyfile(chosen,out/'delivered.py')
        save(out/'comparison.json',dict(status='completed',selected='candidate' if chosen==candidate else 'parent',candidate_id=best['case'],
            source=str(chosen),source_sha256=hashlib.sha256(chosen.read_bytes()).hexdigest(),actual_candidate_over_parent=score,
            parent_geomean=results['parent']['geo_mean'],candidate_geomean=results['candidate']['geo_mean'],cases=cases,
            scope='KGS原合同设备kernel时间；不是完整调用walltime；同case实际延迟比'))
        for label,source in [('parent',parent),('candidate',candidate)]:
            run(label+'-profile',[sys.executable,str(HERE.parent/'roofline/collect_matrix.py'),'--source',str(source),
                '--output',str(b/(label+'-profile')),'--sides',label,'--sizes','1024','2048','--dtypes','float16'])
        state.update(status='completed',selected='candidate' if chosen==candidate else 'parent',actual_speedup=score)
    except Exception as e:
        state.update(status='failed',error=type(e).__name__+': '+str(e));save(out/'state.json',state);raise
    save(out/'state.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
