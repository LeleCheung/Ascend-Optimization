"""开发集冻结选版后一次完整KGS，按相同case实际延迟保留源码。"""
import argparse,hashlib,json,math,shutil,statistics,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();b=a.batch.resolve();out=b/'validation';out.mkdir()
    data=json.loads((b/'search/state.json').read_bytes());assert data['status']=='completed'
    passed=[r for r in data['cases'] if r['status']=='passed']
    if not passed:raise ValueError('无完整开发输入通过候选')
    best=max(passed,key=lambda r:r['speedup']);source=b/'search'/best['case']/'candidate.py'
    (b/'candidate.py').write_bytes(source.read_bytes());assert hashlib.sha256(source.read_bytes()).hexdigest()==best['source_sha256']
    save(out/'selection-before-kgs.json',dict(selected=best,source_sha256=best['source_sha256'],policy='开发集选定后不再调参，完整KGS裁决'))
    state=dict(status='running',steps=[]);save(out/'state.json',state)
    def run(name,command):
        row=dict(step=name,command=command,status='running');state['steps'].append(row);save(out/'state.json',state)
        with (out/(name+'.log')).open('w') as stream:r=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
        row.update(status='completed' if r.returncode==0 else 'failed',returncode=r.returncode);save(out/'state.json',state)
        if r.returncode:raise RuntimeError(name+'未通过')
    try:
        for label in ['seed','candidate']:
            source=b/(label+'.py')
            run(label+'-semantics',[sys.executable,str(HERE.parent/'device_mechanism/check.py'),'--operator','matmul_bias_activation','--source',str(source),'--seed',str(b/'seed.py'),'--output',str(out/(label+'-semantics')),'--semantic-only'])
            run(label+'-large',[sys.executable,str(HERE.parent/'matmul_pipeline/check_large.py'),'--source',str(source),'--output',str(out/(label+'-large'))])
            run(label+'-kgs',[sys.executable,str(PROJECT/'tools/evaluation/evaluate-operator.py'),'--operator','matmul_bias_activation','--source',str(source),
                '--output',str(out/label),'--label',label,'--server','http://127.0.0.1:19659','--repeats','1','--timing-scope','device_kernel'])
        results={s:json.loads((out/s/(s+'-1.result.json')).read_bytes()) for s in ['seed','candidate']}
        maps={s:{r['uuid']:r for r in d['per_workload'] if r['phase']=='timing'} for s,d in results.items()}
        assert all(d['status']=='PASSED' and d['num_passed']==d['num_workloads']==42 for d in results.values())
        assert set(maps['seed'])==set(maps['candidate']) and len(maps['seed'])==15
        rows=[]
        for key,parent in maps['seed'].items():
            candidate=maps['candidate'][key];assert parent['axes']==candidate['axes']
            for item in [parent,candidate]:assert item['status']=='PASSED' and math.isfinite(item['latency_ms']) and item['latency_ms']>0
            rows.append(dict(case_id=key,axes=parent['axes'],seed_ms=parent['latency_ms'],candidate_ms=candidate['latency_ms'],speedup=parent['latency_ms']/candidate['latency_ms']))
        speedup=statistics.geometric_mean(r['speedup'] for r in rows);selected='candidate' if speedup>1.01 else 'seed'
        source=b/(selected+'.py');shutil.copyfile(source,out/'delivered.py')
        save(out/'comparison.json',dict(status='completed',selected=selected,candidate_id=best['case'],actual_candidate_over_seed=speedup,
            seed_pytorch_speedup=results['seed']['geo_mean'],candidate_pytorch_speedup=results['candidate']['geo_mean'],source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            cases=rows,scope='原合同设备kernel时间，非完整调用；实际提升由相同case的延迟比计算'))
        state.update(status='completed',selected=selected,actual_speedup=speedup)
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error));save(out/'state.json',state);raise
    save(out/'state.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
