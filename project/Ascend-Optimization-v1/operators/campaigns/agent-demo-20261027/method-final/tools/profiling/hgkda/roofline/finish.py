"""串行完成模型追加候选、完整评测与最终成对采集；无收益保留父版。"""
import argparse,hashlib,json,math,os,shutil,statistics,subprocess,sys,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args()
    batch=a.batch.resolve();parent=a.parent.resolve();out=batch/'validation';out.mkdir(parents=True)
    state=dict(status='running',steps=[]);save(out/'state.json',state)
    def run(label,arguments):
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for port in [19655,19657,19659]:
            d=json.load(opener.open('http://127.0.0.1:'+str(port)+'/status',timeout=10))['scheduler']
            if d['active'] or d['waiting'] or d['broken'] or d['healthy']!=1:raise ValueError('项目KGS非空闲')
        row=dict(step=label,status='running',command=arguments);state['steps'].append(row);save(out/'state.json',state)
        with (out/(label+'.log')).open('w') as stream:r=subprocess.run(arguments,stdout=stream,stderr=subprocess.STDOUT,timeout=2400)
        row.update(status='completed' if r.returncode==0 else 'failed',returncode=r.returncode);save(out/'state.json',state)
        if r.returncode:raise RuntimeError(label+'失败')
    proposal=json.loads((batch/'agent-review/proposal.json').read_bytes())
    run('search-v3',[sys.executable,str(HERE/'search.py'),'--parent',str(parent),'--output',str(batch/'search-v3'),'--configs',proposal['config_id']])
    choices=[]
    for label in ['search-v2','search-v3']:
        search=json.loads((batch/label/'state.json').read_bytes())
        for row in search['cases']:
            if row['status']=='passed':choices.append((row['speedup'],batch/label/row['case']/'candidate.py'))
    _,candidate=max(choices,key=lambda x:x[0])
    save(out/'selection-before-kgs.json',dict(parent=str(parent),candidate=str(candidate),
        candidates=[dict(development_speedup=v,source=str(s)) for v,s in choices],
        rule='开发集最佳追加候选接受完整评测；最终只有同case实际延迟更快才采用，否则保留父版'))
    for label,source in [('parent',parent),('candidate',candidate)]:
        run(label+'-semantics',[sys.executable,str(HERE.parent/'device_mechanism/check.py'),'--operator','matmul_bias_activation',
            '--source',str(source),'--seed',str(parent),'--output',str(out/(label+'-semantics')),'--semantic-only'])
        run(label+'-kgs',[sys.executable,str(PROJECT/'tools/evaluation/evaluate-operator.py'),'--operator','matmul_bias_activation',
            '--source',str(source),'--output',str(out/label),'--label',label,'--server','http://127.0.0.1:19659',
            '--repeats','1','--timing-scope','device_kernel'])
    results={label:json.loads((out/label/(label+'-1.result.json')).read_bytes()) for label in ['parent','candidate']}
    maps={label:{r['uuid']:r for r in result['per_workload'] if r['phase']=='timing'} for label,result in results.items()}
    if set(maps['parent'])!=set(maps['candidate']) or len(maps['parent'])!=15:raise ValueError('正式计时合同不一致')
    comparisons=[]
    for key,r in maps['parent'].items():
        c=maps['candidate'][key]
        if r['axes']!=c['axes'] or r['status']!='PASSED' or c['status']!='PASSED':raise ValueError('case身份或通过状态不一致')
        values=[r['latency_ms'],c['latency_ms']]
        if any(type(v) not in [int,float] or not math.isfinite(v) or v<=0 for v in values):raise ValueError('无效正式延迟')
        comparisons.append(dict(case_id=key,axes=r['axes'],parent_ms=values[0],candidate_ms=values[1],speedup=values[0]/values[1]))
    speedup=statistics.geometric_mean(r['speedup'] for r in comparisons)
    chosen=candidate if speedup>1.01 else parent
    shutil.copyfile(chosen,out/'delivered.py')
    save(out/'comparison.json',dict(status='completed',actual_candidate_over_parent=speedup,
        parent_geomean=results['parent']['geo_mean'],candidate_geomean=results['candidate']['geo_mean'],
        passed={label:[r['num_passed'],r['num_workloads']] for label,r in results.items()},
        selected='candidate' if chosen==candidate else 'parent',source=str(chosen),
        source_sha256=hashlib.sha256(chosen.read_bytes()).hexdigest(),cases=comparisons,
        scope='KGS原合同设备kernel计时；非完整调用walltime；不以两组对PyTorch汇总分数相除代替实际提升'))
    run('candidate-profile',[sys.executable,str(HERE/'collect_matrix.py'),'--source',str(candidate),
        '--output',str(batch/'candidate-profile'),'--sides','candidate','--sizes','1024','2048','--dtypes','float16','float32'])
    run('empirical-audit',[sys.executable,str(HERE/'empirical_audit.py'),'--output',str(batch/'empirical-audit')])
    state.update(status='completed');save(out/'state.json',state)
    print(json.dumps(dict(status='completed',selected='candidate' if chosen==candidate else 'parent',speedup=speedup)),flush=True)

if __name__=='__main__':main()
