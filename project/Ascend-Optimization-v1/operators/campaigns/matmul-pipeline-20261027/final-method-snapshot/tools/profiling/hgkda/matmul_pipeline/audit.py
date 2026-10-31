"""独立重算候选选择与正式延迟，核对原CSV和冻结模型输入。"""
import argparse,csv,hashlib,json,math,os,statistics
from pathlib import Path,PurePosixPath

def long(p):
    prefix=chr(92)*2+'?'+chr(92)
    return Path(prefix+str(p.resolve())) if os.name=='nt' and not str(p).startswith(prefix) else p
def sha(p):return hashlib.sha256(long(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.batch
    if a.output.exists():raise ValueError('不覆盖已有核验')
    checks=[]
    def ck(value,label):
        if not value:raise ValueError(label)
        checks.append(label)
    for manifest in (b/'transfers').glob('*.manifest.json'):
        for row in json.loads(manifest.read_bytes()):
            path=b/row['path'];ck(sha(path)==row['sha256'] and long(path).stat().st_size==row['bytes'],'传输字节:'+row['path'])
    full=[]
    for label in ['search-v2','search-v3']:
        search=json.loads((b/label/'state.json').read_bytes());plan=json.loads((b/label/'plan.json').read_bytes());ck(search['status']=='completed',label+'完成')
        ck([r['case'] for r in search['cases']]==[c['id'] for c in plan['configs']],label+'候选预算')
        for row in search['cases']:
            ck(sha(b/label/row['case']/'candidate.py')==row['source_sha256'],label+'/'+row['case']+'源码SHA')
            if row['status']=='passed':
                ck({(v['size'],v['dtype']) for v in row['measurements']}=={(v['size'],v['dtype']) for v in plan['extension']},label+'/'+row['case']+'完整开发集')
                score=statistics.geometric_mean(v['speedup'] for v in row['measurements']);ck(math.isclose(score,row['speedup'],rel_tol=1e-12),label+'/'+row['case']+'几何平均')
                full.append(dict(case=row['case'],score=score,source=sha(b/label/row['case']/'candidate.py')))
            for value in row['measurements']:
                case=b/label/row['case']/(value['dtype']+'-'+str(value['size']))
                for repeat in [0,1]:
                    directory=case/('timing-'+str(repeat));metadata=json.loads((directory/'measurement.json').read_bytes());files=list(long(directory).rglob('task_time*.csv'))
                    ck(len(files)==1 and sha(files[0])==metadata['csv_sha256'],'原task CSV:'+str(directory.relative_to(b)))
                    with files[0].open(encoding='utf-8-sig',newline='') as stream:tasks=list(csv.DictReader(stream))[1:-1]
                    ck(len(tasks)==16,'16任务:'+str(directory.relative_to(b)))
                    for i,record in enumerate(metadata['records']):
                        rows=tasks[i*8:(i+1)*8];times=[float(x['task_time(us)']) for x in rows[2:]]
                        ck(all(record['symbol'] in x['kernel_name'] for x in rows),'kernel身份:'+str(directory.relative_to(b))+'/'+record['key'])
                        ck(times==record['active_device_us'] and math.isclose(statistics.mean(times)/1000,record['device_ms'],rel_tol=1e-9),'独立CSV重算:'+str(directory.relative_to(b))+'/'+record['key'])
                        ck(math.isclose(value['times'][record['key']][repeat],record['device_ms'],rel_tol=1e-9),'搜索绑定:'+str(directory.relative_to(b))+'/'+record['key'])
                ck(math.isclose(statistics.median(p/c for p,c in zip(value['times']['parent'],value['times']['candidate'])),value['speedup'],rel_tol=1e-12),'开发加速重算:'+str(case.relative_to(b)))
    comparison=json.loads((b/'validation/comparison.json').read_bytes());selection=json.loads((b/'validation/selection-before-kgs.json').read_bytes())
    winner=max(full,key=lambda r:r['score']);ck(winner['case']==selection['selected']['case']==comparison['candidate_id'],'预登记开发选版')
    ck(sha(b/'validation/candidate/candidate.py')==winner['source'],'正式使用开发优胜源码')
    results={}
    for label in ['parent','candidate']:
        root=b/'validation'/label;result=json.loads((root/(label+'-1.result.json')).read_bytes());request=json.loads((root/(label+'-1.request.json')).read_bytes());source=(root/(label+'.py')).read_bytes()
        ck(request['implementation']['sources'][0]['content'].encode()==source,label+'实际上传源码')
        ck(result['status']=='PASSED' and result['num_passed']==result['num_workloads']==42,label+'42项完整通过')
        results[label]={r['uuid']:r for r in result['per_workload'] if r['phase']=='timing'}
        for suffix,count in [('semantics',18),('large',24)]:
            check=json.loads((b/'validation'/(label+'-'+suffix)/'check.json').read_bytes())
            ck(check['status']=='passed' and check['passed']==check['total']==count and check['source_sha256']==hashlib.sha256(source).hexdigest(),label+'补充检查:'+suffix)
    ck(set(results['parent'])==set(results['candidate']) and len(results['parent'])==15,'正式15项同case')
    scores=[]
    for row in comparison['cases']:
        parent=results['parent'][row['case_id']];candidate=results['candidate'][row['case_id']]
        ck(parent['axes']==candidate['axes']==row['axes'] and parent['latency_ms']==row['parent_ms'] and candidate['latency_ms']==row['candidate_ms'],'正式case来源:'+row['case_id'])
        scores.append(parent['latency_ms']/candidate['latency_ms'])
    ck(math.isclose(statistics.geometric_mean(scores),comparison['actual_candidate_over_parent'],rel_tol=1e-12),'正式真实延迟比')
    ck(sha(b/'validation/delivered.py')==comparison['source_sha256'],'交付SHA')
    pi=json.loads((b/'validation/parent/inspect.json').read_bytes());ci=json.loads((b/'validation/candidate/inspect.json').read_bytes());ck(pi==ci,'正式合同与benchmark fingerprint一致')
    agent=b/'agent-review';frozen=json.loads((agent/'input-manifest.json').read_bytes());snapshot=(agent/'search-snapshot.json').read_bytes()
    ck(hashlib.sha256(snapshot).hexdigest()==frozen['search']['sha256'] and snapshot.decode() in (agent/'prompt.md').read_text(encoding='utf8'),'模型搜索输入快照精确绑定')
    provider=json.loads((agent/'provider.json').read_bytes());ck(any(r.get('http_status')==200 for r in provider['requests']),'真实CC＋DeepSeek请求')
    analysis=json.loads((b/'analysis/mechanism.json').read_bytes());ck(len(analysis['pairs'])==2,'成对机制分析')
    for pair in analysis['pairs']:
        for side,row in pair['records'].items():
            ck(row['facts']['basic']['Device Id']=='7' and row['spec']['source_sha256']==sha(b/'validation'/side/(side+'.py')),'采集源码与设备:'+side+str(pair['size']))
    provenance=json.loads((b/'analysis/roofline/diagnosis.json').read_bytes())
    ck(len(provenance['cases'])==4,'四项Roofline数值点')
    for row in provenance['cases']:
        for artifact in row['facts']['artifacts']:
            relative=PurePosixPath(artifact['path'].replace(chr(92),'/'))
            ck(not relative.is_absolute() and '..' not in relative.parts,'数值附件相对路径:'+row['id'])
            path=(b/row['id']/'profile').joinpath(*relative.parts)
            ck(sha(path)==artifact['sha256'],'数值分析附件绑定:'+row['id']+'/'+artifact['path'])
    confirmation=json.loads((b/'matched-confirmation/state.json').read_bytes())
    ck(confirmation['status']=='completed' and len(confirmation['cases'])==4,'同模板复用方向四项确认完成')
    configs=json.loads((b/'matched-confirmation/plan.json').read_bytes())['configs']
    ck([{k:v for k,v in c.items() if k not in ['id','tm','tn']} for c in configs][0]==[{k:v for k,v in c.items() if k not in ['id','tm','tn']} for c in configs][1],'复用确认其余模板参数相同')
    for name,digest in confirmation['source_sha256'].items():ck(sha(b/'matched-confirmation'/(name+'.py'))==digest,'同模板源码:'+name)
    ck(confirmation['source_sha256']['dual-m-reuse-b']==winner['source'],'确认使用同字节候选')
    for value in confirmation['cases']:
        directory=b/'matched-confirmation'/(value['dtype']+'-'+str(value['size']))
        for repeat in [0,1]:
            root=directory/('timing-'+str(repeat));metadata=json.loads((root/'measurement.json').read_bytes());files=list(long(root).rglob('task_time*.csv'))
            ck(len(files)==1 and sha(files[0])==metadata['csv_sha256'],'确认原CSV:'+str(root.relative_to(b)))
            with files[0].open(encoding='utf-8-sig',newline='') as stream:tasks=list(csv.DictReader(stream))[1:-1]
            ck(len(tasks)==16,'确认16任务:'+str(root.relative_to(b)))
            for i,record in enumerate(metadata['records']):
                ck(all(record['symbol'] in r['kernel_name'] for r in tasks[i*8:(i+1)*8]),'确认kernel身份:'+str(root.relative_to(b))+'/'+record['key'])
                times=[float(r['task_time(us)']) for r in tasks[i*8+2:(i+1)*8]]
                ck(times==record['active_device_us'] and math.isclose(statistics.mean(times)/1000,record['device_ms'],rel_tol=1e-9),'确认独立重算:'+str(root.relative_to(b))+'/'+record['key'])
                ck(math.isclose(value['times'][record['key']][repeat],record['device_ms'],rel_tol=1e-9),'确认结果绑定:'+str(root.relative_to(b))+'/'+record['key'])
        ck(math.isclose(statistics.median(p/c for p,c in zip(value['times']['dual-n-constexpr'],value['times']['dual-m-reuse-b'])),value['m_reuse_over_n_reuse'],rel_tol=1e-12),'复用方向实际加速:'+str(directory.relative_to(b)))
    ck(math.isclose(statistics.geometric_mean(v['m_reuse_over_n_reuse'] for v in confirmation['cases']),confirmation['geomean'],rel_tol=1e-12),'四项复用方向几何平均')
    for name in ['method-snapshot','final-method-snapshot']:
        frozen=json.loads((b/name/'manifest.json').read_bytes())
        for row in frozen if isinstance(frozen,list) else frozen['files']:
            ck(sha(b/name/row['path'])==row['sha256'],name+'方法原件:'+row['path'])
    runtime=b/'runtime-records'
    for row in json.loads((runtime/'manifest.json').read_bytes()):
        ck(sha(runtime/row['name'])==row['sha256'],'运行日志原件:'+row['name'])
    result=dict(status='passed',checks=len(checks),items=checks);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status='passed',checks=len(checks))))

if __name__=='__main__':main()
