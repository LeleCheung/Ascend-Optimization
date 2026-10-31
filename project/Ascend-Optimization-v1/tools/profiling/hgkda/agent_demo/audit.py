"""独立核对工具输入、热点映射、正式上传与原CSV，支持Windows/Linux。"""
import argparse,csv,hashlib,importlib.util,json,math,os,re,statistics,sys
from pathlib import Path

def long(p):
    prefix=chr(92)*2+'?'+chr(92)
    return Path(prefix+str(p.resolve())) if os.name=='nt' and not str(p).startswith(prefix) else p
def sha(p):return hashlib.sha256(long(p).read_bytes()).hexdigest()
def read(p):return json.loads(long(p).read_bytes())
def local_path(b,path):
    text=str(path).replace(chr(92),'/');anchor='agent-demo-20261027/'
    if anchor not in text:raise ValueError('产物不属于本批：'+text)
    return b.joinpath(*text.split(anchor,1)[1].split('/'))

def audit(b):
    checks=[]
    def ck(v,label):
        if not v:raise ValueError(label)
        checks.append(label)
    for manifest in (b/'transfers').glob('*.manifest.json'):
        data=read(manifest)
        archive=manifest.with_name(manifest.name.replace('.tar.manifest.json','.tar.gz'))
        ck(sha(archive)==data['archive_sha256'],'原传输归档:'+archive.name)
        for r in data['files']:
            p=b/r['path'];ck(sha(p)==r['sha256'] and long(p).stat().st_size==r['bytes'],'传输成员:'+r['path'])
    source_sha=sha(b/'seed.py')
    evidence=read(b/'agent-input.json');ck(evidence['source_sha256']==source_sha,'模型起点源码SHA')
    for size in [512,1024]:
        for origin,sides in [('seed-profile',['candidate','pytorch']),('candidate-profile',['candidate'])]:
            state=read(b/origin/'state.json');ck(state['status']=='completed',origin+'完成')
            for side in sides:
                root=b/origin/(side+'-float16-'+str(size));spec=read(root/'spec.json');driver=read(root/'compiled/driver.json');discovery=read(root/'discovery.json')
                ck(spec['shape']==[size]*3 and spec['dtype']=='float16' and driver['status']=='passed' and driver['spec']==spec,'同输入驱动身份:'+origin+'/'+side+str(size))
                ck(spec['source_sha256']==sha(b/('seed.py' if origin=='seed-profile' else 'candidate.py')),'采集源码:'+origin+'/'+side+str(size))
                files=list(long(root/'profile').rglob('OpBasicInfo.csv'));ck(len(files)==1,'目标设备记录唯一:'+origin+'/'+side+str(size))
                with files[0].open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
                ck(len(rows)==1 and rows[0]['Device Id']=='7' and discovery['selected']==[rows[0]['Op Name']],'设备与kernel:'+origin+'/'+side+str(size))
    for label in ['seed','candidate']:
        root=b/('seed-instruction-v3' if label=='seed' else 'candidate-instruction');listing=read(root/'listing.json');summary=read(root/'summary.json');request=read(root/'request.json')
        raw=(root/'stdout.txt').read_text(encoding='utf8',errors='replace')
        partial=any(int(x)!=0 for x in re.findall(r'Child process exited with status (\d+)',raw))
        ck(request['source_sha256']==sha(b/(label+'.py')),'指令源码:'+label)
        count=sum(c['listed_instruction_count'] for k in listing['kernels'] for c in k['cores'])
        mapped=sum(c['mapped_instruction_count'] for k in listing['kernels'] for c in k['cores'])
        ck(count==summary['instructions'] and mapped==summary['mapped_instructions'] and count>0,'真实指令计数:'+label)
        if label=='seed':ck(evidence['instruction']['status']=='partial' if partial else True,'起点模拟器异常未隐藏')
        else:ck(summary['status']=='partial' if partial else summary['status']=='completed','候选模拟器状态正确')
        symbols={k['name'] for k in listing['kernels']}
        ck(all(c['kernel'] in symbols for c in summary['hotspots']['kernels']),'热点只来自候选kernel:'+label)
        lines=(b/(label+'.py')).read_text(encoding='utf8').splitlines();verified=0
        for kernel in summary['hotspots']['kernels']:
            ck(len(kernel['hotspots'])<=12,'热点有限预算:'+label+'/'+kernel['core'])
            ordered=sorted(kernel['hotspots'],key=lambda r:(r.get('running_time_us') or 0,r.get('cycles') or 0),reverse=True)
            ck(ordered==kernel['hotspots'],'热点按实际时间排序:'+label+'/'+kernel['core'])
            for row in kernel['hotspots']:
                for loc in [row.get('source',{})]+row.get('source_stack',[]):
                    if str(loc.get('file','')).replace(chr(92),'/').endswith('/'+label+'.py') and 'code' in loc:
                        line=loc['line'];ck(0<line<=len(lines) and lines[line-1].strip()==loc['code'],'源码行映射:'+label+':'+str(line));verified+=1
        ck(verified>0,'至少一条热点可核对到候选源码:'+label)
    manifest=read(b/'agent/input-manifest.json');ck(manifest['evidence_sha256']==sha(b/'agent-input.json') and manifest['source_sha256']==source_sha,'冻结模型输入')
    route=read(b/'agent/route/decision.json');ck(route['tools']==manifest['tools'],'模型主动工具路由')
    for row in manifest['tool_results']:ck(sha(local_path(b,row['path']))==row['sha256'],'工具原始返回:'+row['path'].split('/')[-1])
    for phase in ['route','plan']:
        provider=read(b/'agent'/phase/'provider.json');ck(any(r.get('http_status')==200 for r in provider['requests']),'真实CC＋DeepSeek:'+phase)
    proposal=read(b/'agent/proposal.json');plan=read(b/'search/plan.json');state=read(b/'search/state.json')
    ck(proposal['config_ids']==[r['id'] for r in plan['configs']],'按模型动作生成配置')
    valid=[]
    for row in state['cases']:
        ck(sha(b/'search'/row['case']/'candidate.py')==row['source_sha256'],'开发源码:'+row['case'])
        ck(row['status']=='passed' and len(row['measurements'])==6,'开发六项完整:'+row['case'])
        score=statistics.geometric_mean(v['speedup'] for v in row['measurements']);ck(math.isclose(score,row['speedup'],rel_tol=1e-12),'开发几何平均:'+row['case']);valid.append(row)
        for value in row['measurements']:
            directory=b/'search'/row['case']/(value['dtype']+'-'+str(value['size']))
            for repeat in [0,1]:
                root=directory/('timing-'+str(repeat));meta=read(root/'measurement.json');fs=list(long(root).rglob('task_time*.csv'))
                ck(len(fs)==1 and sha(fs[0])==meta['csv_sha256'],'原始task CSV:'+str(root.relative_to(b)))
                with fs[0].open(encoding='utf-8-sig',newline='') as f:tasks=list(csv.DictReader(f))[1:-1]
                ck(len(tasks)==16,'原始16任务:'+str(root.relative_to(b)))
                for i,record in enumerate(meta['records']):
                    rows=tasks[i*8:(i+1)*8];times=[float(r['task_time(us)']) for r in rows[2:]]
                    ck(all(record['symbol'] in r['kernel_name'] for r in rows),'任务符号:'+record['key'])
                    ck(times==record['active_device_us'] and math.isclose(statistics.mean(times)/1000,record['device_ms'],rel_tol=1e-9),'设备CSV重算:'+record['key'])
                    ck(math.isclose(value['times'][record['key']][repeat],record['device_ms'],rel_tol=1e-9),'搜索结果绑定:'+record['key'])
            ck(math.isclose(statistics.median(p/c for p,c in zip(value['times']['parent'],value['times']['candidate'])),value['speedup'],rel_tol=1e-12),'开发同case速度:'+str(directory.relative_to(b)))
    selected=read(b/'validation/selection-before-kgs.json');ck(max(valid,key=lambda r:r['speedup'])['case']==selected['selected']['case'],'正式前冻结最优')
    result_maps={}
    for label in ['seed','candidate']:
        root=b/'validation'/label;ev=read(root/(label+'-1.result.json'));req=read(root/(label+'-1.request.json'));raw=(b/(label+'.py')).read_bytes()
        ck(req['implementation']['sources'][0]['content'].encode()==raw,'正式上传精确源码:'+label)
        ck(ev['status']=='PASSED' and ev['num_passed']==ev['num_workloads']==42,'完整KGS42项:'+label)
        result_maps[label]={r['uuid']:r for r in ev['per_workload'] if r['phase']=='timing'}
        for suffix,count in [('semantics',18),('large',24)]:
            d=read(b/'validation'/(label+'-'+suffix)/'check.json');ck(d['status']=='passed' and d['passed']==d['total']==count and d['source_sha256']==hashlib.sha256(raw).hexdigest(),'补充门禁:'+label+'/'+suffix)
    ck(set(result_maps['seed'])==set(result_maps['candidate']) and len(result_maps['seed'])==15,'正式15项一致')
    comparison=read(b/'validation/comparison.json');ratios=[]
    for row in comparison['cases']:
        s=result_maps['seed'][row['case_id']];c=result_maps['candidate'][row['case_id']]
        ck(s['axes']==c['axes']==row['axes'] and s['latency_ms']==row['seed_ms'] and c['latency_ms']==row['candidate_ms'],'逐case来源:'+row['case_id']);ratios.append(s['latency_ms']/c['latency_ms'])
    ck(math.isclose(statistics.geometric_mean(ratios),comparison['actual_candidate_over_seed'],rel_tol=1e-12),'实际完整延迟比')
    ck(sha(b/'validation/delivered.py')==comparison['source_sha256'],'最终交付源码')
    facts=read(b/'feedback/facts.json')
    ck(facts['comparison']==comparison,'反馈采用正式原件')
    source=Path(__file__).resolve().parent.parent/'roofline/profile_data.py'
    module=importlib.util.spec_from_file_location('demo_audit_profile',source);parser=importlib.util.module_from_spec(module);module.loader.exec_module(parser)
    for pair in facts['pairs']:
        values={}
        for side,origin in [('reference','seed-profile'),('candidate','candidate-profile')]:
            directory=b/origin/('candidate-float16-'+str(pair['size']))
            data=parser.parse(long(directory/'profile'))
            values[side]={r['path']:r['official_bytes'] for r in data['curves']}
            ck(math.isclose(data['duration_us'],pair['facts'][side]['duration_us'],rel_tol=1e-12),'反馈原采集时间:'+side+str(pair['size']))
        for path,ratio in pair['traffic_candidate_over_seed'].items():
            ck(math.isclose(values['candidate'][path]/values['reference'][path],ratio,rel_tol=1e-12),'真实组件流量比:'+path+str(pair['size']))
        for label,row in pair['ir'].items():
            file=local_path(b,row['path']);ck(sha(file)==row['sha256'],'反馈IR原件:'+label+str(pair['size']))
            text=file.read_text(encoding='utf8')
            for key,token in [('static_matmul','linalg.matmul'),('static_copy','memref.copy'),('static_alloc','memref.alloc')]:
                ck(text.count(token)==row[key],'反馈静态IR重算:'+label+'/'+key+str(pair['size']))
    provider=read(b/'feedback/agent-review/provider.json');ck(any(r.get('http_status')==200 for r in provider['requests']),'真实CC预测复核')
    prompt=(b/'feedback/agent-review/prompt.md').read_text(encoding='utf8')
    ck(json.dumps(facts,ensure_ascii=False) in prompt,'反馈模型实际读取同批全部事实')
    for name in ['method-executed','method-final']:
        root=b/name
        if not root.exists():continue
        frozen=read(root/'manifest.json')
        for row in frozen:ck(sha(root/row['path'])==row['sha256'],'方法快照:'+name+'/'+row['path'])
    if (b/'report-verified.json').exists():
        for row in read(b/'report-verified.json')['files']:
            ck(sha(b/row['path'])==row['sha256'],'核验版报告:'+row['path'])
    return dict(status='passed',checks=len(checks),items=checks)

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    result=audit(a.batch);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status='passed',checks=result['checks'])))
if __name__=='__main__':main()
