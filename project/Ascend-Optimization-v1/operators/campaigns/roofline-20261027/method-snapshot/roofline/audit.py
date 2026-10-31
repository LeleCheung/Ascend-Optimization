"""独立核对Roofline本批传输、计时CSV、源码、合同和模型输入证据。"""
import argparse,csv,hashlib,json,math,os,statistics
from pathlib import Path
from profile_data import parse

def long(path):
    return Path(chr(92)*2+'?'+chr(92)+str(path.resolve())) if os.name=='nt' else path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit(batch):
    checks=[]
    def check(condition,label):
        if not condition:raise ValueError(label)
        checks.append(label)
    for manifest in sorted((batch/'transfers').glob('*.manifest.json')):
        for row in json.loads(manifest.read_bytes()):
            path=long(batch/row['path'])
            check(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'附件字节:'+row['path'])
    for label in ['search-v2','search-v3']:
        state=json.loads((batch/label/'state.json').read_bytes());check(state['status']=='completed',label+'完成')
        for c in state['cases']:
            check(sha(batch/label/c['case']/'candidate.py')==c['source_sha256'],label+'/'+c['case']+'源码')
            if c['status']!='passed':continue
            for item in c['measurements']:
                for repeat in [0,1]:
                    directory=batch/label/c['case']/(item['dtype']+'-'+str(item['size']))/('timing-'+str(repeat))
                    metadata=json.loads((directory/'measurement.json').read_bytes());files=list(long(directory).rglob('task_time*.csv'))
                    check(len(files)==1,'唯一task CSV:'+str(directory.relative_to(batch)))
                    check(sha(files[0])==metadata['csv_sha256'],'计时CSV SHA:'+str(directory.relative_to(batch)))
                    with files[0].open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))[1:-1]
                    check(len(rows)==16,'设备任务数:'+str(directory.relative_to(batch)))
                    for i,record in enumerate(metadata['records']):
                        values=[float(r['task_time(us)']) for r in rows[i*8+2:(i+1)*8]]
                        check(values==record['active_device_us'],'逐任务时间:'+str(directory.relative_to(batch))+'/'+record['key'])
                        check(math.isclose(statistics.mean(values)/1000,record['device_ms'],rel_tol=1e-9),'CSV重算:'+str(directory.relative_to(batch))+'/'+record['key'])
                        check(math.isclose(item['times'][record['key']][repeat],record['device_ms'],rel_tol=1e-9),'搜索时间绑定:'+str(directory.relative_to(batch))+'/'+record['key'])
    for label in ['parent','candidate']:
        directory=batch/'validation'/label;result=json.loads((directory/(label+'-1.result.json')).read_bytes())
        request=json.loads((directory/(label+'-1.request.json')).read_bytes())
        provenance=json.loads((directory/(label+'.provenance.json')).read_bytes())
        source=(directory/(label+'.py')).read_bytes()
        check(hashlib.sha256(source).hexdigest()==provenance['source_sha256'],label+'正式源码SHA')
        check(request['implementation']['sources'][0]['content'].encode()==source,label+'实际上传源码')
        check(result['status']=='PASSED' and result['num_passed']==result['num_workloads']==42,label+'完整42项通过')
        check(sum(r['phase']=='timing' for r in result['per_workload'])==15,label+'15项计时')
        semantics=json.loads((batch/'validation'/(label+'-semantics/check.json')).read_bytes())
        check(semantics['passed']==semantics['total']==18 and semantics['source_sha256']==provenance['source_sha256'],label+'补充18项与源码绑定')
    comparison=json.loads((batch/'validation/comparison.json').read_bytes())
    check(sha(batch/'validation/delivered.py')==comparison['source_sha256'],'交付源码SHA')
    check(math.isclose(statistics.geometric_mean(c['parent_ms']/c['candidate_ms'] for c in comparison['cases']),comparison['actual_candidate_over_parent'],rel_tol=1e-12),'正式实际加速比重算')
    parent_inspect=json.loads((batch/'validation/parent/inspect.json').read_bytes())
    candidate_inspect=json.loads((batch/'validation/candidate/inspect.json').read_bytes())
    check(parent_inspect['benchmark_fingerprint']==candidate_inspect['benchmark_fingerprint'],'两组benchmark fingerprint一致')
    check(parent_inspect['case_list']==candidate_inspect['case_list'],'两组完整case合同一致')
    for directory in ['matrix-profile','candidate-profile']:
        state=json.loads((batch/directory/'state.json').read_bytes());check(state['status']=='completed',directory+'采集完成')
        for item in state['cases']:
            root=batch/directory/item['case'];facts=parse(root/'profile');spec=json.loads((root/'spec.json').read_bytes())
            driver=json.loads((root/'compiled/driver.json').read_bytes());selected=json.loads((root/'discovery.json').read_bytes())
            check(driver['status']=='passed' and driver['spec']==spec and selected['selected']==[facts['basic']['Op Name']],directory+'/'+item['case']+'输入与kernel绑定')
    for label in ['agent-review','agent-review-v2']:
        root=batch/label;metadata=json.loads((root/'input-manifest.json').read_bytes());provider=json.loads((root/'provider.json').read_bytes())
        check(any(r.get('http_status')==200 for r in provider['requests']),label+'真实模型调用')
        prompt=(root/'prompt.md').read_text(encoding='utf8')
        check('hgkda-roofline-v1' in prompt and '官方GM' in prompt,label+'提示词包含数值证据')
        check((root/'proposal.json').is_file() and len(metadata['role_sha256'])==64,label+'角色与回执')
    empirical=json.loads((batch/'empirical-audit/results.json').read_bytes())
    for record in empirical['records']:
        name=record['kind']+'-'+record['dtype'];files=list(long(batch/'empirical-audit'/name).rglob('task_time*.csv'))
        check(len(files)==1 and sha(files[0])==record['csv_sha256'],'经验参考CSV:'+name)
        with files[0].open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))[1:-1]
        values=[float(r['task_time(us)']) for r in rows[3:]]
        check(values==record['active_device_us'] and math.isclose(statistics.mean(values)/1000,record['device_ms'],rel_tol=1e-9),'经验参考独立重算:'+name)
    summary=json.loads((batch/'final-roofline-summary.json').read_bytes())
    check(summary['source']['sha256']==sha(batch/'analysis-final/diagnosis.json'),'最终摘要绑定完整分析')
    check(len(summary['cases'])==12,'最终12项数值采集')
    final=json.loads((batch/'analysis-final/diagnosis.json').read_bytes())
    gm=final['empirical_reference']['gm_counter_reference']
    check(math.isclose(gm['gbps'],gm['bytes']/gm['duration_us']/1000,rel_tol=1e-12),'同GM事件经验参考单位与重算')
    return dict(status='passed',checks=len(checks),items=checks)

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('不覆盖已有审计')
    result=audit(a.batch);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],checks=result['checks'])))

if __name__=='__main__':main()
