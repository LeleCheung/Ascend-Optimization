"""有操作ID与有限预算的KGS真机/模拟器采集；文件名避免遮蔽stdlib profile。"""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tarfile
import time
import urllib.error
import urllib.request
import uuid


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def summarize(response,artifacts,source):
    symbols={n.name for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and any('jit' in ast.unparse(d) for d in n.decorator_list)}
    def matches(name):return any(name==s or name.startswith(s+'_') for s in symbols)
    result=dict(profile_status=response.get('status'),error=response.get('error'),metrics=response.get('metrics'),
        simulator_hotspots=[],opprof=[],limits=['流水比例不能相加成时间分解','各层内存接口字节不自动等于HBM实际流量','模拟器周期不当作真机延迟'])
    for artifact in artifacts:
        path=Path(artifact['path'])
        if artifact['kind']=='instruction_listing':
            listing=json.loads(path.read_text(encoding='utf-8-sig'))
            for kernel in listing.get('kernels',[]):
                if not matches(kernel.get('name','')):continue
                for core in kernel.get('cores',[]):
                    instructions=core.get('instructions',[])
                    top=sorted(instructions,key=lambda r:(r.get('running_time_us') or 0,r.get('cycles') or 0),reverse=True)[:12]
                    result['simulator_hotspots'].append(dict(kernel=kernel['name'],core=core.get('name'),hotspots=top))
        if artifact['kind']!='vendor_kernel_report':continue
        with tarfile.open(path,'r:*') as tar:
            tables={m.name:list(csv.DictReader(io.StringIO(tar.extractfile(m).read().decode('utf-8-sig')))) for m in tar.getmembers() if m.isfile() and m.name.endswith('.csv')}
        for filename,basic in tables.items():
            if not filename.endswith('/OpBasicInfo.csv') or len(basic)!=1:continue
            kernel=basic[0].get('Op Name','')
            if not matches(kernel):continue
            prefix=filename.rsplit('/',1)[0]+'/'
            facts=dict(kernel=kernel,basic=basic[0],tables={},derived={})
            for full,rows in tables.items():
                if not full.startswith(prefix) or full==filename or not rows:continue
                grouped={}
                for row in rows:grouped.setdefault(row.get('sub_block_id',''),[]).append(row)
                for group,items in grouped.items():
                    fields={}
                    for key in items[0]:
                        if key in ['block_id','sub_block_id']:continue
                        values=[]
                        for item in items:
                            try:
                                value=float(item.get(key,''))
                                if math.isfinite(value):values.append(value)
                            except (ValueError,TypeError):pass
                        fields[key]=dict(valid=len(values),missing=len(items)-len(values),mean=statistics.mean(values) if values else None,
                            min=min(values) if values else None,max=max(values) if values else None,
                            sum=sum(values) if values and len(values)==len(items) else None)
                    short=full[len(prefix):]+'/'+group;facts['tables'][short]=fields
                    if short.startswith('Memory.csv/'):
                        nbytes=fields.get('GM_to_UB_datas(KB)',{}).get('sum');count=fields.get('aiv_mte2_instructions',{}).get('sum')
                        facts['derived'][short+'/bytes_per_mte2']=nbytes*1024/count if nbytes is not None and count and count>0 else None
            result['opprof'].append(facts)
    return result


def collect(evaluation,case_id,out,server,level,budget):
    if out.exists():raise ValueError('采集输出已存在')
    evaluation_data=json.loads(evaluation.read_text());label=evaluation.name.removesuffix('.result.json').rsplit('-',1)[0]
    request=json.loads(evaluation.with_name(evaluation.name.replace('.result.json','.request.json')).read_text())
    inspection=json.loads((evaluation.parent/'inspect.json').read_text())
    if evaluation_data.get('status')!='PASSED' or not any(r['uuid']==case_id and r['phase']=='timing' for r in evaluation_data['per_workload']):
        raise ValueError('只采集完整通过的同身份timing case')
    source=request['implementation']['sources'][0]['content'];digest=hashlib.sha256(source.encode()).hexdigest()
    operation='hanle-device-'+uuid.uuid4().hex
    options=dict(level=level,warmup=1,iterations=2,timeout_sec=budget,
        backend_options={'npu':dict(aic_metrics='PipeUtilization',op_detail=True,
            op_metrics='Default,BasicInfo',simulator=level=='instruction',simulator_core_ids='0',simulator_timeout_min=2)})
    payload=dict(api_version='v6.2',binding=request['binding'],implementation=request['implementation'],
        benchmark_fingerprint=inspection['benchmark_fingerprint'],case_id=case_id,expected_backend='npu',options=options)
    out.mkdir(parents=True);save(out/'request.json',payload)
    save(out/'identity.json',dict(operation_id=operation,source_sha256=digest,case_id=case_id,level=level,budget_seconds=budget))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    command=urllib.request.Request(server+'/profile',data=json.dumps(payload).encode(),
        headers={'Content-Type':'application/json','X-KernelGen-Operation-Id':operation})
    start=time.monotonic()
    try:
        with opener.open(command,timeout=budget+90) as stream:response=json.load(stream)
        save(out/'response.json',response)
    except Exception as error:
        save(out/'request-error.json',dict(error=type(error).__name__+': '+str(error),seconds=time.monotonic()-start))
        try:
            cancel=urllib.request.Request(server+'/operations/'+operation,method='DELETE')
            with opener.open(cancel,timeout=30) as stream:save(out/'cancel.json',json.load(stream))
            for _ in range(20):
                with opener.open(server+'/operations/'+operation,timeout=15) as stream:state=json.load(stream)
                save(out/'operation-final.json',state)
                if state.get('state') in ['CANCELLED','SUCCEEDED','FAILED']:break
                time.sleep(2)
            else:raise RuntimeError('本次操作终态未知，不继续提交设备任务')
        except urllib.error.HTTPError as e:
            if e.code!=404:raise
        raise
    artifacts=[]
    for item in response.get('artifacts',[]):
        url=item.get('download_url','')
        if not url.startswith('/profile_artifacts/'):raise ValueError('附件URL不合法')
        with opener.open(server+url,timeout=120) as stream:data=stream.read()
        if len(data)!=item['size_bytes']:raise ValueError('附件长度不一致')
        path=out/'artifacts'/(item['id']+'-'+Path(item['filename']).name);path.parent.mkdir(exist_ok=True)
        path.write_bytes(data);artifacts.append(dict(kind=item['kind'],path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    save(out/'artifacts.json',artifacts)
    facts=summarize(response,artifacts,source);facts.update(source_sha256=digest,case_id=case_id,seconds=time.monotonic()-start)
    save(out/'facts.json',facts)
    print(json.dumps(dict(status=response.get('status'),opprof=len(facts['opprof']),hotspot_cores=len(facts['simulator_hotspots']),seconds=facts['seconds']),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--case-id',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--server',default='http://127.0.0.1:19659');p.add_argument('--level',choices=['metrics','instruction'],default='metrics')
    p.add_argument('--budget',type=int,default=300);a=p.parse_args();collect(a.evaluation,a.case_id,a.output,a.server,a.level,a.budget)
