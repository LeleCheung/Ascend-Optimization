"""离线汇总完整调用差距、参考波动与独立设备路径，生成下一项可验证动作。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def timing_rows(result):
    rows={}
    for row in result.get('per_workload',[]):
        if row.get('phase')!='timing':continue
        key=row['uuid']
        if key in rows:raise ValueError('重复性能用例 '+key)
        for field in ['latency_ms','reference_latency_ms']:
            value=row.get(field)
            if not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0:raise ValueError('无效耗时 '+key+' '+field)
        if row.get('status')!='PASSED':raise ValueError('性能用例未通过 '+key)
        rows[key]=row
    if not rows:raise ValueError('没有有效性能用例')
    return rows

def analyze(result_path,source,catalog,*,parent=None,paths=None):
    result_path,source,catalog=map(Path,(result_path,source,catalog))
    request_path=result_path.with_name(result_path.name.replace('.result.json','.request.json'))
    if request_path==result_path:raise ValueError('结果文件名必须以.result.json结束')
    request=read(request_path)
    submitted=request['implementation']['sources']
    if len(submitted)!=1 or submitted[0]['content']!=source.read_text(encoding='utf8'):raise ValueError('请求与源码身份不同')
    result=read(result_path);rows=timing_rows(result)
    if result.get('status')!='PASSED' or result.get('num_passed')!=result.get('num_workloads'):raise ValueError('正式完整正确性未通过')
    cases={x['name']:x for x in [json.loads(s) for s in (catalog/'timing.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]}
    if set(cases)!=set(rows):raise ValueError('合同与正式性能用例不匹配')
    old=None
    parent_evidence=None
    if parent:
        parent=Path(parent);parent_value=read(parent)
        parent_request=parent.with_name(parent.name.replace('.result.json','.request.json'))
        parent_payload=read(parent_request)
        if parent_payload.get('binding')!=request.get('binding') or parent_payload.get('settings')!=request.get('settings'):
            raise ValueError('父版与当前的合同绑定或评测参数不同')
        if parent_value.get('status')!='PASSED' or parent_value.get('num_passed')!=parent_value.get('num_workloads'):
            raise ValueError('父版完整正确性未通过')
        old=timing_rows(parent_value)
        parent_evidence=dict(result=str(parent),result_sha256=sha(parent),request=str(parent_request),request_sha256=sha(parent_request))
    if old is not None and set(old)!=set(rows):raise ValueError('父版与当前性能用例集合不同')
    facts=[];count=len(rows)
    for key,row in rows.items():
        spec=cases[key]['inputs']['spec'];speedup=row['reference_latency_ms']/row['latency_ms']
        fact=dict(case=key,shape=spec['shape'],dtype=spec['dtype'],elements=math.prod(spec['shape']),
            candidate_us=row['latency_ms']*1000,reference_us=row['reference_latency_ms']*1000,speedup=speedup,
            log_slowdown_contribution=-math.log(speedup)/count)
        if old is not None:
            fact.update(candidate_time_ratio=row['latency_ms']/old[key]['latency_ms'],
                reference_time_ratio=row['reference_latency_ms']/old[key]['reference_latency_ms'])
        facts.append(fact)
    facts.sort(key=lambda x:x['log_slowdown_contribution'],reverse=True)
    score=math.exp(sum(math.log(x['speedup']) for x in facts)/count)
    evidence=[]
    if paths:
        for path in sorted(Path(paths).rglob('summary.json')):
            value=read(path)
            if value.get('schema')!='ascend-device-path-v2':continue
            trace=Path(value['trace'])
            if not trace.exists() or sha(trace)!=value['trace_sha256']:raise ValueError('设备trace身份无效 '+str(path))
            spec_path=path.parent/'spec.json';spec=read(spec_path)
            bound=Path(spec['source'])
            if sha(bound)!=spec['source_sha256']:raise ValueError('路径源码身份不同')
            evidence.append(dict(summary=str(path),summary_sha256=sha(path),source_sha256=spec['source_sha256'],
                label=spec.get('implementation_label'),case=spec['case'],
                device_union_us=value['device_interval_union_us'],tasks=len(value['device_tasks']),kernel_names=value['kernel_names']))
    maximum=max(x['elements'] for x in facts)
    large=[x for x in facts if x['elements']==maximum]
    small=[x for x in facts if x['elements']<=maximum/16]
    geo=lambda values:math.exp(sum(math.log(v) for v in values)/len(values)) if values else None
    hypotheses=[]
    if large and geo([x['speedup'] for x in large])<.8:
        hypotheses.append(dict(direction='large_input_device_path',evidence_cases=[x['case'] for x in large],
            action='比较同输入设备路径、每核数据范围、搬运块、预取及类型转换；记录组件差异后选择修改',
            prediction='大输入候选实际延迟应下降，参考变慢不计实现收益'))
    if small and geo([x['speedup'] for x in small])<.8:
        hypotheses.append(dict(direction='small_input_host_or_launch',evidence_cases=[x['case'] for x in small],
            action='保持设备源码，减少Python元数据、分配与扩展调用；另测同输入设备时间区分设备原因',
            prediction='完整调用下降；不能从不同采集窗口相减得到精确host耗时'))
    drift=None
    if old is not None:
        candidate_ratio=geo([x['candidate_time_ratio'] for x in facts]);reference_ratio=geo([x['reference_time_ratio'] for x in facts])
        drift=dict(candidate_time_ratio=candidate_ratio,reference_time_ratio=reference_ratio,
            candidate_actual_speedup=1/candidate_ratio,interpretation='两个比值分别表示候选和参考耗时变化；分数变化不能全部归为实现改进')
    return dict(schema='hgkda-execution-diagnosis-v1',operator=read(catalog/'definition.json').get('name',cases[next(iter(cases))]['name'].rsplit('-timing-',1)[0]),
        source=str(source),source_sha256=sha(source),result=str(result_path),result_sha256=sha(result_path),
        request=str(request_path),request_sha256=sha(request_path),
        contracts={p.name:sha(p) for p in [catalog/n for n in ['definition.json','oracle.py','correctness.jsonl','timing.jsonl']]},
        score=score,timing_case_count=count,cases=facts,drift=drift,parent_evidence=parent_evidence,device_evidence=evidence,
        hypotheses=hypotheses,verification=['原KGS完整合同','原扩展语义','真实快速路径特殊值与stream','冻结后新形状'],
        scope='脚本计算事实和待验证假设；Agent结合源码、IR和成对指标选改法。独立设备采集不替代完整调用，不自动证明因果')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['result','source','catalog','output']:p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--parent-result',type=Path);p.add_argument('--paths',type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    value=analyze(a.result,a.source,a.catalog,parent=a.parent_result,paths=a.paths)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:value[k] for k in ['operator','score','timing_case_count','drift','hypotheses']},ensure_ascii=False))

if __name__=='__main__':main()
