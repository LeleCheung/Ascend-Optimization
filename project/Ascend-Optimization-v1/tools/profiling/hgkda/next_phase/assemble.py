"""多kernel成对采集进入紧凑Agent上下文；保留具体执行路径，避免错配。"""
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from evidence import compact,metric_family
from regions import analyze

HERE=Path(__file__).resolve().parent


def stats(capture):
    result={}
    for path in Path(capture['directory']).rglob('*.csv'):
        if path.name=='OpBasicInfo.csv':continue
        with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        groups={r.get('sub_block_id','') for r in rows}
        for group in groups:
            subset=[r for r in rows if r.get('sub_block_id','')==group]
            for metric in subset[0] if subset else []:
                if metric in {'block_id','sub_block_id',''}:continue
                values=[]
                for r in subset:
                    try:
                        import math
                        v=float(r[metric])
                        if math.isfinite(v):values.append(v)
                    except (ValueError,TypeError):pass
                aggregation='sum' if '(KB)' in metric or 'instructions' in metric or 'instr_number' in metric or metric.endswith('_fops') else 'mean'
                value=(sum(values) if aggregation=='sum' else statistics.mean(values)) if len(values)==len(subset) and values else None
                result[(path.name+'/'+group,metric)]=dict(value=value,rows=len(subset),aggregation=aggregation)
    return result


def transfer_granularity(side):
    result={}
    tables={key[0] for key in side if key[0].startswith('Memory.csv/')}
    pairs=[('GM_to_UB_datas(KB)','aiv_mte2_instructions'),('UB_to_GM_datas(KB)','aiv_mte3_instructions'),
           ('GM_to_L1_datas(KB)','aic_mte2_instructions')]
    for table in tables:
        for traffic,counter in pairs:
            if (table,traffic) not in side and (table,counter) not in side:continue
            data=side.get((table,traffic),{}).get('value');count=side.get((table,counter),{}).get('value')
            value=data*1024/count if data is not None and count is not None and count>0 else None
            result[table+'/'+traffic+'/'+counter]=dict(bytes_per_instruction=value,traffic_kb=data,instructions=count,
                interpretation='厂商CSV接口流量与完整计数之比，沿用旧工具KB×1024约定；不是物理HBM事务大小')
    return result


def path_metrics(captures, output):
    """多kernel保留逐项指标；仅可加的完整计数/流量跨kernel求和。"""
    rows=[];totals={};counts={}
    for capture in captures:
        values=stats(capture)
        metrics=[dict(table=k[0],metric=k[1],**v) for k,v in sorted(values.items())]
        rows.append(dict(kernel=capture['kernel'],directory=capture['directory'],metrics=metrics,granularity=transfer_granularity(values)))
        for key,value in values.items():
            if value['aggregation']!='sum':continue
            counts[key]=counts.get(key,0)+1
            if value['value'] is None:totals[key]=None
            elif key not in totals:totals[key]=value['value']
            elif totals[key] is not None:totals[key]+=value['value']
    payload=dict(kernels=rows,aggregate_work=[dict(table=k[0],metric=k[1],value=v if counts[k]==len(rows) else None,observed_sum=v,kernels=counts[k],total_kernels=len(rows),complete=counts[k]==len(rows)) for k,v in sorted(totals.items())],
                 scope='逐kernel时间保持各自统计范围；完整流量和指令计数可加。不将跨核时间均值相加当作端到端关键路径。')
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    concise=[]
    for row in rows:
        selected=[];seen={}
        for metric in row['metrics']:
            name=metric['metric']
            family=metric_family(name,metric['table'])
            if family=='other' or metric['value'] is None or seen.get(family,0)>=3:continue
            seen[family]=seen.get(family,0)+1;selected.append(metric)
        concise.append(dict(kernel=row['kernel'],selected_metrics=selected,all_metric_count=len(row['metrics']),granularity=row['granularity']))
    return dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),kernels=concise,
                aggregate_work=payload['aggregate_work'][:24],scope=payload['scope'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    state=json.loads((a.capture/'state.json').read_text(encoding='utf8'))
    source=Path(state['spec']['source'])
    source_bytes=source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest()!=state['spec']['source_sha256']:
        raise ValueError('采集绑定源码SHA已改变，拒绝关联热点区域')
    source_text=source_bytes.decode('utf8')
    source_context={str(source):dict(text=source_text,sha256=hashlib.sha256(source_text.encode('utf8')).hexdigest())}
    result=dict(schema='hgkda-next-context-v1',spec=state['spec'],status=state['status'],paths={},comparisons=[],regions=[],
                execution_guidance='优先尽早提交第一份完整KGS候选；首轮正式评测前最多两个诊断debug，每次不超过120秒。编译/精度问题先修复；不在未经正式评测的候选上连续做多轮启动微基准。使用真实指标/IR回答明确问题后修改并验证。',
                limits=['PyTorch仍为正式正确性/速度参考；CANN直接入口含生命周期同步，不把此处host开销作为速度成绩','多kernel不按名称相似性盲目配对；只有两侧各一个kernel时输出逐指标比率','数值Roofline不是每个复杂算子必须计算的前置条件，优先使用真实对照'])
    for side,row in state['sides'].items():
        result['paths'][side]=dict(status=row['status'],kernels=[dict(name=c['kernel'],basic=c['basic']) for c in row.get('captures',[])],error=row.get('error'))
        result['paths'][side].update({key:row[key] for key in ['simulator_status','simulator_error','simulator_warnings'] if key in row})
        if row.get('captures'):
            metrics_output=a.output.with_name(a.output.stem+'-'+side+'-path-metrics.json')
            result['paths'][side]['metrics']=path_metrics(row['captures'],metrics_output)
        semantic=a.capture/side/'semantics.json'
        if semantic.exists():
            s=json.loads(semantic.read_text(encoding='utf8'));result['paths'][side]['semantics']=dict(status=s['status'],reference=s.get('reference'))
    candidate=state['sides']['candidate'].get('captures',[])
    for side in ['pytorch','cann','ascendc']:
        reference=state['sides'].get(side,{}).get('captures',[])
        if len(candidate)!=1 or len(reference)!=1:continue
        c,r=stats(candidate[0]),stats(reference[0]);differences=[]
        for key in sorted(set(c)|set(r)):
            cv,rv=c.get(key,{}),r.get(key,{})
            differences.append(dict(table=key[0],metric=key[1],candidate=cv.get('value'),reference=rv.get('value'),
                aggregation=cv.get('aggregation',rv.get('aggregation')),candidate_rows=cv.get('rows'),reference_rows=rv.get('rows'),scope='同一指标/硬件核类别；计数求和，时间均值，不代表端到端关键路径'))
        cg,rg=transfer_granularity(c),transfer_granularity(r);derived={}
        for key in sorted(set(cg)|set(rg)):
            cv=cg.get(key,{}).get('bytes_per_instruction');rv=rg.get(key,{}).get('bytes_per_instruction')
            derived[key]=dict(reference=rv,candidate=cv,absolute_delta=cv-rv if cv is not None and rv is not None else None,
                candidate_over_reference=cv/rv if cv is not None and rv else None,unit='B/instruction',candidate_details=cg.get(key),reference_details=rg.get(key))
        evidence=compact(dict(spec=state['spec'],differences=differences,derived_differences=derived),limit=12)
        comparison_file=a.output.with_name(a.output.stem+'-comparison-'+side+'.json')
        comparison_file.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        evidence.pop('all_metrics')
        evidence['full_comparison']=dict(path=str(comparison_file),sha256=hashlib.sha256(comparison_file.read_bytes()).hexdigest())
        evidence.update(reference_kind=side,candidate_kernel=candidate[0]['kernel'],reference_kernel=reference[0]['kernel'])
        result['comparisons'].append(evidence)
    ir=a.capture/'candidate/ir/manifest.json'
    if not ir.exists() or not json.loads(ir.read_text(encoding='utf8')).get('entries'):
        recovered=a.capture/'candidate/ir-recovered'
        spec_path=a.capture/'ir-recovery-spec.json'
        spec_path.write_text(json.dumps(state['spec'],ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        if not recovered.exists():
            with (a.capture/'candidate/ir-recovery.log').open('w') as log:
                execution=subprocess.run([sys.executable,str(HERE/'driver.py'),'--spec',str(spec_path),'--side','candidate','--ir',str(recovered)],stdout=log,stderr=subprocess.STDOUT,timeout=300)
            if execution.returncode:result['ir_recovery_error']='实际JIT捕获未成功，保留日志，不把空manifest当实际IR'
        if (recovered/'manifest.json').exists():ir=recovered/'manifest.json'
    if ir.exists():result['actual_ir']=json.loads(ir.read_text(encoding='utf8'))
    listing_path=a.capture/'candidate/listing.json'
    if listing_path.exists():
        listing=json.loads(listing_path.read_text(encoding='utf8'))
        for kernel in listing.get('kernels',[]):
            for core in kernel.get('cores',[]):
                trace=next((a.capture/'candidate/simulator').rglob(core['name']+'/trace.json'),None)
                region=analyze(listing,json.loads(trace.read_text()) if trace else None,kernel=kernel['name'],core=core['name'],top=6,sources=source_context)
                for item in region['regions']:
                    item['pc_count']=len(item['pcs']);item['pcs']=item['pcs'][:16]
                    item['source_lines']=item['source_lines'][:16]
                for pipe in region.get('timeline',{}).get('pipelines',[]):
                    pipe['idle_intervals']=pipe.get('idle_intervals',[])[:8]
                result['regions'].append(region)
    source=Path(state['spec']['source'])
    text=source.read_text(encoding='utf8')
    result['source']=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),text=text[:18000],
                          full_characters=len(text),truncated=len(text)>18000,read_full_on_demand=len(text)>18000)
    result['provenance']=dict(capture_state=str(a.capture/'state.json'),sha256=hashlib.sha256((a.capture/'state.json').read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
