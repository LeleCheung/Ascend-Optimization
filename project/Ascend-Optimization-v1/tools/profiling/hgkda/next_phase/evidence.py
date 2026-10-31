"""将成对原件筛选为带单位、绝对差、比率和质量标记的Agent证据。"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def numeric(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def metric_family(metric, table=''):
    name=metric.lower()
    if re.search('wait|conflict|bank',name):return 'wait_conflict'
    if 'scalar' in name:return 'control'
    if re.search('cache|hit_rate|miss_allocate',name) or 'cache' in table.lower():return 'cache'
    if re.search('vec|cube|mac|fops',name):return 'compute'
    if re.search(r'mte|(?:^|_)gm(?:_|$)|(?:^|_)ub(?:_|$)|datas|transfer',name):return 'movement'
    return 'other'


def compact(comparison, limit=16, material_ratio=.15):
    rows=[]
    for item in comparison.get('differences',[]):
        r,c=item.get('reference'),item.get('candidate')
        unit_match=re.search(r'\(([^)]+)\)',item['metric'])
        row={k:item.get(k) for k in ['table','metric','aggregation','reference_rows','candidate_rows','scope']}
        row.update(reference=r,candidate=c,unit=unit_match.group(1) if unit_match else 'ratio' if 'ratio' in item['metric'].lower() else 'count' if re.search('instructions|instr_number',item['metric']) else '未声明，查原始CSV表头',flags=[])
        if not numeric(r) or not numeric(c):
            row.update(status='missing_or_nonfinite',absolute_delta=None,candidate_over_reference=None,material=False)
        elif r==0:
            row.update(status='zero_reference',absolute_delta=c-r,candidate_over_reference=None,material=c!=0)
        else:
            ratio=c/r
            row.update(status='valid',absolute_delta=c-r,candidate_over_reference=ratio,material=abs(ratio-1)>=material_ratio)
        if item.get('reference_rows')!=item.get('candidate_rows'):
            row['flags'].append('记录数量不同；逐记录均值不是整任务时间')
        row['family']=metric_family(item['metric'],item['table'])
        rows.append(row)
    selected=sorted([r for r in rows if r['material']],key=lambda r:abs(r['candidate_over_reference']-1) if r['candidate_over_reference'] is not None else math.inf,reverse=True)
    # 先给每类一项，避免单一指标族挤满上下文；剩余按相对差填充。
    chosen=[];families=set();counts={}
    for row in selected:
        if row['family'] not in families:
            chosen.append(row);families.add(row['family']);counts[row['family']]=1
    for row in selected:
        if len(chosen)>=limit:break
        if row not in chosen and counts.get(row['family'],0)<3:
            chosen.append(row);counts[row['family']]=counts.get(row['family'],0)+1
    chosen=chosen[:limit]
    quality=[r for r in rows if r['status']!='valid']
    return dict(schema='hgkda-paired-context-v2',spec=comparison.get('spec'),selected_metrics=chosen,all_metric_count=len(rows),
                selection=dict(limit=limit,relative_change_threshold=material_ratio,max_per_family=3,policy='经验筛选关注度，非统计显著性检验；按指标族保留并按相对差排序，每族至多3项'),
                derived=comparison.get('derived_differences',{}),quality_flags=quality[:24],quality_flag_count=len(quality),
                references={side:comparison.get(side,{}).get('metadata',{}) for side in ['candidate','reference']},
                interpretation='参考不是理想上限；比率与绝对值、记录范围、源码/IR共同解释；高活跃不能直接证明瓶颈',
                all_metrics=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=16);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.limit<=0:raise ValueError('limit必须大于零')
    d=compact(json.loads(a.comparison.read_text(encoding='utf8')),a.limit);d['original']=dict(path=str(a.comparison),sha256=hashlib.sha256(a.comparison.read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
