"""同一冻结输入的新旧候选指标比较；不将变化幅度自动解释为性能因果。"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from evidence import compact


def checked_json(path, expected_sha=None):
    path=Path(path);raw=path.read_bytes()
    if expected_sha and hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('证据SHA改变：'+str(path))
    return json.loads(raw)


def load_context(path):
    path=Path(path);context=checked_json(path);spec=context['spec']
    source=Path(spec['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=spec['source_sha256']:
        raise ValueError('候选源码SHA改变：'+str(source))
    state=context['provenance']
    checked_json(state['capture_state'],state['sha256'])
    candidate=context['paths']['candidate']
    if candidate['status']!='completed':
        raise ValueError('候选真机采集尚未完整完成')
    metrics=candidate['metrics']
    data=checked_json(metrics['path'],metrics['sha256'])
    identity=dict(context=str(path),context_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source=str(source),source_sha256=spec['source_sha256'],
                  metrics=metrics['path'],metrics_sha256=metrics['sha256'])
    return context,data,identity


def compare(old_context,new_context,old_metrics,new_metrics):
    old=old_context['spec'];new=new_context['spec']
    for field in ['operator','case','contract_sha256']:
        if not old.get(field) or old.get(field)!=new.get(field):
            raise ValueError('新旧证据的冻结输入或合同不同：'+field)
    if len(old_metrics['kernels'])!=1 or len(new_metrics['kernels'])!=1:
        raise ValueError('当前版本比较只支持两侧各一个目标kernel；不猜测多kernel时间对应关系')
    old_kernel=old_metrics['kernels'][0];new_kernel=new_metrics['kernels'][0]
    def index(kernel):
        return {(row['table'],row['metric']):row for row in kernel['metrics']}
    om,nm=index(old_kernel),index(new_kernel);differences=[]
    for table,metric in sorted(set(om)|set(nm)):
        a=om.get((table,metric),{});b=nm.get((table,metric),{})
        if a and b and a['aggregation']!=b['aggregation']:
            raise ValueError('指标聚合方式不同：'+metric)
        differences.append(dict(table=table,metric=metric,reference=a.get('value'),candidate=b.get('value'),
            aggregation=a.get('aggregation',b.get('aggregation')),reference_rows=a.get('rows'),candidate_rows=b.get('rows'),
            scope='新候选相对旧候选；同输入的按核统计，不是端到端关键路径或独立性能复验'))
    derived={}
    og,ng=old_kernel.get('granularity',{}),new_kernel.get('granularity',{})
    for name in sorted(set(og)|set(ng)):
        a=og.get(name,{}).get('bytes_per_instruction');b=ng.get(name,{}).get('bytes_per_instruction')
        derived[name]=dict(reference=a,candidate=b,absolute_delta=b-a if a is not None and b is not None else None,
                           candidate_over_reference=b/a if a and b is not None else None,unit='B/instruction')
    result=compact(dict(spec=dict(operator=new['operator'],case=new['case'],contract_sha256=new['contract_sha256']),
                        differences=differences,derived_differences=derived),limit=16)
    result.update(reference_kind='previous_candidate',reference_kernel=old_kernel['kernel'],candidate_kernel=new_kernel['kernel'],
                  limitations=['比率为新候选÷旧候选；性能成绩另取完整KGS结果',
                               '按核时间的均值不可直接作为整个任务耗时；流量沿用厂商CSV接口定义',
                               '静态IR和指标变化用于验证机制假设，不能仅凭相关性确立因果'])
    return result


def ir_signatures(context):
    ir=context.get('actual_ir',{})
    if not ir.get('entries'):return dict(status='unavailable')
    if ir.get('spec')!=context['spec']:raise ValueError('实际IR与当前采集输入身份不同')
    rows=[]
    for entry in ir['entries']:
        for artifact in entry.get('artifacts',[]):
            if artifact['kind'] not in {'ttir','ttadapter','bcmlir'}:continue
            path=Path(artifact['path']);raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=artifact['sha256']:
                raise ValueError('实际IR内容SHA改变：'+str(path))
            text=raw.decode('utf8');lines=text.splitlines()
            rows.append(dict(symbol=entry['symbol'],kind=artifact['kind'],path=str(path),sha256=artifact['sha256'],
                static_counts=dict(discrete_access_markers=text.count('DiscreteMemAccess'),
                    extracted_load_store_markers=text.count('ExtractedLoadOrStore'),
                    scalar_memref_load_ops=len(re.findall(r'\bmemref\.load\b',text)),
                    single_element_memref_type_occurrences=text.count('memref<1x'),
                    for_loops=len(re.findall(r'\bscf\.for\b',text))),
                tensor_shapes=sorted(set(re.findall(r'tensor<([0-9]+(?:x[0-9]+)*)x(?:f|i|bf)[0-9]+',text))),
                notable_lines=[dict(line=i+1,text=line.strip()[:500]) for i,line in enumerate(lines)
                               if any(marker in line for marker in ['DiscreteMemAccess','ExtractedLoadOrStore','memref.load'])][:16]))
    return dict(status='completed',artifacts=rows,
                scope='静态实际编译IR信号，不是动态指令计数或耗时；各编译阶段分别展示，不相加')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',type=Path,required=True);parser.add_argument('--new',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    old,om,oi=load_context(args.old);new,nm,ni=load_context(args.new)
    result=compare(old,new,om,nm);result['provenance']=dict(old=oi,new=ni)
    result['actual_ir_signatures']=dict(old=ir_signatures(old),new=ir_signatures(new))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status='completed',output=str(args.output),metrics=result['all_metric_count'],
                          selected=len(result['selected_metrics']))))


if __name__=='__main__':main()
