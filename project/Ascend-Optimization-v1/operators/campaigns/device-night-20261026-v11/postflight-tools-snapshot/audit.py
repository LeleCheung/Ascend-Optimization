"""离线核验设备机制冻结、源码谱系、模型来源、正式覆盖及独立检验。"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('device_audit_compare',PROJECT/'tools/evaluation/compare-operator-versions.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


def read(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(batch):
    plan=read(batch/'plan.json');state=read(batch/'state.json');frozen=read(batch/'frozen-method.json')
    if state['status'] not in ['completed','needs_attention']:raise ValueError('批次尚未终态')
    checks=[]
    def check(condition,label):
        if not condition:raise ValueError(label)
        checks.append(label)
    snapshot=batch/'system-snapshot'
    for name,digest in plan['method_sha256'].items():check(sha(batch/'method-snapshot'/name)==digest,'开发预登记方法/'+name)
    for name,digest in frozen['files'].items():check(sha(snapshot/name)==digest,'冻结文件/'+name)
    for name,digest in frozen['shared_dependencies'].items():check(sha(PROJECT/name)==digest,'共享依赖/'+name)
    sys.path.insert(0,str(snapshot))
    # 本轮快照在独立模块目录，必须使用冻结变换重新构建实际候选。
    for name in ['variants','final_variants','diagnose']:sys.modules.pop(name,None)
    spec=importlib.util.spec_from_file_location('device_audit_transform',snapshot/'final_variants.py')
    variant=importlib.util.module_from_spec(spec);spec.loader.exec_module(variant)
    selection=read(batch/'selection-frozen.json')
    check(selection['method_sha256']==sha(batch/'frozen-method.json'),'交付冻结方法身份')
    check(selection['selected']==state['selected'],'独立检验后未改选版')
    rows=[];generation=[];predictions=[];profiles=[]
    for record in plan['records']:
        op=record['operator'];seed=batch/record['seed']
        check(sha(seed)==record['source_sha256'],op+'/父源码')
        for split,contract in record['splits'].items():
            for name,digest in contract['sha256'].items():check(sha(batch/contract['path']/name)==digest,op+'/'+split+'/合同/'+name)
        root=batch/'results'/op
        semantic=read(root/'development/parent-semantics/check.json')
        check(semantic['status']=='passed' and semantic['passed']==semantic['total']==18,op+'/父版补充语义18项')
        check(semantic['source_sha256']==sha(seed) and semantic['checker_sha256']==frozen['files']['check.py'],op+'/父版语义身份')
        for arm in plan['arms']:
            items=[r for r in state['records'] if r['operator']==op and r['arm']==arm]
            check(len(items)==plan['candidates_per_arm'],op+'/'+arm+'/候选上限')
            passed=[]
            for item in items:
                directory=root/'development'/arm/('candidate-'+str(item['index']))
                if item['status']!='passed':
                    check((directory/'failure.json').exists(),op+'/'+arm+'/失败保留/'+str(item['index']))
                    generation.append(dict(operator=op,arm=arm,index=item['index'],status=item['status'],error=item.get('error')));continue
                source=directory/'candidate.py'
                check(sha(source)==item['source_sha256'],op+'/'+arm+'/源码/'+str(item['index']))
                expected=variant.transform(seed.read_text(encoding='utf8'),op,item['config']).encode()
                check(expected==source.read_bytes(),op+'/'+arm+'/冻结模板可重建/'+str(item['index']))
                proposal=read(directory/'proposal.json');check(proposal['config']==item['config'],op+'/'+arm+'/模型配置/'+str(item['index']))
                responses=sorted(directory.glob('response-*.jsonl'));check(1<=len(responses)<=2,op+'/'+arm+'/模型次数/'+str(item['index']))
                valid=[]
                for response in responses:
                    events=[]
                    for line in response.read_text(encoding='utf8').splitlines():
                        try:events.append(json.loads(line))
                        except json.JSONDecodeError:pass
                    for event in events:
                        if event.get('type')=='assistant':
                            check(not any(v.get('type')=='tool_use' for v in event.get('message',{}).get('content',[])),op+'/'+arm+'/无外部工具/'+str(len(checks)))
                        if event.get('type')=='result' and not event.get('is_error'):valid.append(event)
                    provider=read(directory/response.name.replace('response-','provider-').replace('.jsonl','.json'))
                    check(bool(provider['requests']) and all(r.get('http_status')==200 and r.get('forced_thinking_type')=='disabled' for r in provider['requests']),op+'/'+arm+'/真实模型请求/'+response.name)
                check(bool(valid),op+'/'+arm+'/模型有效回执/'+str(item['index']))
                text=valid[-1]['result'].strip()
                if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
                check(json.loads(text)==proposal,op+'/'+arm+'/模型原始建议身份/'+str(item['index']))
                gate=read(directory/'supplementary/check.json')
                check(gate['status']=='passed' and gate['passed']==gate['total']==18,op+'/'+arm+'/补充18项/'+str(item['index']))
                check(gate['source_sha256']==sha(source) and gate['seed_sha256']==sha(seed),op+'/'+arm+'/门禁身份/'+str(item['index']))
                check(gate['checker_sha256']==frozen['files']['check.py'],op+'/'+arm+'/冻结门禁/'+str(item['index']))
                data=comparison.load_result(directory/'formal/candidate-1.result.json')
                baseline=comparison.load_result(root/'development/parent/parent-1.result.json')
                compared=comparison.compare([('parent',baseline),(arm,data)])[1]
                check(data['correctness_count']==18 and len(data['timing'])==6,op+'/'+arm+'/开发完整24项/'+str(item['index']))
                check(data['source_sha256']==sha(source),op+'/'+arm+'/正式源码身份/'+str(item['index']))
                device=[]
                for path in gate['confirmations']:
                    a=path['timings']['parent_ms'];b=path['timings']['candidate_ms']
                    check(len(a)==len(b)==5 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in a+b),op+'/'+arm+'/配对有效/'+str(len(checks)))
                    ratio=statistics.median([x/y for x,y in zip(a,b)])
                    check(math.isclose(path['paired_speedup'],ratio),op+'/'+arm+'/设备比值可重算/'+str(len(checks)))
                    device.append(ratio)
                check(math.isclose(statistics.geometric_mean(device),gate['device_geomean']),op+'/'+arm+'/设备汇总可重算/'+str(item['index']))
                score=statistics.geometric_mean(r['latency_ms'] for r in data['timing'].values())
                check(math.isclose(score,item['development_latency_ms']),op+'/'+arm+'/实际延迟选版/'+str(item['index']))
                passed.append((score,source,item,gate))
                generation.append(dict(operator=op,arm=arm,index=item['index'],status='passed',relative_parent=compared['relative_master'],
                    device_geomean=gate['device_geomean'],model_usage=valid[-1].get('usage'),model_seconds=valid[-1].get('duration_ms',0)/1000,
                    total_seconds=item['seconds'],source_sha256=sha(source)))
            winner=min(passed,key=lambda item:item[0]);chosen=selection['selected'][op+'/'+arm]
            check(sha(winner[1])==chosen['source_sha256'],op+'/'+arm+'/选择最低实际延迟有效候选')
            for path in winner[3]['confirmations']:
                predictions.append(dict(operator=op,arm=arm,shape=path['shape'],dtype=path['dtype'],device_speedup=path['paired_speedup']))
            for split in ['development','heldout']:
                parent=comparison.load_result(root/split/'parent/parent-1.result.json')
                result_path=(winner[1].parent/'formal/candidate-1.result.json') if split=='development' else root/split/arm/(arm+'-1.result.json')
                result=comparison.load_result(result_path);compared=comparison.compare([('parent',parent),(arm,result)])[1]
                check(result['source_sha256']==chosen['source_sha256'],op+'/'+arm+'/'+split+'/冻结交付身份')
                check(result['correctness_count']==18 and len(result['timing'])==6,op+'/'+arm+'/'+split+'/完整覆盖')
                rows.append(dict(operator=op,arm=arm,split=split,relative_parent=compared['relative_master'],relative_pytorch=compared['relative_pytorch'],
                    reference_drift=compared['reference_drift_geomean'],source_sha256=chosen['source_sha256'],config=chosen['config']))
        for path in (root/'development').glob('*/facts.json'):
            if not any(token in path.parent.name for token in ['metrics','instruction']):continue
            facts=read(path);profiles.append(dict(operator=op,name=path.parent.name,status=facts.get('profile_status'),
                opprof_kernels=len(facts.get('opprof',[])),simulator_cores=len(facts.get('simulator_hotspots',[])),seconds=facts.get('seconds')))
    return dict(status='passed',assertions=len(checks),checks=checks,results=rows,generation=generation,predictions=predictions,profiles=profiles,
        original_contracts=state.get('legacy_contracts',[]),
        limits=['有限配置生成，不是任意内核合成','开发集交错设备确认与独立集完整调用分开','真机/模拟器耗时不互相替代','尚无定量Roofline、设备关键路径和RTL执行器'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('核验输出已存在')
    result=audit(a.batch_dir);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],assertions=result['assertions'],results=len(result['results'])),ensure_ascii=False))
