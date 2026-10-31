"""核验机制试验身份、冻结方法、模型来源、正式合同与预测；离线，不调用模型或设备。"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import tarfile

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('mechanism_compare',PROJECT/'tools/evaluation/compare-operator-versions.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))


def audit(batch):
    plan=read(batch/'plan.json');state=read(batch/'state.json')
    assert state['status'] in ['completed','needs_attention'],'队列尚未终态'
    assertions=[]
    def check(condition,label):
        if not condition:raise ValueError(label)
        assertions.append(label)
    method=plan.get('method_sha256',plan.get('tools_sha256'))
    for name,digest in method.items():
        check(sha(batch/'method-snapshot'/name)==digest,'冻结方法/'+name)
    spec=importlib.util.spec_from_file_location('mechanism_frozen_core',batch/'method-snapshot/core.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    rows=[]
    for record in plan['records']:
        op=record['operator'];root=batch/'results'/op
        source=batch/record['seed'];leaf=batch/record['catalog_leaf']
        check(sha(source)==record['source_sha256'],op+'/起点')
        for name,digest in record['contract_sha256'].items():check(sha(leaf/name)==digest,op+'/合同/'+name)
        base_path=root/'FlagGems/baseline-1.result.json'
        if not base_path.exists() or read(base_path).get('status')!='PASSED':
            rows.append(dict(operator=op,role=record['role'],status='baseline_failed_or_not_run',
                             baseline_status=read(base_path).get('status') if base_path.exists() else 'not_run'))
            continue
        baseline=comparison.load_result(base_path)
        check(baseline['source_sha256']==record['source_sha256'],op+'/正式起点身份')
        probe=read(root/'seed-probe/probe.json');diagnosis=read(root/'seed-probe/diagnosis.json')
        check(probe['source_sha256']==record['source_sha256'],op+'/探针身份')
        check(core.diagnose(probe)==diagnosis,op+'/冻结规则可重算')
        check(probe['physical_card']==plan['physical_card'],op+'/设备')
        if not all(c['status']=='passed' for c in probe['cases']):
            rows.append(dict(operator=op,role=record['role'],status='probe_unsupported_or_failed',
                             failures=[c for c in probe['cases'] if c['status']!='passed']))
            continue
        for arm in plan['arms']:
            directory=root/arm
            item=next((r for r in state['records'] if r['operator']==op and r['arm']==arm),
                      dict(status='not_run'))
            row=dict(operator=op,role=record['role'],arm=arm,status=item['status'])
            # 失败的生成与候选同样登记，不能消失于结果表。
            if item['status']!='passed':
                if item['status']=='failed':check((directory/'failure.json').exists(),op+'/'+arm+'/失败保留')
                row['error']=item.get('error');rows.append(row);continue
            candidate=directory/'candidate.py'
            check(sha(candidate)==item['source_sha256'],op+'/'+arm+'/候选身份')
            semantics=read(directory/'semantics.json')
            check(semantics['source_sha256']==sha(candidate),op+'/'+arm+'/补充检查身份')
            check(semantics['status']=='passed' and semantics['passed']==semantics['total']==21,op+'/'+arm+'/补充语义')
            check(semantics['checker_sha256']==method['check.py'],op+'/'+arm+'/冻结检查器')
            result=comparison.load_result(directory/'formal'/(arm+'-1.result.json'))
            check(result['source_sha256']==sha(candidate),op+'/'+arm+'/正式源码身份')
            versions=comparison.compare([('FlagGems',baseline),(arm,result)])
            check(result['correctness_count']==42 and len(result['timing'])==9,op+'/'+arm+'/完整正式覆盖')
            responses=sorted(directory.glob('response-*.jsonl'))
            check(1<=len(responses)<=2,op+'/'+arm+'/模型调用次数')
            outputs=[];usage=[]
            for response in responses:
                events=[]
                for line in response.read_text(encoding='utf8').splitlines():
                    try:events.append(json.loads(line))
                    except json.JSONDecodeError:pass
                outputs.extend(e for e in events if e.get('type')=='result')
                for event in events:
                    if event.get('type')=='assistant':
                        for part in event.get('message',{}).get('content',[]):
                            check(part.get('type')!='tool_use',op+'/'+arm+'/无模型外部工具/'+str(len(assertions)))
                provider=read(directory/response.name.replace('response-','provider-').replace('.jsonl','.json'))
                check(all(r.get('forced_thinking_type')=='disabled' and r.get('http_status')==200 for r in provider['requests']),
                      op+'/'+arm+'/模型真实请求/'+response.name)
            valid=[v for v in outputs if not v.get('is_error')]
            check(bool(valid),op+'/'+arm+'/有效模型回执')
            last=valid[-1]
            proposal=last.get('structured_output')
            if proposal is None:
                text=last.get('result','')
                if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
                proposal=json.loads(text)
            if arm=='mechanism_rewrite':
                check(read(directory/'action.json')==proposal,op+'/'+arm+'/模型动作身份')
                if proposal['action']=='retain':check(candidate.read_bytes()==source.read_bytes(),op+'/'+arm+'/保留起点')
                else:
                    rewrite=read(candidate.with_suffix('.rewrite.json'))
                    check(rewrite['parent_source_sha256']==sha(source),op+'/'+arm+'/变换父源码')
                    check(rewrite['transformer_sha256']==method['rewrite-dispatch.py'],op+'/'+arm+'/冻结变换器')
                    check(rewrite['diagnosis_sha256']==sha(root/'seed-probe/diagnosis.json'),op+'/'+arm+'/变换依据')
                    check(rewrite['source_sha256']==sha(candidate),op+'/'+arm+'/变换源码身份')
                    rs=importlib.util.spec_from_file_location('frozen_rewrite',batch/'method-snapshot/rewrite-dispatch.py')
                    rw=importlib.util.module_from_spec(rs);rs.loader.exec_module(rw)
                    expected=source.read_text(encoding='utf8')+'\n'+rw.TEMPLATE.replace('__OPERATOR__',op)
                    check(expected.encode()==candidate.read_bytes(),op+'/'+arm+'/完整候选可重建且无手工补丁')
            else:check(proposal['source'].encode()==candidate.read_bytes(),op+'/'+arm+'/候选由模型原字节生成')
            prediction=[]
            for path in semantics['path_interventions']:
                w=path['timings']['seed_us'];c=path['timings']['candidate_us']
                check(len(w)==len(c)==9 and all(math.isfinite(v) and v>0 for v in w+c),op+'/'+arm+'/干预计时/'+str(path['shape']))
                prediction.append(dict(shape=path['shape'],jit_calls_before=path['seed_warm_jit_calls'],
                    jit_calls_after=path['candidate_warm_jit_calls'],
                    paired_speedup=statistics.median([a/b for a,b in zip(w,c)]),
                    dispatch_prediction_realized=path['seed_warm_jit_calls']>0 and path['candidate_warm_jit_calls']==0))
            row.update(relative_flaggems=versions[1]['relative_master'],relative_pytorch=versions[1]['relative_pytorch'],
                       formal_correctness=42,formal_timing=9,supplementary=21,prediction_checks=prediction,
                       source_sha256=sha(candidate),model_usage=last.get('usage'),model_seconds=last.get('duration_ms',0)/1000,
                       total_arm_seconds=item['seconds'])
            rows.append(row)
    index=batch/'raw-evidence/archive-index.json'
    if index.exists():
        metadata=read(index);archive=index.parent/metadata['archive']
        check(sha(archive)==metadata['sha256'],'原始profiler归档SHA')
        with tarfile.open(archive,'r:gz') as tar:
            for member in metadata['preserved_members']:
                content=tar.extractfile(member['path']).read()
                check(hashlib.sha256(content).hexdigest()==member['sha256'],'归档成员/'+member['path'])
    return dict(status='passed',assertions=len(assertions),checks=assertions,records=rows,
                limits=['调用机制原型；没有设备指令依赖重建','一次模型机会，不能证明统计显著优势',
                        '共享同探针与知识；不是纯KG原生组','历史优化经验构成先验，独立指本轮冻结后的算子检验'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('核验结果已存在，请使用新输出')
    result=audit(a.batch_dir)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=result['status'],assertions=result['assertions'],records=len(result['records'])),ensure_ascii=False))
