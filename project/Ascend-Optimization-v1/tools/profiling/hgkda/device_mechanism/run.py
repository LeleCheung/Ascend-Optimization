"""夜间设备机制系统：冻结、同空间模型对照、预测确认、完整评测和独立检验。"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import statistics
import sys
import time

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3];PROFILING=PROJECT/'tools/profiling'
sys.path.insert(0,str(PROFILING));sys.path.insert(0,str(HERE))
from deepseek_request_bridge import DeepSeekRequestBridge
from variants import DEFAULTS,validate,AMIN,MATMUL
from final_variants import transform
from diagnose import diagnose,choose_acquisition
spec=importlib.util.spec_from_file_location('device_queue',HERE.parent/'run-nightly.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
spec=importlib.util.spec_from_file_location('device_campaign',PROFILING/'optimization_campaign.py')
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);queue.save(path,value)


def freeze(out):
    snapshot=out/'system-snapshot'
    if snapshot.exists():raise ValueError('系统冻结目录已存在；不重新生成模型请求')
    snapshot.mkdir()
    for path in HERE.iterdir():
        if path.suffix in ['.py','.md']:shutil.copyfile(path,snapshot/path.name)
    for name,source in [('launch-probe.py',HERE.parent/'mechanism/probe.py'),('core.py',HERE.parent/'mechanism/core.py')]:
        shutil.copyfile(source,snapshot/name)
    manifest=dict(frozen_at=queue.now(),files={p.name:sha(p) for p in snapshot.iterdir()},
        shared_dependencies={str(p.relative_to(PROJECT)):sha(p) for p in [HERE.parent/'run-nightly.py',PROFILING/'optimization_campaign.py',PROFILING/'deepseek_request_bridge.py']},
        policy='同配置空间、同模型、同起点、每组2次候选、最多1次格式修正；独立集不参与诊断/选版。',
        hardware_scope='设备kernel探针独立于KGS完整调用；RTL只定义接口，无RTL运行')
    save(out/'frozen-method.json',manifest);return snapshot


def model(directory,op,arm,facts,diagnosis,history,seed,root,model_name):
    directory.mkdir(parents=True)
    # 仅开发集证据进入提示；独立合同、输入与结果不提供给模型。
    compact=json.loads(json.dumps(facts))
    for case in compact['cases']:
        case.pop('parent_launches',None)
        for variant in case['variants']:variant.pop('launches',None)
    lines=['你是Ascend910B设备算子优化角色。不要调用工具；只从冻结的分块/调度空间选择一个新配置。',
        '输出JSON：config（所有配置字段）、hypothesis（中文机制假设）、predictions（待验证指标与变化数组）。',
        '配置将由相同的通用变换器生成完整候选，不修改原数学内核。两个组拥有相同的空间。',
        '矩阵配置：bm=32/64/128，bn=32/64/128，bk32=32/64/128/256，bk16=64/128/256/512，group=0/2/4/8，multi为bool。',
        '归约配置：bo=2/4/8/16/32，last_br=128/256/512/1024/2048，bc=64/128/256/512/1024，middle_br=4/8/16/32/64，native/multi为bool。',
        '仅输出当前算子的字段；native仅影响FP16，BF16/FP32仍为FP32中间值。mask由整除条件自动确定，不准去掉非整除边界。',
        '归约middle_br表示FP32/BF16中间归约片；FP16原生中间值的实际BR为2×middle_br。BO/last_br只影响2D末轴；BC/middle_br只影响3D中间轴。',
        '同组共两次候选，第二次须不同于此前配置。失败和回退不能当成更快设备路径。',
        '为避开历史单K迭代流水超时，正式矩阵新路径要求K至少是2个BK；其他情况走共享兼容修复的父路径。',
        '算子：'+op,'默认配置：'+json.dumps(DEFAULTS[op]),'原始开发集干预证据：'+json.dumps(compact,ensure_ascii=False),
        '同组先前候选：'+json.dumps(history,ensure_ascii=False)]
    if arm=='mechanism':lines+=['机制诊断与下一步补测：'+json.dumps(diagnosis,ensure_ascii=False)]
    # 同一份源码提供给两组，历史人工分析与建议不伪装成新系统自动发现。
    lines+=['最终通用路径模板（占位符由配置替换；新wrapper覆盖父run，未知布局回父路径）：\n'+(AMIN if op=='amin' else MATMUL),
        '父版完整源码（含历史优化，数学计算不得改变）：\n'+seed.read_text(encoding='utf8')]
    prompt='\n\n'.join(lines);(directory/'prompt.md').write_text(prompt,encoding='utf8')
    env=os.environ.copy();env.update(MAX_THINKING_TOKENS='0',CLAUDE_CODE_MAX_OUTPUT_TOKENS='4000',CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING='1')
    for attempt in [1,2]:
        response=directory/f'response-{attempt}.jsonl'
        with DeepSeekRequestBridge(directory/f'provider-{attempt}.json') as bridge:
            settings=dict(disableAllHooks=True,alwaysThinkingEnabled=False,env={'ANTHROPIC_BASE_URL':bridge.base_url})
            command=[str(root/'runtime/kg-controller/claude'),'-p','--model',model_name,'--output-format','stream-json','--verbose',
                '--max-turns','1','--tools','','--strict-mcp-config','--setting-sources','','--settings',json.dumps(settings),
                '--effort','low','--no-session-persistence']
            campaign.run(command,response,prompt=prompt,timeout=1200,cwd=directory,env=env)
        try:
            events=[]
            for line in response.read_text().splitlines():
                try:events.append(json.loads(line))
                except json.JSONDecodeError:pass
            results=[e for e in events if e.get('type')=='result' and not e.get('is_error')]
            if len(results)!=1:raise ValueError('没有唯一有效模型回执')
            text=results[0]['result'].strip()
            if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
            proposal=json.loads(text);validate(op,proposal['config'])
            if any(proposal['config']==r.get('config') for r in history):raise ValueError('配置重复')
            save(directory/'proposal.json',proposal);return proposal
        except (KeyError,ValueError,TypeError) as error:
            save(directory/f'format-error-{attempt}.json',dict(error=str(error)))
            if attempt==2:raise
            prompt+='\n只修正JSON/配置合法性错误：'+str(error)
            (directory/'repair-prompt.md').write_text(prompt,encoding='utf8')


def evaluation_score(result):
    data,_=campaign.checked(result)
    return statistics.geometric_mean(r['latency_ms'] for r in data['timing'].values())


def ir_facts(probe):
    output=[]
    pattern=re.compile(r'cast|convert|transpos|alloc|buffer|load|store|copy|dma|mmad|matmul|wait|set_flag|barrier',re.I)
    for case in probe['cases']:
        for item in [dict(name='parent',launches=case['parent_launches'])]+case['variants']:
            if item.get('status','passed')!='passed':continue
            for launch in item.get('launches',[]):
                summaries=[]
                for artifact in launch['artifacts']:
                    if artifact['path'].endswith('.bin'):continue
                    path=Path(artifact['path'])
                    if sha(path)!=artifact['sha256']:raise ValueError('IR附件身份变化')
                    lines=path.read_text(errors='replace').splitlines();selected=[line.strip() for line in lines if pattern.search(line)]
                    summaries.append(dict(kind=artifact['kind'],sha256=artifact['sha256'],matched_lines=len(selected),examples=selected[:16]))
                output.append(dict(case=case['id'],variant=item['name'],symbol=launch['symbol'],grid=launch['grid'],constants=launch.get('constants',{}),metadata=launch['metadata'],ir=summaries,
                    scope='不同IR表示分别展示；文本匹配次数不是动态指令数，不跨IR相加'))
    return output


def profile_one(q,snapshot,evaluation,out,level='metrics',small=False):
    data=json.loads(evaluation.read_text());rows=[r for r in data['per_workload'] if r['phase']=='timing']
    # 最小case用于受限模拟器；大case用于真机计数。选择规则提前登记。
    row=min(rows,key=lambda r:r['latency_ms']) if small else max(rows,key=lambda r:r['latency_ms'])
    q.idle()
    try:
        q.run('device-profile-'+out.name+'-'+str(len(str(out))),shlex.join([q.py,str(snapshot/'collect_profile.py'),
            '--evaluation',str(evaluation),'--case-id',row['uuid'],'--output',str(out),'--server',q.server,
            '--level',level,'--budget','300' if level=='instruction' else '180']),700)
    except Exception as error:
        save(out/'controller-error.json',dict(error=str(error)))
    q.idle()


def execute(batch,root):
    q=queue.Queue(batch,root,'tle_yy');out=q.batch
    if json.loads((out/'stage-state.json').read_text())['status']!='completed':raise ValueError('开发采集未完整完成')
    snapshot=freeze(out)
    state=dict(status='running',started_at=queue.now(),current='freeze',records=[],selected={})
    save(out/'state.json',state)
    with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for rec in q.plan['records']:
                op=rec['operator'];seed=out/rec['seed'];development=out/'results'/op/'development'
                facts=json.loads((development/'interventions/probe.json').read_text());diagnosis=diagnose(facts)
                facts['compiled_facts']=ir_facts(facts)
                save(development/'diagnosis.json',diagnosis)
                profile_one(q,snapshot,development/'parent/parent-1.result.json',development/'parent-metrics')
                metric_path=development/'parent-metrics/facts.json'
                if metric_path.exists():
                    facts['shared_parent_metrics']=json.loads(metric_path.read_text())
                    diagnosis['available_parent_metrics']=facts['shared_parent_metrics']
                decision=choose_acquisition(diagnosis,facts.get('shared_parent_metrics'))
                save(development/'acquisition-decision.json',decision);diagnosis['acquisition_decision']=decision
                if decision['next_probe']=='instruction':
                    profile_one(q,snapshot,development/'parent/parent-1.result.json',development/'parent-instruction',level='instruction',small=True)
                instruction_path=development/'parent-instruction/facts.json'
                if instruction_path.exists():facts['shared_parent_instruction']=json.loads(instruction_path.read_text())
                diagnosis['probe_sha256']=sha(development/'interventions/probe.json')
                save(development/'diagnosis.json',diagnosis)
                save(development/'analysis-input.json',facts)
                # 原始组与方法组获得相同新增真机数据；只有诊断结构不同。
                arms=q.plan['arms'] if op=='matmul_bias_activation' else list(reversed(q.plan['arms']))
                for arm in arms:
                    history=[];candidates=[]
                    for index in range(1,q.plan['candidates_per_arm']+1):
                        directory=development/arm/f'candidate-{index}'
                        state['current']=op+':'+arm+':'+str(index);save(out/'state.json',state)
                        row=dict(operator=op,arm=arm,index=index,status='running');state['records'].append(row);start=time.monotonic()
                        try:
                            proposal=model(directory,op,arm,facts,diagnosis,history,seed,root,q.plan['model'])
                            row['config']=proposal['config']
                            candidate=directory/'candidate.py';candidate.write_bytes(transform(seed.read_text(),op,row['config']).encode())
                            row['source_sha256']=sha(candidate)
                            q.idle();q.run('device-check-'+op+'-'+arm+'-'+str(index),shlex.join([q.py,str(snapshot/'check.py'),
                                '--operator',op,'--source',str(candidate),'--seed',str(seed),'--output',str(directory/'supplementary')]),3600)
                            gate=json.loads((directory/'supplementary/check.json').read_text())
                            row['semantic_cases']=gate['cases']
                            row['device_cases']=[{k:v for k,v in item.items() if k in ['shape','dtype','paired_speedup']} for item in gate['confirmations']]
                            if gate['status']!='passed':raise ValueError('补充检查失败')
                            row.update(supplementary=gate['passed'],device_geomean=gate['device_geomean'])
                            row['formal']=q.evaluate(op,candidate,directory/'formal','candidate',1)
                            row['development_latency_ms']=evaluation_score(directory/'formal/candidate-1.result.json')
                            row['status']='passed';candidates.append((row,candidate,directory))
                        except Exception as error:
                            row.update(status='failed',error=type(error).__name__+': '+str(error));save(directory/'failure.json',dict(error=row['error']))
                            q.idle()
                        row['seconds']=time.monotonic()-start;history.append(dict(row));save(out/'state.json',state)
                    if not candidates:raise ValueError('此组没有完整通过候选：'+op+'/'+arm)
                    # 开发集完整调用延迟选版，设备变化另外报告；不根据相对PyTorch分数择优。
                    winner=min(candidates,key=lambda item:item[0]['development_latency_ms'])
                    row,source,directory=winner
                    state['selected'][op+'/'+arm]=dict(source=str(source.relative_to(out)),source_sha256=sha(source),
                        config=row['config'],device_geomean=row['device_geomean'],development_latency_ms=row['development_latency_ms'])
                    save(out/'state.json',state)
                # 设备计数与模拟器只对冻结的机制组胜出源码做最终机制核验。
                winner=state['selected'][op+'/mechanism'];path=out/winner['source']
                profile_one(q,snapshot,path.parent/'formal/candidate-1.result.json',development/'winner-metrics')
                if instruction_path.exists() and json.loads(instruction_path.read_text()).get('simulator_hotspots'):
                    profile_one(q,snapshot,path.parent/'formal/candidate-1.result.json',development/'winner-instruction',level='instruction',small=True)
            # 在看到独立集结果前冻结两组交付身份；此后不再生成或调参。
            save(out/'selection-frozen.json',dict(frozen_at=queue.now(),selected=state['selected'],method_sha256=sha(out/'frozen-method.json')))
            for rec in q.plan['records']:
                op=rec['operator'];state['current']=op+':heldout';save(out/'state.json',state)
                for name,digest in rec['splits']['heldout']['sha256'].items():
                    if sha(out/rec['splits']['heldout']['path']/name)!=digest:raise ValueError('独立合同改变')
                leaf=q.records[op]['catalog_leaf'];q.records[op]['catalog_leaf']=rec['splits']['heldout']['path']
                try:
                    directory=out/'results'/op/'heldout'
                    q.evaluate(op,out/rec['seed'],directory/'parent','parent',1)
                    for arm in list(reversed(q.plan['arms'])):
                        chosen=state['selected'][op+'/'+arm];source=out/chosen['source']
                        if sha(source)!=chosen['source_sha256']:raise ValueError('冻结候选改变')
                        q.evaluate(op,source,directory/arm,arm,1)
                finally:q.records[op]['catalog_leaf']=leaf
            # 新形状检验后，按原项目FlagGems合同补充完整覆盖，不据此再次调参。
            state['legacy_contracts']=[]
            for rec in q.plan['records']:
                op=rec['operator']
                for version,source in [('parent',out/rec['seed'])]+[(arm,out/state['selected'][op+'/'+arm]['source']) for arm in q.plan['arms']]:
                    state['current']=op+':original-contract:'+version;save(out/'state.json',state)
                    directory=out/'results'/op/'original-contract'/version
                    record=dict(operator=op,version=version,source_sha256=sha(source),status='running')
                    state['legacy_contracts'].append(record)
                    try:
                        q.idle();q.run('device-original-'+op+'-'+version,shlex.join([q.py,str(PROJECT/'tools/evaluation/evaluate-operator.py'),
                            '--operator',op,'--source',str(source),'--output',str(directory),'--label','original',
                            '--server',q.server,'--repeats','1','--timing-scope','device_kernel']),4200)
                        result=json.loads((directory/'original-1.result.json').read_text())
                        record.update(status='passed',workloads=result['num_workloads'])
                    except Exception as error:record.update(status='failed',error=str(error));q.idle()
                    save(out/'state.json',state)
            state.update(status='completed',current=None,finished_at=queue.now())
        except BaseException as error:state.update(status='needs_attention',error=str(error));raise
        finally:save(out/'state.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True)
    p.add_argument('--root',type=Path,default=Path('/data/hanle/ascend-optimization'));a=p.parse_args();execute(a.batch,a.root)
