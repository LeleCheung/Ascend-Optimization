"""持卡锁的冻结机制实验：同证据对照、模型生成、语义门禁及KGS完整评测。"""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
TOOLS=HERE.parent
PROFILE=TOOLS.parent
PROJECT=PROFILE.parents[1]
sys.path.insert(0,str(PROFILE))
from core import diagnose
from deepseek_request_bridge import DeepSeekRequestBridge
spec=importlib.util.spec_from_file_location('mechanism_queue',TOOLS/'run-nightly.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)
spec=importlib.util.spec_from_file_location('mechanism_campaign',PROFILE/'optimization_campaign.py')
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt(seed,op,probe,diagnosis,structured):
    # 两组拥有完全相同的测量和接口知识，仅诊断结构与动作建议不同。
    compact=json.loads(json.dumps(probe))
    for case in compact['cases']:
        for launch in case.get('launches',[]):
            launch.pop('compiled_artifact_sha256',None)
            launch.pop('source_file',None)
    common=(HERE/'launch-interface.md').read_text(encoding='utf8')
    lines=[
        '你是Ascend 910B算子优化工程师。只根据提供的源码和测量，生成一个更快的完整单文件候选。',
        '输出JSON，仅含source（完整Python源码）、hypothesis（中文）、expected_effect（中文）。',
        '不得调用工具、读取文件或联网。保留def run(A)的原地返回别名和完整shape/dtype/stride语义。',
        '不能调用目标算子的PyTorch或FlagGems作为新捷径。原代码已有的非连续fallback可以原样保留。允许torch分配、设备上下文和stream接口。',
        '不能删用例、读取评测模式、缓存计算结果、缓存输入Tensor或复用旧输入指针。浮点运算和特殊值必须正确。',
        '缓存编译产物时每次重新绑定本次输入、当前stream及配置。使用当前后端实际接口；不新增CUDA专有选项。',
        '只允许一次候选机会。输出完整有效源码，不要Markdown，不要占位符。',
        '算子：'+op,'共享接口知识：\n'+common,
        '原始执行观测：\n'+json.dumps(compact,ensure_ascii=False),
    ]
    if structured:lines+=['自动机制诊断（待验证的原因与方向）：\n'+json.dumps(diagnosis,ensure_ascii=False)]
    else:lines+=['根据以上原始数据自行分析并选择改法；没有额外的机制图或自动诊断建议。']
    lines+=['固定FlagGems起点：\n'+seed]
    return '\n\n'.join(lines)


def generate(directory,prompt_text,source,model,claude):
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'prompt.md').write_text(prompt_text,encoding='utf8')
    env=os.environ.copy();env.update(MAX_THINKING_TOKENS='0',CLAUDE_CODE_MAX_OUTPUT_TOKENS='10000',
                                   CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING='1')
    for attempt in [1,2]:
        response=directory/('response-%d.jsonl'%attempt)
        if response.exists():raise ValueError('存在模型回执，不重复调用')
        with DeepSeekRequestBridge(directory/('provider-%d.json'%attempt)) as bridge:
            settings=dict(disableAllHooks=True,alwaysThinkingEnabled=False,env={'ANTHROPIC_BASE_URL':bridge.base_url})
            command=[claude,'-p','--model',model,'--output-format','stream-json','--verbose','--max-turns','1',
                     '--tools','','--strict-mcp-config','--setting-sources','','--settings',json.dumps(settings),
                     '--effort','low','--no-session-persistence']
            campaign.run(command,response,prompt=prompt_text,timeout=1200,cwd=directory,env=env)
        try:
            proposal=campaign.parse_candidate(response.read_text(encoding='utf8'),source)
            candidate=directory/'candidate.py';candidate.write_bytes(proposal['source'].encode())
            save(directory/'proposal.json',{k:v for k,v in proposal.items() if k!='source'})
            return candidate
        except (ValueError,KeyError,TypeError,SyntaxError) as e:
            save(directory/('format-error-%d.json'%attempt),dict(error=str(e)))
            if attempt==2:raise
            prompt_text+='\n上次格式校验失败：'+str(e)+'。仅修复输出格式与完整源码，不新增优化机会。'
            (directory/'repair-prompt.md').write_text(prompt_text,encoding='utf8')


def execute(batch,root):
    q=queue.Queue(batch,root,'tle_yy')
    plan=q.plan;out=q.batch
    # 实验入口、规则、探针和提示知识均按字节冻结；独立检验期间不可悄悄更新。
    for name,digest in plan['tools_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('方法文件改变，需另设版本：'+name)
    (out/'runtime-env.json').write_text(json.dumps(dict(host_python=sys.version,model=plan['model'],
        entry_sha256=sha(Path(__file__)),started=datetime.datetime.utcnow().isoformat()+'Z'),indent=2))
    state=dict(status='running',jobs={},records=[])
    save(out/'state.json',state)
    with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for record in plan['records']:
                op=record['operator'];source=out/record['seed'];operator_dir=out/'results'/op
                state['current']=op;save(out/'state.json',state)
                started=time.monotonic()
                baseline=q.evaluate(op,source,operator_dir/'FlagGems','baseline',1)
                baseline['seconds']=time.monotonic()-started
                save(operator_dir/'baseline-summary.json',baseline)
                shapes=[[1024],[1048576]] if record['role']=='development' else [[1537],[1572864]]
                q.idle()
                probe_dir=operator_dir/'seed-probe'
                args=[q.py,str(HERE/'probe.py'),'--operator',op,'--source',str(source),'--output',str(probe_dir),
                      '--shapes',json.dumps(shapes),'--trace']
                start=time.monotonic();q.run('mechanism-probe-'+op,shlex.join(args),budget=900)
                save(probe_dir/'cost.json',dict(seconds=time.monotonic()-start,scope='采集及进程启动，不是纯设备时间'))
                facts=json.loads((probe_dir/'probe.json').read_text())
                diagnoses=json.loads((probe_dir/'diagnosis.json').read_text())
                if not all(c['status']=='passed' for c in facts['cases']):raise ValueError('起点探针不完整：'+op)
                # 运行顺序预先固定并反转，减少方法组总在后面的系统漂移。
                arms=plan['arms'] if record['role']=='development' else list(reversed(plan['arms']))
                for arm in arms:
                    row=dict(operator=op,role=record['role'],arm=arm,status='running');state['records'].append(row)
                    directory=operator_dir/arm
                    state['current']=op+':'+arm;save(out/'state.json',state)
                    start=time.monotonic()
                    try:
                        text=prompt(source.read_text(encoding='utf8'),op,facts,diagnoses,arm=='mechanism')
                        candidate=generate(directory,text,source.read_text(encoding='utf8'),plan['model'],
                                           str(root/'runtime/kg-controller/claude'))
                        row['source_sha256']=sha(candidate)
                        q.idle()
                        check_path=directory/'semantics.json'
                        q.run('mechanism-check-'+op+'-'+arm,shlex.join([q.py,str(HERE/'check.py'),
                              '--operator',op,'--source',str(candidate),'--seed',str(source),'--output',str(check_path)]),1800)
                        semantics=json.loads(check_path.read_text())
                        if semantics['status']!='passed':raise ValueError('补充语义失败')
                        row['semantics']=dict(passed=semantics['passed'],total=semantics['total'])
                        row['formal']=q.evaluate(op,candidate,directory/'formal',arm,1)
                        row['status']='passed'
                    except Exception as e:
                        row.update(status='failed',error=type(e).__name__+': '+str(e))
                        save(directory/'failure.json',dict(error=row['error']))
                        q.idle()
                    row['seconds']=time.monotonic()-start
                    save(out/'state.json',state)
            state.update(status='completed',current=None)
        except BaseException as e:
            state.update(status='needs_attention',error=type(e).__name__+': '+str(e));raise
        finally:save(out/'state.json',state)
    summarize(out)


def summarize(out):
    state=json.loads((out/'state.json').read_text())
    rows=[]
    for item in state['records']:
        directory=out/'results'/item['operator']/item['arm']
        row=dict(item)
        if item['status']=='passed':
            # 不猜测KGS结果字段；通用正式比较器负责合同匹配与实际延迟。
            loaded,score=campaign.checked(directory/'formal'/(item['arm']+'-1.result.json'))
            base,_=campaign.checked(out/'results'/item['operator']/'FlagGems/baseline-1.result.json')
            campaign.comparison.compare([('parent',base),('candidate',loaded)])
            ratios=[base['timing'][k]['latency_ms']/v['latency_ms'] for k,v in loaded['timing'].items()]
            row.update(speedup_vs_flaggems=statistics.geometric_mean(ratios),speedup_vs_pytorch=score)
            check=json.loads((directory/'semantics.json').read_text())
            row['prediction_checks']=[dict(shape=c['shape'],seed_warm_jit_calls=c['seed_warm_jit_calls'],
                candidate_warm_jit_calls=c['candidate_warm_jit_calls'],
                paired_speedup=statistics.median([x/y for x,y in zip(c['timings']['seed_us'],c['timings']['candidate_us'])]))
                for c in check['path_interventions']]
        rows.append(row)
    save(out/'summary.json',dict(status=state['status'],records=rows,
        limits=['每组一次模型候选，不能推断统计显著优势','相同候选次数及探针，不宣称等实际墙钟',
                '仅调用机制图，不具备设备指令依赖重建','KG/KGS评测复用，不标为纯KG原生对照']))
    lines=['# 机制原型试验结果','','固定FlagGems起点；一次模型候选；正式KGS与调用路径探针分开。','',
           '| 算子 | 角色 | 反馈 | 状态 | 相对FlagGems | 相对PyTorch |','|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| %s | %s | %s | %s | %s | %s |'%(r['operator'],r['role'],r['arm'],r['status'],
            '%.3f×'%r['speedup_vs_flaggems'] if 'speedup_vs_flaggems' in r else '—',
            '%.3f×'%r['speedup_vs_pytorch'] if 'speedup_vs_pytorch' in r else '—'))
    (out/'结果简报.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',default='mechanism-pilot-20261026')
    p.add_argument('--root',type=Path,default=Path('/data/hanle/ascend-optimization'))
    p.add_argument('--summarize-only',action='store_true');a=p.parse_args()
    if a.summarize_only:summarize(a.root/'Ascend-Optimization/project/Ascend-Optimization-v1/operators/campaigns'/a.batch)
    else:execute(a.batch,a.root)
