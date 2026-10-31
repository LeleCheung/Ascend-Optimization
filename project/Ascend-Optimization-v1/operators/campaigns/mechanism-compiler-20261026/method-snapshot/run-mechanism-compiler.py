"""冻结后的机制→受约束代码变换与自由源码生成对照，宿主tmux串行运行。"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'mechanism'))
spec=importlib.util.spec_from_file_location('mechanism_runtime',HERE/'mechanism/run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def action(directory,op,facts,diagnosis,model,claude,common):
    text='\n\n'.join(['你是硬件证据驱动的算子优化分析角色。禁止调用任何工具，只根据冻结证据选择一项动作。',
        '仅输出JSON：action为cache_compiled_dispatch或retain；evidence_cases为case id数组；expected_effect为中文说明。',
        'cache_compiled_dispatch由通用变换器实现，只支持同输入原地单JIT路径，首调用执行一次；后续绑定本次指针和当前stream。',
        '该变换不改数学公式或分块，不宣称改善设备指令；没有分派干预支持时应retain。',
        '算子：'+op,'同组共享接口知识：'+common,'原始观测：'+json.dumps(facts,ensure_ascii=False),
        '自动诊断：'+json.dumps(diagnosis,ensure_ascii=False)])
    directory.mkdir(parents=True);(directory/'prompt.md').write_text(text,encoding='utf8')
    env=m.os.environ.copy();env.update(MAX_THINKING_TOKENS='0',CLAUDE_CODE_MAX_OUTPUT_TOKENS='10000',CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING='1')
    for attempt in [1,2]:
        response=directory/('response-%d.jsonl'%attempt)
        with m.DeepSeekRequestBridge(directory/('provider-%d.json'%attempt)) as bridge:
            settings=dict(disableAllHooks=True,alwaysThinkingEnabled=False,env={'ANTHROPIC_BASE_URL':bridge.base_url})
            command=[claude,'-p','--model',model,'--output-format','stream-json','--verbose','--max-turns','1',
                     '--tools','','--strict-mcp-config','--setting-sources','','--settings',json.dumps(settings),
                     '--effort','low','--no-session-persistence']
            m.campaign.run(command,response,prompt=text,timeout=1200,cwd=directory,env=env)
        try:
            entries=[]
            for line in response.read_text().splitlines():
                try:value=json.loads(line)
                except json.JSONDecodeError:continue
                if value.get('type')=='result':entries.append(value)
            if len(entries)!=1 or entries[0].get('is_error'):raise ValueError('无唯一有效模型回执')
            proposal=json.loads(entries[0]['result'].strip().removeprefix('```json').removesuffix('```').strip())
            if proposal['action'] not in ['cache_compiled_dispatch','retain']:raise ValueError('不支持的动作')
            allowed={d['case'] for d in diagnosis['decisions']}
            if not set(proposal['evidence_cases'])<=allowed:raise ValueError('引用了不存在的证据')
            m.save(directory/'action.json',proposal);return proposal
        except (KeyError,ValueError,TypeError) as error:
            m.save(directory/('format-error-%d.json'%attempt),dict(error=str(error)))
            if attempt==2:raise
            text+='\n格式错误：'+str(error)+'；只修正JSON与字段。'


def execute(batch,root):
    q=m.queue.Queue(batch,root,'tle_yy');out=q.batch;plan=q.plan;snapshot=out/'method-snapshot'
    for name,digest in plan['method_sha256'].items():
        if sha(snapshot/name)!=digest:raise ValueError('冻结方法改变：'+name)
    # 实际probe/check与变换均从本批快照运行；自由生成提示也绑定快照知识。
    m.HERE=snapshot
    state=dict(status='running',records=[],started_at=m.queue.now());m.save(out/'state.json',state)
    with (root/'runs/physical-card-7-campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for rec in plan['records']:
                op=rec['operator'];seed=out/rec['seed'];directory=out/'results'/op
                state['current']=op;m.save(out/'state.json',state)
                base=q.evaluate(op,seed,directory/'FlagGems','baseline',1)
                m.save(directory/'baseline-summary.json',base)
                shapes=[[1024],[1048576]] if rec['role']=='development' else [[2305],[1310720]]
                probe=directory/'seed-probe';q.idle()
                start=time.monotonic()
                q.run('compiler-probe-'+op,shlex.join([q.py,str(snapshot/'probe.py'),'--operator',op,'--source',str(seed),
                    '--output',str(probe),'--shapes',json.dumps(shapes),'--trace']),900)
                facts=json.loads((probe/'probe.json').read_text());diagnosis=json.loads((probe/'diagnosis.json').read_text())
                m.save(probe/'cost.json',dict(seconds=time.monotonic()-start))
                if not all(c['status']=='passed' for c in facts['cases']):raise ValueError('探针不完整')
                arms=plan['arms'] if rec['role']=='development' else list(reversed(plan['arms']))
                for arm in arms:
                    output=directory/arm;row=dict(operator=op,role=rec['role'],arm=arm,status='running')
                    state['records'].append(row);state['current']=op+':'+arm;m.save(out/'state.json',state)
                    start=time.monotonic()
                    try:
                        if arm=='raw_evidence':
                            text=m.prompt(seed.read_text(),op,facts,diagnosis,False)
                            candidate=m.generate(output,text,seed.read_text(),plan['model'],str(root/'runtime/kg-controller/claude'))
                        else:
                            proposal=action(output,op,facts,diagnosis,plan['model'],str(root/'runtime/kg-controller/claude'),
                                            (snapshot/'launch-interface.md').read_text())
                            candidate=output/'candidate.py'
                            if proposal['action']=='retain':candidate.write_bytes(seed.read_bytes())
                            else:
                                q.run('compiler-rewrite-'+op,shlex.join([q.py,str(snapshot/'rewrite-dispatch.py'),
                                    '--operator',op,'--source',str(seed),'--diagnosis',str(probe/'diagnosis.json'),
                                    '--output',str(candidate)]),120)
                        row['source_sha256']=sha(candidate);q.idle()
                        q.run('compiler-check-'+op+'-'+arm,shlex.join([q.py,str(snapshot/'check.py'),'--operator',op,
                            '--source',str(candidate),'--seed',str(seed),'--output',str(output/'semantics.json')]),1800)
                        gate=json.loads((output/'semantics.json').read_text())
                        if gate['status']!='passed':raise ValueError('补充语义失败')
                        row['semantics']=dict(passed=gate['passed'],total=gate['total'])
                        row['formal']=q.evaluate(op,candidate,output/'formal',arm,1)
                        row['status']='passed'
                    except Exception as e:
                        row.update(status='failed',error=type(e).__name__+': '+str(e));m.save(output/'failure.json',dict(error=row['error']));q.idle()
                    row['seconds']=time.monotonic()-start;m.save(out/'state.json',state)
            state.update(status='completed',current=None,finished_at=m.queue.now())
        except BaseException as e:state.update(status='needs_attention',error=str(e));raise
        finally:m.save(out/'state.json',state)
    m.summarize(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',default='mechanism-compiler-20261026')
    p.add_argument('--root',type=Path,default=Path('/data/hanle/ascend-optimization'));a=p.parse_args();execute(a.batch,a.root)
