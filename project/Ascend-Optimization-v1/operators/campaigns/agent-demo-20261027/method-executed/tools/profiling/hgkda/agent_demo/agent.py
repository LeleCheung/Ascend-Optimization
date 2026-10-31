"""模型先选择证据工具，再读取工具原件作有限优化决策；保留真实CC请求。"""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
sys.path.insert(0,str(PROJECT/'tools/profiling'))
from deepseek_request_bridge import DeepSeekRequestBridge

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def invoke(prompt,out):
    out.mkdir()
    (out/'prompt.md').write_text(prompt,encoding='utf8')
    s=importlib.util.spec_from_file_location('demo_campaign',PROJECT/'tools/profiling/optimization_campaign.py');campaign=importlib.util.module_from_spec(s);s.loader.exec_module(campaign)
    env=os.environ.copy();env.update(MAX_THINKING_TOKENS='0',CLAUDE_CODE_MAX_OUTPUT_TOKENS='5000',CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING='1')
    with DeepSeekRequestBridge(out/'provider.json') as bridge:
        settings=dict(disableAllHooks=True,alwaysThinkingEnabled=False,env={'ANTHROPIC_BASE_URL':bridge.base_url})
        command=['/data/hanle/ascend-optimization/runtime/kg-controller/claude','-p','--model','deepseek-flash','--output-format','stream-json','--verbose',
            '--max-turns','1','--tools','','--strict-mcp-config','--setting-sources','','--settings',json.dumps(settings),'--effort','low','--no-session-persistence']
        campaign.run(command,out/'response.jsonl',prompt=prompt,timeout=900,cwd=out,env=env)
    events=[]
    for line in (out/'response.jsonl').read_text(encoding='utf8').splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:pass
    result=[e for e in events if e.get('type')=='result' and not e.get('is_error')]
    if len(result)!=1:raise ValueError('CC有效回执不唯一')
    text=result[0]['result'].strip()
    if text.startswith(chr(96)*3):text='\n'.join(text.splitlines()[1:-1])
    proposal=json.loads(text);save(out/'decision.json',proposal);return proposal

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();b=a.batch.resolve();root=b/'agent'
    assert not root.exists();root.mkdir()
    evidence=json.loads((b/'agent-input.json').read_bytes())
    compact=[]
    for pair in evidence['pairs']:
        compact.append(dict(size=pair['size'],records={side:dict(duration_us=v['duration_us'],kernel=v['basic']['Op Name'],pipeline=v['pipeline'],
            roofline=v['roofline']) for side,v in pair['records'].items()}))
    tools=['read_source','read_ir','read_hotspots','read_roofline','read_paired_metrics','read_knowledge']
    first=('你是HG-KDA的Ascend910B性能诊断角色。当前只有一个算子matmul_bias_activation，起点已含历史优化。'
        '先选择需要进一步检查的证据工具，随后控制器会实际读取原件并返回。仅输出JSON：tools（从允许项选2至6个不重复字符串）、'
        'questions（每个工具要回答的问题）、hypotheses（尚待验证假设）。不要调用其他工具，不输出优化源码。'
        '允许工具：'+json.dumps(tools)+'。PyTorch指标是GEMM组件，候选融合bias/ReLU，不能据此给出完整算子速度。'
        'Roofline为经验参考，不是物理峰值；流水比例不可相加。\n当前真机摘要：\n'+json.dumps(compact,ensure_ascii=False)+
        '\n模拟器状态：'+json.dumps({k:v for k,v in (evidence.get('instruction') or {}).items() if k!='hotspots'},ensure_ascii=False))
    route=invoke(first,root/'route')
    selected=route.get('tools')
    if not isinstance(selected,list) or not 2<=len(selected)<=6 or len(set(selected))!=len(selected) or any(t not in tools for t in selected):raise ValueError('工具选择超出合同')
    outputs=[]
    for tool in selected:
        if tool=='read_source':
            path=b/'seed.py';value=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),text=path.read_text(encoding='utf8'))
        elif tool=='read_ir':
            value=evidence['ir']
        elif tool=='read_hotspots':value=evidence['instruction']
        elif tool=='read_roofline':value=[dict(size=q['size'],sides={s:r['roofline'] for s,r in q['records'].items()}) for q in evidence['pairs']]
        elif tool=='read_paired_metrics':value=[dict(size=q['size'],differences=q['differences'],derived=q['derived']) for q in evidence['pairs']]
        else:
            path=Path(evidence['knowledge']['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==evidence['knowledge']['sha256']
            value=dict(path=str(path),text=path.read_text(encoding='utf8'),prior='此前真实优化总结，非本批自动发现')
        record=dict(tool=tool,result=value);save(root/'tool-results'/(tool+'.json'),record);outputs.append(record)
    sys.path.insert(0,str(HERE.parent/'matmul_pipeline'));from templates import CONFIGS
    choices=[c for c in CONFIGS if c['id'] in ['dual-n-constexpr','dual-m-reuse-b','dual-n-saving','dual-n-interleave']]
    second=('根据刚才主动选择的工具结果，提出本批有限优化实验。只输出JSON：config_ids（选择1至2个配置，不重复）、'
        'diagnosis（中文具体原因）、evidence（具体文件/指标/IR行号/指令；证据不支持则说待验证）、'
        'predictions（修改后预期可测变化，数组）、limitations（数组）。'
        '配置为控制器已有通用变换，Codex实现源码模板，模型负责选动作，不声称自主发明模板。'
        '允许使用历史成功方法，本批作为新采集与方法复现，不假装未知历史。'
        '必须区分模拟器热点时间、实际设备时间、静态IR、真实搬运；不要由比例武断断言唯一瓶颈。'
        '只允许这些配置：'+json.dumps(choices,ensure_ascii=False)+'\n工具路由：'+json.dumps(route,ensure_ascii=False)+
        '\n实际工具返回：\n'+json.dumps(outputs,ensure_ascii=False))
    proposal=invoke(second,root/'plan');ids=proposal.get('config_ids');allowed={c['id'] for c in choices}
    if not isinstance(ids,list) or not 1<=len(ids)<=2 or len(set(ids))!=len(ids) or any(i not in allowed for i in ids):raise ValueError('候选选择超预算')
    for key in ['diagnosis','evidence','predictions','limitations']:
        if not proposal.get(key):raise ValueError('诊断字段缺失：'+key)
    save(root/'input-manifest.json',dict(evidence_sha256=hashlib.sha256((b/'agent-input.json').read_bytes()).hexdigest(),
        source_sha256=evidence['source_sha256'],tools=selected,tool_results=[dict(path=str(root/'tool-results'/(t+'.json')),sha256=hashlib.sha256((root/'tool-results'/(t+'.json')).read_bytes()).hexdigest()) for t in selected]))
    save(root/'proposal.json',proposal);print(json.dumps(proposal,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
