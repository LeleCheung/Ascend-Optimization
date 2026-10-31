"""让CC/DeepSeek实际消费KG角色中的Roofline摘要，选择一次追加实验。"""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
sys.path.insert(0,str(PROJECT/'tools/profiling'))
from deepseek_request_bridge import DeepSeekRequestBridge

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def main():
    p=argparse.ArgumentParser()
    for name in ['overlay','search','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--decision-only',action='store_true')
    p.add_argument('--root',type=Path,default=Path('/data/hanle/ascend-optimization'));a=p.parse_args()
    a.output.mkdir(parents=True)
    role=a.overlay/'kernelgen/.kernelgen/agents/kernel-profile-analyzer.md'
    evidence=a.overlay/'kernelgen/.kernelgen/skills/hgkda/roofline-evidence.json'
    history=json.loads(a.search.read_bytes())
    prompt=('你是固定910B栈的性能分析角色。不要调用工具。阅读以下已加载KG分析角色和本批数值证据，'
        '结合真实失败结果选择一项追加实验。只输出JSON：config_id、hypothesis（中文）、'
        'predictions（指标及预期方向数组）、limitations（数组）。'
        'config_id只能为single64x128-k256、single128x64-k256、single64x256-k256之一。'
        '三个选项均为单输出tile，K块256，启用multibuffer；不修改数学公式。'
        '历史128x256/256x128均UB超限，128x128单块可编译但比双输出父版慢。'
        '追加候选由相同模板生成，完整KGS仍是裁决；不要宣称选择即获得提升。\n\n'
        +role.read_text(encoding='utf8')+'\n\n本批真实搜索结果：\n'
        +json.dumps(history,ensure_ascii=False))
    if a.decision_only:
        prompt=('你是Ascend910B性能分析角色。不要调用工具，不生成候选。阅读以下已修正为同GM事件口径的数值证据，'
            '结合已测搜索结果给出最终裁决。只输出JSON：decision（retain_parent或adopt_candidate）、'
            'hypothesis（中文）、predictions（数组）、limitations（数组）。'
            '父版已完整42/42；现有单输出候选全部比父版慢，正式KGS结果单独归档。'
            '本次是证据修订后的角色消费验证，不代表重新跑了一轮完整KG。'
            '不要把原始活跃比直接叫峰值利用率，不把有效算法FLOPs和内部操作混淆。\n\n'
            +role.read_text(encoding='utf8')+'\n\n真实搜索结果：\n'+json.dumps(history,ensure_ascii=False))
    (a.output/'prompt.md').write_text(prompt,encoding='utf8')
    manifest=dict(role_sha256=hashlib.sha256(role.read_bytes()).hexdigest(),
        roofline_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        search_sha256=hashlib.sha256(a.search.read_bytes()).hexdigest(),
        budget='一次模型请求；只选配置，不允许调用工具；Codex提出模板与搜索空间')
    (a.output/'input-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    campaign=module(PROJECT/'tools/profiling/optimization_campaign.py','roof_review_campaign')
    env=os.environ.copy();env.update(MAX_THINKING_TOKENS='0',CLAUDE_CODE_MAX_OUTPUT_TOKENS='4000',CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING='1')
    with DeepSeekRequestBridge(a.output/'provider.json') as bridge:
        settings=dict(disableAllHooks=True,alwaysThinkingEnabled=False,env={'ANTHROPIC_BASE_URL':bridge.base_url})
        command=[str(a.root/'runtime/kg-controller/claude'),'-p','--model','deepseek-flash','--output-format','stream-json','--verbose',
            '--max-turns','1','--tools','','--strict-mcp-config','--setting-sources','','--settings',json.dumps(settings),
            '--effort','low','--no-session-persistence']
        campaign.run(command,a.output/'response.jsonl',prompt=prompt,timeout=900,cwd=a.output.resolve(),env=env)
    events=[]
    for line in (a.output/'response.jsonl').read_text(encoding='utf8').splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:pass
    results=[e for e in events if e.get('type')=='result' and not e.get('is_error')]
    if len(results)!=1:raise ValueError('无唯一有效模型输出')
    text=results[0]['result'].strip()
    if text.startswith('~~~') or text.startswith(chr(96)*3):text='\n'.join(text.splitlines()[1:-1])
    proposal=json.loads(text)
    if a.decision_only:
        if proposal.get('decision') not in ['retain_parent','adopt_candidate']:raise ValueError('无效裁决')
    elif proposal['config_id'] not in ['single64x128-k256','single128x64-k256','single64x256-k256']:raise ValueError('配置超出预算')
    for key in ['hypothesis','predictions','limitations']:
        if not proposal.get(key):raise ValueError('缺少诊断字段：'+key)
    (a.output/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(proposal,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
