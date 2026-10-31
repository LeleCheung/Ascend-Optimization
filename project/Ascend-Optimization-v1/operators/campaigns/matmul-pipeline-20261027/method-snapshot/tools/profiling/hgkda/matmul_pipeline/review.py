"""CC＋DeepSeek读取冻结证据，选择有限追加配置；保留原回执。"""
import argparse,hashlib,importlib.util,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]
sys.path.insert(0,str(PROJECT/'tools/profiling'))
from deepseek_request_bridge import DeepSeekRequestBridge

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();b=a.batch.resolve();out=b/'agent-review';out.mkdir()
    parent=PROJECT/'operators/campaigns/roofline-20261027/validation/delivered.py'
    summary=PROJECT/'operators/campaigns/roofline-20261027/final-roofline-summary.json'
    template=HERE/'templates.py';search=b/'search-v2/state.json'
    choices=['dual-n-group4','dual-n-persistent20','dual-n-stage1']
    prompt=('你是Ascend910B性能分析角色。不要调用工具。当前目标是实际加速matmul_bias_activation，'
        '保留多输出小tile，避免上一轮单输出变慢。请根据已有Roofline、源码IR事实与本轮真实搜索记录，'
        '选择最多两项尚未测的追加配置。只输出JSON：config_ids（不重复数组）、hypothesis（中文）、'
        'predictions（指标与预期方向数组）、limitations（数组）。config_ids只能来自'+json.dumps(choices)+'。'
        '不可把limiting_side=compute直接解释为实际计算受限，不可把流水活跃比例相加或等同整卡利用率。'
        'GM、L2事件模型、逻辑字节与算法量区分。既有UB超限与单输出回归不得重新包装为收益。'
        '本轮候选仅半精度快路径，FP32回父版。依据大形状交错真实设备计时选版，一次完整KGS裁决。\n\n'
        'Roofline冻结证据：\n'+summary.read_text(encoding='utf8')+'\n\n'
        '父版源码：\n'+parent.read_text(encoding='utf8')+'\n\n'
        '通用候选空间与模板：\n'+template.read_text(encoding='utf8')+'\n\n'
        '本轮实际已测结果（可能仍在筛选，未完成项目不推断收益）：\n'+search.read_text(encoding='utf8')+'\n\n'
        'IR已核对：父版FP16存在两只128x128xf32累加器，A的128x256面板显式memref.copy一次，'
        'B两只256x128面板分别copy，随后两个linalg.matmul；文本次数不等于动态指令数。')
    (out/'prompt.md').write_text(prompt,encoding='utf8')
    frozen={name:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in [('parent',parent),('roofline',summary),('template',template),('search',search)]}
    (out/'search-snapshot.json').write_bytes(search.read_bytes())
    (out/'input-manifest.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    spec=importlib.util.spec_from_file_location('pipeline_campaign',PROJECT/'tools/profiling/optimization_campaign.py');campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)
    env=os.environ.copy();env.update(MAX_THINKING_TOKENS='0',CLAUDE_CODE_MAX_OUTPUT_TOKENS='4000',CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING='1')
    with DeepSeekRequestBridge(out/'provider.json') as bridge:
        settings=dict(disableAllHooks=True,alwaysThinkingEnabled=False,env={'ANTHROPIC_BASE_URL':bridge.base_url})
        command=['/data/hanle/ascend-optimization/runtime/kg-controller/claude','-p','--model','deepseek-flash',
            '--output-format','stream-json','--verbose','--max-turns','1','--tools','','--strict-mcp-config',
            '--setting-sources','','--settings',json.dumps(settings),'--effort','low','--no-session-persistence']
        campaign.run(command,out/'response.jsonl',prompt=prompt,timeout=900,cwd=out,env=env)
    events=[]
    for line in (out/'response.jsonl').read_text(encoding='utf8').splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:pass
    results=[r for r in events if r.get('type')=='result' and not r.get('is_error')]
    if len(results)!=1:raise ValueError('模型回执不唯一')
    text=results[0]['result'].strip()
    if text.startswith(chr(96)*3):text='\n'.join(text.splitlines()[1:-1])
    proposal=json.loads(text);ids=proposal['config_ids']
    if not isinstance(ids,list) or not 1<=len(ids)<=2 or len(set(ids))!=len(ids) or any(x not in choices for x in ids):raise ValueError('追加配置不符合预算')
    for key in ['hypothesis','predictions','limitations']:
        if not proposal.get(key):raise ValueError('缺少诊断字段')
    (out/'proposal.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(proposal,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
