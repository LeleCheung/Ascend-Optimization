"""Draw a paired-profiling companion to the simulator evidence diagram."""
from __future__ import annotations

import hashlib
import html
import json
import shutil
from pathlib import Path

from build_simulator_flow_draft_20261122 import DELIVERY, MAIN, text

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'development/reports/paired-profile-flow-draft-20261123'
MATERIAL = DELIVERY / 'material/06_成对Profiling'
STEM = '成对profiling-简图草案'


def card(key, x, y, w, h, lines, dark=False):
    fill, color = ('#133F6D', '#FFFFFF') if dark else ('#E7EFF7', '#133F6D')
    pieces = [f'<g class="node" data-node="{key}" tabindex="0" role="button" aria-label="{html.escape(DETAILS[key][0])}">',
              f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="17" fill="{fill}" stroke="{"#133F6D" if dark else "#CAD5E2"}" stroke-width="1.6"/>']
    for value,offset in lines:
        pieces.append(text(x+w/2,y+offset,value,26,color=color,weight=600))
    return '\n'.join(pieces+['</g>'])


DETAILS = {
    'candidate': ('当前实现', '待继续优化的算子版本。与参考实现在同设备、同一输入规格和相同指标组下采集。', 'runtime/collect_pair.py'),
    'reference': ('参考实现', '当前实验的参考实现，例如CANN。分别保存两侧全部相关kernel任务与指标，避免只挑某个子kernel比较。', 'runtime/collect_pair.py'),
    'collect': ('成对采集', 'msprof op分别运行当前实现与参考实现，采集OpBasicInfo、PipeUtilization、Memory等实际支持的统计表。', 'runtime/collect_pair.py'),
    'movement': ('数据搬运', '将搬运量与搬运粒度归入同一组：总字节数回答搬了多少，搬运指令数与平均粒度回答分多少次搬、每次搬多少。按对应GM、UB、L1等接口分别对比，保留总量、均值、极值和记录数；不把接口字节自动等同物理HBM字节。', 'compare.py / read_side / derived_differences'),
    'pipeline': ('流水活动', '回答Vector、Scalar、MTE及适用Cube流水忙了多久。逐block保留时间、活跃比例及其均值、极值和记录数；流水可能重叠，活跃比例不直接等于性能好坏。此页用数据搬运与流水活动两组组织讲述；profiler已有等待/冲突统计仍可用，模拟器再展开时间位置与依赖关系。', 'compare.py / differences'),
    'gap': ('差距标准化 / 方案预览', '逐指标将与CANN的差异大小映射到[0,1]。非负指标可用abs(当前-CANN)/max(当前,CANN)，两侧均0时为0。图中删除方向字样，明细仍保留两侧原值、Delta和比值。该归一化层尚未接入HG-KDA后端，当前代码输出原始统计与比值。', '现有：compare.py / 拟增加：差距标准化'),
    'agent': ('Analyst Agent / Coder Agent', 'Analyst Agent解释硬件证据、结合知识库形成具体改法；Coder Agent实现Triton、Host或IR修改，之后真机验证。图中使用职责名称表达分析和编码分工，不据此声称已实现独立的多Agent编排模块。', 'context.py / 既有分析与实现分工'),
    'validation': ('编译验证', '候选先编译，再核对正确性与性能，结果进入下一轮迭代；图上在Agent框下面单独放置淡蓝色小框。', '既有编译与评测流程'),
}


def build_svg():
    return '\n'.join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-labelledby="title desc">',
        '<title id="title">技术点1：成对profiling</title>',
        '<desc id="desc">msprof op成对采集当前与CANN参考实现。对比数据搬运与流水活动两组指标，再经差距标准化映射到[0,1]。Analyst Agent分析、Coder Agent实现修改，下面单独连接编译验证，再反馈迭代。</desc>',
        '<defs><marker id="arrow" markerWidth="13" markerHeight="13" refX="11" refY="6.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L12 6.5 L0 13 Z" fill="#606870"/></marker>',
        '<marker id="feedback" markerWidth="13" markerHeight="13" refX="11" refY="6.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L12 6.5 L0 13 Z" fill="#133F6D"/></marker></defs>',
        '<rect width="1600" height="900" fill="#FFFFFF"/>',
        '<g font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">',
        text(65,90,'技术点 1：成对 profiling',44,color='#14181C',weight=700,anchor='start'),
        '<rect x="550" y="150" width="680" height="580" rx="14" fill="#F3F4F5"/>',
        text(580,192,'两组指标对比',26,color='#606870',anchor='start'),
        '<g fill="none" stroke="#606870" stroke-width="2.8" stroke-linejoin="round">',
        '<path d="M240 325 H270 V452.5"/>',
        '<path d="M240 587.5 H270 V452.5"/>',
        '<path d="M270 452.5 H300" marker-end="url(#arrow)"/>',
        '<path d="M500 452.5 H530"/>',
        '<path d="M530 330 V590"/>',
        *[f'<path d="M530 {y} H580" marker-end="url(#arrow)"/>' for y in (330,590)],
        *[f'<path d="M900 {y} H935"/>' for y in (330,590)],
        '<path d="M935 330 V590"/>',
        '<path d="M935 452.5 H970" marker-end="url(#arrow)"/>',
        '<path d="M1200 452.5 H1270" marker-end="url(#arrow)"/>',
        '</g>',
        '<path d="M1395 505 V565" fill="none" stroke="#606870" stroke-width="2.8" marker-end="url(#arrow)"/>',
        '<path d="M1395 650 V790 H35 V325 H60" fill="none" stroke="#133F6D" stroke-width="2.4" stroke-dasharray="10 10" marker-end="url(#feedback)"/>',
        text(790,770,'迭代',26,weight=500),
        card('candidate',60,275,180,100,[('当前实现',60)]),
        card('reference',60,537.5,180,100,[('CANN 参考',60)]),
        card('collect',300,390,200,125,[('msprof op',51),('成对采集',91)]),
        card('movement',580,225,320,210,[('数据搬运',51),('总字节数 / 指令数',105),('平均搬运粒度',159)]),
        card('pipeline',580,510,320,160,[('流水活动',60),('时间 / 活跃比例',110)]),
        card('gap',970,390,230,125,[('差距标准化',51),('[0,1]',91)]),
        card('agent',1270,350,250,155,[('Analyst Agent',61),('Coder Agent',110)],True),
        card('validation',1270,565,250,85,[('编译验证',53)]),
        '<g class="impl-labels" visibility="hidden" aria-hidden="true">',
        text(400,365,'collect_pair.py',18,color='#606870',weight=450),
        text(1085,355,'差距标准化方案',18,color='#606870',weight=450),
        text(740,705,'compare.py',18,color='#606870',weight=450),
        '</g></g></svg>',
    ])


MERMAID = '''flowchart LR
    C[当前实现] --> P
    R[CANN参考] --> P["msprof op<br/>成对采集"]
    subgraph PAIR["两组指标对比"]
        V["数据搬运<br/>总字节数 / 指令数<br/>平均搬运粒度"]
        L["流水活动<br/>时间 / 活跃比例"]
        V --> D["差距标准化<br/>[0,1]"]
        L --> D
    end
    P --> V
    P --> L
    D --> A["Analyst Agent<br/>Coder Agent"]
    A --> VAL[编译验证]
    VAL -. 迭代 .-> C
    classDef default fill:#E7EFF7,stroke:#CAD5E2,color:#133F6D;
    classDef agent fill:#133F6D,stroke:#133F6D,color:#FFFFFF;
    class A agent;
    style PAIR fill:#F3F4F5,stroke:#F3F4F5;
'''


def main():
    before=hashlib.sha256(MAIN.read_bytes()).hexdigest()
    REPORT.mkdir(parents=True,exist_ok=True)
    MATERIAL.mkdir(parents=True,exist_ok=True)
    svg=build_svg()
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>成对 profiling / 简图草案</title>
<style>*{box-sizing:border-box}body{margin:0;background:#edf1f5;color:#14181c;font:17px/1.65 "Microsoft YaHei",sans-serif}main{max-width:1640px;margin:auto;padding:18px 20px 40px}nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}nav span{color:#606870;margin-right:auto}button,a{font:inherit;color:#133f6d;background:white;border:1px solid #cad5e2;border-radius:7px;padding:6px 16px;text-decoration:none;cursor:pointer}button[aria-pressed=true]{background:#133f6d;color:#fff}.slide{background:white;border-radius:10px;overflow:hidden}.slide svg{display:block;width:100%;height:auto}.node{cursor:pointer}.node:hover>rect,.node:focus-visible>rect{stroke:#377db5;stroke-width:3}.node:focus{outline:none}.detail{margin-top:18px;padding:18px 24px;border:1px solid #d7e0e9;border-radius:9px;background:white}.detail h2{font-size:21px;color:#133f6d;margin:0 0 8px}.detail p{margin:0}.detail code{color:#606870;font-size:14px}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #3b82c4;outline-offset:3px}.form{display:flex;gap:20px;flex-wrap:wrap;margin:18px 0}.form label{display:flex;flex-direction:column;gap:5px}.form input,.form select{font:inherit;border:1px solid #cad5e2;border-radius:6px;padding:6px 10px;max-width:100%}.form input{width:170px}.result{display:flex;gap:26px;flex-wrap:wrap;background:#e7eff7;border-radius:8px;padding:16px 20px;margin:14px 0}.result strong{font-size:26px;color:#133f6d}.fine{color:#606870;font-size:15px}.formula{font-family:Consolas,"Microsoft YaHei",sans-serif}@media print{nav,.detail{display:none}body{background:#fff}main{padding:0}}</style></head>
<body><main><nav><span>技术点 1 / 成对 profiling草图</span><button id="scripts" aria-pressed="false">显示脚本</button><a href="../03_模拟器/模拟器分析-简图草案.html">查看模拟器图</a><a href="成对profiling-简图草案.mmd" download>Mermaid源图</a></nav><section class="slide">__SVG__</section><section class="detail" aria-live="polite"><h2 id="detail-title">数据搬运 / 流水活动</h2><p id="body">按优化问题整理为数据搬运、流水活动两组；搬运量与粒度放在同一组内。现有代码输出原始统计和当前/CANN比值；图中的[0,1]差距标准化是待接入方案，下面可以查看它的定义与示例。</p><code id="file"></code></section>
<section class="detail" aria-labelledby="encoding-title"><h2 id="encoding-title">差距标准化 [0,1] / 方案预览</h2><p>各指标分别标准化，明细保留两侧原值、Delta和比值，不将两组压成一个分数。</p><div class="form"><label>示例<select id="preset"><option value="gqa">GQA实测 / GM→L1搬运量</option><option value="gap07">差异度0.7 / 数值示意</option><option value="equal">两侧相同 / 数值示意</option><option value="zero">两侧均为0 / 数值示意</option></select></label><label>当前值<input id="candidate" type="number" min="0" step="any" value="112"></label><label>CANN参考值<input id="reference" type="number" min="0" step="any" value="64.296875"></label></div>
<p id="example-source" class="fine"></p><div class="result" aria-live="polite"><span>差异度 <strong id="score"></strong></span><span id="delta"></span><span id="ratio"></span></div><p class="formula">d = |当前 − CANN| / max(当前, CANN)</p><p class="fine">仅用于非负、同单位、同统计范围的指标；两侧均0时d=0，缺失时记为未采得。d=0.7表示按此定义得到的差异大小，不代表70%性能损失或瓶颈概率。分数只表达差异大小；两侧原值用于判断哪个更高。MTE活跃比例更高不自动等于性能更差。</p></section>
</main><script>
const details=__DETAILS__;
function select(key){const d=details[key];document.getElementById('detail-title').textContent=d[0];document.getElementById('body').textContent=d[1];document.getElementById('file').textContent=d[2];}
document.querySelectorAll('[data-node]').forEach(node=>{node.addEventListener('click',()=>select(node.dataset.node));node.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(node.dataset.node);}});});
document.getElementById('scripts').addEventListener('click',e=>{const show=e.currentTarget.getAttribute('aria-pressed')!=='true';e.currentTarget.setAttribute('aria-pressed',String(show));e.currentTarget.textContent=show?'隐藏脚本':'显示脚本';document.querySelector('.impl-labels').setAttribute('visibility',show?'visible':'hidden');document.querySelector('.impl-labels').setAttribute('aria-hidden',String(!show));});
const presets={gqa:{c:112,r:64.296875,source:'实际数据：attempt-006修改前成对profiling。当前114688 KiB=112 MiB；CANN 65840 KiB=64.296875 MiB。数值来自comparison.json，归一化分数为本页提出的编码。'},gap07:{c:100,r:30,source:'数值示意 / 不是实验成绩。当前100、参考30，差异度0.7，当前/参考比值3.333。'},equal:{c:100,r:100,source:'数值示意 / 两侧相同。'},zero:{c:0,r:0,source:'数值示意 / 两侧都明确测得为0；与缺失不同。'}};
function encode(){const cText=document.getElementById('candidate').value,rText=document.getElementById('reference').value,c=Number(cText),r=Number(rText);const valid=cText!==''&&rText!==''&&Number.isFinite(c)&&Number.isFinite(r)&&c>=0&&r>=0;document.getElementById('score').textContent=valid?(Math.max(c,r)===0?0:Math.abs(c-r)/Math.max(c,r)).toFixed(3):'未采得';document.getElementById('delta').textContent=valid?'Delta '+(c-r).toFixed(3):'';document.getElementById('ratio').textContent=valid?'当前 / CANN '+(r>0?(c/r).toFixed(3)+'×':'未定义'):'';}
document.getElementById('preset').addEventListener('change',e=>{const item=presets[e.target.value];document.getElementById('candidate').value=item.c;document.getElementById('reference').value=item.r;document.getElementById('example-source').textContent=item.source;encode();});
['candidate','reference'].forEach(id=>document.getElementById(id).addEventListener('input',encode));document.getElementById('example-source').textContent=presets.gqa.source;encode();
</script></body></html>'''.replace('__SVG__',svg).replace('__DETAILS__',json.dumps(DETAILS,ensure_ascii=False))
    notes='''# 成对profiling配套草图

技术点1为成对profiling；技术点2为模拟器热点区域定位；技术点3为模拟器流水时序分析。两页均用正文26px、标题44px、可选脚本18px。

图上删除“同输入/同设备/同采集配置”和“瓶颈假设”方框。msprof op成对采集后，对比两组指标：数据搬运（总字节数/指令数/平均搬运粒度），流水活动（时间/活跃比例）。搬运量与粒度放进同一组，右侧Analyst Agent和Coder Agent下面单独连接淡蓝色编译验证小框，再进入迭代。

现有compare.py输出两侧原值、统计量、candidate_over_reference与derived_differences，还没有统一[0,1]归一化。图中差距标准化为方案，不修改现有研究后端或历史证据。

建议非负可比指标按d=abs(c-r)/max(c,r)编码差异大小，两侧均0时为0，缺失为未采得，明细保留raw_delta、candidate、reference、ratio、unit与aggregation，图中只写[0,1]。d=0.7不代表70%性能损失或瓶颈概率；对耗时/字节成本与流水活跃比例分别解释方向，不给一个万能好坏分数。每条具体指标分别编码，不将MTE时间、字节、比例混成单一0.7。

HTML提供真实GQA112MiB对64.296875MiB的差异编码，以及100对30得到0.7的示意。原始GQA数据来自development/reports/gqa-goal-20261121/gqa/iterations/attempt-006/paired-parent/comparison.json。归一化均为此草图新提出的计算，未声称历史Agent已使用。图中Analyst Agent与Coder Agent表示既有分析和实现职责，不额外声称独立多Agent编排。

等待/冲突从本页图中删除以突出主线，profiler仍可能提供对应汇总指标；模拟器增加的是具体时间位置、同步与重叠关系。采集入口collect_pair.py，比较compare.py，上下文与启发式假设context.py、diagnostics.py。Profiler比例形成排查方向，不能直接推出精确依赖；不同program数下的逐block均值保持各自范围。正式加速比使用独立真机评测。
'''
    for suffix,content in (('.svg',svg),('.html',page),('.mmd',MERMAID),('.md',notes)):
        path=REPORT/(STEM+suffix)
        path.write_text(content,encoding='utf-8')
        shutil.copy2(path,MATERIAL/path.name)
    assert hashlib.sha256(MAIN.read_bytes()).hexdigest()==before
    (REPORT/'validation.json').write_text(json.dumps({'status':'generated','main_ppt_preserved':True,'main_ppt_sha256':before,'diagram_nodes':len(DETAILS),'delivery':str(MATERIAL/(STEM+'.html'))},ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(MATERIAL/(STEM+'.html')))


if __name__=='__main__':
    main()
