"""Build a standalone, interactive simulator-flow proposal without editing PPT."""
from __future__ import annotations

import hashlib
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'development/reports/simulator-flow-draft-20261122'
DELIVERY = Path('E:/0_XSAI/00_KernelAgent/0_HGKDA_docs/BAAI-reports/06_work')
MATERIAL = DELIVERY / 'material/03_模拟器'
STEM = '模拟器分析-简图草案'
MAIN = DELIVERY / '智源20261124汇报.pptx'

DETAILS = {
    'model': {
        'title': '左侧合成一个仿真入口',
        'body': '“当前算子仿真”把当前算子与运行入口收进同一个框。CANN CA model执行编译后的NPU二进制，运行入口提供输入数据、启动参数和运行时上下文。修改Triton或IR后，都要重新编译并继续重放；图上回路缩写为“迭代”。',
        'files': ['hgkda/src/hgkda/runtime/collect_simulator.py', 'hgkda/src/hgkda/simulator.py'],
    },
    'inst': {
        'title': 'Inst duration：指令耗时',
        'body': '图上采用“Inst duration”，直接说明这项输入是指令耗时。既有CSV含PC、指令、流水、调用次数、cycles与running_time(us)。simulator.py将它与trace中的源码位置和内联栈关联，形成逐核指令清单。',
        'files': ['hgkda/src/hgkda/simulator.py:75'],
    },
    'topk': {
        'title': '统计Top-K指令',
        'body': 'hotspots.py在候选kernel的各硬件核内，以running_time_us为主、cycles为次排序，返回Top-K。排名回答“先关注哪些指令”；不同流水可能重叠，累计指令时间不能相加为真机总耗时，也不直接等于关键路径。',
        'files': ['hgkda/src/hgkda/hotspots.py:4'],
    },
    'mapping': {
        'title': '编译调用栈 / AST',
        'body': '这里的编译调用栈指编译器生成的源码位置和内联源码调用栈，用于把指令PC对应到源码行；AST再提供函数、循环和条件的范围。它与Top-K指令作为两项输入，共同支持热点区域定位。对应IR操作继续结合编译产物的位置信息核对。',
        'files': ['hgkda/src/hgkda/simulator.py:75', 'hgkda/src/hgkda/regions.py:81'],
    },
    'region': {
        'title': '热点区域定位',
        'body': '指令PC先通过编译器源码位置与内联调用栈定位源码行，regions.py再用AST归入函数、循环或条件区域。对应IR操作需结合编译产物的loc与数据流核对。这里画成一条简洁的代码关联路径；实际区域聚合也会使用完整指令清单，不只处理Top-K。',
        'files': ['hgkda/src/hgkda/simulator.py:75', 'hgkda/src/hgkda/regions.py:81', 'development/reports/gqa-meeting-traces-20261122/source-mapping-and-compilation.md'],
    },
    'trace': {
        'title': '执行trace：保留先后与重叠关系',
        'body': 'trace提供指令执行区间、流水与同步事件。它同时也是PC到源码位置信息的重要来源；上下两路是为了讲清分析目的，并不是互不相干的两套原始数据。',
        'files': ['hgkda/src/hgkda/simulator.py', 'hgkda/src/hgkda/timeline.py'],
    },
    'pipeline': {
        'title': '流水分析：等待、空泡与重叠',
        'body': 'regions.py对已观测窗口内的执行、WAIT/BAR与空闲区间做分析，timeline.py组织同核时间线，gqa_simulator_schedule.py进一步提取源码对应的逐次指令区间。图中“空泡”指可观察的空闲；原因还需要结合事件依赖、指令与源码/IR核对。',
        'files': ['hgkda/src/hgkda/regions.py:117', 'hgkda/src/hgkda/timeline.py:9', 'development/tools/research/gqa_simulator_schedule.py:10'],
    },
    'agent': {
        'title': 'Analyst Agent / Coder Agent',
        'body': 'Analyst Agent结合两路证据及优化知识库形成改法，Coder Agent实现Triton、Host或单算子IR修改，再用真机验收。名称表达分析与实现职责，不额外声称独立多Agent编排。知识库先放在讲述中，不再增加方框。仿真与真机结果反馈到下一轮；图上只保留“迭代”两个字。',
        'files': ['hgkda/src/hgkda/context.py'],
    },
    'validation': {
        'title': '编译验证',
        'body': '候选先编译，再做正确性和性能验证；修改后的Triton与IR继续进行模拟器重放，核对预期的指令或流水变化，并据结果进入下一轮。图上以独立淡蓝色小框概括。',
        'files': ['hgkda/src/hgkda/runtime/collect_simulator.py'],
    },
}


def text(x, y, value, size=28, *, color='#133F6D', weight=600, anchor='middle', cls=''):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{color}" class="{cls}">{html.escape(value)}</text>'


def card(key, x, y, w, h, lines, *, dark=False):
    body = [f'<g class="node" data-node="{key}" tabindex="0" role="button" aria-label="{html.escape(DETAILS[key]["title"])}">',
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="17" fill="{"#133F6D" if dark else "#E7EFF7"}" stroke="{"#133F6D" if dark else "#CAD5E2"}" stroke-width="1.6"/>']
    for value, offset, size, weight in lines:
        body.append(text(x+w/2, y+offset, value, size, color='#FFFFFF' if dark else '#133F6D', weight=weight))
    body.append('</g>')
    return '\n'.join(body)


def build_svg():
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-labelledby="diagram-title diagram-desc">',
         '<title id="diagram-title">模拟器分析：指令热点与流水时序</title>',
         '<desc id="diagram-desc">当前算子在CANN CA model仿真。技术点2使用Top-K指令与编译调用栈/AST共同定位热点区域；技术点3分析有效执行、等待、空泡与重叠。两路证据进入Agent分析修改及真机验证，并反馈迭代。</desc>',
         '<defs><marker id="arrow" markerWidth="13" markerHeight="13" refX="11" refY="6.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L12,6.5 L0,13 Z" fill="#606870"/></marker>',
         '<marker id="feedback-arrow" markerWidth="13" markerHeight="13" refX="11" refY="6.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L12,6.5 L0,13 Z" fill="#133F6D"/></marker></defs>',
         '<rect width="1600" height="900" fill="#FFFFFF"/>',
         '<g font-family="Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif">',
         text(65, 90, '模拟器分析', 44, color='#14181C', weight=700, anchor='start'),
         '<rect class="technical-panel" data-technique="1" x="365" y="155" width="820" height="330" rx="14" fill="#F3F4F5"/>',
         '<rect class="technical-panel" data-technique="2" x="365" y="520" width="820" height="205" rx="14" fill="#F3F4F5"/>',
         text(390, 200, '技术点 2：热点区域定位', 26, color='#606870', weight=600, anchor='start'),
         text(390, 560, '技术点 3：流水时序分析', 26, color='#606870', weight=600, anchor='start'),
         '<g fill="none" stroke="#606870" stroke-width="2.8" stroke-linejoin="round">',
         '<path d="M300 450 H345"/>',
         '<path d="M345 450 V397.5 H390" marker-end="url(#arrow)"/>',
         '<path d="M345 450 V642.5 H390" marker-end="url(#arrow)"/>',
         '<path d="M595 397.5 H670" marker-end="url(#arrow)"/>',
         '<path d="M880 397.5 H950" marker-end="url(#arrow)"/>',
         '<path d="M900 267 H915 V397.5"/>',
         '<path d="M580 642.5 H695" marker-end="url(#arrow)"/>',
         '<path d="M1160 397.5 H1210 V452.5"/>',
         '<path d="M1160 642.5 H1210 V452.5"/>',
         '<path d="M1210 452.5 H1270" marker-end="url(#arrow)"/>',
         '</g>',
         '<path d="M1395 505 V565" fill="none" stroke="#606870" stroke-width="2.8" marker-end="url(#arrow)"/>',
         '<path d="M1395 650 V790 H190 V515" fill="none" stroke="#133F6D" stroke-width="2.4" stroke-dasharray="10 10" marker-end="url(#feedback-arrow)"/>',
         text(790, 770, '迭代', 26, weight=500),
         card('model', 80,385,220,130,[('当前算子仿真',53,26,600),('CANN CA model',91,26,500)]),
         card('inst',390,345,205,105,[('Inst duration',63,26,600)]),
         card('topk',670,345,210,105,[('Top-K 指令',63,26,600)]),
         card('mapping',655,232,245,70,[('编译调用栈 / AST',44,26,600)]),
         card('region',950,345,210,105,[('热点区域',63,26,600)]),
         card('trace',390,590,190,105,[('执行 trace',63,26,600)]),
         card('pipeline',695,590,465,105,[('有效执行 / 等待 / 空泡 / 重叠',63,26,600)]),
         card('agent',1270,350,250,155,[('Analyst Agent',61,26,600),('Coder Agent',110,26,600)],dark=True),
         card('validation',1270,565,250,85,[('编译验证',53,26,600)]),
         '<g class="impl-labels" aria-hidden="true" visibility="hidden">',
         text(190,365,'collect_simulator.py',18,color='#606870',weight=450),
         text(775,331,'hotspots.py',18,color='#606870',weight=450),
         text(1055,331,'regions.py',18,color='#606870',weight=450),
         text(925,578,'timeline.py / regions.py',18,color='#606870',weight=450),
         '</g></g></svg>']
    return '\n'.join(s)


MERMAID = '''flowchart LR
    S["当前算子仿真<br/>CANN CA model"]
    subgraph TECH1["技术点 2：热点区域定位"]
        I["Inst duration"] --> K["Top-K指令"] --> H["热点区域"]
        C["编译调用栈 / AST"] --> H
    end
    subgraph TECH2["技术点 3：流水时序分析"]
        T["执行trace"] --> P["有效执行 / 等待 / 空泡 / 重叠"]
    end
    S --> I
    S --> T
    H --> A["Analyst Agent<br/>Coder Agent"]
    P --> A
    A --> V[编译验证]
    V -. 迭代 .-> S
    classDef default fill:#E7EFF7,stroke:#CAD5E2,color:#133F6D;
    classDef agent fill:#133F6D,stroke:#133F6D,color:#FFFFFF;
    class A agent;
    style TECH1 fill:#F3F4F5,stroke:#F3F4F5;
    style TECH2 fill:#F3F4F5,stroke:#F3F4F5;
'''

NOTES = '''# 模拟器分析：简图草案

只生成HTML、SVG、PNG和Mermaid供讨论，不修改主PPT。

## 第4页

建议先保留静态总图。它展示整体关系，讲述时按“输入 → 三路证据 → Agent修改 → 验证与输出”引导即可。如果需要动画，最多分两步：先显示主流程，再淡入反馈虚线；不逐个方框飞入。

## 第5页的取舍

- 左侧合并为“当前算子仿真 / CANN CA model”，保留算子入口含义。实际执行编译后的NPU二进制。
- 成对profiling是技术点1；模拟器两路顺延为技术点2和3。
- 上路用浅灰矩形包住，标题为“技术点 2：热点区域定位”。保留“Inst duration → Top-K指令 → 热点区域”，Top-K上方增加“编译调用栈 / AST”，两项输入汇合后定位热点区域。
- 下路用浅灰矩形包住，标题为“技术点 3：流水时序分析”。路径为“执行trace → 有效执行 / 等待 / 空泡 / 重叠”。
- 热点区域框只保留“热点区域”四字。图上分隔圆点统一替换为斜杠。
- 右侧深蓝框保留“Analyst Agent / Coder Agent”，下面单独连接淡蓝色“编译验证”小框，再进入反馈迭代。知识库作为分析背景，口头说明，不再加外部方框。
- 反馈回路只标“迭代”，讲述中说明修改后重新编译并重放，IR修改也继续重放。
- 脚本名和技术细节从主图移到HTML的可展开说明；可打开“显示脚本”查看小字。
- 图中正文、节点与技术点标题统一26px；页面标题44px，可选脚本说明统一18px。“Inst duration”直接表达指令耗时，不再拆成“报告”和“duration”两行。

## 口头讲法

这里我们把模拟器的信息分成两路来用。

上面这路，先看指令耗时，找出Top-K热点指令，再结合编译信息和源码AST，把它对应到具体的源码或IR区域。

下面这路，沿着执行时间线，看哪些流水在等、哪里有空泡，以及计算和搬运有没有重叠。

这两路证据交给Agent，结合知识库分析和修改。修改后继续仿真，再用真机验证收益，进入下一轮。

## 实现对应

- `simulator.py`：读取指令CSV与trace，按PC关联源码位置/内联栈。
- `hotspots.py`：候选kernel逐核Top-K指令排序。
- `regions.py`：源码AST区域聚合、执行/等待/空闲区间分析。
- `timeline.py`：同核观测窗口内的流水时间线组织。
- `gqa_simulator_schedule.py`：实际GQA源码对应指令的逐次执行区间提取。

AST负责源码结构归属；IR关联仍需编译位置信息和实际操作核对。图中上路是讲述顺序，完整区域分析并不限于Top-K。两路原始数据存在交叉，trace也提供源码位置信息。Top-K累计时间不直接等于关键路径；空泡为已观测窗口内的可观察空闲，原因另结合事件和代码分析。
'''


def main():
    before = hashlib.sha256(MAIN.read_bytes()).hexdigest()
    REPORT.mkdir(parents=True,exist_ok=True)
    MATERIAL.mkdir(parents=True,exist_ok=True)
    svg = build_svg()
    page = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>模拟器分析 / 简图草案</title>
<style>
:root{color-scheme:light;--navy:#133f6d;--muted:#606870;--border:#cad5e2}*{box-sizing:border-box}
body{margin:0;background:#edf1f5;color:#14181c;font:17px/1.65 "Microsoft YaHei",sans-serif}
main{max-width:1640px;margin:auto;padding:18px 20px 40px}nav{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:16px}
nav span{color:var(--muted);margin-right:auto}button,a.action{font:inherit;border:1px solid var(--border);background:#fff;color:var(--navy);border-radius:7px;padding:6px 16px;cursor:pointer;text-decoration:none}
button[aria-pressed=true]{background:var(--navy);color:#fff}button:focus-visible,a:focus-visible{outline:3px solid #3b82c4;outline-offset:3px}
.slide{background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 8px 30px #133f6d0a}.slide svg{width:100%;height:auto;display:block}
.node{cursor:pointer}.node:hover>rect,.node:focus-visible>rect{stroke:#377db5;stroke-width:3}.node:focus{outline:none}
.details{background:#fff;border:1px solid #d7e0e9;border-radius:9px;margin-top:18px;padding:18px 24px;min-height:120px}
.details h2{font-size:21px;color:var(--navy);margin:0 0 8px}.details p{margin:0}.files{display:flex;gap:6px 18px;flex-wrap:wrap;margin-top:12px;font:14px/1.8 Consolas,monospace;color:var(--muted);overflow-wrap:anywhere}.files span{max-width:100%}
details{margin-top:18px;background:#fff;border:1px solid #d7e0e9;border-radius:9px;padding:12px 24px}summary{cursor:pointer;color:var(--navy)}details p{margin:12px 0}
@media(max-width:700px){main{padding:10px}nav span{width:100%}.details{padding:14px}nav{font-size:14px}}
@media print{body{background:#fff}main{padding:0}nav,.details,details{display:none}.slide{box-shadow:none;border-radius:0}}
</style></head><body><main>
<nav><span>第5页草图 / 点选方框可查看对应实现</span><button id="scripts" aria-pressed="false">显示脚本</button><button id="clear">回到简图说明</button><a class="action" href="模拟器分析-简图草案.mmd" download>Mermaid源图</a><a class="action" href="模拟器分析-简图草案.svg" download>SVG源图</a></nav>
<section class="slide">__SVG__</section>
<section class="details" aria-live="polite"><h2 id="detail-title">两路证据，一个修改入口</h2><p id="detail-body">上路定位哪里最耗时，下路分析执行先后、等待与重叠。知识库先放在口头说明中，右侧合并分析、修改和真机验证。</p><div class="files" id="files"></div></section>
<details><summary>讲述与技术细节</summary><p>上路：指令PC → 编译器源码位置 / 内联栈 → 源码行 → AST区域；对应IR操作结合编译位置信息核对。图上用“热点区域”概括。</p><p>下路：读取执行区间、同步和搬运事件，分析已观测窗口内的等待、可观察空闲与重叠；完整因果判断还需要结合代码和事件依赖。</p><p>“迭代”包括修改后重新编译、模拟器重放和真机验证；所有IR修改也继续重放。图中两路是分析目的的划分，底层数据存在交叉。</p><p>第4页总图建议暂时保持静态；若加动画，最多分“主流程”和“反馈回路”两步。</p></details>
</main><script>
const details=__DETAILS__;
const title=document.getElementById('detail-title'),body=document.getElementById('detail-body'),files=document.getElementById('files');
function select(key){const item=details[key];title.textContent=item.title;body.textContent=item.body;files.replaceChildren(...item.files.map(value=>{const s=document.createElement('span');s.textContent=value;return s;}));}
document.querySelectorAll('[data-node]').forEach(node=>{node.addEventListener('click',()=>select(node.dataset.node));node.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(node.dataset.node);}});});
document.getElementById('scripts').addEventListener('click',e=>{const show=e.currentTarget.getAttribute('aria-pressed')!=='true';e.currentTarget.setAttribute('aria-pressed',String(show));e.currentTarget.textContent=show?'隐藏脚本':'显示脚本';document.querySelector('.impl-labels').setAttribute('visibility',show?'visible':'hidden');document.querySelector('.impl-labels').setAttribute('aria-hidden',String(!show));});
document.getElementById('clear').addEventListener('click',()=>{title.textContent='两路证据，一个修改入口';body.textContent='上路定位哪里最耗时，下路分析执行先后、等待与重叠。知识库先放在口头说明中，右侧合并分析、修改和真机验证。';files.replaceChildren();});
</script></body></html>'''.replace('__SVG__',svg).replace('__DETAILS__',json.dumps(DETAILS,ensure_ascii=False))
    for suffix,content in (('.svg',svg),('.html',page),('.mmd',MERMAID),('.md',NOTES)):
        path=REPORT/(STEM+suffix)
        path.write_text(content,encoding='utf-8')
        shutil.copy2(path,MATERIAL/path.name)
    assert hashlib.sha256(MAIN.read_bytes()).hexdigest()==before
    validation={'main_ppt_preserved':True,'main_ppt_sha256':before,'diagram_nodes':len(DETAILS),'standalone_html':True,'external_dependencies':False,'status':'generated','delivery':str(MATERIAL/(STEM+'.html'))}
    (REPORT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(validation,ensure_ascii=False))


if __name__=='__main__':
    main()
