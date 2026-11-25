"""从冻结 GQA 主线生成完整与重点案例汇报图，不改写研究原件。"""
import hashlib
import html
import json
from pathlib import Path

from .research_progress import milestones
from .selection import remeasurement_indices

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'development/reports/gqa-goal-20261121/milestone-over-cann-final/trace/mainline-presentation.json'
OUT = ROOT / 'development/reports/gqa-meeting-traces-20261122'
COLORS = {'paired': '#2368b1', 'sim': '#cc641e', 'knowledge': '#7852a6'}
LABELS = {'paired': '成对 Profiling', 'sim': '白盒模拟器', 'knowledge': '知识与代码推理'}
INK, MUTED = '#202e40', '#627185'
COMPACT_INDICES = [0, 6, 7, 11, 19, 22, 23]
PRIMARY_INDICES = [7, 11, 22]
METHOD_LABELS = {
    6: ['整块 KV', '单轮特化'],
    7: ['共享 KV', '持久复用'],
    11: ['QK 交接', 'FP16 缩窄'],
    19: ['下一块', 'QK 前瞻'],
    22: ['分母归约', '移入等待期'],
    23: ['下一块', '概率前瞻'],
}
STAGES = [
    (1, 6, '分块复用与 PV 简化', ['扩大 query / KV 分块', '两项 PV → 单次 PV；单轮特化']),
    (7, 10, 'KV 复用与计算精简', ['持久复用；常驻程序均分 tile', '公共偏移替代逐行 max 归约']),
    (11, 12, 'QK 交接缩窄与 P 前发', ['FP16 交接；恢复 FP32 softmax', '提前发送概率，推进 PV']),
    (13, 19, '路径精简与 QK 前瞻', ['host 缓存；BF16 PV 与值域简化', 'PV 交接缩窄；多缓冲与 QK 前瞻']),
    (20, 22, '倒数与接收／归约前移', ['原生倒数；复用缓冲提前接收 QK', 'sum / VREC 移入发送与等待期']),
    (23, 23, '下一块概率前瞻', ['调整临时缓冲，保持概率存活', '转换与 exp 重叠当前 PV 等待']),
]


def esc(s):
    return html.escape(str(s), quote=True)


def text(x, y, s, size=24, color=INK, weight=400, anchor='start'):
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{esc(s)}</text>'


def mark(x, y, layer, color, r=11):
    attrs = f'fill="{color}" stroke="white" stroke-width="2.5"'
    if layer == 'host':
        return f'<rect x="{x-r}" y="{y-r}" width="{2*r}" height="{2*r}" rx="1" {attrs}/>'
    if layer == 'ir':
        return f'<path d="M{x},{y-r*1.3} L{x+r*1.2},{y+r} L{x-r*1.2},{y+r} Z" {attrs}/>'
    if layer == 'mixed':
        return f'<path d="M{x},{y-r*1.35} L{x+r*1.35},{y} L{x},{y+r*1.35} L{x-r*1.35},{y} Z" {attrs}/>'
    return f'<circle cx="{x}" cy="{y}" r="{r}" {attrs}/>'


def node(e, x, y, r=11):
    tip = f"进展 {e['progress_index']:02d} · {e['label']}\n{e['latency_ms']*1000:.5f} μs · {e['score']:.6f}×\n{e.get('method','')}\n{e.get('attribution','')}"
    return f'<g class="node" tabindex="0" role="button" data-progress="{e["progress_index"]}" aria-label="{esc(tip)}"><title>{esc(tip)}</title>' + mark(x, y, e['location'], COLORS[e['display_category']], r) + '</g>'


def chart(nodes, compact=False, summary=None):
    summary = summary or {}
    bg, green = '#ffffff', '#168358'
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">',
           f'<rect width="1920" height="1080" fill="{bg}"/>', '<defs>']
    for cat,color in COLORS.items():
        out.append(f'<marker id="arrow-{cat}" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L9 4.5 L0 9 Z" fill="{color}"/></marker>')
    out += ['</defs><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif">',
            text(70, 82, summary.get('title', 'GQA case 4 优化轨迹'), 42, INK, 700),
            text(70, 137, summary.get('headline', '1.058× CANN'), 32, green, 700)]
    if summary.get('subtitle'):
        out.append(text(70, 182, summary['subtitle'], 24, INK))
    # 图例统一右对齐，来源一行、修改类别一行。
    for k, (cat, label) in enumerate(LABELS.items()):
        x = 1190+k*225
        out += [f'<rect x="{x}" y="57" width="16" height="16" rx="2" fill="{COLORS[cat]}"/>',
                text(x+25, 72, label, 19, COLORS[cat], 600)]
    for x,layer,label in [(1198,'host','Host'),(1320,'device','Device Triton'),(1525,'ir','Device IR'),(1710,'mixed','Mixed')]:
        out += [mark(x, 112, layer, INK, 8),text(x+20,119,label,19,INK,600)]
    left, right, top, bottom = 145, 1795, 315, 850
    yy = lambda v: bottom-v/1.2*(bottom-top)
    out.append(text(70, 275, '加速比', 27, INK, 600))
    for value in [0,.2,.4,.6,.8,1.,1.2]:
        y=yy(value)
        dash='stroke-dasharray="8 7"' if value==1 else ''
        out += [f'<path d="M{left} {y}H{right}" stroke="{ "#aab4bf" if value==1 else "#e7ebef"}" stroke-width="1.3" {dash}/>',
                text(left-25,y+8,f'{value:.1f}×',23,INK,anchor='end')]
    chosen=summary.get('displayed_indices', COMPACT_INDICES if compact else list(range(len(nodes))))
    focus=set(summary.get('focus_indices', COMPACT_INDICES[1:]))
    xs=[left+(right-left)*(k/(len(chosen)-1) if compact else ix/(len(nodes)-1)) for k,ix in enumerate(chosen)]
    for j in range(1,len(chosen)):
        x0,x1=xs[j-1],xs[j]
        y0,y1=yy(nodes[chosen[j-1]]['score']),yy(nodes[chosen[j]]['score'])
        dx=(x1-x0)*.42
        out.append(f'<path d="M{x0} {y0} C{x0+dx} {y0} {x1-dx} {y1} {x1} {y1}" fill="none" stroke="{green}" stroke-width="4"/>')
    for j,ix in enumerate(chosen):
        x,e=xs[j],nodes[ix]
        y=yy(e['score'])
        if compact and ix in PRIMARY_INDICES:
            out.append(f'<rect class="primary-frame" data-progress="{ix}" x="{x-110}" y="{y-66}" width="220" height="216" fill="none" stroke="{COLORS[e["display_category"]]}" stroke-width="2" pointer-events="none"/>')
        if ix==0:
            tip=f'原始 Triton · 官方 checker 4/4 通过 · {e.get("metric_name", "case 4 完整调用")} {e["latency_ms"]*1000:.5f} μs'
            out += [f'<g class="baseline-node" tabindex="0" role="button" data-progress="0"><title>{esc(tip)}</title><circle cx="{x}" cy="{y}" r="10" fill="{bg}" stroke="{INK}" stroke-width="3"/></g>',
                    text(x+21,y-55,f'{e["score"]:.3f}×',25,INK,700)]
            out.append(text(x,992,'0',23,INK,600,'middle'))
        else:
            out.append(node(e,x,y,13 if compact else 11))
            out.append(text(x,992,str(ix),23,INK,600,'middle'))
            if ix in focus:
                default_position=(x,y-29,'middle')
                label_x,label_y,anchor=summary.get('label_positions',{}).get(ix,default_position)
                out.append(f'<g class="score-label" data-progress="{ix}">'+text(label_x,label_y,f'{e["score"]:.3f}×',28,INK,700,anchor)+'</g>')
            if compact:
                cat=e['display_category'];color=COLORS[cat]
                out.append(f'<path class="method-arrow" d="M{x} {y+60}V{y+20}" stroke="{color}" stroke-width="2" fill="none" marker-end="url(#arrow-{cat})"/>')
                for line,label in enumerate(METHOD_LABELS[ix]):
                    out.append(text(x,y+94+line*32,label,25,color,600,'middle'))
    if summary.get('displayed_indices'):
        for ix in sorted(set(range(len(nodes)))-set(chosen)):
            x=left+(right-left)*ix/(len(nodes)-1)
            out.append(text(x,992,str(ix),23,INK,600,'middle'))
    out += ['</g></svg>']
    return '\n'.join(out)


def page(svg, name, data):
    payload = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    data_file = 'geomean-data.json' if name == 'geomean' else 'trace-data.json'
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GQA优化轨迹</title><style>
body{margin:0;background:#ffffff;color:#202e40;font-family:"Microsoft YaHei",sans-serif}nav,aside,details{max-width:1500px;margin:20px auto;padding:0 24px}nav{display:flex;gap:24px;align-items:center}a{color:#2368b1}svg{display:block;width:100%;height:auto;background:#ffffff}.node,.baseline-node{cursor:pointer}.node:hover,.baseline-node:hover{filter:brightness(1.25)}aside{background:white;padding:24px;border-radius:12px;line-height:1.9}pre{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:8px 15px;background:white;border:1px solid #cbd5e1;border-radius:6px;cursor:pointer}
</style><nav><b>GQA</b><a href="full.html">完整轨迹</a><a href="compact.html">重点轨迹</a><a href="geomean.html">四用例几何平均</a>''' + f'<a href="{name}.png">PNG</a><a href="{name}.svg">SVG</a><a href="{data_file}">数据与阶段对应</a><button onclick="document.documentElement.requestFullscreen()">全屏</button></nav>' + svg + '''<details id="node-details"><summary>节点与测量详情</summary><aside id="info">点击图中节点查看。</aside></details>
<details><summary>完整历史记录（包括未进入主线的实验）</summary><pre id="history"></pre></details>
<script>const DATA=''' + payload + ''';
const info=document.getElementById('info');
function show(el){document.getElementById('node-details').open=true;const n=DATA.nodes[+el.dataset.progress];info.replaceChildren();const h=document.createElement('h3');h.textContent=`进展 ${n.progress_index} · ${n.label}`;info.append(h);for(const t of [`${n.metric_name||'case 4 完整调用'}：${(n.latency_ms*1000).toFixed(5)} μs · ${n.score.toFixed(6)}×（固定参考）`,n.method,`修改类别：${n.display_layer}；指导来源：${n.display_category_label}`,n.attribution||'原始基线',`研究记录索引：${n.event_index}`]){const p=document.createElement('div');p.textContent=t;info.append(p)}if(n.case_results){for(const c of n.case_results){const row=document.createElement('div');row.textContent=`case ${c.case_id} · ${c.candidate_us.toFixed(5)} μs · 同轮 CANN ${c.paired_speedup.toFixed(6)}× · 正确性${c.passed?'通过':'未通过'}`;info.append(row)}}const stage=DATA.stages.find(s=>n.progress_index>=s.start_progress&&n.progress_index<=s.end_progress);if(stage){const p=document.createElement('p');p.textContent=`阶段 ${stage.stage}：${stage.name}（进展 ${stage.start_progress}–${stage.end_progress}）`;info.append(p);for(let i=stage.start_progress;i<=stage.end_progress;i++){const e=DATA.nodes[i];const row=document.createElement('div');row.textContent=`${i} · ${e.method.replaceAll('\\n',' / ')} · ${e.display_category_label} · ${e.display_layer}`;row.style.color=DATA.colors[e.display_category];info.append(row)}}}
document.querySelectorAll('.node,.baseline-node').forEach(el=>{el.onclick=()=>show(el);el.onkeydown=e=>{if(e.key==='Enter')show(el)}});
document.getElementById('history').textContent=JSON.stringify(DATA.original,null,2);
</script></html>'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = SOURCE.read_bytes()
    original = json.loads(raw)
    selection = milestones(original)
    assert selection['displayed_indices'] == original['presentation_selection']['displayed_indices']
    nodes = []
    for n, idx in enumerate(selection['displayed_indices']):
        e = dict(original['events'][idx], progress_index=n, event_index=idx)
        e['display_category'] = 'knowledge' if e['category'] in ('ir', 'knowledge') else e['category']
        e['display_category_label'] = LABELS[e['display_category']]
        e['display_layer'] = {'host':'Host','device':'Device Triton','ir':'Device IR','mixed':'Mixed'}[e['location']]
        nodes.append(e)
    stages = [dict(stage=k+1, start_progress=a, end_progress=b, name=n, methods=lines,
                   event_indices=[nodes[i]['event_index'] for i in range(a,b+1)],
                   reduction_percent=(1-nodes[b]['latency_ms']/nodes[a-1]['latency_ms'])*100)
              for k,(a,b,n,lines) in enumerate(STAGES)]
    data = dict(source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(raw).hexdigest(),
                colors=COLORS, color_labels=LABELS, nodes=nodes, stages=stages, original=original,
                compact_indices=COMPACT_INDICES, primary_indices=PRIMARY_INDICES,
                method_labels=METHOD_LABELS,
                note='展示类别合并，不修改原始归因；进展序号不等于实验轮次；压缩阶段保留全部内部来源。')
    assert len(nodes) == 24
    assert all(a['score'] < b['score'] for a,b in zip(nodes,nodes[1:]))
    assert not set(selection['displayed_indices']) & remeasurement_indices(original['events'])
    assert [i for a,b,_,_ in STAGES for i in range(a,b+1)] == list(range(1,len(nodes)))
    assert all(abs(e['score']-original['cann_anchor_us']/(e['latency_ms']*1000))<1e-10 for e in nodes)
    assert all(e['display_category']=='sim' for e in nodes if e['location']=='ir')
    (OUT/'trace-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name, compact in [('full',False),('compact',True)]:
        svg=chart(nodes,compact)
        (OUT/f'{name}.svg').write_text(svg,encoding='utf-8')
        (OUT/f'{name}.html').write_text(page(svg,name,data),encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=full.html"><a href="full.html">完整轨迹</a> · <a href="compact.html">重点轨迹</a>',encoding='utf-8')
    from playwright.sync_api import sync_playwright
    browser_checks=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch()
        tab=browser.new_page(viewport={'width':1920,'height':1200},device_scale_factor=1)
        for name in ['full','compact']:
            errors=[]
            tab.on('pageerror',lambda e:errors.append(str(e)))
            tab.goto((OUT/f'{name}.html').as_uri())
            tab.evaluate('document.fonts.ready')
            displayed=tab.locator('.node,.baseline-node').evaluate_all('(els)=>els.map(e=>+e.dataset.progress)')
            frames=tab.locator('.primary-frame').evaluate_all('(els)=>els.map(e=>+e.dataset.progress)')
            labels=tab.locator('.score-label').evaluate_all('(els)=>els.map(e=>+e.dataset.progress)')
            assert displayed == (COMPACT_INDICES if name=='compact' else list(range(24)))
            assert frames == (PRIMARY_INDICES if name=='compact' else [])
            assert labels == COMPACT_INDICES[1:]
            tab.locator('svg').screenshot(path=str(OUT/f'{name}.png'))
            tab.locator('.node').first.click()
            assert '进展 1' in tab.locator('#info').inner_text() if name=='full' else '进展 6' in tab.locator('#info').inner_text()
            overflow=tab.locator('svg').evaluate('(s)=>[...s.querySelectorAll("text")].filter(t=>{let b=t.getBBox();return b.x<0||b.y<0||b.x+b.width>1920||b.y+b.height>1080}).map(t=>t.textContent)')
            assert not errors and not overflow, (errors,overflow)
            browser_checks.append(dict(view=name,js_errors=errors,text_outside_canvas=overflow,click_details='passed',
                                       displayed_indices=displayed,primary_frames=frames,score_labels=labels))
        browser.close()
    assert SOURCE.read_bytes()==raw
    (OUT/'validation.json').write_text(json.dumps(dict(status='passed',nodes=24,stages=6,monotone=True,
        frozen_source_unchanged=True,remeasurements_hidden=True,stage_coverage='all 23 progress nodes exactly once',
        compact_indices=COMPACT_INDICES,primary_indices=PRIMARY_INDICES,
        browser=browser_checks),ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
