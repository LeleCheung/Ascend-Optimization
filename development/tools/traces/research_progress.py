"""Render one source-bound research trajectory using the gallery's visual language."""
import argparse
import html
import json
import math
from pathlib import Path

from .render import BG, GREEN, COLORS, text, node_shape
from .selection import remeasurement_indices


def milestones(data):
    """Select measured improvements; retain every original event in the input."""
    selected = []
    hidden = []
    best = None
    remeasurements = remeasurement_indices(data['events'])
    for index, event in enumerate(data['events']):
        score = event.get('score')
        if index == 0:
            selected.append(index)
            best = score
        elif index in remeasurements:
            hidden.append(index)
        elif not event.get('presentation_eligible', True):
            hidden.append(index)
        elif event.get('qualified') and score is not None and (best is None or score > best):
            selected.append(index)
            best = score
        else:
            hidden.append(index)
    labels = {}
    progress_round = 0
    for index in selected:
        if index == 0:
            labels[str(index)] = data.get('baseline_axis_label', 'FlagGems')
        else:
            progress_round += 1
            labels[str(index)] = '第' + str(progress_round) + '轮'
    return dict(policy='measured best progress; remeasurements hidden; full events preserved',
        displayed_indices=selected, detail_indices=hidden, axis_labels=labels)


def render(data):
    selection = milestones(data)
    events = [dict(data['events'][i], axis_label=selection['axis_labels'][str(i)])
        for i in selection['displayed_indices']]
    assert events and all(e['qualified'] or (i == 0 and e['score'] is None) for i,e in enumerate(events))
    width, height = max(1600, 320 + 310 * (len(events)-1)), 900 if data.get('compact') else 1060
    label_y = 850 if data.get('compact') else 1005
    valid = [e['score'] for e in events if e['score'] is not None]
    logarithmic = max(valid)/min(valid)>100
    high = data.get('axis_max', max(.5, math.ceil(max(valid)*1.18*10)/10))
    assert high > max(valid)
    xs = [180 + (width - 305) * i / max(1, len(events) - 1) for i in range(len(events))]
    if logarithmic:
        low=math.floor(math.log10(min(valid))); high=math.ceil(math.log10(max(valid)*1.15))
        y=lambda v:735-(math.log10(v)-low)/(high-low)*380
        ticks=[10**i for i in range(low,high+1)]
    else:
        y=lambda v:735-v/high*380
        ticks=[high*i/4 for i in range(1,5)]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           f'<rect width="{width}" height="{height}" fill="{BG}"/>', '<defs>']
    for cat, color in COLORS.items():
        out.append(f'<marker id="tip-{cat}" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6Z" fill="{color}"/></marker>')
    out += ['</defs><g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif">',
            text(52, 62, data['operator'], min(42,(width-650)/(len(data['operator'])*.58))),
            text(52, 112, ('固定shape ' if data.get('metric') == 'fixed_case_cann_speedup' else '全用例几何平均 ')+f'{max(valid):.3f}× CANN', 30, GREEN)]
    for k, (cat, label) in enumerate([('paired','Profiling'),('knowledge','知识／经验'),('ir','源码／编译分析'),('sim','模拟器')]):
        x, yy = width - 520 + k % 2 * 255, 55 + k // 2 * 40
        out += [f'<path d="M{x} {yy}h23" stroke="{COLORS[cat]}" stroke-width="3"/>', text(x+32, yy+8, label, 23, COLORS[cat])]
    for k,(location,label) in enumerate([('device','设备kernel'),('host','宿主侧'),
                                       ('mixed','两者都有'),('ir','IR层'),('baseline','起点')]):
        x=width-640+k%3*205; yy=145+k//3*40
        mark=(f'<circle cx="{x+10}" cy="{yy}" r="8" fill="#92908a"/>'
              if location=='baseline' else node_shape(x+10,yy,location,8))
        out += [mark,text(x+32,yy+8,label,23)]
    if data.get('optimization_loop'):
        out.append(text(52,225,data['optimization_loop'],24,COLORS['sim']))
    out.append(text(52,320,'加速比'+('（对数刻度）' if logarithmic else ''),27))
    if data.get('compact') and data.get('shape'):
        out.append(text(52,260,data['shape']+' · '+data.get('case',''),25))
    for tick in ticks:
        tick_label=f'{tick:g}×' if logarithmic else f'{tick:.2f}×'
        out += [f'<path d="M150 {y(tick)}H{width-65}" stroke="#ded7c9"/>',text(125,y(tick)+9,tick_label,25,anchor='end')]
    best = events[0]['score']
    for i, event in enumerate(events[1:], 1):
        score = event['score']
        if best is None:
            out.append(f'<path class="source-lineage" d="M{xs[i-1]} 785L{xs[i]} {y(score)}" fill="none" stroke="{GREEN}" stroke-width="3.5"><title>源码演进连线；失败起点无性能数值</title></path>')
            best=score
            continue
        out.append(f'<path class="retained-best" d="M{xs[i-1]} {y(best)}L{xs[i]} {y(max(best,score))}" fill="none" stroke="{GREEN}" stroke-width="3.5"/>')
        if score < best:
            out.append(f'<path class="exploration" d="M{xs[i-1]} {y(best)}L{xs[i]} {y(score)}" fill="none" stroke="{GREEN}" stroke-width="3" stroke-dasharray="8 8"/>')
            out.append(f'<circle cx="{xs[i]}" cy="{y(best)}" r="8" fill="{GREEN}"/>')
        best = max(best,score)
    for i, event in enumerate(events):
        x, yy = xs[i], y(event['score']) if event['score'] is not None else 785
        if event['score'] is None:
            out += [f'<path d="M{x-9} {yy-9}L{x+9} {yy+9}M{x-9} {yy+9}L{x+9} {yy-9}" stroke="#b84c42" stroke-width="4"/>',text(x,yy-25,'未通过',27,anchor='middle'),text(x,label_y,data.get('baseline_axis_label', 'FlagGems'),25,anchor='middle')]
            continue
        out += ['<g><title>'+html.escape(json.dumps(event,ensure_ascii=False))+'</title>',
                (f'<circle cx="{x}" cy="{yy}" r="9" fill="#92908a" stroke="{BG}" stroke-width="2"/>' if i==0 else node_shape(x,yy,event.get('location','device'),9)),
                text(x+12 if i==0 else x,yy-25,('≈' if event.get('score_is_rebased') else '')+(f'{event["score"]:.4f}×' if event['score']<.01 else f'{event["score"]:.3f}×'),27,anchor='start' if i==0 else 'middle'),
                text(x,label_y,event.get('axis_label',event['label'] if i==0 else f'V{i}'),25,anchor='middle')]
        if event.get('method'):
            cat = event['category']; color = COLORS[cat]
            above=yy>620
            lines=[]
            for paragraph in event['method'].split('\n'):
                line='';units=0
                for char in paragraph:
                    step=28 if ord(char)>127 else 16
                    if units+step>290 and line:lines.append(line);line='';units=0
                    line+=char;units+=step
                if line:lines.append(line)
            tx=max(175,min(width-175,x))
            for j, line in enumerate(lines):
                out.append(text(tx,yy-105+j*33 if above else yy+85+j*33,line,28,color,anchor='middle'))
            out.append(f'<path class="method-arrow" d="M{x} {yy-75 if above else yy+50}L{x} {yy-45 if above else yy+18}" stroke="{color}" stroke-width="2" fill="none" marker-end="url(#tip-{cat})"/>')
        out.append('</g>')
    out.append('</g></svg>')
    return '\n'.join(out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();write_report(json.loads(a.data.read_bytes()),a.output)


def write_report(data, output):
    from types import SimpleNamespace
    a=SimpleNamespace(output=output)
    a.output.mkdir(parents=True,exist_ok=True)
    data=dict(data,presentation_selection=milestones(data))
    chart=render(data);(a.output/'mainline.svg').write_text(chart,encoding='utf8')
    (a.output/'mainline-presentation.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    css='body{margin:24px;background:#fff7e8;color:#1b2434;font:20px/1.6 "Microsoft YaHei",sans-serif}svg{width:100%;height:auto;display:block}a{color:#276980;text-decoration:none;padding:8px 14px;background:#eee7d9;border-radius:8px}details{margin-top:18px}table{border-collapse:collapse;width:100%;font-size:17px}td,th{padding:9px;border-bottom:1px solid #ded7c9;text-align:left}code{overflow-wrap:anywhere}'
    detail=['<details><summary>轮次映射与原始记录</summary><p>'+html.escape(data['scope'])+'</p>'
        '<p>图中轮次为最佳进展编号；原始事件与筛选映射分别保存。</p>']
    if data.get('historical_presentation'):
        detail.append('<p><a href="'+html.escape(data['historical_presentation'],quote=True)+'">统一复测前的全部历史成绩</a></p>')
    for link in data.get('history_links', []):
        detail.append('<p><a href="'+html.escape(link['href'],quote=True)+'">'+html.escape(link['label'])+'</a></p>')
    detail.append('<table><tr><th>记录</th><th>图中加速比</th><th>候选耗时μs</th><th>源码／IR身份</th><th>结果</th></tr>')
    for event in data['events']:
        score=f'{event["score"]:.6f}×' if event.get('score') is not None else '无有效成绩'
        latency=f'{event["latency_ms"]*1000:.3f}' if event.get('latency_ms') is not None else '—'
        replay=event.get('replay',{});result=replay.get('result')
        link='<a href="../'+html.escape(result,quote=True)+'">原始结果</a>' if result else '完整记录中保存'
        identity=html.escape(event.get('source_sha256',''))
        if event.get('ir_sha256'):
            identity+='</code><br>IR <code>'+html.escape(event['ir_sha256'])
        detail.append('<tr><td>'+html.escape(event['label'])+'</td><td>'+score+'</td><td>'+latency+'</td><td><code>'+
            identity+'</code></td><td>'+link+'</td></tr>')
    detail.append('</table></details>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(data['operator'])+'</title><style>'+css+'</style><p><a href="mainline.png">PNG</a> <a href="mainline.svg">SVG</a> <a href="mainline-presentation.json">完整记录</a></p>'+chart+''.join(detail)+'</html>'
    (a.output/'index.html').write_text(page,encoding='utf8')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser=pw.chromium.launch();tab=browser.new_page(viewport=dict(width=1800,height=1100),device_scale_factor=1)
        tab.goto((a.output/'index.html').resolve().as_uri());tab.locator('svg').screenshot(path=str(a.output/'mainline.png'));browser.close()


if __name__=='__main__':main()
