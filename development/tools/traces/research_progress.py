"""Render one source-bound research trajectory using the gallery's visual language."""
import argparse
import html
import json
import math
from pathlib import Path

from .render import BG, GREEN, COLORS, text, node_shape


def render(data):
    events = data['events']
    assert events and all(e['qualified'] for e in events)
    width, height = max(1600, 380 * len(events)), 1000
    high = max(1, math.ceil(max(e['score'] for e in events) * 1.2 * 4) / 4)
    xs = [180 + (width - 340) * i / max(1, len(events) - 1) for i in range(len(events))]
    y = lambda v: 720 - v * 420 / high
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           f'<rect width="{width}" height="{height}" fill="{BG}"/>', '<defs>']
    for cat, color in COLORS.items():
        out.append(f'<marker id="tip-{cat}" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6Z" fill="{color}"/></marker>')
    out += ['</defs><g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif">',
            text(52, 62, data['operator'], 38),
            text(52, 112, f'保留 {max(e["score"] for e in events):.3f}× CANN', 30, GREEN)]
    for k, (cat, label) in enumerate([('paired','Profiling'),('knowledge','知识／经验'),('ir','源码／IR分析'),('sim','模拟器')]):
        x, yy = width - 520 + k % 2 * 255, 55 + k // 2 * 40
        out += [f'<path d="M{x} {yy}h23" stroke="{COLORS[cat]}" stroke-width="3"/>', text(x+32, yy+8, label, 23, COLORS[cat])]
    out += [f'<circle cx="{width-510}" cy="145" r="8" fill="{GREEN}"/>', text(width-488,153,'设备kernel',23), text(52,265,'加速比',27)]
    if any(e.get('location')=='mixed' for e in events):
        out += [node_shape(width-255,145,'mixed',8),text(width-232,153,'设备＋启动',23)]
    for i in range(5):
        tick = high * i / 4
        out += [f'<path d="M150 {y(tick)}H{width-65}" stroke="#ded7c9"/>',text(125,y(tick)+9,f'{tick:.2f}×',25,anchor='end')]
    best = events[0]['score']
    for i, event in enumerate(events[1:], 1):
        score = event['score']
        out.append(f'<path class="retained-best" d="M{xs[i-1]} {y(best)}L{xs[i]} {y(max(best,score))}" fill="none" stroke="{GREEN}" stroke-width="3.5"/>')
        if score < best:
            out.append(f'<path class="exploration" d="M{xs[i-1]} {y(best)}L{xs[i]} {y(score)}" fill="none" stroke="{GREEN}" stroke-width="3" stroke-dasharray="8 8"/>')
            out.append(f'<circle cx="{xs[i]}" cy="{y(best)}" r="8" fill="{GREEN}"/>')
        best = max(best,score)
    for i, event in enumerate(events):
        x, yy = xs[i], y(event['score'])
        out += ['<g><title>'+html.escape(json.dumps(event,ensure_ascii=False))+'</title>',
                node_shape(x,yy,event.get('location','device'),9),
                text(x,yy-25,f'{event["score"]:.4f}×' if event['score']<.01 else f'{event["score"]:.3f}×',27,anchor='middle'),
                text(x,945,event['label'],25,anchor='middle')]
        if event.get('method'):
            cat = event['category']; color = COLORS[cat]
            for j, line in enumerate(event['method'].split('\n')):
                out.append(text(x,yy+85+j*34,line,28,color,anchor='middle'))
            out.append(f'<path class="method-arrow" d="M{x} {yy+50}L{x} {yy+18}" stroke="{color}" stroke-width="2" fill="none" marker-end="url(#tip-{cat})"/>')
        out.append('</g>')
    out.append('</g></svg>')
    return '\n'.join(out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.data.read_bytes());a.output.mkdir(parents=True,exist_ok=True)
    chart=render(data);(a.output/'mainline.svg').write_text(chart,encoding='utf8')
    (a.output/'mainline-presentation.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    css='body{margin:24px;background:#fff7e8;color:#1b2434;font:20px/1.6 "Microsoft YaHei",sans-serif}svg{width:100%;height:auto;display:block}a{color:#276980;text-decoration:none;padding:8px 14px;background:#eee7d9;border-radius:8px}details{margin-top:18px}'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(data['operator'])+'</title><style>'+css+'</style><p><a href="mainline.png">PNG</a> <a href="mainline.svg">SVG</a> <a href="mainline-presentation.json">完整记录</a></p>'+chart+'<details><summary>轮次映射与原始记录</summary><p>'+html.escape(data['scope'])+'</p></details></html>'
    (a.output/'index.html').write_text(page,encoding='utf8')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser=pw.chromium.launch();tab=browser.new_page(viewport=dict(width=1800,height=1100),device_scale_factor=1)
        tab.goto((a.output/'index.html').resolve().as_uri());tab.locator('svg').screenshot(path=str(a.output/'mainline.png'));browser.close()


if __name__=='__main__':main()
