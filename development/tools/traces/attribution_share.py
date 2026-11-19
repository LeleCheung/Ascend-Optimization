"""Render mainline primary-attribution shares, including inference counts."""
import collections
import json
import math
from pathlib import Path

from .stitched import COLORS, BG, text


def main():
    out=Path('development/reports/operator-traces-stitched-20261117').resolve()
    rows=json.loads((out/'cann-view.json').read_bytes())['operators']
    records=[a for r in rows for a in r['mainline_attribution_review']]
    counts=collections.Counter(a['primary'] for a in records)
    confirmed=collections.Counter(a['primary'] for a in records if a['confidence']=='confirmed')
    total=len(records)
    names={'ir':'源码／IR分析','paired':'Profiling','knowledge':'知识／经验','sim':'模拟器'}
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="820" viewBox="0 0 1400 820"><rect width="1400" height="820" fill="{BG}"/><g font-family="Microsoft YaHei,sans-serif">',
           text(55,75,'69个修改节点：主来源占比',40),
           text(55,130,'15个日志确认 · 54个推断；每个节点只计一个主来源',27)]
    angle=-math.pi/2;cx,cy,r=355,440,225
    summary=[]
    for i,key in enumerate(names):
        count=counts[key];fraction=count/total;end=angle+fraction*2*math.pi
        if count:
            x1,y1=cx+r*math.cos(angle),cy+r*math.sin(angle)
            x2,y2=cx+r*math.cos(end),cy+r*math.sin(end)
            parts.append(f'<path d="M{cx} {cy}L{x1} {y1}A{r} {r} 0 {int(fraction>.5)} 1 {x2} {y2}Z" fill="{COLORS[key]}" stroke="{BG}" stroke-width="4"/>')
            mid=(angle+end)/2
            parts.append(text(cx+r*.68*math.cos(mid),cy+r*.68*math.sin(mid)+10,f'{fraction:.1%}',30,'white',anchor='middle'))
        angle=end
        yy=270+i*105
        parts.append(f'<rect x="680" y="{yy-24}" width="22" height="22" rx="4" fill="{COLORS[key]}"/>')
        parts.append(text(720,yy,f'{names[key]}：{count}个 · {fraction:.1%}',31))
        parts.append(text(720,yy+39,f'日志确认 {confirmed[key]} · 推断 {count-confirmed[key]}',25))
        summary.append(dict(category=key,count=count,percent=100*fraction,confirmed=confirmed[key],inferred=count-confirmed[key]))
    parts.extend([text(55,745,'统计的是当前主图归因，不是工具调用频率或性能收益贡献。',27),
                  text(55,787,'源码／IR的48项均为推断；模拟器0项不包含历史分支和GQA新实验。',27),'</g></svg>'])
    svg=out/'attribution-share.svg';svg.write_text('\n'.join(parts),encoding='utf-8')
    (out/'attribution-share.json').write_text(json.dumps({'total':total,'categories':summary},ensure_ascii=False,indent=2),encoding='utf-8')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        page=browser.new_page(viewport={'width':1400,'height':820})
        page.goto(svg.as_uri());page.screenshot(path=str(svg.with_suffix('.png')))
        browser.close()
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
