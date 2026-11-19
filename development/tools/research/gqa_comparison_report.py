"""Render the frozen GQA comparison; failed attempts have no invented score."""
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EVIDENCE=ROOT/'development/experiments/gqa-comparison-20261117'
OUT=ROOT/'development/reports/gqa-simulator-comparison-20261117'


def text(x,y,value,size=30,color='#253e4c',anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" font-weight="650">{html.escape(str(value))}</text>'


def chart(arm,sim,baseline_us,cann_us):
    green='#13865c';color='#cc6000' if sim else '#596170'
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="820" viewBox="0 0 1500 820">',
      '<rect width="1500" height="820" fill="#fff7e8"/><defs><marker id="method-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="context-stroke"/></marker></defs>','<g font-family="Microsoft YaHei,sans-serif">',
      text(55,70,'GQA · '+('有模拟器' if sim else '无模拟器'),44),
      text(55,120,f"{cann_us / baseline_us * arm['retained_speedup']:.3f}× CANN",32,green),
      text(1050,65,'Profiling',25,'#2368b1'),text(1260,65,'知识／经验',25,'#8752ae'),
      text(1050,110,'源码／IR',25,'#596170'),text(1260,110,'模拟器',25,'#cc6000'),
      text(55,205,'相对 CANN 的加速比 · case 4',28)]
    xs=[180,710,1270];y=lambda value:570-value/.12*300
    for value in [0,.04,.08,.12]:
        yy=y(value);parts.extend([f'<path d="M135 {yy}H1430" stroke="#ddd5c4"/>',text(115,yy+10,f'{value:.2f}×',25,anchor='end')])
    points=[(0,cann_us/baseline_us)]+[(i+1,cann_us/r['target_us']) for i,r in enumerate(arm['rounds']) if r['passed']==r['total'] and r['target_us']]
    for (i,a),(j,b) in zip(points,points[1:]):parts.append(f'<path d="M{xs[i]} {y(a)}L{xs[j]} {y(b)}" fill="none" stroke="{green}" stroke-width="4"/>')
    for i,v in points:parts.extend([f'<circle cx="{xs[i]}" cy="{y(v)}" r="10" fill="{green}"/>',text(xs[i],y(v)-25,f'{v:.3f}×',32,anchor='middle')])
    for i,name in enumerate(['起点','第1次修改','第2次修改']):parts.append(text(xs[i],635,name,29,anchor='middle'))
    methods=['重排矩阵乘累加','减少一次概率矩阵乘'] if sim else ['调整编译参数','增大分块：编译失败']
    for i,method in enumerate(methods,1):
        parts.append(text(xs[i],740,method,29,color,anchor='middle'))
        point=next((v for j,v in points if j==i),None)
        if point is not None:parts.append(f'<path d="M{xs[i]} 700Q{xs[i]-380} 635 {xs[i]-18} {y(point)+18}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#method-arrow)"/>')
    parts.extend(['</g></svg>']);return '\n'.join(parts)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=json.loads((EVIDENCE/'comparison-results.json').read_bytes())
    anchor=next(c['native_us'] for c in data['arms']['with-simulator-v2']['rounds'][-1]['cases'] if c['case_id']==data['target_case'])
    view={'case_id':data['target_case'],'cann_anchor_us':anchor,'anchor_origin':'with-simulator-v2/round-02','baseline_cann_speedup':anchor/data['baseline_target_us'],'arms':{}}
    for key,arm in data['arms'].items():
        view['arms'][key]=[{'round':r['round'],'cann_speedup':anchor/r['target_us'] if r['passed']==r['total'] and r['target_us'] else None} for r in arm['rounds']]
    (OUT/'cann-view.json').write_text(json.dumps(view,indent=2),encoding='utf-8')
    cards=[]
    for key,title,sim in [('no-simulator-v2','无模拟器',False),('with-simulator-v2','有模拟器',True)]:
        svg=chart(data['arms'][key],sim,data['baseline_target_us'],anchor);name='with-simulator' if sim else 'no-simulator'
        (OUT/(name+'.svg')).write_text(svg,encoding='utf-8')
        cards.append('<div class="card">'+svg+'<p><a href="'+name+'.png">PNG</a> · <a href="'+name+'.svg">SVG</a></p></div>')
    rows=[]
    for key,title in [('no-simulator-v2','无模拟器'),('with-simulator-v2','有模拟器')]:
        for step in data['arms'][key]['rounds']:
            rows.append(f'<tr><td>{title}</td><td>{step["round"]}</td><td>{step["passed"]}/{step["total"]}</td><td>'+ (f'{step["target_us"]:.1f} μs' if step['passed']==step['total'] else '编译失败')+'</td></tr>')
    css='body{margin:28px;background:#fff7e8;color:#253e4c;font:20px/1.6 "Microsoft YaHei",sans-serif}h1{font-size:36px}a{color:#276980}.cards{display:grid;grid-template-columns:1fr;gap:24px}.card{border:1px solid #ddd5c4;padding:12px;border-radius:12px}.card svg{width:100%;height:auto;display:block}table{border-collapse:collapse}td,th{padding:12px 24px;border-bottom:1px solid #ddd5c4;text-align:left}@media(min-width:1200px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}}'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>GQA：有／无模拟器对照</title><style>'+css+'</style><h1>GQA：模拟器如何影响改法</h1><p>同一起点、模型与最多两次修改；目标为case 4，四项用例共同验收。</p><div class="cards">'+''.join(cards)+'</div><h2>本次结果</h2><p>有模拟器一组：1900.7 → 1378.7 μs，耗时降低27.5%；四项检查通过。无模拟器一组保留起点。</p><table><tr><th>组别</th><th>修改</th><th>通过</th><th>目标耗时</th></tr>'+''.join(rows)+'</table><h2>修改后的证据</h2><p>IR内矩阵乘4→3；观察窗口中Cube的FIX_L0C_TO_DST执行17→13次，WAIT_FLAG_DEVI执行33→24次；真机kernel耗时1894.0→1369.0 μs。</p><p>模拟器应用超时，以上是已采到的局部窗口，不是完整调用时间。此次为单任务、每组一条优化过程；包含Codex代码审查反馈，不宣称模拟器独立贡献已被统计证明。</p><p><a href="../attribution-and-gqa-20261117.md">完整说明</a> · <a href="../../experiments/gqa-comparison-20261117/comparison-results.json">原始结果与模型成本</a> · <a href="../operator-traces-stitched-20261117/index.html">26算子轨迹</a></p></html>'
    page=page.replace('<h2>本次结果</h2>',f'<h2>本次结果</h2><p>相对 CANN：起点 {anchor/data["baseline_target_us"]:.3f}× → 有模拟器终点 {anchor/data["arms"]["with-simulator-v2"]["rounds"][-1]["target_us"]:.3f}×，仍慢于 CANN。</p><details><summary>计量口径</summary><p>两图统一使用最终轮 case 4 的 CANN 耗时 {anchor:.4f} μs 除以各版本候选耗时。原始各轮 CANN 测量保留在结果 JSON；这里不是四项用例的平均加速比。</p></details>')
    (OUT/'index.html').write_text(page,encoding='utf-8')
    from playwright.sync_api import sync_playwright
    checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True);page=browser.new_page(viewport={'width':1500,'height':820})
        for f in OUT.glob('*.svg'):
            page.goto(f.as_uri());page.screenshot(path=str(f.with_suffix('.png')))
            overflow=page.locator('svg').evaluate('s=>[...s.querySelectorAll("text")].filter(e=>{const b=e.getBBox();return b.x<0||b.x+b.width>1501||b.y+b.height>821}).map(e=>e.textContent)')
            checks.append({'file':f.name,'overflow':overflow})
        browser.close()
    (OUT/'presentation-check.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    assert all(not c['overflow'] for c in checks)
    print(OUT)


if __name__=='__main__':main()
