"""Render complete study attempts, including failures, from the current inventory."""
import argparse
import html
import json
import math
from pathlib import Path


def comparison_tables(out):
    path=out/'results-summary.json'
    if not path.exists():return ''
    data=json.loads(path.read_bytes());comparison=data.get('comparison',{})
    parts=['<h2>完整性能与实际成本</h2><p>各组保留完整六项几何平均最好的有效候选；两组均须通过正式合同和双方补充检查。</p>',
           '<table><tr><th>性能用例</th><th>无模拟器 · 相对CANN</th><th>有模拟器 · 相对CANN</th><th>候选耗时比</th></tr>']
    for row in comparison.get('cases',[]):
        parts.append('<tr><td>'+html.escape(row['case'])+'</td>'+''.join('<td>'+f'{row[key]:.3f}×'+'</td>' for key in ['no_simulator_cann_speedup','with_simulator_cann_speedup','candidate_time_ratio'])+'</tr>')
    parts.append('</table><p>候选耗时比＝无模拟器耗时÷有模拟器耗时。各轮CANN参考存在测量波动，原始参考时间保留在JSON。</p>')
    parts.append('<table><tr><th>组别</th><th>完成轮数</th><th>实际工具轮数</th><th>输出token</th><th>模型成本估计</th></tr>')
    for arm,label in [('no-simulator','无模拟器'),('with-simulator','有模拟器'),('compatibility','兼容准备（另计）')]:
        row=data['arms'][arm]
        parts.append(f'<tr><td>{label}</td><td>{row["completed_sessions"]}</td><td>{row["model_turns"]}</td><td>{row["usage"]["output_tokens"]:,}</td><td>${row["estimated_usd"]:.3f}</td></tr>')
    parts.append('</table><p>成本为宿主报告估计，包含失败调用；不含Codex、硬件与采集成本，亦非账单金额。相同最大预算不代表实际token相同。</p><p><a href="results-summary.json">逐项成绩与成本JSON</a></p>')
    if (out/'trace-rounds.csv').exists():parts.append('<p><a href="trace-rounds.csv">16轮CSV索引</a></p>')
    return ''.join(parts)


def final_replay_panel(out):
    path=out/'results-summary.json'
    if not path.exists():return ''
    data=json.loads(path.read_bytes());replays=data.get('final_replay',{})
    if len(replays)!=2:return ''
    parts=['<h2>最终保留版本：固定源码收尾复测</h2><p>两组均完成8次生成后按完整合同成绩选择版本；复测不增加生成轮次，原搜索成绩保留在下方。</p>',
           '<table><tr><th>组别</th><th>正确性</th><th>目标耗时</th><th>目标相对CANN</th><th>六项几何平均</th></tr>']
    for arm,label in [('no-simulator','无模拟器'),('with-simulator','有模拟器')]:
        row=replays[arm];t=next(x for x in row['timings'] if x['uuid']=='upsample_bilinear2d-timing-1')
        parts.append(f'<tr><td>{label}</td><td>{row["num_passed"]}/{row["num_workloads"]}</td><td>{t["latency_ms"]*1000:.1f} μs</td><td>{t["speedup"]:.3f}×</td><td>{row["geo_mean"]:.3f}×</td></tr>')
    parts.append('</table><p>计时为KGS profiler设备任务时间，启用L2清理；不是CPU侧walltime。该结果是一个算子的单次搜索对照，不代表多任务或多随机种子的统计结论。</p>')
    return ''.join(parts)


def mechanism_panel(out):
    path=out/'mechanism-findings.json'
    if not path.exists():return ''
    data=json.loads(path.read_bytes());esc=html.escape
    parts=['<h2>模拟器怎样影响了修改</h2>']
    for step in data['steps']:
        links=' · '.join('<a href="'+esc(x['path'])+'">'+esc(x['label'])+'</a>' for x in step.get('evidence_links',[]) if (out/x['path']).exists())
        parts.append('<article><h3>'+esc(step['round'])+'</h3><p>'+esc(step['observation'])+'</p><p>修改：'+esc(step['action'])+'</p><p>验证：'+esc(step['outcome'])+'</p><p>'+links+'</p></article>')
    parts.extend(['<p>'+esc(data['lesson'])+'</p>','<details><summary>观察范围与修改层次</summary><p>'+esc(data['limits'])+'</p></details>'])
    return ''.join(parts)


def progress_chart(data):
    """Plot retained target scores; failed calls consume budget but add no score."""
    parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 400" role="img" aria-label="各轮最佳已验证目标加速比">']
    x=lambda n:90+105*n
    y=lambda s:320-75*(math.log10(max(s,0.001))+3)
    for score in [0.001,0.01,0.1,1,10]:
        yy=y(score)
        parts.append(f'<path d="M90 {yy}H970" stroke="#d7dce0"/><text x="15" y="{yy+6}" font-size="18">{score:g}×</text>')
    for n in range(1,9):
        parts.append(f'<text x="{x(n)-10}" y="355" font-size="18">{n}</text>')
    parts.append('<text x="380" y="390" font-size="20">生成轮次（失败尝试同样计入）</text>')
    for arm,color,label in [('no-simulator','#287ab3','无模拟器'),('with-simulator','#dc7b22','有模拟器')]:
        best=0;points=[]
        for n,row in enumerate(data['arms'][arm],1):
            if not row.get('output_sha256') or not row.get('model_result'):continue
            f=row.get('formal',{})
            target=next((v for v in row.get('timings',[]) if v['uuid']=='upsample_bilinear2d-timing-1'),{})
            supplements=row.get('supplementary',{})
            checks_ok=all(supplements.get(k,{}).get('status')=='passed' and supplements[k].get('source_sha256')==row.get('output_sha256') for k in ['supplementary.json','cann_supplementary.json'])
            if f.get('status')=='PASSED' and f.get('num_passed')==24 and checks_ok and row.get('review_feedback',{}).get('status')!='repair_required':
                best=max(best,target.get('speedup') or 0)
            if best>0:points.append((x(n),y(best),best))
        if points:
            coords=' '.join(f'{xx},{yy}' for xx,yy,_ in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
            for xx,yy,s in points:
                parts.append(f'<circle cx="{xx}" cy="{yy}" r="5" fill="{color}"><title>{label}：截至该轮最佳 {s:.6f}× CANN</title></circle>')
        else:
            parts.append(f'<text x="330" y="{80 if arm=="with-simulator" else 110}" fill="{color}" font-size="20">{label}：尚无全合同通过成绩</text>')
    parts.append('<text x="670" y="30" fill="#287ab3" font-size="20">● 无模拟器</text><text x="830" y="30" fill="#dc7b22" font-size="20">● 有模拟器</text></svg>')
    return '<h2>截至每轮的最佳已验证目标成绩</h2><p>相对CANN，对数坐标。平段表示保留已有成绩，不是重复测量；原生起点未通过完整合同，单独列示。</p>'+''.join(parts)


def search_trace(data,labels):
    """Draw all source-parent relationships, including the simulator-arm rollback."""
    esc=html.escape
    parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1630" role="img" aria-label="两组完整八轮源码与验证轨迹">',
           '<rect width="1600" height="1630" fill="#fff7e8"/>',
           '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#67757c"/></marker></defs>',
           '<g font-family="Microsoft YaHei,sans-serif" fill="#253e4c" text-anchor="middle">',
           '<text x="800" y="50" font-size="36" font-weight="bold">同一起点 · 完整八轮优化轨迹</text>',
           '<rect x="440" y="80" width="720" height="100" rx="16" fill="#eef2e9" stroke="#67757c" stroke-width="2"/>',
           '<text x="800" y="120" font-size="27">原生 FlagGems · 完整合同 18/24</text>',
           '<text x="800" y="158" font-size="25">目标 0.000967× CANN（局部可测，整体未通过）</text>']
    for arm,x,color,title in [('no-simulator',410,'#287ab3','无模拟器'),('with-simulator',1190,'#dc7b22','增加模拟器')]:
        rows=data['arms'][arm];known={}
        parts.append(f'<text x="{x}" y="218" font-size="28" fill="{color}">{title}</text>')
        for i,row in enumerate(rows):
            y=250+i*160;parent=known.get(row['input_sha256'])
            if parent is None:
                path=f'M{650 if x<800 else 950} 180 L{x} {y-6}'
            elif parent==i-1:
                path=f'M{x} {250+parent*160+135} L{x} {y-6}'
            else:
                edge=x+315;outside=x+385;py=250+parent*160+65
                path=f'M{edge} {py} C{outside} {py},{outside} {y+65},{edge} {y+65}'
            parts.append(f'<path d="{path}" fill="none" stroke="#67757c" stroke-width="2" marker-end="url(#arrow)"/>')
            method=labels.get(arm,{}).get(row['round'],{}).get('method','待生成')
            chunks=[method[n:n+23] for n in range(0,len(method),23)][:2]
            formal=row.get('formal',{});target=next((v for v in row.get('timings',[]) if v['uuid']=='upsample_bilinear2d-timing-1'),{})
            if formal.get('status')=='PASSED':status=f"24/24 · 目标 {target.get('speedup',0):.3f}× CANN"
            elif formal:status='正式评测未通过'
            elif row.get('preflight',{}).get('status')=='failed':status='预检未通过'
            else:status='验证进行中'
            fill='#fffaf0' if formal.get('status')=='PASSED' else '#f3eee7'
            parts.append(f'<rect x="{x-310}" y="{y}" width="620" height="135" rx="14" fill="{fill}" stroke="{color}" stroke-width="2"/>')
            parts.append(f'<text x="{x}" y="{y+29}" font-size="25" font-weight="bold">第{i+1}轮</text>')
            for n,chunk in enumerate(chunks):parts.append(f'<text x="{x}" y="{y+61+n*26}" font-size="23">{esc(chunk)}</text>')
            parts.append(f'<text x="{x}" y="{y+119}" font-size="24" fill="{color}">{esc(status)}</text>')
            if row.get('output_sha256'):known[row['output_sha256']]=i
    parts.append('<text x="800" y="1580" font-size="25">连线按实际源码继承；所有失败与退步保留，保留最佳成绩另见性能曲线。</text></g></svg>')
    return ''.join(parts)


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--complete',action='store_true');a=p.parse_args()
    data=json.loads(a.inventory.read_bytes());out=a.output;out.mkdir(parents=True,exist_ok=True)
    label_path=out/'labels.json'
    labels=json.loads(label_path.read_bytes()) if label_path.exists() else {}
    esc=lambda x:html.escape(str(x))
    cards=[]
    for arm,title in [('no-simulator','无模拟器'),('with-simulator','有模拟器')]:
        rows=[]
        for r in data['arms'][arm]:
            label=labels.get(arm,{}).get(r['round'],{})
            if label and not (r.get('output_sha256') or '').startswith(label['source_prefix']):label={}
            formal=r.get('formal');target=next((x for x in r.get('timings',[]) if x['uuid']=='upsample_bilinear2d-timing-1'),{})
            methods=[]
            for d in r['decisions']:
                record=d['record'];sources='、'.join(dict.fromkeys(e['category'] for e in record['evidence']))
                sim=[e['finding'] for e in record['evidence'] if e['category']=='simulator']
                methods.append('<details><summary>修改前原始判断 · '+esc(sources)+'</summary><p>'+esc(record['hypothesis'])+'</p><p>'+esc(record['change'])+'</p>'+(''.join('<p class="sim">模拟器：'+esc(x)+'</p>' for x in sim))+'</details>')
            status=(str(formal['num_passed'])+'/'+str(formal['num_workloads'])+' · '+formal['status']) if formal else ('预检 '+r.get('preflight',{}).get('status','待完成'))
            score=f"{target['speedup']:.6f}× CANN" if target.get('speedup') else '无有效性能分数'
            cost=r.get('model_result',{}).get('total_cost_usd')
            parent='<p>起点 '+esc(r['input_sha256'][:12])+' → '+esc((r.get('output_sha256') or '待生成')[:12])+'</p>'
            feedback=r.get('review_feedback',{})
            warning=('<details><summary>源码审查／失败反馈</summary><pre>'+esc(json.dumps(feedback,ensure_ascii=False,indent=2))+'</pre></details>') if feedback else ''
            failure=r.get('formal_failure',{}).get('log') or (r.get('preflight',{}).get('log_tail') if r.get('preflight',{}).get('status')=='failed' else None)
            if failure:
                kind='入口冲突：尚未执行正式测试' if 'conflicting public symbol' in failure else '本轮失败原始记录'
                warning+='<details><summary>'+esc(kind)+'</summary><pre>'+esc(failure)+'</pre></details>'
            short=('<p>'+esc(label['method'])+'</p><p>判断依据：'+esc(label['evidence'])+'</p>') if label else ''
            artifacts=[]
            call=Path('evidence')/arm/'agent-calls'/r['round']
            for name,title in [('output-source.py','源码'),('candidate.diff','diff'),('agent-events.jsonl','模型原始记录'),('prompt.md','输入提示'),('model-usage.json','模型成本')]:
                relative=call/name
                if (out/relative).exists():artifacts.append('<a href="'+esc(relative.as_posix())+'">'+title+'</a>')
            if r.get('output_sha256'):
                relative=Path('evidence')/arm/('formal-'+r['output_sha256'][:12])/'evaluation/result.json'
                if (out/relative).exists():artifacts.append('<a href="'+esc(relative.as_posix())+'">正式评测原件</a>')
            links='<p>'+' · '.join(artifacts)+'</p>' if artifacts else ''
            rows.append('<article><h3>'+esc(r['round'])+'</h3>'+short+'<p>'+esc(status)+' · '+esc(score)+'</p>'+parent+links+'<p>模型记录成本 '+esc(f'${cost:.3f}' if cost is not None else '待返回')+'</p>'+warning+''.join(methods)+'</article>')
        cards.append('<section><h2>'+title+'</h2>'+''.join(rows)+'</section>')
    css='body{background:#fff7e8;color:#253e4c;font:20px/1.6 "Microsoft YaHei",sans-serif;margin:30px}h1{font-size:36px}.arms{display:grid;grid-template-columns:1fr;gap:24px}section,article{border:1px solid #ddd5c4;border-radius:12px;padding:18px;margin-bottom:18px}p,pre{overflow-wrap:anywhere}pre{white-space:pre-wrap;font:inherit}summary{cursor:pointer;font-weight:bold}.sim{color:#b45708}a{color:#2368b1}@media(min-width:1300px){.arms{grid-template-columns:repeat(2,minmax(0,1fr))}}'
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>原生上采样优化对照 · 运行中</title><style>'+css+'</style><h1>原生上采样：有／无模拟器优化对照</h1><p>运行中。共同起点为固定FlagGems原生实现；18/24通过，目标case约253.949ms、0.000967× CANN。保留全部修改、失败和修复，尚不作两组胜负结论。</p><p>每轮先保存判断，再修改；模拟器组必须读取当前源码绑定的模拟器报告。局部窗口与完整调用分别呈现。</p><div class="arms">'+''.join(cards)+'</div><p><a href="trace-inventory.json">完整结构化记录</a></p></html>'
    chart=progress_chart(data)
    page=page.replace('局部窗口与完整调用分别呈现。','局部模拟窗口与真机设备任务时间分别呈现。')
    (out/'search-trace.svg').write_text(search_trace(data,labels),encoding='utf8')
    chart+='<p><a href="search-trace.svg">打开完整八轮源码轨迹图（含失败、退步与回退）</a></p>'
    page=page.replace('<div class="arms">',final_replay_panel(out)+chart+comparison_tables(out)+mechanism_panel(out)+'<div class="arms">',1)
    if a.complete:
        summary=json.loads((out/'results-summary.json').read_bytes())
        assert all(summary['arms'][arm]['completed_sessions']==8 for arm in ['no-simulator','with-simulator'])
        assert len(summary.get('final_replay',{}))==2
        assert all(row['status']=='PASSED' and row['num_passed']==24 and all(c['status']=='passed' for c in row['checks'].values()) for row in summary['final_replay'].values())
        page=page.replace('原生上采样优化对照 · 运行中','原生上采样优化对照 · 完整记录')
        page=page.replace('运行中。共同起点','两组各8轮已完成。共同起点')
        page=page.replace('尚不作两组胜负结论。','最终成绩与完整失败过程同时保留。')
    page=page.replace('</style>','table{border-collapse:collapse;width:100%;margin:20px 0}th,td{text-align:left;border-bottom:1px solid #ddd5c4;padding:10px}svg{max-width:100%;height:auto}</style>')
    (out/'progress.svg').write_text(chart[chart.index('<svg'):chart.index('</svg>')+6],encoding='utf-8')
    destination=out/('details.html' if (out/'mainline-presentation.json').exists() else 'index.html')
    destination.write_text(page,encoding='utf-8');(out/'trace-inventory.json').write_bytes(a.inventory.read_bytes())
    print(destination)


if __name__=='__main__':main()
