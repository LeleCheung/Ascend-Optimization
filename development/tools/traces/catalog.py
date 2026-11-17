"""从既有交付与冻结账本摘要生成26项图册；缺失轨迹不补造轮次。"""
import argparse
import html
import json
import math
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .importer import digest, read
from .index import save
from .render import draw, CSS


# 展示选线及简短中文方法。原始计划、来源和成绩仍逐条保留在catalog.json。
SELECTIONS = {
 'selu_':('c3e781', [('直接启动精简宿主路径','host','probe'),('固定网格流式循环','device','probe')]),
 'rad2deg_':('b20dea', [('预绑定编译后启动器','host','probe'),('按规模调整分块','device','probe'),('保留输入精度计算','device','ir')]),
 'uniform':('20d459', [('向量化 Philox 与状态访问','mixed','probe'),('缓存编译后直接启动','host','paired')]),
 'value_selecting_reduction_backward':('d9428d', [('固定网格流式回填梯度','mixed','unknown')]),
 'scatter_add':('42950a', [('排序分段归约','device','knowledge'),('将查找移至片上','device','probe')]),
 'upsample_bilinear2d':('c34839', [('错位全局读取改为片上 gather','device','knowledge'),('缓存形状与启动配置','host','knowledge'),('去除冗余掩码：未提升','device','probe')]),
 '_conj_physical':('62faff',[('预绑定编译后启动器','host','probe'),('简化输出分配路径','host','probe'),('增加安全回退：退步','host','unknown')]),
 'negative_':('033b6a',[('直接启动已编译 kernel','host','probe'),('每个 program 处理64KB','device','paired')]),
 'xlogy_scalar_tensor':('016df1',[('缓存启动计划与输出分配','host','probe')]),
 'nan_to_num_':('0af95a',[('简化启动参数','host','probe'),('区分大小输入路径','mixed','probe'),('单阶段多块循环','device','probe'),('原精度计算尝试：未提升','device','ir')]),
 'cross_out':('023367',[('重组索引与邻居访问','device','probe'),('扩大可驻留分块','device','probe'),('简化启动与修复边界','mixed','probe'),('扩大32位数据分块','device','ir')]),
 'hardshrink_backward':('e0a716',[('固定网格无掩码循环','device','probe'),('简化启动与单块特化','mixed','probe'),('只让尾块执行边界处理','device','probe')]),
 'subtract_':('dcd20c',[('直接启动已编译 kernel','host','probe')]),
 'native_group_norm':('092021',[('缓存启动与合并分配','host','unknown'),('缩减启动参数：仍退步','host','probe')]),
 'log2_':('ddeb4e',[]),
 'fix':('a2d5ed',[('固定网格多块循环：退步','device','probe'),('调整流水与启动回退','mixed','probe'),('修正启动参数数量','host','ir')]),
 'round_out':('009743',[('流式循环与舍入简化','device','probe'),('双阶段流水尝试','device','probe'),('扩大16位分块：未提升','device','probe')]),
 'special_round_out':('cec136',[('缓存直接启动路径','host','probe'),('简化 FP32 舍入','device','ir'),('16位舍入尝试：失败','device','ir')]),
 'special_xlogy_':('a754ce',[('缓存网格与启动参数','host','probe'),('精简宿主查找与查询','host','probe')]),
 'hardswish':('1c3071',[('连续块去除掩码','device','probe'),('缩减小输入启动开销','host','paired'),('原精度向量运算：退步','device','ir')]),
 'fmod_tensor_':('aca31f',[('精简宿主元数据处理','host','probe')]),
}


def make_catalog(root, output):
    delivery = read(root/'operators/index.json')
    matrix_path = root/'development/reports/trace-attribution-matrix-20261116.json'
    matrix = {r['operator']:r for r in read(matrix_path)['rows']}
    accepted = [r for r in delivery['rows'] if r['scope']=='main_30' and r['accepted']]
    rows=[]
    for selected in accepted:
        op=selected['operator']; archived=matrix[op]
        if op=='hardswish_out':
            bases=[root/'development/experiments/regressions/20261109'/batch/'operators'/op
                   for batch in ('fixed-25','hardswish-native-low-precision')]
            requests=[read(b/'evaluation/request.json') for b in bases]
            assert requests[0]['binding']==requests[1]['binding']
            assert requests[0].get('settings')==requests[1].get('settings')
            reviews=[read(b/'review.json') for b in bases]
            assert reviews[1]['parent_source_sha256']==reviews[0]['source_sha256']
            points=[]
            for n,(base,req,review) in enumerate(zip(bases,requests,reviews)):
                source=(base/'candidate.py').read_bytes()
                assert digest(source)==review['source_sha256']
                assert req['implementation']['sources'][0]['content'].encode('utf8')==source
                result=read(base/'evaluation/result.json')
                points.append(dict(id=op+':repair-'+str(n),score=result['geo_mean'],source_sha256=digest(source),
                    method='修复前' if n==0 else '调整低精度计算与分块',
                    result=(base/'evaluation/result.json').relative_to(root).as_posix(),review=review))
            view=dict(id='mainline',session=op,title=op+' 优化轨迹',subtitle='相对 CANN · 同合同修复前后',ylim=[0,1.1],
                annotations=[dict(point=2,method='调整低精度计算与分块',color='ir',position=[750,285],arrow=[1200,325])],
                nodes=[dict(point=1,location='mixed'),dict(point=2,location='device')],
                line_status='下图：交付版本修复过程')
            record=dict(operator=op,kind='verified_delivery_repair',points=points,view=view,
                        history=archived['historical_rounds'],gap=None,selected_source=[str(b.relative_to(root)) for b in bases])
        elif op=='log10_':
            pilot=read(root/'development/manifests/traces/log10_/index.json')
            view=next(v for v in pilot['views'] if v['id']=='mainline')
            points=[p for p in pilot['evaluations'] if p['session_id']==view['session']]
            svg=draw(pilot,view)
            record=dict(operator=op,kind='recovered_continuous_trace',selected_source='development/manifests/traces/log10_/index.json',
                points=points,view=view,history=archived['historical_rounds'],gap=None)
        elif op in SELECTIONS:
            prefix,labels=SELECTIONS[op]
            points_raw=[r for r in archived['historical_rounds'] if r['ledger_sha256'].startswith(prefix)]
            assert len(points_raw)==len(labels)+1,(op,len(points_raw),len(labels))
            points=[]
            for n,p in enumerate(points_raw):
                points.append(dict(id=op+':round-'+str(p['round']),score=p['evaluation'].get('geo_mean'),
                    source_sha256=p.get('source_sha256'),method='本批起点' if n==0 else labels[n-1][0],raw=p))
            high=max(p['score'] or 0 for p in points)*1.16
            high=max(.5,math.ceil(high*10)/10)
            annotations=[]
            # 最多4个方法箭头；位置来自版式模板，不取代原始方法记录。
            for i,label in enumerate(labels[:4],2):
                placements={1:([220,285],[540,325]),2:([930,285],[1210,325]),3:([220,775],[540,730]),4:([930,775],[1200,730])}
                if len(labels)==1:position,arrow=[550,285],[950,325]
                else:position,arrow=placements[i-1]
                annotations.append(dict(point=i,method=label[0],color=label[2],position=position,arrow=arrow))
            legend=['paired','sim','knowledge','probe','ir']
            if any(a['color']=='unknown' for a in annotations):legend=['paired','sim','knowledge','probe','unknown']
            gap='仅找到1次正式测量，优化过程记录待补' if len(points)==1 else None
            view=dict(id='mainline',session=op,title=op+' 优化轨迹',subtitle='历史 KGS 参考 · 本批 '+str(len(points))+' 次正式评测',
                ylim=[0,high],annotations=annotations,legend=legend,gap=gap,x_label='本批正式评测顺序',plot_bottom=650 if len(labels)>2 else 745,
                nodes=[dict(point=1,location='mixed')]+[dict(point=i,location=label[1]) for i,label in enumerate(labels,2)])
            if op=='_conj_physical':view['subtitle']='历史参考可能含 CPU 回退 · 本批 '+str(len(points))+' 次正式评测'
            svg=draw(dict(evaluations=[dict(p,session_id=op) for p in points],sessions=[dict(id=op,counts={})]),view)
            ledger=next(l for l in archived['ledger_files'] if l['sha256'].startswith(prefix))
            record=dict(operator=op,kind='saved_ledger_attempt_order',selected_source=ledger,points=points,view=view,
                history=archived['historical_rounds'],gap=gap,
                relation_note='按同一原始账本正式轮次展示；未逐编辑恢复的关系不声称直接源码继承。')
        else:
            result=read(root/'operators'/op/'evidence/retained/evaluation/result.json')
            points=[dict(id=op+':retained-evaluation',score=result['geo_mean'],method='已交付源码回归',source_sha256=archived['delivery_source_sha256'])]
            gap='已保存交付结果；原优化轮次记录待补'
            view=dict(id='mainline',session=op,title=op+' 记录缺口',subtitle='相对 CANN · 当前保留版测量',
                ylim=[0,max(1.,math.ceil(result['geo_mean']*1.3*10)/10)],annotations=[],gap=gap,
                x_label='保留版测量（不冒充优化过程）',nodes=[dict(point=1,location='mixed')])
            svg=draw(dict(evaluations=[dict(p,session_id=op) for p in points],sessions=[dict(id=op,counts={})]),view)
            record=dict(operator=op,kind='delivery_snapshot_only',points=points,view=view,history=[],gap=gap,
                selected_source='operators/'+op+'/evidence/retained/evaluation/result.json')
        manifest=read(root/selected['manifest'] if selected['manifest'].startswith('operators/') else root/'operators'/selected['manifest'])
        source=root/manifest['selected']['path']
        assert digest(source.read_bytes())==manifest['selected']['sha256']
        result_path=root/'operators'/op/'evidence/retained/evaluation/result.json'
        result=read(result_path)
        assert result['status']=='PASSED' and result['num_passed']==result['num_workloads']
        assert abs(result['geo_mean']-selected['cann_score'])<1e-10 and selected['cann_score']>=.8
        record['delivery']=dict(score=selected['cann_score'],source=manifest['selected']['path'],
            source_sha256=manifest['selected']['sha256'],passed=result['num_passed'],total=result['num_workloads'],
            result=result_path.relative_to(root).as_posix(),result_sha256=digest(result_path.read_bytes()),
            endpoint_same_source=points[-1].get('source_sha256')==manifest['selected']['sha256'])
        view=record['view']
        view['delivery']=record['delivery']
        view['result_only']=len(points)==1
        if view['result_only']:view['title']=op+' 交付结果'
        elif not view.get('line_status'):
            view['line_status']='下图：独立重生成实验' if op=='log10_' else '下图：历史优化片段'
        svg=draw(dict(evaluations=[dict(p,session_id=view['session']) for p in points],sessions=[dict(id=view['session'],counts={})]),view)
        directory=output/op;directory.mkdir(parents=True,exist_ok=True)
        (directory/'mainline.svg').write_text(svg,encoding='utf8')
        record.update(current_cann_score=selected['cann_score'],historical_simulator_files=len(archived['simulator_files']))
        if op=='_conj_physical':
            record['reference_limitation']='历史参考路径可能包含CPU回退；该批巨大比值不代表相对原生CANN的加速，不能与当前CANN分数比较。'
        save(directory/'trace.json',record)
        details=html.escape(json.dumps(record,ensure_ascii=False,indent=2))
        (directory/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><style>'+CSS+'</style><main>'
            '<p><a href="../index.html">返回26项总览</a>　<a href="mainline.png">PNG</a>　<a href="mainline.svg">SVG</a>　<a href="trace.json">轨迹数据</a></p>'
            +'<figure>'+svg+'</figure><details><summary>评测口径与图例说明</summary><p>'+html.escape(record['view']['subtitle'])+'</p><p>纵轴为该实验参考实现耗时÷候选耗时，沿用原始KGS统计口径。横轴只编号本图展示的已评测版本，不是全部模型调用次数。Profiling包含成对采集、宿主剖析和运行探测；代码分析表示记录中直接引用源码或实际编译IR，不能据此推断整轮未使用其他工具。颜色保留主要记录类别，不证明独立因果贡献。</p></details>'
            +'<p><a href="../../../../'+html.escape(record['delivery']['source'])+'">当前交付源码</a> · <a href="../../../../'+html.escape(record['delivery']['result'])+'">交付评测原件</a></p>'
            +('<p class="note">'+html.escape(record['gap'])+'</p>' if record['gap'] else '')
            +('<p class="note">'+html.escape(record['reference_limitation'])+'</p>' if record.get('reference_limitation') else '')
            +'<details><summary>选线、全部已收集历史和原始计划</summary><pre>'+details+'</pre></details></main></html>',encoding='utf8')
        rows.append(record)
    assert len(rows)==26
    result=dict(schema='hgkda.trace-catalog/v1',operators=rows,source=dict(path=matrix_path.relative_to(root).as_posix(),sha256=digest(matrix_path.read_bytes())),
        counts=dict(total=26,multinode=sum(len(r['points'])>1 for r in rows),single_ledger=sum(r['kind']=='saved_ledger_attempt_order' and len(r['points'])==1 for r in rows),snapshot_only=sum(r['kind']=='delivery_snapshot_only' for r in rows)))
    save(output/'catalog.json',result)
    cards=[]
    for r in rows:
        op=r['operator']
        cards.append('<article><h2><a href="'+op+'/index.html">'+html.escape(op)+'</a></h2><a href="'+op+'/index.html"><img loading="lazy" src="'+op+'/mainline.svg"></a><p>'
            +('优化过程待补' if r['gap'] else '<a href="'+op+'/index.html">查看详细记录</a>')+'</p></article>')
    (output/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>26项算子优化轨迹</title><style>'+CSS+
        'section{display:grid;grid-template-columns:repeat(auto-fit,minmax(540px,1fr));gap:24px}article{border:1px solid #cec7b8;padding:12px;border-radius:8px}img{width:100%}article h2{font-size:24px;overflow-wrap:anywhere}</style><main><h1>26项算子优化轨迹</h1><p>每项展示一条主要优化过程，点击查看完整记录。</p><p><a href="catalog.json">轨迹数据</a>　<a href="../operator-traces-history-20261116/index.html">更多历史轨迹</a></p><section>'+''.join(cards)+'</section></main></html>',encoding='utf8')
    notes=['# 26项优化轨迹图册','',
           '[打开图册](index.html) · [完整数据索引](catalog.json) · [版式及链接检查](presentation-check.json)','',
           '22项有多个展示节点，4项以交付结果卡展示。hardswish_out已从回归／修复目录恢复同合同两点；log2_保留早期单点在数据中，图面展示当前交付结果。','',
           '每图选择同一原始账本的一条主要尝试线；log10使用已恢复的连续主线。账本尝试顺序不等于逐次源码直接继承。不同参考、合同、起点和批次不拼接。每项HTML展开区及trace.json保存全部已收集历史与原计划；主线节点少不代表该算子总共只优化了这些轮次。','',
           '箭头颜色表示记录中的判断依据类别，节点形状表示修改位置。来源不明保留待核验；历史归类、模型声明和工具共现不能证明独立因果。缺失轨迹及历史参考限制保留在对应页面。','',
           '| 算子 | 主线节点 | 已收集历史记录 | 范围 |','| --- | ---: | ---: | --- |']
    for row in rows:
        notes.append('| ['+row['operator']+']('+row['operator']+'/index.html) | '+str(len(row['points']))+' | '+str(len(row['history']))+' | '+(row.get('gap') or '已保留选线与历史来源')+' |')
    notes+=['','## 重建','',
            '```powershell','python -m development.tools.traces.catalog --output development/reports/operator-traces-26-20261116 --png','```','',
            'SVG／JSON仅需Python标准库；PNG及版式检查需要可选Playwright和Edge。图例缩小为24像素置于右上角，仅显示本图使用的类别；原始参考口径与记录数量保留在详情及JSON。Profiling合并展示成对采集、宿主剖析和运行探测，原始分类不改写。方法标注不相互重叠，曲线不用阶梯转角。','']
    (output/'README.md').write_text('\n'.join(notes),encoding='utf8')
    print(json.dumps(result['counts']))


def export_and_check(output):
    """导出可截图的26张PNG，检查文本边界、方法标注重叠及本地链接。"""
    from playwright.sync_api import sync_playwright
    checks=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        try:
            page=browser.new_page(viewport=dict(width=1600,height=900))
            for path in sorted(output.glob('*/mainline.svg')):
                page.goto(path.as_uri());page.screenshot(path=str(path.with_suffix('.png')))
                issues=page.locator('svg').evaluate('''svg=>{
                  const bad=[...svg.querySelectorAll('text')].filter(e=>{const b=e.getBBox();return b.x<0||b.x+b.width>1601||b.y<0||b.y+b.height>901}).map(e=>'overflow: '+e.textContent);
                  const texts=[...svg.querySelectorAll('.method-annotation text')];
                  for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){
                    const a=texts[i].getBBox(),b=texts[j].getBBox();
                    if(a.x<b.x+b.width&&b.x<a.x+a.width&&a.y<b.y+b.height&&b.y<a.y+a.height)bad.push('method overlap: '+texts[i].textContent+' / '+texts[j].textContent)
                  }return bad}''')
                checks.append(dict(operator=path.parent.name,issues=issues))
        finally:browser.close()
    missing=[]
    for doc in output.rglob('*.html'):
        for target in re.findall(r'(?:href|src)="([^"]+)"',doc.read_text(encoding='utf8')):
            url=urlsplit(target)
            if not url.scheme and url.path and not (doc.parent/unquote(url.path)).resolve().exists():
                missing.append(dict(file=doc.relative_to(output).as_posix(),target=target))
    result=dict(operators=len(checks),pngs=len(list(output.glob('*/mainline.png'))),
                issues=[r for r in checks if r['issues']],missing_links=missing)
    save(output/'presentation-check.json',result)
    assert len(checks)==26 and not result['issues'] and not missing,result
    print(json.dumps(result,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True)
    p.add_argument('--png',action='store_true',help='使用可选Playwright/Edge导出并验证PNG')
    a=p.parse_args();make_catalog(a.root.resolve(),a.output.resolve())
    if a.png:export_and_check(a.output.resolve())


if __name__=='__main__':main()
