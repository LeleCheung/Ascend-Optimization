"""从 GQA 已验收主线的四用例原始测量构建几何平均进展图。"""
import copy
import hashlib
import json
import math

from .gqa_meeting_views import OUT, ROOT, chart, page

CASE_IDS = (1, 2, 4, 7)


def geomean(values):
    assert all(v > 0 for v in values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main():
    source = json.loads((OUT/'trace-data.json').read_text(encoding='utf-8'))
    nodes = copy.deepcopy(source['nodes'])
    sources, documents = [], []
    for node in nodes:
        path = (ROOT/'development/reports'/node['replay']['result']).resolve()
        raw = path.read_bytes()
        doc = json.loads(raw)
        cases = {c['case_id']: c for c in doc['cases']}
        assert set(cases) == set(CASE_IDS)
        assert doc['source_sha256'] == node['source_sha256']
        node['case_results'] = []
        for case_id in CASE_IDS:
            c = cases[case_id]
            passed = c['candidate_correctness']['official_checker']['passed']
            assert passed and c['status'] == 'measured'
            candidate = c['candidate_timing']['full_call_us']
            native = c['native_timing']['full_call_us']
            assert candidate > 0 and native > 0
            node['case_results'].append(dict(case_id=case_id, candidate_us=candidate, cann_us=native,
                                            paired_speedup=native/candidate, passed=passed,
                                            shape=c['shape'], dtype=c['dtype'], attrs=c['attrs']))
        sources.append(dict(progress_index=node['progress_index'], path=str(path.relative_to(ROOT)),
                            sha256=hashlib.sha256(raw).hexdigest()))
        documents.append(doc)
    references = {c['case_id']:c['cann_us'] for c in nodes[-1]['case_results']}
    initial = {c['case_id']:c['candidate_us'] for c in nodes[0]['case_results']}
    for node in nodes:
        cases = node['case_results']
        node['case4_original_score'] = node['score']
        node['case4_original_latency_ms'] = node['latency_ms']
        node['score'] = geomean([references[c['case_id']]/c['candidate_us'] for c in cases])
        node['latency_ms'] = geomean([c['candidate_us'] for c in cases])/1000
        node['metric_name'] = '四用例耗时几何平均'
        node['paired_cann_geomean'] = geomean([c['paired_speedup'] for c in cases])
        node['speedup_vs_original'] = geomean([initial[c['case_id']]/c['candidate_us'] for c in cases])
    selected = [0]
    for i,node in enumerate(nodes[1:],1):
        if node['score'] > nodes[selected[-1]]['score']:
            selected.append(i)
    assert selected[-1] == 23
    assert all(nodes[b]['score'] > nodes[a]['score'] for a,b in zip(selected,selected[1:]))
    assert abs(nodes[-1]['score']-nodes[-1]['paired_cann_geomean']) < 1e-12
    for doc in documents:
        # 验证原始用例语义与 checker 沿主线一致。
        assert doc['golden_sha256'] == documents[0]['golden_sha256']
        assert doc['official_checker_identity']['sha256'] == documents[0]['official_checker_identity']['sha256']
        for case,base in zip(sorted(doc['cases'],key=lambda c:c['case_id']),
                             sorted(documents[0]['cases'],key=lambda c:c['case_id'])):
            for key in ['case_id','shape','dtype','attrs']:
                assert case[key] == base[key]
    comparison = []
    for c in nodes[-1]['case_results']:
        comparison.append(dict(case_id=c['case_id'], initial_us=initial[c['case_id']], final_us=c['candidate_us'],
                               speedup_vs_original=initial[c['case_id']]/c['candidate_us'],
                               final_cann_speedup=c['paired_speedup']))
    other_speedup=geomean([r['speedup_vs_original'] for r in comparison if r['case_id']!=4])
    data = dict(nodes=nodes, stages=source['stages'], colors=source['colors'],
                color_labels=source['color_labels'], original=dict(mainline=source['original'], results=documents),
                sources=sources, case_ids=CASE_IDS, fixed_cann_reference_us=references,
                displayed_indices=selected, detail_indices=sorted(set(range(24))-set(selected)),
                final_comparison=comparison, final_geomean_vs_cann=nodes[-1]['score'],
                final_geomean_vs_original=nodes[-1]['speedup_vs_original'],
                other_three_geomean_vs_original=other_speedup,
                scope='沿已验收case4主线24个版本，重算case1/2/4/7完整调用加速比的等权几何平均。',
                selection='仅绘制四用例几何平均推进历史最佳的真实版本，其余逐轮成绩保存在详情；没有拼接不同版本的单case最好成绩。',
                interpretation='四用例从优化过程中共同参与验证；不能视为冻结workflow后对未见用例的独立迁移实验。')
    (OUT/'geomean-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    svg = chart(nodes, summary=dict(title='GQA 四用例几何平均轨迹', headline=f'{nodes[-1]["score"]:.3f}× CANN',
                                   displayed_indices=selected, focus_indices=[6,12,13,18,23],
                                   label_positions={12:(990,550,'end')}))
    (OUT/'geomean.svg').write_text(svg,encoding='utf-8')
    (OUT/'geomean.html').write_text(page(svg,'geomean',data),encoding='utf-8')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        tab = browser.new_page(viewport=dict(width=1920,height=1200))
        errors=[]
        tab.on('pageerror',lambda e:errors.append(str(e)))
        tab.goto((OUT/'geomean.html').as_uri())
        tab.evaluate('document.fonts.ready')
        tab.locator('svg').screenshot(path=str(OUT/'geomean.png'))
        tab.locator('.baseline-node').click()
        assert 'case 4' in tab.locator('#info').inner_text()
        tab.locator('.node').last.click()
        assert '四用例耗时几何平均' in tab.locator('#info').inner_text()
        assert 'case 7' in tab.locator('#info').inner_text()
        overflow=tab.locator('svg').evaluate('(s)=>[...s.querySelectorAll("text")].filter(t=>{let b=t.getBBox();return b.x<0||b.y<0||b.x+b.width>1920||b.y+b.height>1080}).map(t=>t.textContent)')
        assert not errors and not overflow
        browser.close()
    # 确认只读来源没有改变。
    assert all(hashlib.sha256((ROOT/s['path']).read_bytes()).hexdigest()==s['sha256'] for s in sources)
    (OUT/'geomean-validation.json').write_text(json.dumps(dict(status='passed',cases=list(CASE_IDS),
        source_versions=len(nodes),displayed_indices=selected,correctness='96/96 case-version checks passed',
        identical_case_definitions=True,identical_checker_and_golden=True,sources_unchanged=True,
        latest_is_best_in_this_line=True,final_geomean_vs_cann=nodes[-1]['score'],
        browser_errors=errors,text_overflow=overflow),ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# GQA 四用例成绩与汇报口径','',
           '[四用例几何平均 HTML](geomean.html) · [PNG](geomean.png) · [SVG](geomean.svg)','',
           '主线以case4为重点研究对象，每个完整候选同时评测case1/2/4/7；这些用例从优化过程中已参与验证。',
           '方法包括目标shape特化和共用host路径改善，不能把当前结果描述为先完成case4后，再向未参与优化的新用例迁移。','',
           '| 用例 | 原始完整调用 μs | 最终完整调用 μs | 相对原始加速 | 最终同轮CANN比值 |',
           '|---|---:|---:|---:|---:|']
    for c in comparison:
        lines.append(f'| case {c["case_id"]} | {c["initial_us"]:.5f} | {c["final_us"]:.5f} | {c["speedup_vs_original"]:.4f}× | {c["final_cann_speedup"]:.4f}× |')
    lines += ['',f'四用例相对原始版本加速比的等权几何平均：**{nodes[-1]["speedup_vs_original"]:.6f}×**。',
              f'四用例相对最终同轮CANN的等权几何平均：**{nodes[-1]["score"]:.6f}×**。',
              f'除case4外，其余三个用例相对原始版本的几何平均：**{other_speedup:.6f}×**。','',
              '图中对每个case固定最终同轮CANN耗时为参考，再计算四个比值的几何平均。曲线选择沿已展示主线取得几何平均新最佳的版本，保留原0–23索引；24版本逐case原始测量及历史同轮参考均存入数据文件。没有把不同候选的单case最好成绩拼起来。','',
              '重点节点13的host编译产物与元数据缓存使三个较小用例同步改善。其他节点存在不同程度的回退与测量变化，原记录保留。','',
              '可用于汇报的表述：以case4深入展示热点、流水与IR优化过程，同时在四个不同配置上持续验证；最终四用例相对原始版本几何平均加速3.44倍，相对同轮CANN达到1.036倍。后续将扩展至完整20项，并单独评价新增用例上的方法迁移效果。','',
              '## 起点与shape','',
              '空心圈为原始Triton基线，索引0不是一次优化实验；索引1为第一次有效改进。原始四用例官方checker均通过。case4原始完整调用1900.72105μs；按最终CANN参考127.9771μs换算为0.067330816×。','',
              'case4：Q [4,256,64,128]；K/V各[4,256,8,128]。布局维度为batch、sequence、heads、head_dim；64个Q头、8个KV头，每组8个Q头；BF16、非因果，默认scaleValue=-1.0。','',
              '## 再生成','',
              '```powershell','python -m development.tools.traces.gqa_meeting_views','python -m development.tools.traces.gqa_geomean_view','```']
    (OUT/'four-case-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(displayed_indices=selected,final_geomean_vs_cann=nodes[-1]['score'],
                          final_geomean_vs_original=nodes[-1]['speedup_vs_original'],other_three=other_speedup)))


if __name__ == '__main__':
    main()
