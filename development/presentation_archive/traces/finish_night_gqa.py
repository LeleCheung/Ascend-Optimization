"""Publish the accepted GQA continuation from immutable four-case evidence."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .check_presentation import geometry
from .research_progress import write_report
from .selection import remeasurement_indices


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latency(result, case=4, side='candidate'):
    return next(r for r in result['cases'] if r['case_id'] == case)[side + '_timing']['full_call_us']


def main():
    root = Path.cwd()
    reports = root / 'development/reports'
    campaign = reports / 'night-whitebox-gqa-20261120'
    run = campaign / 'gqa'
    historical = root / 'development/experiments/gqa-comparison-20261117'
    old_calls = historical / 'with-simulator-v2/agent-calls'
    baseline = read(historical / 'official-check-01.json')
    round1 = read(historical / 'with-simulator-v2/evaluation-round-01/results.json')
    round2 = read(historical / 'with-simulator-v2/evaluation-round-02/results.json')
    parent = read(run / 'parent-validation/results.json')
    failed = read(run / 'iterations/attempt-001/parent-validation/results.json')
    current = read(run / 'iterations/attempt-002/parent-validation/results.json')
    before = read(run / 'binary-binding-and-facts.json')
    after = read(run / 'iterations/attempt-002/binary-binding-and-facts.json')
    contract = read(run / 'contract.json')
    assert len({r['contract_sha256'] for r in [baseline, round1, round2, parent, failed, current]}) == 1
    assert len({r['harness_sha256'] for r in [baseline, round1, round2, parent, failed, current]}) == 1
    assert current['contract_sha256'] == sha(run / 'contract.json')
    for result in [baseline, round1, round2, parent, current]:
        assert {r['case_id'] for r in result['cases']} == {1, 2, 4, 7}
        for row in result['cases']:
            candidate = row.get('candidate_correctness', row.get('nearest_representable_correctness'))
            assert candidate['official_checker']['passed'] and row['native_correctness']['official_checker']['passed']
    for call in ['attempt-001', 'attempt-002']:
        folder = run / 'agent-calls' / call
        review = read(folder / 'host-review.json')
        assert review['status'] == 'preedit_reads_and_semantics_approved'
        assert review['decision_result_event'] < review['first_source_edit']
        assert review['source_sha256'] == sha(folder / 'output-source.py')
        assert review['parent_source_sha256'] == sha(folder / 'input-source.py')
    chain_sources = [old_calls / 'round-01/input-source.py', old_calls / 'round-01/output-source.py',
                     old_calls / 'round-02/output-source.py', run / 'agent-calls/attempt-001/output-source.py',
                     run / 'agent-calls/attempt-002/output-source.py']
    identities = [sha(p) for p in chain_sources]
    assert identities[:3] == [r['source_sha256'] for r in [baseline, round1, round2]]
    assert identities[2] == sha(run / 'agent-calls/attempt-001/input-source.py')
    assert identities[3] == sha(run / 'agent-calls/attempt-002/input-source.py')
    # The candidate author changes only the launch choice; inspect the actual function AST.
    def kernel_ast(path):
        return ast.dump(next(n for n in ast.parse(path.read_text(encoding='utf8')).body
                             if isinstance(n, ast.FunctionDef) and n.name == '_gqa_fwd_kernel'), include_attributes=False)
    assert len({kernel_ast(p) for p in chain_sources[2:]}) == 1
    assert before['source_sha256'] == identities[2] and after['source_sha256'] == identities[4]
    assert before['status'] == after['status'] == 'formal_profile_simulator_binary_verified'
    assert not before['application_completed'] and not after['application_completed']

    anchor = latency(current, side='native')
    cases = []
    for old, new in zip(parent['cases'], current['cases']):
        assert old['case_id'] == new['case_id']
        a, b = old['candidate_timing']['full_call_us'], new['candidate_timing']['full_call_us']
        cases.append(dict(case_id=old['case_id'], parent_us=a, candidate_us=b,
                          latency_reduction_pct=(1-b/a)*100,
                          cann_us=new['native_timing']['full_call_us'],
                          cann_speedup=new['native_timing']['full_call_us']/b,
                          original_trials_us=old['candidate_timing']['trials_us'],
                          candidate_trials_us=new['candidate_timing']['trials_us']))
    geomean_speedup = __import__('math').prod(r['parent_us']/r['candidate_us'] for r in cases)**.25
    selection = dict(status='retained_complete_four_case_subset', timestamp=datetime.now(timezone.utc).isoformat(),
        invocation='attempt-002', source_sha256=identities[4], retained_parent_source_sha256=identities[2],
        immediate_source_parent_sha256=identities[3], contract_sha256=current['contract_sha256'],
        checker=current['official_checker_identity'], target_case=4,
        historical_retained_us=latency(round2), original_same_source_us=latency(parent),
        target_latency_us=latency(current), target_cann_us=anchor,
        target_cann_speedup=anchor/latency(current),
        target_reduction_vs_historical_pct=(1-latency(current)/latency(round2))*100,
        cases=cases, four_case_latency_geomean_speedup=geomean_speedup,
        four_case_latency_geomean_reduction_pct=(1-1/geomean_speedup)*100,
        selected_layer='triton_host_launch', kernel_ast_unchanged=True,
        modification='For BLOCK_D<=128 use BLOCK_N64 for causal inputs and128 for noncausal inputs; BLOCK_M64 retained.',
        decision_inputs=['paired profiling', 'partial simulator hotspot and pipeline evidence', 'actual TTIR and BCMLIR',
                         'compiler UB overflow feedback for the causal repair'],
        lineage=[dict(parent=a, child=b) for a, b in zip(identities, identities[1:])],
        binary_binding=dict(before='binary-binding-and-facts.json', after='iterations/attempt-002/binary-binding-and-facts.json'),
        abba=dict(performed=False, reason='Three target trials separate by about43%; parent agrees with frozen result. No measurement contradiction.'),
        scope='Fixed official cases1/2/4/7. Full20-case CANNBench qualification remains open. Simulator windows are partial '
              'and separately normalized. Triton launch modification is accepted; no manually edited GQA IR is claimed.')
    save(run / 'selection.json', selection)
    save(run / 'retained-current.json', dict(source_sha256=identities[4], source='agent-calls/attempt-002/output-source.py',
                                          selection='selection.json', target_case=4, target_us=latency(current)))

    def event(label, result, index, qualified=True, method='', location='device', category='sim', **kw):
        value = dict(label=label, source_sha256=identities[index], case='case4',
                     qualified=qualified, score=anchor/latency(result) if qualified else None,
                     latency_ms=latency(result)/1000, score_is_rebased=result is not current,
                     method=method, location=location, category=category)
        value.update(kw)
        return value
    events = [
        event('原始Triton', baseline, 0),
        event('历史round-01：完整通过、耗时上升', round1, 1),
        event('历史round-02：两项概率分解', round2, 2, method='两项概率分解\n减少PV交接',
              attribution='历史模拟器组续接；模型读取及Codex审查反馈共同记录。原始两组对照材料独立保留。'),
        event('当前同源码测量', parent, 2, selected_layer='remeasurement'),
        event('attempt-001：case1 UB溢出', failed, 3, qualified=False, location='host',
              failure='case1 requires231488B while196608B available; three noncausal cases pass.',
              method='非因果KV分块扩大；完整四项待修复'),
        event('attempt-002：因果分流修复、四项通过', current, 4, method='KV分块64→128\n按因果条件分流', location='host',
              selected_layer='triton_host_launch',
              attribution='模拟器热点与流水等待支持减少KV循环交接；profiling和IR关联机制。模型先登记后编辑；'
                          '首版case1编译溢出，依据编译反馈恢复其原分块。核内AST保持，实际修改为宿主分块选择。')]
    paths = ['../../experiments/gqa-comparison-20261117/official-check-01.json',
             '../../experiments/gqa-comparison-20261117/with-simulator-v2/evaluation-round-01/results.json',
             '../../experiments/gqa-comparison-20261117/with-simulator-v2/evaluation-round-02/results.json',
             'night-whitebox-gqa-20261120/gqa/parent-validation/results.json',
             'night-whitebox-gqa-20261120/gqa/iterations/attempt-001/parent-validation/results.json',
             'night-whitebox-gqa-20261120/gqa/iterations/attempt-002/parent-validation/results.json']
    # research_progress's evidence links are relative to the reports directory.
    for e, path in zip(events, paths):
        if path.startswith('../../experiments/'):
            path = '../experiments/' + path[len('../../experiments/'):]
        e['replay'] = dict(result=path)
    data = dict(operator='CANNBench Level4 GQA', metric='fixed_case_cann_speedup', baseline_axis_label='原始Triton',
        compact=True, axis_max=.2, case='case4 · BF16 · 非因果', shape='Q [4,256,64,128] · K/V [4,256,8,128] · G8',
        cann_anchor_us=anchor, contract=contract, events=events,
        optimization_loop='模拟器重放 → 选择Triton／IR层 → 模型修改 → 真机验证 → 再重放',
        source_lineage=selection['lineage'], scope=selection['scope'] + ' 全部历史使用当前case4 CANN锚点换算；'
        '历史时间与设备环境差异保留。隐藏同实现测量、完整通过但下降的探索及编译失败，详情保存实际父链。',
        history_links=[dict(href='../night-whitebox-gqa-20261120/gqa/README.md', label='四项结果、机制与证据边界'),
                       dict(href='../night-whitebox-gqa-20261120/gqa/selection.json', label='保留决定与实际源码父链'),
                       dict(href='../night-whitebox-gqa-20261120/gqa/trace/rounds/attempt-001/round.json', label='首版完整调用记录'),
                       dict(href='../night-whitebox-gqa-20261120/gqa/trace/rounds/attempt-002/round.json', label='修复版完整调用记录')])
    figure = reports / 'gqa-simulator-20261120'
    write_report(data, figure)
    rendered = read(figure / 'mainline-presentation.json')
    assert rendered['presentation_selection']['displayed_indices'] == [0, 2, 5]
    assert not set(rendered['presentation_selection']['displayed_indices']) & remeasurement_indices(events)
    segments = geometry(figure / 'mainline.svg')
    save(campaign / 'gqa-trace-validation.json', dict(native_origin='original Triton', segments=segments,
        displayed=[0, 2, 5], hidden=[1, 3, 4], source_lineage_verified=True,
        continuous=True, monotonic=True, remeasurements_hidden=True, latest_node_location='host',
        latest_method_category='sim', unified_target_cann_us=anchor))

    table = '\n'.join(f"| {r['case_id']} | {r['parent_us']:.5f} | {r['candidate_us']:.5f} | "
                      f"{r['latency_reduction_pct']:.2f}% | {r['cann_us']:.5f} | {r['cann_speedup']:.6f}× |"
                      for r in cases)
    note = f'''# GQA：减少KV循环交接的完整四项案例

目标case4：**{latency(round2):.5f} → {latency(current):.5f} μs，耗时下降{selection['target_reduction_vs_historical_pct']:.2f}%**。
BF16非因果，Q `[4,256,64,128]`、K/V `[4,256,8,128]`、G8。新版本相对本轮CANN为{selection['target_cann_speedup']:.6f}×。
固定官方case1/2/4/7全部通过；完整20项验收仍待后续开展。

| case | 本轮原保留源码 μs | 新源码 μs | 耗时下降 | 本轮CANN μs | 相对CANN |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

四项候选耗时几何平均下降{selection['four_case_latency_geomean_reduction_pct']:.2f}%，几何加速{geomean_speedup:.6f}倍。
两边合同、harness、官方checker完全相同：warmup5、trials3、iterations20。目标三次原版为
{cases[2]['original_trials_us']}μs，新版为{cases[2]['candidate_trials_us']}μs；本轮原版1378.74995μs与历史成绩接近。
当前差距明确，按条件策略省略新增ABBA。

修改前模拟器的热点为WAIT_FLAG_DEVI与FIX_L0C_TO_DST，Cube窗口productive4.74%、FLOWCTRL等待89.79%；
Vector0窗口productive35.98%、等待29.87%。结合实际TTIR与BCMLIR，模型选择减少KV块循环的Cube／Vector交接。
`BLOCK_M=64`保持，D≤128的非因果输入将`BLOCK_N=64→128`，case4的KV轮数4→2。
实际改动为宿主分块选择，kernel函数AST及两项概率分解表达式保持；在线softmax归约分组变化由官方checker验证。

首版全部D≤128都用BN128，case1编译需要231488B UB、可用196608B，超出34.0625KiB。
修复版按真实`is_causal`恢复因果BN64、保留非因果BN128。模型对资源峰值的其他估计作为假设保留。
两次调用均成功读取相应证据，并在编辑之前登记决策；[首版审查](agent-calls/attempt-001/host-review.json)、
[修复审查](agent-calls/attempt-002/host-review.json)、diff、完整反馈与真实父链均已保存。

修改后profiling：kernel1368.927→791.996μs；每program MTE1指令45→23、MTE2指令18→10，
Vector0服务9.344→6.780μs、Scalar等待5.809→4.366μs，L0C→GM320→192KiB，GM→UB320→192KiB。
该数据支持减少交接与中间数据搬运。硬件服务时间、单kernel耗时和正式完整调用耗时分别记录。

前后三核重放时钟均核验通过：原版19/19、41/41、38/38，新版19/19、43/43、40/40个PC。
新版Cube窗口productive5.19%、FLOWCTRL等待83.51%；Vector0窗口productive29.17%、等待22.87%。
各窗口分别归一化，VECTOR的可观测空闲从34.15%增至47.96%，其他等待与占比变化也同时保留。
两次应用均未完整结束；部分窗口比例用于解释局部执行，完整收益采用真机四项结果。
每个版本的正式评测cache、硬件profiling与模拟器cache中目标npubin字节一致，见
[修改前绑定](binary-binding-and-facts.json)、[修改后绑定](iterations/attempt-002/binary-binding-and-facts.json)。
GQA实际修改位于Triton宿主侧，IR用于机制分析；后重放持续执行。

[连续最佳曲线](../../gqa-simulator-20261120/index.html) · [保留决定](selection.json) ·
[完整四项结果](iterations/attempt-002/parent-validation/results.json) · [完整源码／决策trace](trace/rounds/attempt-002/round.json)

原始大trace继续保存在远端`/data/hanle/ascend-optimization/runs/night-whitebox-20261120/gqa/`及
`iterations/attempt-002/simulator-parent/`。本地保存源码、实际IR、目标二进制、完整结果、profile表与重放摘要。
原始模型结果回读及[宿主纠正](../model-result-review-host-note.md)共同保留。
'''
    (run / 'README.md').write_text(note, encoding='utf8')
    append_slides(reports, campaign, selection, figure)
    print(json.dumps(dict(target_us=latency(current), reduction_pct=selection['target_reduction_vs_historical_pct'],
                          four_case_geomean_reduction_pct=selection['four_case_latency_geomean_reduction_pct'],
                          figure=str(figure), stages='statistics complete; IR2/2; GQA4/4'), ensure_ascii=False))


def append_slides(reports, campaign, selection, figure):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches
    from .meeting_slides import BG, GREEN, text
    source = reports / 'group-meeting-20261119/whitebox-simulator-ascend-20261120-night.pptx'
    destination = source.with_name('whitebox-simulator-ascend-20261120-night-final.pptx')
    p = Presentation(source)
    assert len(p.slides) == 26
    graph = p.slides.add_slide(p.slide_layouts[6])
    graph.background.fill.solid()
    graph.background.fill.fore_color.rgb = RGBColor.from_string(BG)
    graph.shapes.add_picture(str(figure / 'mainline.png'), Inches(.35), Inches(.15), width=Inches(12.6))
    text(graph, '固定官方case1／2／4／7通过；目标case4；当前CANN统一锚点；完整20项待验证',
         .6, 7.0, 12.1, .35, 14)
    graph.notes_slide.notes_text_frame.text = ('原始Triton起点，包含历史两轮及本轮两次候选的完整父链。'
        '同实现测量、完整通过但退步的探索和编译失败保留详情。主图最新方形节点表示宿主分块选择。'
        + selection['scope'])
    detail = p.slides.add_slide(p.slide_layouts[6])
    detail.background.fill.solid()
    detail.background.fill.fore_color.rgb = RGBColor.from_string(BG)
    text(detail, 'GQA：减少KV循环中的Cube／Vector交接', .55, .35, 12.1, .8, 31, bold=True)
    text(detail, f"1378.737 → 784.638 μs · 下降{selection['target_reduction_vs_historical_pct']:.2f}%",
         .7, 1.45, 11.9, .75, 32, GREEN, True)
    text(detail, '模拟器：WAIT_FLAG与FIXP热点，关联双dot的流水交接\n'
         '改Triton启动分块：非因果KV 64→128；因果保留64\n'
         '硬件每程序：MTE1指令45→23；Vector服务9.344→6.780μs\n'
         '首版因果UB超限；修复后固定四项官方checker全部通过\n'
         '持续重放新产物：三核时钟通过，部分窗口分别解释',
         .8, 2.6, 11.6, 3.65, 23)
    text(detail, '本例实际修改为Triton宿主分块；IR用于解释；完整20项验收留作后续', .75, 6.55, 11.8, .5, 17)
    detail.notes_slide.notes_text_frame.text = ('前后模拟器均为部分窗口，应用未完整结束。'
        '原19/19、41/41、38/38；新19/19、43/43、40/40。'
        'Vector窗口productive35.98→29.17%，wait29.87→22.87%，idle34.15→47.96%，不按单项比例外推全调用。'
        '四项耗时几何平均下降' + f"{selection['four_case_latency_geomean_reduction_pct']:.2f}%。"
        '首版UB231488B，可用196608B；超34.0625KiB。'
        '完整证据：../night-whitebox-gqa-20261120/gqa/README.md。')
    for number in [27, 28]:
        text(p.slides[number-1], str(number), 12.3, 7.12, .5, .25, 11)
    for slide in p.slides:
        for shape in slide.shapes:
            assert shape.left >= 0 and shape.top >= 0
            assert shape.left + shape.width <= p.slide_width + Inches(.01)
            assert shape.top + shape.height <= p.slide_height + Inches(.01)
    p.save(destination)
    save(campaign / 'completion-slides.json', dict(slides=len(p.slides), file=str(destination.relative_to(Path.cwd())),
        preserved_26_slide_source=str(source.relative_to(Path.cwd())), added=[27,28], bounds_passed=True,
        native_render_passed=False, native_visual_review_passed=False))


if __name__ == '__main__':
    main()
