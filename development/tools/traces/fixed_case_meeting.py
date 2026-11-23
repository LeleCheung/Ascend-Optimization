"""Build the four source-bound fixed-input figures and meeting webpage."""
import argparse
import copy
import html
import hashlib
import json
from pathlib import Path
from .research_progress import milestones, render


def read(path):
    return json.loads(path.read_bytes())


CONFIG = [
    ('batchnorm', 'BatchNorm', 'batchnorm-simulator-20261118',
     'batch_norm_with_update_functional-timing-7', 'FP32 · NCHW [8,64,64,64]',
     '循环等待 → 提前加载下一块 → 真机验证等待下降'),
    ('upsample', 'upsample_bilinear2d', 'native-upsample-contrast-20261118',
     'upsample_bilinear2d-timing-1', 'FP16 · [4,32,128,128] → [4,32,256,256]',
     '交织热点 → 打包写回与低位掩码 → 向量工作减少'),
    ('qkv', 'QKV', 'qkv-simulator-20261118',
     'transform_bias_rescale_qkv-timing-2', 'FP16 · [8,512,3072] · 16 heads',
     '标量除模热点 → 二维仿射分块 → 调整分块与流水'),
    ('scatter', 'scatter_add', 'scatter-simulator-20261118',
     'scatter_add-timing-8', 'FP32 · [2048,2048] · dim=-1',
     '扫描原语退步 → 模拟器和IR检查 → 精简索引与类型转换'),
]


def fixed_curve(base, name, title, folder, case, shape, story):
    data = read(base / folder / 'mainline-presentation.json')
    events = copy.deepcopy(data['events'])
    contract = base.parents[1] / 'operators' / data['operator'] / 'contracts/timing.jsonl'
    case_spec = next(row for row in [json.loads(line) for line in contract.read_text(encoding='utf8').splitlines() if line.strip()]
        if row['name'] == case)
    if name == 'upsample':
        inventory = read(base / folder / 'trace-inventory.json')
        events[0]['source_sha256'] = inventory['arms']['with-simulator'][0]['input_sha256']
        timings = {}
        for row in inventory['arms']['with-simulator']:
            target = next((t for t in row.get('timings', []) if t['uuid'] == case), None)
            if target:
                timings[row['output_sha256']] = target
        latest = read(base / 'upsample-performance-20261118/continuation.json')['qualified_rounds'][-1]
        timings[latest['source_sha256']] = dict(latency_ms=latest['target_latency_ms'])
        for event in events[1:]:
            if not event.get('ir_sha256'):
                event['latency_ms'] = timings[event['source_sha256']]['latency_ms']
        if events[-1].get('ir_sha256'):
            anchor_ms = events[-1]['latency_ms'] * events[-1]['target_case_speedup']
        else:
            anchor_ms = latest['target_latency_ms'] * latest['target_speedup']
    else:
        assert not any(e.get('historical') for e in events), 'Import completed profiler replay first'
        latest = [e for e in events if e.get('qualified')][-1]
        anchor_ms = latest['latency_ms'] * latest['target_case_speedup']
    for index, event in enumerate(events):
        event['aggregate_score'] = event.get('geo_mean', event.get('score'))
        event['score'] = anchor_ms / event['latency_ms'] if event.get('qualified') else None
        event['raw_target_speedup'] = event.get('raw_target_case_speedup', event.get('target_case_speedup'))
        event['score_is_rebased'] = index != len(events) - 1 and event.get('qualified', False)
        event['axis_label'] = 'FlagGems' if index == 0 else f'V{index}'
        event['case'] = case
        event.setdefault('measurement_transition', False)
    curve = dict(operator=title, metric='fixed_case_cann_speedup', case=case, shape=shape, compact=True,
        scope='固定输入、profiler计时；最新合格版本的逐case CANN参考耗时为共同锚点。'
              '各版本以实测候选耗时换算；原始逐轮成绩、失败和源码接续保留于完整记录。',
        events=events, cann_anchor_ms=anchor_ms, aggregate_score=events[-1]['aggregate_score'],
        source_report='../' + folder + '/index.html', story=data.get('ir_case_story', story),
        optimization_loop=data.get('optimization_loop'),
        reference_kind='CANN_FP32_state_compatibility_composition' if name == 'batchnorm' else 'CANN',
        case_spec=case_spec, timing_contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest())
    curve['presentation_selection'] = milestones(curve)
    selected = [events[i] for i in curve['presentation_selection']['displayed_indices']]
    scores = [e['score'] for e in selected if e['score'] is not None]
    assert all(a <= b for a, b in zip(scores, scores[1:]))
    assert selected[0]['source_sha256'], 'Native source identity is required'
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    base = args.root / 'development/reports'
    output = base / 'group-meeting-20261119'
    output.mkdir(exist_ok=True)
    from .research_routes import write_routes
    curves = [fixed_curve(base, *row) for row in CONFIG]
    full_curves = [read(base / row[2] / 'mainline-presentation.json') for row in CONFIG]
    evidence_notes = read(output / 'case-evidence-notes.json')['operators']
    figure_index = []
    ir_count = sum(any(e.get('selected_layer') == 'single_operator_ir' for e in curve['events']) for curve in curves)
    write_routes(output, ir_count)
    checks = {'batchnorm': '27/27；补测各36/36', 'upsample': '24/24；补测各30/30',
        'qkv': '21/21；补测各48/48', 'scatter': '18/18；补测各21/21；专项3/3'}
    css = ('body{max-width:1500px;margin:32px auto;padding:0 24px;background:#fff7e8;'
           'color:#1b2434;font:21px/1.65 "Microsoft YaHei",sans-serif}'
           'section{padding:22px;margin:24px 0;border:1px solid #ded7c9;border-radius:12px}'
           'svg{width:100%;height:auto}a{color:#2368b1}td,th{padding:10px;border-bottom:1px solid #ded7c9}'
           'table{border-collapse:collapse;width:100%}h1{font-size:34px}h2{font-size:28px}')
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>大组会 · 白盒模拟器驱动的多层次优化</title><style>' + css + '</style>',
        '<h1>白盒模拟器驱动的多层次优化：以昇腾算子优化为例</h1>',
        '<p>继续模拟器重放 → 定位热点与等待 → 选择Triton／IR修改层 → 编译新产物并再次重放 → 真机验收 → 下一轮</p>',
        '<p>八张图分成两组：<a href="#aggregate">四张全用例geomean曲线</a>与'
        '<a href="#fixed-shapes">四张固定shape机制曲线</a>。均从原生FlagGems开始，'
        '实线连续连接完整合格的最佳成绩；失败、退步与复测保留详情。</p>',
        '<p>橙色方法文字与箭头表示模拟器指导；绿色节点形状表示修改位置：圆形设备kernel、方形host、菱形both、三角形IR层。'
        '模拟器重放先后指导Triton源码及IR层修改，具体证据与其他贡献保留详情。</p>',
        '<section><h2>当前保留成绩：同时标清两种指标</h2>'
        '<p>geomean覆盖完整合同的全部性能用例；固定shape用于讲解具体瓶颈与改法。'
        f'当前四项中{ir_count}项已有完整验收的IR新增收益；逐项结果与探索保留详情。</p>',
        '<table><tr><th>算子</th><th>全用例geomean / CANN</th><th>固定shape / CANN</th>'
        '<th>IR进展</th><th>完整检查</th></tr>']
    for row, curve in zip(CONFIG, curves):
        parts.append(f'<tr><td>{html.escape(row[1])}</td><td>{curve["aggregate_score"]:.3f}×</td>'
            f'<td>{curve["events"][-1]["score"]:.3f}×</td>'
            f'<td>{html.escape(evidence_notes[row[0]]["ir_status"])}</td><td>{checks[row[0]]}</td></tr>')
    parts.append('</table></section>')
    parts.append('<h2 id="aggregate">第一组：四个算子的全用例geomean进展</h2>'
        '<p>直接使用26算子主图中这四项的最新数据；从原生出发持续模拟器重放，按问题选择Triton或IR修改层。'
        '不同轮次采用共同逐case CANN参考重算，≈表示历史重算；同源码复测与原始配对成绩可展开查看。</p>')
    for row, full in zip(CONFIG, full_curves):
        name = row[0]
        drawing = render(full)
        (output / (name + '-geomean.svg')).write_text(drawing, encoding='utf8')
        source = base / row[2] / 'mainline-presentation.json'
        figure_index.append(dict(operator=full['operator'], group='aggregate',
            metric='full_contract_cann_geomean', svg=name + '-geomean.svg', png=name + '-geomean.png',
            source=str(source.relative_to(base)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            native_source_sha256=full['events'][0]['source_sha256'],
            displayed_indices=milestones(full)['displayed_indices']))
        gallery = '../operator-traces-stitched-20261117/' + full['operator'] + '/index.html'
        parts += [f'<section id="{name}-geomean"><h2>{html.escape(row[1])} · 全用例geomean</h2>',
            drawing, '<p>' + html.escape(evidence_notes[name]['ir_status']) + '</p>',
            f'<p><a href="{name}-geomean.svg">SVG</a> · <a href="{name}-geomean.png">PNG</a> · '
            f'<a href="{gallery}">26算子中的原主图与完整记录</a> · '
            f'<a href="#{name}">对应固定shape与机制</a></p></section>']
    parts.append('<h2 id="fixed-shapes">第二组：四张固定shape机制曲线</h2>'
        '<p>图中方法箭头沿用真实判断来源；图下逐项列出模拟器观察、其他指导、修改层和真机结果。'
        'IR阶段继续重放当前版本，并在修改后重放新二进制；模拟器、profiling、源码／IR及Codex的分析贡献按记录说明。'
        + html.escape(read(output / 'case-evidence-notes.json')['ordering_note']) + '</p>')
    for row, curve, full in zip(CONFIG, curves, full_curves):
        name = row[0]
        drawing = render(curve)
        (output / (name + '.svg')).write_text(drawing, encoding='utf8')
        figure_index.append(dict(operator=full['operator'], group='fixed_shape',
            metric=curve['metric'], case=curve['case'], shape=curve['shape'], svg=name + '.svg', png=name + '.png',
            native_source_sha256=curve['events'][0]['source_sha256'],
            displayed_indices=curve['presentation_selection']['displayed_indices']))
        parts += [f'<section id="{name}"><h2>{html.escape(curve["operator"])}</h2>',
            '<p>' + html.escape(curve['shape'] + ' · ' + curve['case']) + '</p>',
            '<p>' + html.escape(curve['story']) + '</p>', drawing,
            f'<p>CANN固定参考耗时：{curve["cann_anchor_ms"] * 1000:.3f} μs。'
            '节点由共同CANN锚点与各版本候选实测耗时换算；≈标记采用共同参考重算的点。</p>']
        parts.append('<table><tr><th>模拟器重放与判断依据</th><th>修改层及改法</th><th>真机结果与证据范围</th></tr>')
        for step in evidence_notes[name]['steps']:
            replay = ('<strong>继续重放闭环</strong><br>' + html.escape(step['replay']) + '<br>'
                '<a href="' + html.escape(step['before_decision'], quote=True) + '">修改前读取与决策</a> · '
                '<a href="' + html.escape(step['after_replay'], quote=True) + '">修改后二进制重放</a><br>'
                if step.get('replay') else '')
            parts.append('<tr><td>' + replay + html.escape(step['observation']) + '</td><td><strong>' +
                html.escape(step['layer']) + '</strong><br>' +
                ('<span style="color:#c55a18">' + html.escape(step['change']) + '</span>'
                 if step.get('replay') else html.escape(step['change'])) + '</td><td>' +
                html.escape(step['result']) + '<br><small>' + html.escape(step['scope']) +
                '</small><br><a href="' + html.escape(step['href'], quote=True) + '">对应证据</a></td></tr>')
        parts.append('</table>')
        if name == 'batchnorm':
            parts.append('<p>BatchNorm参考包含FP32状态兼容处理，具体调用链保留于成对采集记录。</p>'
                '<p>预取步骤：71.551→55.622 μs；真机每程序Scalar等待减少47.1%。'
                '模拟器部分窗口Vector等待30.89%→3.58%，窗口分别归一化。</p>'
                '<p><a href="../batchnorm-simulator-20261118/interactive.html">热点与流水交互演示</a></p>')
        parts.append('<details><summary>全部测量、筛选映射与源码</summary><table>'
            '<tr><th>版本</th><th>候选耗时μs</th><th>相对CANN</th><th>源码SHA</th></tr>')
        for event in curve['events']:
            latency = f'{event["latency_ms"] * 1000:.3f}' if event.get('qualified') else '未通过'
            score = f'{event["score"]:.4f}×' if event.get('qualified') else '无性能数值'
            parts.append(f'<tr><td>{html.escape(event["label"])}</td><td>{latency}</td>'
                f'<td>{score}</td><td>{html.escape(event["source_sha256"][:12])}</td></tr>')
        parts.append('</table><p>' + html.escape(curve['scope']) + '</p></details>')
        parts.append(f'<p><a href="{name}.svg">SVG</a> · <a href="{name}.png">PNG</a> · '
            f'<a href="{curve["source_report"]}">全合同主图与完整记录</a></p></section>')
    layer_text = (f'四项均已有Triton源码层实验；其中{ir_count}项已补充完整合格的单算子IR收益。'
                  if ir_count else '当前四项实际修改均在Triton源码层。')
    parts.append('<section id="when-ir"><h2>什么时候考虑修改IR？</h2>'
        '<p>模拟器显示当前生成代码存在有成本的热点、重复工作或等待；实际IR与低层产物能定位其形成方式，且存在语义可验证的局部改法。</p>'
        '<p>先绑定当前最佳的case、输入和二进制，再检查热点与流水、确认低层仍保留该成本，登记预计减少的指令／等待／轮次。'
        '固定源码修改IR后，重新编译并重放，以完整真机合同及既有补测确认收益；测量出现疑点时再做交错复测。</p>'
        '<p>上采样：宽位选择缩窄；scatter：冗余gather尾步折叠；BatchNorm：减少输出tile轮次。'
        '算法与启动意图需重设计时优先改Triton；同类编译缺陷跨多个算子重复后评估通用pass。</p>'
        '<p><a href="../when-to-edit-ir-20261120.md">选层条件、观察分类与案例</a></p></section>')
    parts.append('<section><h2>当前研究层次与下一步</h2><p>' + layer_text +
        '模拟器指令和流水用于提出机制假设，实际IR用于检查编译结果，真机评测用于确认收益。'
        'IR修改纳入持续模拟器重放：读取当前热点与等待，固定源码修改IR，编译新二进制后再次重放，并结合完整真机验收保留最佳。'
        '在重复缺陷出现时评估通用pass；修改、正确性与真实收益分别核验。</p>'
        '<p>模型负责候选实现；Codex负责证据分析、调度与核验。模拟器窗口和绝对时钟范围逐项保留。</p>'
        '<img src="research-routes.svg" alt="模拟器证据驱动的编译优化线与能力扩展线" style="width:100%;height:auto">'
        '<p><a href="research-routes.svg">路线图SVG</a> · <a href="research-routes.png">路线图PNG</a> · '
        '<a href="whitebox-simulator-ascend-20261119.pptx">组会PowerPoint</a> · <a href="talk.md">讲稿</a></p></section></html>')
    if ir_count:
        parts.insert(-1, '<p><a href="whitebox-simulator-ascend-20261120-replay.pptx">持续模拟器重放闭环 · 更新PPT</a></p>')
    (output / 'index.html').write_text(''.join(parts), encoding='utf8')
    (output / 'cases.json').write_text(json.dumps(curves, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (output / 'eight-figures.json').write_text(json.dumps(dict(aggregate_count=4, fixed_shape_count=4,
        figures=figure_index, evidence_notes='case-evidence-notes.json',
        scope='Same four aggregate trajectories as the26-operator gallery, plus four fixed-case views; '
              f'{ir_count} operators have accepted IR improvements; original measurements and explorations remain in details.'),
        ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        tab = browser.new_page(viewport=dict(width=1700, height=1200), device_scale_factor=1)
        tab.goto((output / 'index.html').resolve().as_uri())
        for row in CONFIG:
            tab.locator('#' + row[0] + ' svg').screenshot(path=str(output / (row[0] + '.png')))
            tab.locator('#' + row[0] + '-geomean svg').screenshot(path=str(output / (row[0] + '-geomean.png')))
        browser.close()
    print(output / 'index.html')


if __name__ == '__main__':
    main()
