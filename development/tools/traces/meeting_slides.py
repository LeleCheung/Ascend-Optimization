"""Create the group meeting PowerPoint from accepted figures and recorded evidence."""
import json
from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


BG = 'FFF7E8'
INK = '1B2434'
GREEN = '168358'
ORANGE = 'C55A18'
BLUE = '2368B1'


def text(slide, value, x, y, w, h, size=25, color=INK, bold=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.word_wrap = True
    for index, line in enumerate(value.split('\n')):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.space_after = Pt(15)
        for run in paragraph.runs:
            run.font.name = 'Microsoft YaHei'
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = RGBColor.from_string(color)
    return shape


def main():
    output = Path.cwd() / 'development/reports/group-meeting-20261119'
    acceptance = json.loads((output / 'acceptance.json').read_bytes())
    assert len(acceptance['operator_figures']) == 26 and len(acceptance['fixed_case_figures']) == 4
    curves = json.loads((output / 'cases.json').read_bytes())
    evidence_notes = json.loads((output / 'case-evidence-notes.json').read_bytes())['operators']
    by_name = dict(zip(['batchnorm', 'upsample', 'qkv', 'scatter'], curves))
    ir_count = sum(any(e.get('selected_layer') == 'single_operator_ir' for e in curve['events']) for curve in curves)
    layer_status = (f'四项源码层案例已验证；{ir_count}项固定源码IR收益已验收'
                    if ir_count else '当前四项有效修改位于Triton源码层，单算子IR进入下一阶段')
    presentation = Presentation()
    presentation.slide_width = Inches(13.333)
    presentation.slide_height = Inches(7.5)

    def slide(title, body=None, notes=''):
        current = presentation.slides.add_slide(presentation.slide_layouts[6])
        current.background.fill.solid()
        current.background.fill.fore_color.rgb = RGBColor.from_string(BG)
        text(current, title, .55, .35, 12.2, .9, 33, bold=True)
        if body:
            text(current, body, .7, 1.6, 11.9, 5.1, 25)
        text(current, 'HG-KDA · Ascend 910B · 2026-11-19', .6, 7.12, 10, .25, 11, '596170')
        text(current, str(len(presentation.slides)), 12.2, 7.08, .5, .3, 12, '596170')
        current.notes_slide.notes_text_frame.text = notes
        return current

    cover = slide('白盒模拟器驱动的多层次优化：\n以昇腾算子优化为例')
    cover.shapes[0].height = Inches(1.8)
    text(cover, '从原生FlagGems出发，记录判断、修改与真机收益', .75, 2.75, 11.9, 1, 31, GREEN, True)
    text(cover, '四个案例：BatchNorm · 双线性上采样 · QKV · scatter_add\n下周大组会材料', .8, 4.5, 11.7, 1.4, 25)
    slide('研究问题：Agent怎样选改法，又怎样验证？',
        '已有成果：30项中保留26项交付，本批深入研究4项\n'
        'Profiling指出差距；模拟器进一步定位指令热点与流水等待\n'
        '源码和实际IR确定修改位置；完整真机评测确认正确性与收益\n'
        + layer_status)
    slide('持续模拟器重放：从Triton修改延续到IR',
        '重放当前最佳二进制，定位热点、流水空泡与等待\n'
        '关联源码与实际IR → 登记假设 → 选择修改层\n'
        '模型修改Triton／IR → 编译新二进制 → 继续重放\n'
        '结合完整真机验收保留最佳，进入下一轮重放',
        '模型负责候选；Codex负责诊断、工具、调度和审查。既有案例先真机评测，再做修改后profiling和模拟器重放；实际顺序按日志保存。模拟器执行绑定IR编译产物、case与真实输入。')
    routes = slide('')
    routes.shapes.add_picture(str(output / 'research-routes.png'), Inches(.52), Inches(.15), width=Inches(12.3))
    routes.notes_slide.notes_text_frame.text = '模拟器重放贯穿每轮；Triton／IR／pass是按证据选择的修改层。' + layer_status + '；通用pass和Ascend C按具体缺陷及能力缺口推进。'
    overview = slide('四项结果：单输入曲线与全部用例分开呈现')
    rows = [['算子', '全部用例相对CANN', '完整检查']]
    checks = ['27/27；补测各36/36', '24/24；补测各30/30', '21/21；补测各48/48', '18/18；补测各21/21；专项3/3']
    for curve, check in zip(curves, checks):
        rows.append([curve['operator'], f'{curve["aggregate_score"]:.3f}×', check])
    table = overview.shapes.add_table(5, 3, Inches(.7), Inches(1.65), Inches(11.9), Inches(3.9)).table
    for column, width in zip(table.columns, [5.6, 2.7, 3.6]):
        column.width = Inches(width)
    for row_index, values in enumerate(rows):
        for column_index, value in enumerate(values):
            cell = table.cell(row_index, column_index)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(GREEN if row_index == 0 else 'FFFAF0')
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.name = 'Microsoft YaHei'
                    run.font.size = Pt(20)
                    run.font.bold = row_index == 0
                    run.font.color.rgb = RGBColor.from_string('FFFFFF' if row_index == 0 else INK)
    text(overview, '八张图：4张全用例geomean进展＋4张固定shape机制；纵轴均相对CANN。', .75, 6.15, 11.9, .7, 22, GREEN)

    def figure(name, aggregate=False):
        from PIL import Image
        curve = by_name[name]
        current = slide('')
        # The figure contains its own title, key, numerical axis and source-bound nodes.
        path = output / (name + ('-geomean' if aggregate else '') + '.png')
        with Image.open(path) as picture:
            scale = min(12.3 / picture.width, 6.7 / picture.height)
            width, height = picture.width * scale, picture.height * scale
        current.shapes.add_picture(str(path), Inches((13.333-width)/2), Inches(.15),
            width=Inches(width), height=Inches(height))
        current.notes_slide.notes_text_frame.text = ('全用例geomean，使用26算子主图同一份数据。'
            if aggregate else curve['scope'] + '\n完整数据：cases.json；case：' + curve['case'])
        explorations = [step for step in evidence_notes[name]['steps'] if '探索' in step['layer']]
        if explorations:
            current.notes_slide.notes_text_frame.text += '\n\n探索详情：\n' + '\n\n'.join(
                '\n'.join(step[key] for key in ['observation', 'change', 'result', 'scope', 'href'])
                for step in explorations)
        return current

    def ir_figure_detail(name):
        events = [e for e in by_name[name]['events'] if e.get('selected_layer') == 'single_operator_ir']
        if not events:
            return
        event = events[-1]
        selection = json.loads((output.parent / event['selection']).read_bytes())
        original = selection.get('historical_retained_latency_us', selection['original_ir_latency_us'])
        candidate = selection['target_latency_us']
        title = {'batchnorm': 'BatchNorm', 'upsample': '双线性上采样', 'qkv': 'QKV', 'scatter': 'scatter_add'}[name]
        current = slide(title + '：继续模拟器重放，固定源码改IR')
        text(current, f'{original:.3f} → {candidate:.3f} μs · 耗时下降{(1-candidate/original)*100:.2f}%',
             .8, 1.55, 11.8, .8, 33, GREEN, True)
        samples = '／'.join(f'{row["median_us"]:.3f}' for row in selection.get('stability', {}).get('records', []))
        checks = event['replay']['supplementary']
        supplement = '；'.join(f'{row["passed"]}/{row["total"]}' for row in checks.values())
        comparison = '源码、IR和二进制分别绑定身份；原始成绩与窗口范围分别解释'
        if name == 'qkv' and selection.get('historical_retained_latency_us'):
            historical_gain = (1-candidate/selection['historical_retained_latency_us'])*100
            comparison = f'较历史最佳耗时下降{historical_gain:.2f}%；全合同geomean基本持平'
        if name == 'batchnorm' and selection.get('numerical_diagnostic'):
            diagnostic = selection['numerical_diagnostic']
            comparison = (f'冻结诊断{diagnostic["original_passed"]}→{diagnostic["candidate_passed"]}/{diagnostic["total"]}；'
                f'新增回归{diagnostic["new_regressions"]}；窄分支继承边界保留')
        text(current, event['method'].replace('\n', '：') + '\n'
            f'完整{event["replay"]["num_passed"]}/{event["replay"]["num_workloads"]}；补充检查{supplement}\n'
            '修改后profiling与模拟器重放已保存\n'
            '读取当前重放 → 修改IR并编译 → 验收后重放新二进制\n'
            + comparison,
             .85, 2.65, 11.6, 3.8, 23)
        current.notes_slide.notes_text_frame.text = (event['attribution'] + '\n' + selection['scope']
            + ('\n历史ABBA：' + samples + ' μs' if samples else '\n测量按条件策略执行，完整记录见selection.json'))

    for name in by_name:
        figure(name, aggregate=True)
    figure('batchnorm')
    bn = slide('BatchNorm：预取步骤的两类证据与真机验证')
    text(bn, '71.551 → 55.622 μs', .8, 1.55, 11.7, .9, 39, GREEN, True)
    text(bn, '模拟器部分窗口：Vector等待 30.89% → 3.58%\n'
        '真机每程序：Scalar等待 19.661 → 10.401 μs，减少47.1%\n'
        '最新整除去掩码版继续降低耗时；其窗口等待上升也保留\n'
        '窗口比例分别归一化，完整调用收益采用KGS结果', .85, 2.9, 11.5, 3.5, 25)
    bn.notes_slide.notes_text_frame.text = '本页采用11月18日原始机制验证，和11月19日展示复测分开记录。现场演示interactive.html。'
    if any(e.get('selected_layer') == 'single_operator_ir' for e in by_name['batchnorm']['events']):
        ir_bn = slide('BatchNorm：继续模拟器重放，修改输出IR')
        text(ir_bn, '42.955 → 39.261 μs · 耗时下降8.60%', .8, 1.55, 11.8, .8, 33, GREEN, True)
        text(ir_bn, '归一化／写回分块：8×128 → 8×256；保留原分支兜底\n'
            '统计／Welford／归约与逐元素FP32运算顺序保持固定\n'
            '完整27/27、双方各36/36；其余27个AST二进制逐一相同\n'
            '修改后重放：33个PC时钟核对通过\n'
            '读取当前局部重放 → 改IR并编译 → 验收后重放新产物',
            .85, 2.65, 11.6, 3.8, 23)
        ir_bn.notes_slide.notes_text_frame.text = ('证据与贡献：修改前模拟器提供统计局部窗口内同步节奏线索；'
            'Codex结合真实IR与UB失败提出输出加宽方向，CC／DeepSeek登记并修改。'
            '直接PC源码映射缺失，保存PC热点与源码IR语义对应。'
            '修改前局部统计窗口与修改后完整窗口分别解释；后重放33个PC时钟通过。'
            '动态同步节省的同范围前后量化仍待原版完整窗口补采。'
            '验收：../ir-performance-20261119/batchnorm/selection.json。'
            '历史ABBA：44.948／40.058／39.022／42.704 μs。')
    if any(e.get('selected_layer') == 'single_operator_ir' and
           e.get('selection', '').startswith('midday-whitebox-20261120/')
           for e in by_name['batchnorm']['events']):
        ir_figure_detail('batchnorm')
    figure('upsample')
    ir_figure_detail('upsample')
    figure('qkv')
    ir_figure_detail('qkv')
    figure('scatter')
    ir_figure_detail('scatter')
    negative = slide('scatter：模拟器也帮助解释退步')
    text(negative, '扫描原语被后端降成逐元素串行循环', .8, 1.55, 11.7, .7, 30, ORANGE, True)
    text(negative, '实际IR：scf.for，从1迭代到2048，循环内读、加、选择、写\n'
        '模拟器：最热LD/ST映射到扫描调用行\n'
        '恢复最佳后改为精简索引、掩码和类型转换\n'
        '最终主要收益位于写回kernel，其直接依据为profiling和IR', .85, 2.65, 11.5, 3.6, 25)
    slide('归因与验证：每个判断保留适用范围',
        '模拟器热点／指令和流水／等待均作为核心输入\n'
        '实际IR与源码绑定，历史方法推断用星号标明\n'
        '部分窗口采用相对占比；真实收益采用完整设备评测\n'
        '方法复用归知识／经验；知识库读取按日志确认\n'
        '历史对照、本批性能续接与展示复测分别记录')
    slide('什么时候考虑修改IR？',
        '模拟器：当前生成代码存在有成本的热点、重复工作或等待\n'
        '实际IR与低层产物：对应问题仍保留，局部变换语义可验\n'
        '固定源码，登记预期减少的指令、等待或tile轮次\n'
        '模型改IR → 编译并重放 → 固定合同真机验收\n'
        '源码意图需重设计时先改Triton；重复编译缺陷再评估pass',
        '图例：橙色表示模拟器指导；绿色圆／方／菱形为设备kernel／host／both，三角形为IR层。'
        '案例与条件见../when-to-edit-ir-20261120.md；能力缺口进入独立custom op／Ascend C线。')
    slide('IR收益续接：沿当前最佳继续模拟器重放',
        f'夜间三个案例已验收；当前四算子中{ir_count}项已有IR收益\n'
        '从当前最佳继续：BatchNorm、上采样、QKV、scatter候选池\n'
        '固定Triton源码：读取当前重放 → 模型改IR → 编译新产物\n'
        '完整数值／性能验收后继续模拟器重放，检查机制\n'
        '历史退步恢复与编译配置试验单列，持续保留最佳')
    slide('可复核的交付材料',
        '四项固定输入曲线：index.html、cases.json、SVG／PNG\n'
        '26项主图：优化总览与完整历史入口\n'
        'BatchNorm交互演示：热点、流水、源码映射\n'
        '源码SHA、原始成绩、展示复测与完整验收\n'
        '讲稿：talk.md；下一阶段：ir-study-protocol-20261119.md')
    destination = output / ('whitebox-simulator-ascend-20261120-replay.pptx' if ir_count
                            else 'whitebox-simulator-ascend-20261119.pptx')
    presentation.save(destination)
    print(destination)


if __name__ == '__main__':
    main()
