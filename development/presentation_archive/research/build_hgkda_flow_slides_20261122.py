"""Generate concise editable transition, overview, and simulator flow slides."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

from build_hgkda_overview_20261122 import NAVY, PALE, INK, MUTED, BORDER, edge as base_edge, font, textbox
from presentation_materials_20261122 import copy_material


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'development/reports/hgkda-flow-slides-20261122'

TRANSITION_NOTES = '''好，谈到算子优化，或者让Agent自动优化算子，其实有一个很基本的过程。

我们先有一个当前版本，把它放到真机或者模拟器里执行，拿到反馈，再根据反馈去分析和修改，然后继续下一轮。

那这里头很关键的问题就是：反馈的信息，究竟可以有哪些？为此，我们进行了广泛的调研，下一页把这些信息分成五类。

本页先概括优化的基本回路。两种执行路径按问题需要使用；当前优化版本也可以经源码、IR与已有知识分析得到改法。'''

OVERVIEW_NOTES = '''那这些信息，具体怎么用到我们的HG-KDA里呢？

左边是输入，也就是我们要继续优化的当前算子版本，以及相应的输入规格、参考实现和验收要求。

中间有三路依据。成对profiling，主要比较当前实现和参考实现的性能差距。模拟器分析，进一步观察热点、等待、空泡和重叠，并关联到源码和IR。知识库和代码推理，则结合已有方法、适用条件和历史失败，检查具体可以怎么改。

这些依据交给Agent分析，提出修改。能在Triton或Host层表达的，优先修改源码；生成代码有具体可干预问题时，再研究单算子IR。

候选重新编译后，继续模拟器重放，检查预期的变化，并以真机确认正确性和性能。合格且有收益的版本保留，失败和退化也记录。

右边是输出：经过验收的优化版本及其性能和证据报告。下方这条虚线表示继续迭代，保留或回退后再分析下一轮。

三路依据按照当前信息缺口选择，图中并列位置表示各自的作用。编译与验证包括修改后的profiling和模拟器重放；具体顺序按实际实验记录。'''

SIMULATOR_NOTES = '''接下来具体看模拟器这一路。

我们从当前算子开始，编译后在CANN CA model环境中运行。实际模拟器执行的是编译后的NPU二进制指令，输入数据、启动参数和运行时上下文由运行入口提供。msprof op simulator组织运行、采集和解析。图里用“当前算子 → 编译后运行”概括这一步。

上面这条是指令报告：先提取热点指令，再通过源码位置和内联调用栈定位源码行，用AST确定它属于哪个函数、循环或分支；结合对应编译产物和IR的位置信息确认IR操作，帮助Agent定位修改位置。AST承担源码结构定位，IR关联使用编译映射与具体操作核对。

下面这条是执行trace：解析时间线事件，按流水区分执行、等待及已观测窗口内的可观察空闲，再结合指令、同步和搬运事件判断原因。图中用一条“时序分析”箭头概括这组处理。

这两路证据与优化知识库共同支持Agent分析与修改。候选重新编译后，继续做模拟器重放，检查预期变化，再结合真机KGS确认正确性和性能。下方虚线表示进入下一轮。

文件格式放在备注即可：本项目既有采集包括trace.json、instr_exe.csv和code_exe.csv。trace.json记录时间线事件，CSV提供指令和代码执行数据；具体字段随工具版本和采集配置核对。本页使用“指令报告”和“执行trace”表达数据的作用。

模拟器执行与真机执行的范围分别保留；在已采核和观测窗口内分析执行、等待与可观察空闲。IR修改编译后继续模拟器重放。'''

SCRIPT = '''# 开场与前五页讲稿

## 页面顺序

1. 标题页。
2. 算子优化的基本回路：当前版本、执行与反馈、分析修改。
3. 优化证据信息表：三大类、五小类。
4. 芯片仿真与性能建模：重点说明第三类时序仿真。
5. HG-KDA Overview：三路依据、修改、验证和迭代。

模拟器双分支图放在后面的模拟器技术部分，再展开热点和流水分析的案例。命令与文件格式留在备注及问答中。

## 第一页：开场

尊敬的林院长，各位老师、同学们，大家早上好！

我是来自中国科学院计算技术研究所的张汉乐。

今天我汇报的题目是《白盒模拟器驱动的多层次优化：以昇腾算子优化为例》。

## 第二页：算子优化的基本回路

好，谈到算子优化，或者让Agent自动优化算子，其实有一个很基本的过程。

我们先有一个当前版本，把它放到真机或者模拟器里执行，拿到反馈，再根据反馈去分析和修改，然后继续下一轮。

**那这里头很关键的问题就是：反馈的信息，究竟可以有哪些？**

为此，我们进行了广泛的调研，下一页把这些信息分成五类。

## 第三页：优化证据分类

这张表总结了人和Kernel Design Agent，在不同平台上能够取得哪些优化证据。

我们分成应用层、系统层、微架构层三大类。

应用层主要看耗时，这里列的是Host调用和kernel执行时间。

系统层进一步包括计算量、访存量、并行度，以及利用率。

微架构层，我们又分成硬件规格、统计指标和时序指标。

硬件规格，就是芯片有哪些资源、容量多大，帮助我们选择分块、数据复用和资源分配的策略。

**我们的HG-KDA，会重点关注后面的统计指标和时序指标。**

统计指标其实已经很有用了。比如，它可以告诉我们，计算单元利用率比较低，数据搬运占了很大比例。

但这些通常是一个kernel或一段时间内的汇总结果，比如总量、平均值和比例。

我们还会关心：**利用率为什么低？它究竟在等谁？**

是数据没搬过来，还是某一个同步阻塞了执行？有没有一些计算可以提前做，和搬运重叠起来？

这些问题就需要进一步看时序信息，可以来自模拟器，也可以来自真机采样。

NVIDIA的工具链比较成熟，可以结合真机采样和SASS分析，提供指令热点以及一部分依赖和等待信息；对详细流水时序的覆盖仍有边界。

昇腾这边的一个特点，是提供了CANN CA model，可以输出指令和流水时序。

**那这里说的模拟器，具体属于哪一类？下一页简单对齐一下概念。**

## 第四页：芯片仿真与性能建模

芯片设计和算子优化都会用到仿真或性能模型，不过关注的问题会有些不同。

这里按主要用途列了四类。功能仿真关注执行对不对，解析模型帮助估算性能，时序仿真展开指令和流水的执行过程，RTL仿真主要检查硬件逻辑。

**刚才提到的CA model，就属于第三类。** 我们主要利用它输出的执行事件，分析热点、同步等待，以及计算和搬运的重叠情况。

有了这些能力，接下来看看，具体怎么用到HG-KDA里。

## 第五页：HG-KDA Overview

''' + OVERVIEW_NOTES + '''

## 后续模拟器技术页

''' + SIMULATOR_NOTES + '''

## 画图与讲述口径

- 过渡页保持两个框和一条反馈回路。反馈既来自真机，也可以来自模拟器；按具体分析任务选择。
- Overview每个框保留一个主要概念，输入和输出显式标识；三路依据的细节在口头说明。迭代回路用虚线。
- 模拟器页延续用户提供的双分支结构；主图保留“指令报告”“执行trace”，JSON与CSV文件名放备注。
- 三页箭头统一加粗并放大箭头头部。模拟器名称统一为表格中的“CANN CA model”。输入用“当前算子”，以“编译后运行”说明执行入口。
- Agent在图上表示分析和修改职责，沿用“Agent分析与修改”。Analyst/Coder若用作名称，需说明其为角色分工；当前图保持单一Agent模块及知识库输入。
- 绿色、黄色及厂商规格的型号范围继续沿用已对齐的信息表口径。统计指标通常按kernel或窗口聚合；profiler也可提供时间线和采样。
- 当前脚本按本次用户口述保留姓名“张汉乐”；原汇报封面未修改。
'''

MMD = {
    '算子优化-基本回路': '''flowchart LR
    V[当前算子版本] -->|真机 / 模拟器| E[执行与反馈]
    E -. 分析反馈，修改算子 .-> V
''',
    'HG-KDA-overview-简版': '''flowchart LR
    IN([输入：当前算子]) --> P[成对 profiling]
    IN --> S[模拟器分析]
    IN --> K[知识库＋代码推理]
    P --> A[Agent分析与修改]
    S --> A
    K --> A
    A --> V[编译与验证]
    V --> OUT([输出：优化后算子])
    V -. 反馈，继续迭代 .-> IN
''',
    '模拟器-证据与优化': '''flowchart LR
    K[当前算子] -->|编译后运行| S[CANN CA model]
    S --> I[指令报告]
    I --> H[热点指令]
    H --> R[源码区域 / IR]
    S --> T[执行 trace]
    T -->|时序分析| P[流水执行 / 等待 / 空闲]
    R --> A[Agent分析与修改]
    P --> A
    KB[优化知识库] --> A
    A --> V[真机KGS验证]
    V -. 修改后重新编译与重放 .-> K
'''
}


def edge(slide, points, **kwargs):
    return base_edge(slide, points, line_width=2.1, arrow_size='lg', **kwargs)


def node(slide, value, x, y, w, h, endpoint=False, size=23):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shape.name = value.replace('\n', ' ')
    shape.adjustments[0] = .12
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(NAVY if endpoint else PALE)
    shape.line.color.rgb = RGBColor.from_string(NAVY if endpoint else BORDER)
    shape.line.width = Pt(1)
    shape._element.spPr.append(OxmlElement('a:effectLst'))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = False
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = Inches(.05)
    frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(value.split('\n')):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        p.line_spacing = 1.12
        p.space_before = p.space_after = Pt(0)
        run = p.add_run()
        run.text = line
        font(run, size, 'FFFFFF' if endpoint else NAVY, True)
    return shape


def new_deck(template):
    deck = Presentation(template)
    for item in list(deck.slides._sldIdLst):
        deck.part.drop_rel(item.rId)
        deck.slides._sldIdLst.remove(item)
    deck.core_properties.author = 'HG-KDA'
    return deck


def new_slide(deck, title, notes):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string('FFFFFF')
    textbox(slide, title, .55, .48, 12.23, .65, 32, INK, True)
    notes_slide = slide.notes_slide
    if notes_slide.notes_text_frame is None:
        notes_slide.shapes._spTree.add_placeholder(notes_slide.shapes._next_shape_id,
            'Notes Text', PP_PLACEHOLDER.BODY, 'horz', 'full', 13)
    notes_slide.notes_text_frame.text = notes
    return slide


def transition(deck):
    slide = new_slide(deck, '算子优化：从一个版本到下一个版本', TRANSITION_NOTES)
    edge(slide, [(4.70, 3.55), (8.63, 3.55)])
    edge(slide, [(10.28, 4.20), (10.28, 5.63), (3.05, 5.63), (3.05, 4.20)],
         dashed=True, name='反馈回路')
    node(slide, '当前算子版本', 1.40, 2.90, 3.30, 1.30, size=28)
    node(slide, '执行与反馈', 8.63, 2.90, 3.30, 1.30, size=28)
    textbox(slide, '真机 / 模拟器', 5.20, 2.90, 2.93, .42, 24, MUTED, center=True)
    textbox(slide, '分析反馈，修改算子', 4.62, 5.04, 4.09, .42, 24, NAVY, center=True)
    return slide


def overview(deck):
    slide = new_slide(deck, 'HG-KDA：从硬件证据到优化决策', OVERVIEW_NOTES)
    center = 3.88
    edge(slide, [(2.35, center), (2.62, center)], arrow=False)
    edge(slide, [(2.62, 2.23), (2.62, 5.53)], arrow=False)
    for y in [2.23, center, 5.53]:
        edge(slide, [(2.62, y), (2.90, y)])
        edge(slide, [(5.65, y), (5.94, y)], arrow=False)
    edge(slide, [(5.94, 2.23), (5.94, 5.53)], arrow=False)
    edge(slide, [(5.94, center), (6.25, center)])
    edge(slide, [(8.05, center), (8.55, center)])
    edge(slide, [(10.40, center), (10.95, center)])
    edge(slide, [(9.48, 4.38), (9.48, 6.45), (1.45, 6.45), (1.45, 4.38)],
         dashed=True, name='验证反馈与下一轮')
    textbox(slide, '输入', .55, 2.88, 1.80, .35, 20, NAVY, True, True)
    textbox(slide, '输出', 10.95, 2.88, 1.83, .35, 20, NAVY, True, True)
    node(slide, '当前算子', .55, 3.38, 1.80, 1.0, True, 23)
    node(slide, '成对 profiling', 2.90, 1.73, 2.75, 1.0, size=23)
    node(slide, '模拟器分析', 2.90, 3.38, 2.75, 1.0, size=23)
    node(slide, '知识库＋代码推理', 2.90, 5.03, 2.75, 1.0, size=21)
    node(slide, 'Agent\n分析与修改', 6.25, 3.38, 1.80, 1.0, size=23)
    node(slide, '编译与验证', 8.55, 3.38, 1.85, 1.0, size=22)
    node(slide, '优化后算子', 10.95, 3.38, 1.83, 1.0, True, 22)
    textbox(slide, '反馈，继续迭代', 5.83, 5.96, 3.45, .38, 21, NAVY, center=True)
    return slide


def simulator(deck):
    slide = new_slide(deck, '模拟器证据怎样指导修改', SIMULATOR_NOTES)
    center = 2.75
    for start, stop in [(1.85, 2.20), (4.00, 4.35), (5.60, 5.95),
                        (7.20, 7.55), (9.25, 9.60), (11.10, 11.45)]:
        edge(slide, [(start, center), (stop, center)])
    edge(slide, [(4.00, 3.02), (4.17, 3.02), (4.17, 4.90), (4.35, 4.90)])
    edge(slide, [(5.60, 4.90), (7.55, 4.90)])
    edge(slide, [(9.25, 4.90), (9.42, 4.90), (9.42, 3.02), (9.60, 3.02)])
    edge(slide, [(10.35, 1.85), (10.35, 2.20)])
    edge(slide, [(12.18, 3.30), (12.18, 6.28), (.98, 6.28), (.98, 3.30)],
         dashed=True, name='候选重新编译与模拟器重放')
    node(slide, '当前算子', .45, 2.20, 1.40, 1.10, size=21)
    node(slide, 'CANN CA\nmodel', 2.20, 2.20, 1.80, 1.10, size=22)
    node(slide, '指令报告', 4.35, 2.20, 1.25, 1.10, size=20)
    node(slide, '热点指令', 5.95, 2.20, 1.25, 1.10, size=20)
    node(slide, '源码区域\n/ IR', 7.55, 2.20, 1.70, 1.10, size=21)
    node(slide, '执行\ntrace', 4.35, 4.35, 1.25, 1.10, size=21)
    node(slide, '流水执行 /\n等待 / 空闲', 7.55, 4.35, 1.70, 1.10, size=19)
    node(slide, 'Agent\n分析与修改', 9.60, 2.20, 1.50, 1.10, size=20)
    node(slide, '真机KGS\n验证', 11.45, 2.20, 1.45, 1.10, size=20)
    node(slide, '优化知识库', 9.60, 1.28, 1.50, .57, size=19)
    textbox(slide, '编译后运行', 1.10, 1.65, 1.85, .35, 18, MUTED, center=True)
    textbox(slide, '位置映射＋AST', 7.15, 1.65, 2.50, .35, 18, MUTED, center=True)
    textbox(slide, '时序分析', 5.86, 4.32, 1.44, .35, 19, MUTED, center=True)
    textbox(slide, '修改后重新编译与重放', 4.02, 5.75, 5.20, .42, 22, NAVY, center=True)
    return slide


BUILDERS = [('算子优化-基本回路', transition),
            ('HG-KDA-overview-简版', overview),
            ('模拟器-证据与优化', simulator)]


def check(deck):
    result = []
    for i, slide in enumerate(deck.slides, 1):
        minimum = 100
        for shape in slide.shapes:
            assert min(shape.left, shape.top) >= 0
            assert shape.left + shape.width <= deck.slide_width
            assert shape.top + shape.height <= deck.slide_height
            if shape.has_text_frame:
                for p in shape.text_frame.paragraphs:
                    for run in p.runs:
                        if run.text.strip() and run.font.size:
                            minimum = min(minimum, run.font.size.pt)
        assert minimum >= 18
        result.append({'slide': i, 'minimum_font_pt': minimum,
                       'native_editable_shapes': True, 'speaker_notes': True,
                       'bounds_passed': True})
    return result


def main(template: Path, delivery: Path):
    REPORT.mkdir(parents=True, exist_ok=True)
    before = hashlib.sha256(template.read_bytes()).hexdigest()
    outputs = []
    pack = new_deck(template)
    for stem, builder in BUILDERS:
        builder(pack)
        single = new_deck(template)
        builder(single)
        single.core_properties.title = stem
        check(single)
        path = REPORT / (stem + '.pptx')
        single.save(path)
        outputs.append(path)
        mmd = REPORT / (stem + '.mmd')
        mmd.write_text(MMD[stem] +
            '    classDef default fill:#E7EFF7,stroke:#CAD5E2,color:#133F6D;\n', encoding='utf-8')
        outputs.append(mmd)
    path = REPORT / 'HG-KDA-过渡与流程-三页.pptx'
    pack.save(path)
    outputs.append(path)
    check(pack)
    script = REPORT / '开场与前五页讲稿.md'
    script.write_text(SCRIPT, encoding='utf-8')
    outputs.append(script)
    assert hashlib.sha256(template.read_bytes()).hexdigest() == before
    validation = {'status': 'structure_passed', 'pages': check(pack),
                  'template': str(template), 'template_sha256': before,
                  'template_preserved': True, 'overview_inputs_outputs_explicit': True,
                  'loops_dashed': True, 'file_formats_in_notes_only': True,
                  'minimum_body_font_pt': 18, 'native_export': 'pending'}
    validation.update(arrow_line_width_pt=2.1, arrow_head='large',
                      simulator_name='CANN CA model', explicit_hot_instruction_step=True,
                      agent_roles='single analysis-and-modification module',
                      knowledge_base_input=True)
    (REPORT / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    delivery.mkdir(parents=True, exist_ok=True)
    for p in outputs:
        copy_material(p, delivery)
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--delivery-dir', type=Path, required=True)
    args = parser.parse_args()
    main(args.template.resolve(), args.delivery_dir.resolve())
