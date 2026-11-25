"""Create a first overview slide with explicit inputs, outputs, and feedback."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt
from presentation_materials_20261122 import copy_material


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / 'development/reports/hgkda-overview-20261122'
NAVY, PALE, INK, MUTED, BORDER = '133F6D', 'E7EFF7', '14181C', '606870', 'CAD5E2'
FONT = 'Microsoft YaHei'

MERMAID = '''flowchart LR
    IN([输入：初始算子源码、shape/dtype、参考实现与验收要求]) --> CUR[当前版本：源码 / IR / 编译产物]
    CUR --> P[成对 profiling：当前实现 vs 参考实现]
    CUR --> S[模拟器重放：热点、等待、空泡与重叠]
    CUR --> K[知识库与代码推理：已有方法、适用约束、源码与IR]
    P --> D[瓶颈假设与修改决策]
    S --> D
    K --> D
    D --> M[Agent修改：Triton / Host / 单算子IR]
    M --> V[编译与验证：模拟器重放、真机正确性与性能]
    V --> OUT([输出：已验收最佳实现、性能及证据报告])
    V -. 保留或回退，继续下一轮 .-> CUR
    classDef default fill:#E7EFF7,stroke:#CAD5E2,color:#133F6D;
    classDef endpoint fill:#133F6D,stroke:#133F6D,color:#FFFFFF;
    class IN,OUT endpoint;
'''

OVERVIEW_NOTES = '''那这些信息，具体怎么用到HG-KDA里呢？

先看左边的输入：已有的算子实现、输入规格、参考实现，以及固定的正确性和性能要求。我们从这个版本开始迭代。

中间主要有三路依据。第一路是成对profiling，把当前版本和参考实现放在一起看，先找计算、搬运以及启动等方面的差距。

第二路是模拟器，把编译后的kernel重放，分析指令热点、流水等待和空泡，再把这些现象对应到源码和IR。

第三路是知识库和代码推理。我们结合已验证的方法、适用条件和历史失败，检查当前代码里有哪些可以修改的地方。这三路根据当前问题选择和配合使用。

这些信息共同支持瓶颈假设和修改决策。能在Triton或者Host层表达的，我们优先修改源码；发现具体生成问题，或者源码难以控制的调度时，再研究单算子IR。

候选编译之后继续做模拟器重放，检查预期变化，同时用真机检查正确性和性能。符合要求的收益版本保留下来；负结果也记录。

右边是最终输出：已验收的最佳实现，以及对应的性能和证据报告。下方这条虚线表示继续迭代：保留或者回退后，再分析下一轮。'''

OPENING = '''# 开场与前四页讲稿

## 顺序

封面 → 算子优化可获得的硬件证据 → 芯片仿真与性能建模 → HG-KDA整体流程。

先提出优化中的信息需求，再解释我们选择的模拟器类型，随后进入方法。仿真分类页以半分钟左右为宜；msprof入口名称放在Overview讲解中带过。

## 第一页：封面

尊敬的林院长，各位老师、同学们，大家早上好！

我是来自中国科学院计算技术研究所的张汉乐。

今天我汇报的题目是《白盒模拟器驱动的多层次优化：以昇腾算子优化为例》。

## 第二页：能取得哪些优化证据

好，谈到算子优化，或者让Agent自动优化算子，有一个很关键的问题：**当我们已经写好一个版本之后，可以从芯片平台拿到哪些信息，来帮助我们继续优化？**

为此，我们进行了广泛的调研，把信息整理成了五类，也就是表里的应用层、系统层，以及微架构层下的硬件规格、统计指标和时序指标。

前面这些信息可以告诉我们耗时多少、资源有哪些，以及计算和访存的大致情况。我们这次重点看后面的统计指标和时序指标。

统计指标其实已经很有用了。比如，它能告诉我们计算单元利用率低，或者数据搬运占了很大比例。但这些通常是一个kernel或者一段时间内的汇总结果。

我们还会继续问：**它究竟在等谁？是数据没搬过来，还是某个同步阻塞了执行？有没有一些计算可以提前做，和搬运重叠起来？**

这些问题就需要进一步看时序证据。NVIDIA主要通过成熟的真机分析工具，提供指令采样和等待归因。昇腾这边还有一条我们可以实际使用的路径，就是CANN CA model模拟器，可以输出指令和流水时序。

那这里所说的模拟器，具体属于哪一类？下一页我们先简单对齐一下概念。

## 第三页：芯片仿真与性能建模

芯片设计和算子优化都会用到仿真或性能模型，不过关注的问题会有些不同。

这里我们按主要用途列了四类：功能仿真关注执行对不对，解析模型帮助估算性能，时序仿真展开指令和流水的执行过程，RTL仿真主要检查硬件逻辑。

**刚才提到的CANN CA model，就在第三类。** 我们主要利用它输出的指令和流水事件，分析热点、同步等待，以及计算和搬运的重叠情况。

有了这些原生能力，接下来看看，我们怎样把它用到HG-KDA里。

## 第四页：HG-KDA整体流程

''' + OVERVIEW_NOTES + '''

## 口径与待核对事项

- 当前会话中已看到模拟器双分支截图。包含成对profiling、知识库和代码推理的另一会话原图尚未取得；本Overview按本次用户描述绘制，具体模块关系可继续对照调整。
- 本页明确输入、输出，实线表示主要数据与决策路径，虚线表示迭代。多路依据按问题选择，图中并列位置表达各自作用。
- 入口名称：`msprof op`用于真机算子profiling，`msprof op simulator`用于模拟器采集。成对比较、代码区域关联、诊断整理和实验决策由工作流组织。
- 标题中的白盒指执行行为可观测、可分析。CA model的二进制动态库交付形式与其时序观测能力分别说明。
- 仿真分类按主要用途组织，实际工具能力可交叉；模拟器输出范围和模型误差需结合真机结果解释。
- 口述姓名按本次用户输入“张汉乐”；现有封面为“张韩乐”，此稿未修改原封面。
'''


def font(run, size, color, bold=False):
    run.font.name, run.font.size, run.font.bold = FONT, Pt(size), bold
    run.font.color.rgb = RGBColor.from_string(color)
    props = run._r.get_or_add_rPr()
    for tag in ('a:latin', 'a:ea', 'a:cs'):
        element = props.find(tag, props.nsmap)
        if element is None:
            element = OxmlElement(tag)
            props.append(element)
        element.set('typeface', FONT)


def textbox(slide, value, x, y, w, h, size=18, color=INK, bold=False, center=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    shape.name = value.replace('\n', ' ')
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    for i, line in enumerate(value.split('\n')):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = PP_ALIGN.CENTER if center else PP_ALIGN.LEFT
        p.space_before = p.space_after = Pt(0)
        p.line_spacing = 1.1
        run = p.add_run()
        run.text = line
        font(run, size, color, bold)
    return shape


def box(slide, name, title, body, x, y, w, h, endpoint=False, title_size=20, body_size=16):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shape.name = name
    shape.adjustments[0] = .12
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(NAVY if endpoint else PALE)
    shape.line.color.rgb = RGBColor.from_string(NAVY if endpoint else BORDER)
    shape.line.width = Pt(1)
    shape._element.spPr.append(OxmlElement('a:effectLst'))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = Inches(.07)
    frame.margin_top = frame.margin_bottom = Inches(.04)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate([title] + body.split('\n')):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        p.space_before = Pt(0)
        p.space_after = Pt(4 if i == 0 else 0)
        p.line_spacing = 1.05
        run = p.add_run()
        run.text = line
        font(run, title_size if i == 0 else body_size,
             'FFFFFF' if endpoint else NAVY if i == 0 else INK, i == 0)
    return shape


def edge(slide, points, dashed=False, arrow=True, name='流程连接',
         line_width=1.4, arrow_size='sm'):
    shapes = []
    for i, (start, end) in enumerate(zip(points, points[1:])):
        shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
            Inches(start[0]), Inches(start[1]), Inches(end[0]), Inches(end[1]))
        shape.name = name
        shape.line.color.rgb = RGBColor.from_string(NAVY if dashed else MUTED)
        shape.line.width = Pt(line_width)
        shape._element.spPr.append(OxmlElement('a:effectLst'))
        if dashed:
            shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        if arrow and i == len(points) - 2:
            node = OxmlElement('a:tailEnd')
            node.set('type', 'triangle')
            node.set('w', arrow_size)
            node.set('len', arrow_size)
            shape.line._get_or_add_ln().append(node)
        shapes.append(shape)
    return shapes


def main(template: Path, delivery: Path):
    REPORT.mkdir(parents=True, exist_ok=True)
    before = hashlib.sha256(template.read_bytes()).hexdigest()
    deck = Presentation(template)
    for item in list(deck.slides._sldIdLst):
        deck.part.drop_rel(item.rId)
        deck.slides._sldIdLst.remove(item)
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string('FFFFFF')
    textbox(slide, 'HG-KDA：从硬件证据到优化决策', .55, .48, 12.23, .65, 32, INK, True)
    textbox(slide, '输入', .45, 1.37, 1.7, .3, 16, NAVY, True)
    textbox(slide, '按当前问题选择分析依据', 2.57, 1.28, 3.05, .35, 17, MUTED, center=True)

    # Main paths stay solid; the single cycle is drawn below the working area.
    edge(slide, [(1.28, 2.85), (1.28, 3.30)])
    edge(slide, [(2.10, 3.97), (2.30, 3.97)], arrow=False)
    edge(slide, [(2.30, 2.265), (2.30, 5.485)], arrow=False)
    for y in [2.265, 3.875, 5.485]:
        edge(slide, [(2.30, y), (2.57, y)])
        edge(slide, [(5.62, y), (5.88, y)], arrow=False)
    edge(slide, [(5.88, 2.265), (5.88, 5.485)], arrow=False)
    edge(slide, [(5.88, 3.97), (6.17, 3.97)])
    edge(slide, [(7.77, 3.97), (8.12, 3.97)])
    edge(slide, [(9.77, 3.97), (10.12, 3.97)])
    edge(slide, [(11.42, 4.65), (11.42, 5.35)])
    edge(slide, [(10.52, 4.65), (10.52, 4.95), (9.95, 4.95),
                 (9.95, 6.83), (1.28, 6.83), (1.28, 4.65)],
         dashed=True, name='迭代反馈：保留或回退')

    box(slide, '输入：初始算子与任务约束', '初始算子',
        'shape / dtype\n参考实现\n验收要求', .45, 1.77, 1.65, 1.08,
        endpoint=True, title_size=20, body_size=11.5)
    box(slide, '当前迭代版本', '当前版本', '源码 / IR\n编译产物',
        .45, 3.30, 1.65, 1.35, title_size=20, body_size=16)
    box(slide, '分析依据1：成对profiling', '成对 profiling', '当前实现 vs 参考实现\n计算 / 搬运 / 启动差距',
        2.57, 1.69, 3.05, 1.15, title_size=21, body_size=16)
    box(slide, '分析依据2：模拟器重放', '模拟器重放', '指令热点 → 源码 / IR\n流水等待 / 空泡 / 重叠',
        2.57, 3.30, 3.05, 1.15, title_size=21, body_size=16)
    box(slide, '分析依据3：知识库与代码推理', '知识库＋代码推理', '已验证的方法与约束\n当前源码 / IR分析',
        2.57, 4.91, 3.05, 1.15, title_size=20, body_size=16)
    box(slide, '瓶颈假设与修改决策', '瓶颈假设', '对应具体代码\n确定修改位置',
        6.17, 3.30, 1.60, 1.35, title_size=20, body_size=15)
    box(slide, 'Agent修改', 'Agent修改', 'Triton / Host\n单算子 IR',
        8.12, 3.30, 1.65, 1.35, title_size=20, body_size=15)
    box(slide, '候选编译与验证', '编译与验证', '模拟器再次重放\n真机正确性 / 性能',
        10.12, 3.30, 2.60, 1.35, title_size=22, body_size=17)
    box(slide, '输出：已验收最佳实现与报告', '输出：已验收最佳', '算子实现＋性能与证据报告',
        10.12, 5.35, 2.60, .95, endpoint=True, title_size=20, body_size=14)
    textbox(slide, '保留 / 回退，继续下一轮', 6.02, 6.42, 3.70, .32, 17, NAVY, center=True)

    notes = slide.notes_slide
    if notes.notes_text_frame is None:
        notes.shapes._spTree.add_placeholder(notes.shapes._next_shape_id,
            'Notes Text', PP_PLACEHOLDER.BODY, 'horz', 'full', 13)
    notes.notes_text_frame.text = OVERVIEW_NOTES
    deck.core_properties.title = 'HG-KDA：从硬件证据到优化决策'
    deck.core_properties.subject = '输入输出明确的Overview初稿；三路依据与虚线迭代'
    deck.core_properties.author = 'HG-KDA'
    output = REPORT / 'HG-KDA-overview-初稿.pptx'
    deck.save(output)
    (REPORT / 'HG-KDA-overview-初稿.mmd').write_text(MERMAID, encoding='utf-8')
    (REPORT / '开场与前四页讲稿.md').write_text(OPENING, encoding='utf-8')
    result = Presentation(output)
    assert len(result.slides) == 1
    for shape in result.slides[0].shapes:
        assert min(shape.left, shape.top) >= 0
        assert shape.left + shape.width <= result.slide_width
        assert shape.top + shape.height <= result.slide_height
    assert hashlib.sha256(template.read_bytes()).hexdigest() == before
    validation = {'status': 'structure_passed', 'editable_native_shapes': True,
                  'inputs_explicit': True, 'outputs_explicit': True,
                  'three_analysis_routes': ['paired_profiling', 'simulator', 'knowledge_code_reasoning'],
                  'feedback_dashed': True, 'post_change_simulator_replay': True,
                  'template': str(template), 'template_sha256': before,
                  'template_preserved': True, 'visual_review': 'pending',
                  'original_other_session_diagram': 'unavailable; inferred from current user description'}
    (REPORT / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    delivery.mkdir(parents=True, exist_ok=True)
    for p in [output, REPORT / 'HG-KDA-overview-初稿.mmd', REPORT / '开场与前四页讲稿.md']:
        copy_material(p, delivery)
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--delivery-dir', type=Path, required=True)
    args = parser.parse_args()
    main(args.template.resolve(), args.delivery_dir.resolve())
