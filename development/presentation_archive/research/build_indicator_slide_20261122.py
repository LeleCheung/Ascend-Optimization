"""Build an editable comparison slide using the current reporting template."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

import build_indicator_comparison_20261121 as matrix
from presentation_materials_20261122 import copy_material


TITLE = '算子优化可获得的硬件证据'
SPEECH = '''这张表总结了人和Kernel Design Agent在不同平台上能够取得哪些优化证据。我们分成应用层、系统层、微架构层三大类。

应用层首先看耗时，这张表列的是Host调用和kernel执行时间；系统层进一步看计算量、访存量、并行度及利用率。微架构层又分三类：硬件规格、统计指标和时序指标。硬件规格给出容量、带宽和算力等资源约束，帮助Agent判断分块、数据复用及资源分配策略。

我们重点关注后面的统计指标和时序指标。统计指标能提示计算、访存方面的问题，但通常给出某个kernel或采集窗口内的总量、平均值和比例，具体发生在哪一段、怎样影响执行，还需要进一步分析。

例如，流水线利用率低，我们还想知道：它在等哪次数据搬运？哪条同步依赖阻塞了执行？哪些计算原本可以和搬运重叠？这些问题就需要进一步看真机采样或模拟器提供的时序证据。

接下来看NVIDIA和昇腾这两列。NVIDIA的工具链比较成熟，既有丰富的统计指标，也能通过真机采样定位指令热点、分析等待原因。本表把流水依赖和空泡标为部分，表示对详细流水时序的覆盖仍有边界。

昇腾提供了另一条可用的路径：CANN CA model。它能输出指令和流水时序，让我们观察热点、同步等待，以及计算和搬运的重叠情况。我们把这些事件关联到源码和IR，再据此提出具体修改。

所以，HG-KDA关注的是：怎样把硬件证据变成可检验的优化决策，并用真机确认收益。下面就用昇腾的具体案例，说明时序证据怎样带来新的判断和改法。'''

DISCUSSION = '''# 指标比较页讲稿

## 建议的中心句

统计指标提示瓶颈方向，时序证据进一步解释等待与重叠；KDA据此提出改法，再以真机验证。

## 口头讲稿

''' + SPEECH + '''

## 这一页的讲述顺序

用前半分钟交代三大类五小类，然后只举“流水线在等谁”一个例子。介绍NVIDIA的真机采样路径、昇腾的模拟器路径后，转到具体优化案例。其他厂商用一句“各平台可获得证据的覆盖和粒度存在差异，会影响Agent的优化成本”带过。

两列讲述可以缩成三句话：

> NVIDIA工具链比较成熟，既能给出统计指标，也能通过真机采样分析指令热点和等待。
>
> 昇腾提供CA model这条路径，让我们观察指令和流水时序，并关联到源码、IR。
>
> 我们接下来展示，Agent怎样把这些时序证据变成具体改法，并取得真机收益。

## 被追问时再展开

- **绿色/黄色**：主表沿用小组1122的厂商能力类别口径；绿色数量只表达表内类别覆盖。NVIDIA工具链的成熟度、精度、接口质量等，需要各自证据说明。
- **910B与950**：950已公开更多寄存器和峰值资料。910B4-1的完整子规格仍有缺项；910B实际优化已用到存储容量、核数、profiling和模拟器执行事件，改法和收益有真机验证。
- **应用层耗时**：本表列Host调用与kernel耗时。端到端延迟另需明确调用、同步、数据搬运等计时范围。
- **已有工作的覆盖**：“多数KDA覆盖前3类”需要对应论文或系统调研支持；本页直接说明我们关注的证据及作用。
- **Profiler与时序**：本表统计项通常按kernel/窗口聚合。Profiler也可提供时间线、PC采样或PM采样；证据粒度按具体数据说明。
- **缓存命中率例子**：仅有聚合命中率时，仍需定位到具体访问；后续若展示缓存机制，应提供对应地址、缓存事件或采样证据。本页使用已在案例中观察到的流水等待作例子。
- **模拟器的范围**：CA model能给出所采核和窗口的指令/流水事件；模拟器时序及模型假设需结合实际设备结果解释。时序深度、覆盖范围、采集成本分别评估。
- **国产平台**：按当前可获得数据说明缺项和接入困难。公开模拟器、采样和工具接口逐家核实，避免把文档缺项推广成硬件缺失。

## 产物使用

PPT页面只保留标题和Excel主表对应的原生可编辑表格，保留1122的43组合并，并附演讲者备注。可复制整页到现有汇报。PDF和PNG为该页的预览。现有《智源20261124汇报.pptx》保持原文件。
'''


def set_run_font(run, size, color, bold=False):
    run.font.name = 'Microsoft YaHei'
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color.lstrip('#'))
    properties = run._r.get_or_add_rPr()
    for tag in ('a:ea', 'a:cs'):
        element = properties.find(tag, properties.nsmap)
        if element is None:
            element = OxmlElement(tag)
            properties.append(element)
        element.set('typeface', 'Microsoft YaHei')


def text(slide, content, x, y, width, height, size, color, bold=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = False
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    paragraph = frame.paragraphs[0]
    paragraph.space_before = paragraph.space_after = Pt(0)
    paragraph.line_spacing = 1.0
    run = paragraph.add_run()
    run.text = content
    set_run_font(run, size, color, bold)
    return shape


def border(cell):
    properties = cell._tc.get_or_add_tcPr()
    for index, name in enumerate(('lnL', 'lnR', 'lnT', 'lnB')):
        old = properties.find(f'a:{name}', properties.nsmap)
        if old is not None:
            properties.remove(old)
        line = OxmlElement('a:' + name)
        line.set('w', '9525')
        fill = OxmlElement('a:solidFill')
        rgb = OxmlElement('a:srgbClr')
        rgb.set('val', matrix.THEME['border'].lstrip('#'))
        fill.append(rgb)
        line.append(fill)
        # DrawingML requires border elements before the fill. PowerPoint can
        # ignore borders placed after solidFill and show the style's white line.
        properties.insert(index, line)


def build(template: Path, delivery: Path):
    report = matrix.REPORT
    template_hash = hashlib.sha256(template.read_bytes()).hexdigest()
    source = matrix.load_workbook(matrix.OUTPUT)
    sheet = source['指标比较']
    expanded = matrix.expanded(sheet)
    deck = Presentation(template)
    for slide_id in list(deck.slides._sldIdLst):
        deck.part.drop_rel(slide_id.rId)
        deck.slides._sldIdLst.remove(slide_id)
    assert abs(deck.slide_width / 914400 - 13.33333333) < .001
    assert abs(deck.slide_height / 914400 - 7.5) < .001
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
    ink = matrix.THEME['ink']
    text(slide, TITLE, .55, .48, 12.23, .65, 32, ink, True)

    frame = slide.shapes.add_table(21, 9, Inches(.55), Inches(1.30),
                                   Inches(12.23), Inches(5.85))
    frame.name = '指标比较：与Excel一致的原生表格'
    table = frame.table
    table.first_row = table.first_col = table.horz_banding = table.vert_banding = False
    widths = [1.7, 3.43, 1.03, 1.04, 1.01, 1.01, .97, .97, 1.07]
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    for index, row in enumerate(table.rows):
        row.height = Inches(.37 if index == 0 else .274)
    group_names = {
        'A9': '微架构层\n硬件信息\n（硬件 Spec）',
        'A16': '微架构层\n统计指标\n（Profiler）',
        'A19': '微架构层\n时序指标\n（模拟器/真机采样）',
    }
    for row in range(1, 22):
        for col in range(1, 10):
            coordinate = sheet.cell(row, col).coordinate
            table.cell(row - 1, col - 1).text = group_names.get(
                coordinate, str(sheet.cell(row, col).value or ''))
    for area in sorted(sheet.merged_cells.ranges, key=lambda a: (a.min_row, a.min_col)):
        table.cell(area.min_row - 1, area.min_col - 1).merge(
            table.cell(area.max_row - 1, area.max_col - 1))

    for row in range(1, 22):
        for col in range(1, 10):
            cell = table.cell(row - 1, col - 1)
            if cell.is_spanned:
                continue
            source_cell = sheet.cell(row, col)
            cell.fill.solid()
            fill_color = source_cell.fill.fgColor.rgb[-6:] if source_cell.fill.fill_type == 'solid' else 'FFFFFF'
            cell.fill.fore_color.rgb = RGBColor.from_string(fill_color)
            cell.margin_left = Inches(.045)
            cell.margin_right = Inches(.035)
            cell.margin_top = cell.margin_bottom = 0
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.text_frame.word_wrap = True
            cell.text_frame.auto_size = MSO_AUTO_SIZE.NONE
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.LEFT if col == 2 and row > 1 else PP_ALIGN.CENTER
                paragraph.line_spacing = 1.0
                paragraph.space_before = paragraph.space_after = Pt(0)
                for run in paragraph.runs:
                    size = 14 if row == 1 else 11 if col == 1 else 12 if col == 2 else 14
                    color = 'FFFFFF' if row == 1 else source_cell.font.color.rgb[-6:]
                    set_run_font(run, size, color, row == 1 or col == 1 or (col >= 3 and cell.text != matrix.PART))
            border(cell)

    notes = slide.notes_slide
    if notes.notes_text_frame is None:
        notes.shapes._spTree.add_placeholder(notes.shapes._next_shape_id,
            'Notes Text', PP_PLACEHOLDER.BODY, 'horz', 'full', 13)
    notes.notes_text_frame.text = SPEECH
    deck.core_properties.title = TITLE
    deck.core_properties.subject = '对齐06_work深蓝模板；单页原生可编辑表格'
    deck.core_properties.author = 'HG-KDA'
    output = report / '指标比较-证据覆盖.pptx'
    deck.save(output)
    (report / '指标比较-讲稿.md').write_text(DISCUSSION, encoding='utf-8')

    result = Presentation(output)
    assert len(result.slides) == 1
    assert len(result.slides[0].shapes) == 2
    native = next(shape.table for shape in result.slides[0].shapes if shape.has_table)
    origins = {(area.min_row - 1, area.min_col - 1): area for area in sheet.merged_cells.ranges}
    for row in range(2, 22):
        for col in range(3, 10):
            expected = expanded[sheet.cell(row, col).coordinate]
            if native.cell(row - 1, col - 1).is_spanned:
                area = next(a for a in sheet.merged_cells.ranges if a.min_row <= row <= a.max_row
                            and a.min_col <= col <= a.max_col)
                actual = native.cell(area.min_row - 1, area.min_col - 1).text
            else:
                actual = native.cell(row - 1, col - 1).text
            assert actual == expected, (row, col, actual, expected)
    merge_count = sum(cell.is_merge_origin for row in native.rows for cell in row.cells)
    assert merge_count == len(origins) == 43
    for row in range(1, 21):
        properties = native.cell(row, 1)._tc.get_or_add_tcPr()
        assert [element.tag.rsplit('}', 1)[-1] for element in properties][:4] == ['lnL', 'lnR', 'lnT', 'lnB']
        assert properties.find('a:lnB/a:solidFill/a:srgbClr', properties.nsmap).get('val') == matrix.THEME['border'][1:]
    for shape in result.slides[0].shapes:
        assert shape.left >= 0 and shape.top >= 0
        assert shape.left + shape.width <= result.slide_width + 1
        assert shape.top + shape.height <= result.slide_height + 1
    assert hashlib.sha256(template.read_bytes()).hexdigest() == template_hash
    validation = {'status': 'passed', 'slides': 1, 'native_editable_table': True,
                  'visible_shapes': 2, 'layout': 'title_and_excel_table_only',
                  'metric_column_horizontal_borders': 'explicit; correct DrawingML order',
                  'expanded_conclusions_checked': 140, 'merged_ranges': merge_count,
                  'slide_bounds': 'passed', 'speaker_notes': True,
                  'template': str(template), 'template_sha256': template_hash,
                  'template_preserved': True, 'theme': matrix.THEME,
                  'visual_review': 'pending_native_export',
                  'delivery_directory': str(delivery)}
    (report / 'slide-validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    delivery.mkdir(parents=True, exist_ok=True)
    for path in [matrix.OUTPUT, output, report / '指标比较-讲稿.md']:
        copy_material(path, delivery)
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--delivery-dir', type=Path, required=True)
    args = parser.parse_args()
    build(args.template.resolve(), args.delivery_dir.resolve())
