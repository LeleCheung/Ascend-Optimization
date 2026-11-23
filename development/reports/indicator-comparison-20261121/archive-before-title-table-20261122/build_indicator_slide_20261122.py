"""Build an editable comparison slide using the current reporting template."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.enum.shapes import PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

import build_indicator_comparison_20261121 as matrix


TITLE = '算子优化可获得的硬件证据'
TAKEAWAY = 'HG-KDA：硬件证据 → 瓶颈假设 → 源码/IR修改 → 真机验证'
FOOTNOTE = '厂商级能力概览；具体型号及版本范围见依据页。资料：小组《指标比较-1122》及补充复核。'
SPEECH = '''这张表回答的是：Kernel Design Agent在不同平台上，能拿到哪些优化证据。我们分成应用层、系统层、微架构层三大类，其中微架构层又分硬件规格、统计指标和时序指标，共五类。

应用层先看Host调用和kernel耗时；系统层看计算量、访存量、并行度及利用率。硬件规格给出容量、带宽和算力等资源约束，帮助Agent判断分块、数据复用和资源分配的可行范围。

这页的统计指标通常按kernel或采集窗口聚合，能指出计算、搬运等大方向，但仅有总量和平均值，往往难以解释某段流水为什么等待、两条流水有没有重叠。这时需要进一步看真机采样或模拟器时序。Profiler本身也可以提供时间线和采样；这里按证据内容分层。

NVIDIA的工具链覆盖广，PC采样、SASS分析和等待归因提供了较细的信息。依赖与空泡标成部分，体现它们在本表所关注的详细流水时序上的覆盖边界。

昇腾的特点是提供CANN CA model。即使模型以动态库形式交付，我们仍可以利用输出的指令、同步等待和流水事件进行分析，再关联到源码与IR。当前910B4-1的部分完整规格仍需核实；950已有更丰富的公开资料，所以这张厂商级能力表沿用小组判定，具体型号范围单独保留。

HG-KDA的重点，是把这些证据转成可检验的优化假设，指导Triton或IR修改，编译后继续模拟器重放，并用真机确认正确性和性能。其余平台也有各自的分析工具；可获取证据的覆盖、粒度和接口稳定性，会影响Agent判断瓶颈与筛选方案的成本。'''

DISCUSSION = '''# 指标比较页讲稿

## 建议的中心句

KDA取得证据，形成瓶颈假设，修改算子，再以正确性和实际性能验证。证据的覆盖与粒度影响优化判断及试错成本。

## 口头讲稿

''' + SPEECH + '''

## 汇报时的具体口径

- **绿色/黄色**：主表沿用小组1122的厂商能力类别口径；绿色数量只表达表内类别覆盖。NVIDIA工具链的成熟度、精度、接口质量等，需要各自证据说明。
- **910B与950**：950已公开更多寄存器和峰值资料。910B4-1的完整子规格仍有缺项；910B实际优化已用到存储容量、核数、profiling和模拟器执行事件，改法和收益有真机验证。
- **应用层耗时**：本表列Host调用与kernel耗时。端到端延迟另需明确调用、同步、数据搬运等计时范围。
- **Profiler与时序**：本表统计项通常按kernel/窗口聚合。Profiler也可提供时间线、PC采样或PM采样；证据粒度按具体数据说明。
- **模拟器的范围**：CA model能给出所采核和窗口的指令/流水事件；模拟器时序及模型假设需结合实际设备结果解释。时序深度、覆盖范围、采集成本分别评估。
- **国产平台**：按当前可获得数据说明缺项和接入困难。公开模拟器、采样和工具接口逐家核实，避免把文档缺项推广成硬件缺失。

## 产物使用

PPT为单页原生可编辑表格，保留1122的43组合并，可复制整页到现有汇报。PDF可作为图片或对象插入，排版通常更适合使用原生PPT页或高清PNG。现有《智源20261124汇报.pptx》保持原文件。
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
    for name in ('lnL', 'lnR', 'lnT', 'lnB'):
        old = properties.find(f'a:{name}', properties.nsmap)
        if old is not None:
            properties.remove(old)
        line = OxmlElement('a:' + name)
        line.set('w', '6350')
        fill = OxmlElement('a:solidFill')
        rgb = OxmlElement('a:srgbClr')
        rgb.set('val', matrix.THEME['border'].lstrip('#'))
        fill.append(rgb)
        line.append(fill)
        properties.append(line)


def build(template: Path, delivery: Path):
    root = matrix.ROOT
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
    navy, ink, muted = [matrix.THEME[x] for x in ('navy', 'ink', 'muted')]
    text(slide, TITLE, .55, .48, 12.23, .65, 32, ink, True)
    rule = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
        Inches(.55), Inches(1.05), Inches(12.78), Inches(1.05))
    rule.line.color.rgb = RGBColor.from_string('E8ECF0')
    rule.line.width = Pt(1)
    text(slide, '应用层、系统层、微架构层：从耗时与资源约束，到统计与时序证据',
         .55, 1.18, 12.23, .27, 15, muted)

    frame = slide.shapes.add_table(21, 9, Inches(.55), Inches(1.58),
                                   Inches(12.23), Inches(4.99))
    frame.name = '指标比较：与Excel一致的原生表格'
    table = frame.table
    table.first_row = table.first_col = table.horz_banding = table.vert_banding = False
    widths = [1.7, 3.43, 1.03, 1.04, 1.01, 1.01, .97, .97, 1.07]
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    for index, row in enumerate(table.rows):
        row.height = Inches(.35 if index == 0 else .232)
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
                    size = 13 if row == 1 else 10.5 if col == 1 else 11.5 if col == 2 else 13
                    color = 'FFFFFF' if row == 1 else source_cell.font.color.rgb[-6:]
                    set_run_font(run, size, color, row == 1 or col == 1 or (col >= 3 and cell.text != matrix.PART))
            border(cell)

    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
        Inches(.55), Inches(6.68), Inches(12.78), Inches(6.68))
    line.line.color.rgb = RGBColor.from_string(navy.lstrip('#'))
    line.line.width = Pt(1.1)
    text(slide, TAKEAWAY, .55, 6.81, 12.23, .35, 20, navy, True)
    text(slide, FOOTNOTE, .55, 7.25, 12.23, .17, 9, muted)
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
    for shape in result.slides[0].shapes:
        assert shape.left >= 0 and shape.top >= 0
        assert shape.left + shape.width <= result.slide_width + 1
        assert shape.top + shape.height <= result.slide_height + 1
    assert hashlib.sha256(template.read_bytes()).hexdigest() == template_hash
    validation = {'status': 'passed', 'slides': 1, 'native_editable_table': True,
                  'expanded_conclusions_checked': 140, 'merged_ranges': merge_count,
                  'slide_bounds': 'passed', 'speaker_notes': True,
                  'template': str(template), 'template_sha256': template_hash,
                  'template_preserved': True, 'theme': matrix.THEME,
                  'visual_review': 'pending_native_export',
                  'delivery_directory': str(delivery)}
    (report / 'slide-validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    delivery.mkdir(parents=True, exist_ok=True)
    for path in [matrix.OUTPUT, output, report / '指标比较-讲稿.md']:
        shutil.copy2(path, delivery / path.name)
    print(json.dumps(validation, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--delivery-dir', type=Path, required=True)
    args = parser.parse_args()
    build(args.template.resolve(), args.delivery_dir.resolve())
