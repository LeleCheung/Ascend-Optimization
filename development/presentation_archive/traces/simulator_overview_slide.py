"""生成芯片仿真分类与本文定位的独立开场幻灯片。"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "development/reports/group-meeting-20261119"
DECK = OUTPUT / "simulator-overview-blue-20261122.pptx"
FONT = "Microsoft YaHei"
BG = "FFFFFF"
INK = "14181C"
BLUE = "133F6D"
PALE_BLUE = "E7EFF7"
PALE_GRAY = "F3F5F7"


def font(run, size, color=INK, bold=False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)
    props = run._r.get_or_add_rPr()
    for tag in ("a:latin", "a:ea", "a:cs"):
        node = props.find(tag, props.nsmap)
        if node is None:
            node = OxmlElement(tag)
            props.append(node)
        node.set("typeface", FONT)


def text(slide, value, x, y, w, h, size=20, color=INK, bold=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.word_wrap = False
    frame.margin_left = frame.margin_right = Inches(0)
    frame.margin_top = frame.margin_bottom = Inches(0)
    for i, line in enumerate(value.split("\n")):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        p.line_spacing = 1.1
        run = p.add_run()
        run.text = line
        font(run, size, color, bold)
    return shape


def fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor.from_string(color)
    shape.line.fill.background()


def main():
    p = Presentation()
    p.slide_width = Inches(13.333)
    p.slide_height = Inches(7.5)
    p.core_properties.title = "芯片仿真与性能建模：四类工具与本文定位"
    p.core_properties.subject = "白盒模拟器驱动的多层次优化，以昇腾算子优化为例"
    p.core_properties.author = "HG-KDA"
    slide = p.slides.add_slide(p.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string(BG)

    text(slide, "芯片仿真与性能建模", 0.55, 0.48, 12.2, 0.65, 32, INK, True)

    rows = [
        ["类型", "主要关注", "代表工具"],
        ["功能仿真", "功能正确性", "QEMU"],
        ["性能解析模型", "性能估算", "NPUMeter"],
        ["时序仿真", "指令时序与流水", "gem5、CANN CA model"],
        ["RTL 仿真", "硬件逻辑正确性", "VCS"],
    ]
    heights = [0.65, 0.85, 0.85, 0.85, 0.85]
    shape = slide.shapes.add_table(
        5, 3, Inches(0.55), Inches(1.58), Inches(12.23), Inches(sum(heights))
    )
    table = shape.table
    table.first_row = True
    table.horz_banding = False
    for column, width in zip(table.columns, [2.90, 3.60, 5.73]):
        column.width = Inches(width)
    for i, values in enumerate(rows):
        table.rows[i].height = Inches(heights[i])
        for j, value in enumerate(values):
            cell = table.cell(i, j)
            cell.text = value
            cell.margin_left = Inches(0.22)
            cell.margin_right = Inches(0.15)
            cell.margin_top = Inches(0.06)
            cell.margin_bottom = Inches(0.06)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.text_frame.word_wrap = False
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor.from_string(
                BLUE if i == 0 else PALE_BLUE if i == 3 else PALE_GRAY if i == 2 else BG
            )
            for paragraph in cell.text_frame.paragraphs:
                paragraph.space_before = Pt(0)
                paragraph.space_after = Pt(0)
                paragraph.line_spacing = 1.1
                for run in paragraph.runs:
                    font(run, 25 if i else 24, "FFFFFF" if i == 0 else BLUE if i == 3 else INK,
                         i == 0 or j == 0)
            props = cell._tc.get_or_add_tcPr()
            for edge in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                line = OxmlElement(edge)
                line.set("w", "12700")
                solid = OxmlElement("a:solidFill")
                color = OxmlElement("a:srgbClr")
                color.set("val", BG)
                solid.append(color)
                line.append(solid)
                props.append(line)

    rule = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(5.98), Inches(12.23), Inches(0.025)
    )
    fill(rule, BLUE)
    text(slide, "本工作采用 CANN CA model，分析指令热点与流水时序。",
         0.65, 6.28, 12.0, 0.62, 26, BLUE, True)

    slide.notes_slide.notes_text_frame.text = """建议放置：汇报封面后的背景介绍页，随后进入研究问题与优化闭环。

口播：先用一页回顾芯片仿真与性能建模工具。功能仿真关注程序是否正确执行，解析模型用于估算性能与搜索参数，时序模拟器展开执行中的流水、依赖和资源竞争，RTL 仿真面向硬件实现验证。本文通过昇腾算子指令级性能仿真，观察热点和可重叠窗口，指导 Triton 源码及单算子 IR 优化，并通过真机验证收益。

名称与工具关系：本页 NPU-Simulator 对应用户指定的昇腾官方仓库 https://gitcode.com/cann/npu-simulator 。按用户说明，其底层模型以动态链接库形式提供。msOpProf simulator 是调用 CANN 仿真库并采集、解析性能数据的工具入口。本工程使用 CANN 9.0.0 的 msprof op simulator 命令与 Ascend910B4 仿真库。CA model 的名称在本页按用户提供的术语记录。

本文的白盒含义：执行过程可观测、可分析，具体体现为指令热点、源码映射和流水时序。库的交付形式与观测能力分别描述。

代表工具主要能力可以交叉。表中 gem5 指其详细时序模型；NPUMeter 归入解析性能模型与自动参数搜索。功能仿真在本页概括为功能正确性，RTL 仿真关注硬件逻辑正确性。

版式：独立单页，16:9；采用参考汇报的深蓝色 #133F6D、白色与浅灰色。主表为三列四类，所有单元格保持单行；正文为 25 pt，表头为 24 pt，下方本文采用为 26 pt。来源与工具关系保存于讲者备注。

来源：
NPU-Simulator（用户提供）：https://gitcode.com/cann/npu-simulator
msOpProf 官方指令级性能仿真说明：https://github.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_simulator_performance_data.md
NPUMeter，ACM TACO 2026：https://doi.org/10.1145/3820380
gem5，The gem5 Simulator: Version 20.0+：https://arxiv.org/abs/2007.03152
QEMU：https://www.qemu.org/docs/master/about/index.html
"""
    for item in slide.shapes:
        assert item.left >= 0 and item.top >= 0
        assert item.left + item.width <= p.slide_width + Inches(0.01)
        assert item.top + item.height <= p.slide_height + Inches(0.01)
    p.save(DECK)
    print(DECK)


if __name__ == "__main__":
    main()
