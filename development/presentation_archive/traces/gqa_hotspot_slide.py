"""GQA 节点11热点定位图：深蓝汇报风格、原生可编辑形状。"""
from pathlib import Path
from html import escape
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement

OUT = Path('E:/0_XSAI/00_KernelAgent/0_HGKDA_docs/BAAI-reports/06_work/material/04_算子案例/GQA')
STEM = 'GQA-热点到源码定位'
NAVY, PALE, BORDER, INK = '133F6D', 'E7EFF7', 'D3DCE6', '14181C'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333333), Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string('FFFFFF')
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">',
           '<rect width="1920" height="1080" fill="white"/>']

    def box(x, y, w, h):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, *[Inches(v / 144) for v in (x, y, w, h)])
        shape.adjustments[0] = .08
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(PALE)
        shape.line.color.rgb = RGBColor.from_string(BORDER)
        shape.line.width = Pt(1)
        shape._element.spPr.append(OxmlElement('a:effectLst'))
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#{PALE}" stroke="#{BORDER}" stroke-width="2"/>')

    def text(x, y, w, h, value, size=38, bold=False, color=NAVY, font='Microsoft YaHei', center=True):
        shape = slide.shapes.add_textbox(*[Inches(v / 144) for v in (x, y, w, h)])
        frame = shape.text_frame
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        frame.word_wrap = False
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if center else PP_ALIGN.LEFT
        r = p.add_run()
        r.text = value
        r.font.name, r.font.size, r.font.bold = font, Pt(size / 2), bold
        r.font.color.rgb = RGBColor.from_string(color)
        anchor = 'middle' if center else 'start'
        svg.append(f'<text x="{x+w/2 if center else x}" y="{y+h/2}" dominant-baseline="central" text-anchor="{anchor}" font-family="{font}" font-size="{size}" font-weight="{700 if bold else 400}" fill="#{color}">{escape(value)}</text>')

    def arrow(y, h, label):
        shape = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, *[Inches(v/144) for v in (947, y, 26, h)])
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(NAVY)
        shape.line.fill.background()
        shape._element.spPr.append(OxmlElement('a:effectLst'))
        svg.append(f'<path d="M955 {y}H965V{y+h-16}H973L960 {y+h}L947 {y+h-16}H955Z" fill="#{NAVY}"/>')
        text(1000, y+(h-50)/2, 350, 50, label, 32, center=False)

    text(80, 65, 1760, 85, '从热点指令定位 Triton 源码', 58, True, INK, center=False)
    box(360, 210, 1200, 205)
    text(390, 230, 1140, 52, '模拟器热点', 36, True)
    text(390, 288, 1140, 55, 'FIX_L0C_TO_DST  ·  FIXP', 42, True, font='Consolas')
    text(390, 349, 1140, 47, 'PC  0x10d11568', 34, font='Consolas')
    arrow(427, 85, '源码调用栈')
    box(360, 525, 1200, 160)
    text(390, 540, 1140, 52, 'candidate.py · 第 217 行', 36, True)
    text(390, 609, 1140, 50, 'qk = tl.dot(q, tl.trans(k)) * qk_scale', 40, font='Consolas')
    arrow(697, 85, 'AST 源码结构')
    box(360, 795, 1200, 160)
    text(390, 813, 1140, 55, '_gqa_tile_part_body', 40, True, font='Consolas')
    text(390, 882, 1140, 50, 'query tile 循环', 38, True)

    slide.notes_slide.notes_text_frame.text = (
        '对应紧凑trace进展11：attempt-033，QK交接FP16缩窄。热点采自修改前attempt-031。'
        'PC 0x10d11568通过完整source_stack对应candidate.py:217；AST核查确定函数及query tile循环。'
        '详见Ascend-Optimization/development/reports/gqa-meeting-traces-20261122/source-mapping-and-compilation.md。'
        '配色取自智源20261124汇报.pptx：133F6D / E7EFF7 / D3DCE6。')
    prs.save(OUT / f'{STEM}.pptx')
    svg.append('</svg>')
    content = '\n'.join(svg)
    (OUT / f'{STEM}.svg').write_text(content, encoding='utf-8')
    (OUT / f'{STEM}.html').write_text('<!doctype html><meta charset="utf-8"><title>GQA 热点定位</title><style>body{margin:0;background:white}svg{width:100%;height:auto;display:block}</style>'+content, encoding='utf-8')
    print(OUT / f'{STEM}.pptx')


if __name__ == '__main__':
    main()
