"""GQA 三步计算与抽象执行单元：单页原生 PowerPoint。"""
from pathlib import Path
from html import escape
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement

OUT = Path('E:/0_XSAI/00_KernelAgent/0_HGKDA_docs/BAAI-reports/06_work/material/04_算子案例/GQA')
STEM = 'GQA-三步计算与硬件执行'
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

    def box(x, y, w, h, fill=PALE, stroke=BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, *[Inches(v/144) for v in (x,y,w,h)])
        shape.adjustments[0] = .07
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(fill)
        shape.line.color.rgb = RGBColor.from_string(stroke)
        shape.line.width = Pt(1)
        shape._element.spPr.append(OxmlElement('a:effectLst'))
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#{fill}" stroke="#{stroke}" stroke-width="2"/>')

    def text(x,y,w,h,value,size=36,bold=False,color=NAVY,font='Microsoft YaHei',center=True):
        shape = slide.shapes.add_textbox(*[Inches(v/144) for v in (x,y,w,h)])
        frame = shape.text_frame
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        frame.word_wrap = False
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if center else PP_ALIGN.LEFT
        r = p.add_run()
        r.text = value
        r.font.name, r.font.size, r.font.bold = font, Pt(size/2), bold
        r.font.color.rgb = RGBColor.from_string(color)
        svg.append(f'<text x="{x+w/2 if center else x}" y="{y+h/2}" dominant-baseline="central" text-anchor="{"middle" if center else "start"}" font-family="{font}" font-size="{size}" font-weight="{700 if bold else 400}" fill="#{color}">{escape(value)}</text>')

    def arrow(points, label=None, label_box=None):
        for i, ((x0,y0),(x1,y1)) in enumerate(zip(points,points[1:])):
            shape=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,*[Inches(v/144) for v in (x0,y0,x1,y1)])
            shape.line.color.rgb=RGBColor.from_string(NAVY)
            shape.line.width=Pt(1.2)
            shape._element.spPr.append(OxmlElement('a:effectLst'))
            if i==len(points)-2:
                end=OxmlElement('a:tailEnd')
                end.set('type','triangle');end.set('w','lg');end.set('len','lg')
                shape._element.spPr.find('{http://schemas.openxmlformats.org/drawingml/2006/main}ln').append(end)
        coords=' '.join(f'{x},{y}' for x,y in points)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="#{NAVY}" stroke-width="2.4" marker-end="url(#tip)"/>')
        if label: text(*label_box,label,28)

    svg.append(f'<defs><marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#{NAVY}"/></marker></defs>')
    text(80,55,1760,80,'GQA：三步计算与硬件执行',56,True,INK,center=False)
    text(80,145,1760,50,'case 4：每 8 个 Q 头共享一组 K / V',33,center=False)
    for x,title,line1,line2 in [
        (80,'① QK 得分','S = scale × QKᵀ','scale = 1 / √D'),
        (680,'② 指数权重','m = row_max(S)','P = exp(S − m)'),
        (1280,'③ 加权汇总与归一化','d = row_sum(P)','O = PV / d'),
    ]:
        box(x,225,560,205)
        text(x+15,240,530,50,title,35,True)
        text(x+15,310,530,45,line1,34,font='Consolas')
        text(x+15,365,530,45,line2,34,font='Consolas')

    text(80,465,1760,50,'Cube / Vector 执行与存储路径',36,True,center=False)
    # GM在两处表示同一全局地址空间中的输入及交接区。
    for x,w,title,line1,line2 in [
        (80,200,'GM','Q / K / V','P'),
        (390,180,'L1','Q / K / V','P'),
        (680,180,'L0A / L0B','',''),
        (970,300,'Cube Unit','QKᵀ','PV'),
        (1380,180,'L0C','QKᵀ','PV'),
        (1670,170,'GM','交接缓冲',''),
    ]:
        box(x,585,w,165)
        text(x+8,595,w-16,48,title,32,True)
        if line1:text(x+8,647,w-16,42,line1,30)
        if line2:text(x+8,696,w-16,42,line2,30)
    for x0,x1,label in [(280,390,'MTE2'),(570,680,'MTE1'),(860,970,None),(1270,1380,None),(1560,1670,'FIXP')]:
        arrow([(x0+8,665),(x1-8,665)],label,(x0-8,607,x1-x0+16,40) if label else None)

    box(80,815,620,175,'FFFFFF')
    text(100,827,580,45,'FLOWCTRL / 同步',33,True)
    text(100,885,580,40,'SET_FLAG / WAIT_FLAG',29,font='Consolas')
    text(100,930,580,40,'sync_block_set / sync_block_wait',28,font='Consolas')
    box(780,810,520,190)
    text(800,822,480,50,'Vector Unit',38,True)
    text(800,883,480,42,'× scale    row_max    exp',30,font='Consolas')
    text(800,938,480,42,'row_sum    ÷ d',32,font='Consolas')
    box(1460,835,180,145)
    text(1470,845,160,45,'UB',35,True)
    text(1470,905,160,50,'QKᵀ / PV / P',25)
    arrow([(1745,750),(1745,865),(1645,865)],'MTE2',(1625,775,110,40))
    arrow([(1645,950),(1800,950),(1800,755)],'MTE3',(1650,960,180,40))
    arrow([(1450,865),(1310,865)],'QKᵀ / PV',(1300,815,160,40))
    arrow([(1310,960),(1450,960)],'P',(1300,970,160,40))
    slide.notes_slide.notes_text_frame.text = (
        'case4：Q=[4,256,64,128]，K/V=[4,256,8,128]，BF16，非因果，scale=1/sqrt(128)。'
        '数学公式等价于softmax(QK^T/sqrt(D))V。P为未归一化指数权重，d逐行广播。'
        '左GM与右GM表示同一全局内存地址空间，右侧突出交接缓冲；写入的P可由左侧GM→L1路径送回Cube。'
        '图为相关资源和数据路径，重复L1中的P/QKV表示不同矩阵乘的操作数，不表示同时驻留全部数据。'
        '节点7对应GM→L1重复加载；节点11对应L0C→FIXP→GM的QK交接类型；节点22对应Vector独立归约跨同步等待。'
        'MTE2负责GM输入搬运，MTE1负责L1到L0输入，MTE3负责UB输出，FIXP负责L0C结果搬出。'
        'FLOWCTRL/sync_block为控制/同步含义，SET_FLAG/WAIT_FLAG可关联不同生产消费流水，不表示这些指令全位于同一物理流水。'
        '省略输出O写回路径；实际PV接收后Vector完成除以d并输出。')
    prs.save(OUT / f'{STEM}.pptx')
    svg.append('</svg>')
    content = '\n'.join(svg)
    (OUT / f'{STEM}.svg').write_text(content,encoding='utf-8')
    (OUT / f'{STEM}.html').write_text('<!doctype html><meta charset="utf-8"><title>GQA 三步计算与硬件执行</title><style>body{margin:0;background:white}svg{width:100%;height:auto;display:block}</style>'+content,encoding='utf-8')
    print('Created editable GQA introduction slide')


if __name__ == '__main__':
    main()
