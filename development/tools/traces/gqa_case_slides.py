"""节点11/22讲解页，原生可编辑图形与真实重放窗口。"""
from pathlib import Path
import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement

ROOT = Path(__file__).resolve().parents[3]
OUT = Path('E:/0_XSAI/00_KernelAgent/0_HGKDA_docs/BAAI-reports/06_work/material/04_算子案例/GQA')
STEM = 'GQA-节点11热点与节点22等待'
NAVY, PALE, BORDER, INK = '133F6D', 'E7EFF7', 'D3DCE6', '14181C'


class Canvas:
    def __init__(self, prs):
        self.slide = prs.slides.add_slide(prs.slide_layouts[6])
        self.slide.background.fill.solid()
        self.slide.background.fill.fore_color.rgb = RGBColor.from_string('FFFFFF')

    def shape(self, kind, x,y,w,h,fill=PALE,line=BORDER):
        s = self.slide.shapes.add_shape(kind,*[Inches(v/144) for v in (x,y,w,h)])
        s.fill.solid();s.fill.fore_color.rgb=RGBColor.from_string(fill)
        if line:
            s.line.color.rgb=RGBColor.from_string(line);s.line.width=Pt(.9)
        else:s.line.fill.background()
        s._element.spPr.append(OxmlElement('a:effectLst'))
        return s

    def box(self,x,y,w,h,fill=PALE,line=BORDER):
        s=self.shape(MSO_SHAPE.ROUNDED_RECTANGLE,x,y,w,h,fill,line)
        s.adjustments[0]=.06
        return s

    def text(self,x,y,w,h,value,size=34,bold=False,color=NAVY,center=False,code=False):
        s=self.slide.shapes.add_textbox(*[Inches(v/144) for v in (x,y,w,h)])
        tf=s.text_frame;tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0
        tf.word_wrap=False;tf.vertical_anchor=MSO_ANCHOR.MIDDLE
        for i,line in enumerate(value.split('\n')):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
            p.alignment=PP_ALIGN.CENTER if center else PP_ALIGN.LEFT
            p.space_after=Pt(3)
            r=p.add_run();r.text=line
            r.font.name='Consolas' if code else 'Microsoft YaHei'
            r.font.size=Pt(size/2);r.font.bold=bold;r.font.color.rgb=RGBColor.from_string(color)
        return s

    def arrow(self,x,y,w):
        s=self.shape(MSO_SHAPE.RIGHT_ARROW,x,y,w,24,NAVY,None)
        s.adjustments[0]=.14
        s.adjustments[1]=.8


def main():
    prs=Presentation();prs.slide_width=Inches(13.333333);prs.slide_height=Inches(7.5)
    a=Canvas(prs)
    a.text(80,60,1760,80,'11  热点区域定位：缩窄 QK 交接',54,True,INK)
    a.box(80,210,520,335)
    a.text(110,225,460,55,'CSV：热点指令统计',36,True)
    a.text(110,300,460,115,'PC  0x10d11568\nFIX_L0C_TO_DST\nFIXP',34,True,code=True)
    a.text(110,455,460,55,'83.49 μs / 19 次',38,True)
    a.arrow(620,365,60)
    a.box(700,210,1140,335)
    a.text(730,225,1080,55,'源码区域定位',36,True)
    a.text(730,295,1080,50,'trace.args.code 行号栈 → candidate.py:217',32)
    a.text(730,355,1080,55,'qk = tl.dot(q, tl.trans(k)) * qk_scale',36,code=True)
    a.text(730,435,1080,65,'QK 计算区域\n源码行号栈 + AST 结构',32)
    # 这些数值来自同一父版；量与窗口时间分别展示。
    a.text(80,580,1760,50,'IR 确认：QK 为 128×256 FP32，PV 为 128×128 FP32',35,True)
    for x,label,dtype,amount in [(80,'修改前','FP32 交接','128 KiB / 块'),(1050,'修改后','FP16 交接 → FP32 计算','64 KiB / 块')]:
        a.box(x,670,790,190)
        a.text(x+25,685,740,45,label,32,True)
        a.text(x+25,742,740,50,dtype,38,True)
        a.text(x+25,800,740,45,amount,34)
    a.arrow(900,748,120)
    a.text(80,890,1760,50,'值域检查 → dot 后缩窄 → 验证 FIXP 转换与搬运量',34,True)
    a.text(80,985,1760,60,'L0C→GM：96 → 64 MiB        完整调用：205.284 → 196.158 μs',38,True)
    a.slide.notes_slide.notes_text_frame.text=(
        '对应attempt-033。CSV按PC统计指令次数和运行时间，trace.args.code提供编译源码位置/内联栈，按PC关联。'
        'AST解析所属函数/循环；历史自动区域摘要曾取外层288行，内层217行来自完整source_stack与IR核对。'
        '本图将源码行号栈与AST作为区域定位能力合并展示，不改写历史自动分组结果。'
        'QK128KiB/块，PV64KiB/块，不能把83.49us/19次和29.22us/18次当作等字节效率比较。'
        'case4输入输出BF16，rawQK FP32；固定输入[-1,1]，D128，绝对值界128，FP16范围足够，舍入由原checker验证。'
        '记录显示考虑同一GQA早期028的FP16分数负结果，没有证明来自26项某个FlagGems的直接迁移。'
        '详情：development/reports/gqa-meeting-traces-20261122/hotspot-and-wait-explanation.md。')

    b=Canvas(prs)
    b.text(80,60,1760,80,'22  等待窗口利用：提前计算分母与倒数',52,True,INK)
    # 当前实现的物理交接路径；GM以两个逻辑交接区呈现。
    for x,w,title,expr in [(80,260,'Vector / UB','P'),(430,180,'GM','P'),
                           (700,360,'Cube / L0C','PV'),(1150,180,'GM','PV'),
                           (1420,420,'Vector / UB','O = PV × inv')]:
        b.box(x,205,w,120)
        b.text(x+10,212,w-20,45,title,30,True,center=True)
        b.text(x+10,263,w-20,50,expr,36,True,center=True)
    for x,w,label in [(350,70,'MTE3'),(620,70,'MTE2'),(1070,70,'FIXP'),(1340,70,'MTE2')]:
        b.text(x-10,168,w+20,32,label,28,center=True)
        b.arrow(x,258,w)
    for x,title,lines in [
        (80,'修改前','send(P)\nwait(PV) → receive(PV)\nd = row_sum(P); inv = 1/d'),
        (1000,'IR 重排后','send(P)\nd = row_sum(P); inv = 1/d\nwait(PV) → receive(PV)')]:
        b.text(x,355,840,45,title,34,True)
        b.box(x,410,840,165)
        b.text(x+25,422,790,140,lines,32,code=True)
    b.text(80,592,1760,50,'重放时间轴：以各自 P 搬出起点为 0，单位 μs',30,True)

    root=ROOT/'development/reports/gqa-goal-20261121/gqa-ir-qk-receive-prefetch'
    before=json.loads((root/'post-ir-003/qk-receive-mechanism.json').read_text(encoding='utf-8'))['example_us']
    after=json.loads((root/'post-ir-004/sum-reciprocal-mechanism.json').read_text(encoding='utf-8'))['example_us']
    before['sum_inv']=[before['denominator_reduce'][0],46.900]
    after['sum_inv']=[after['sum'][0],after['reciprocal'][1]]
    def timeline(x,data):
        start=data['p_send'][0];px=x+205;scale=96
        for tick in [0,2,4,6]:
            xx=px+tick*scale
            b.shape(MSO_SHAPE.RECTANGLE,xx,675,1,230,BORDER,None)
            b.text(xx-15,643,40,30,str(tick),28,center=True)
        rows=[('MTE3', 'p_send', 'P',NAVY),
              ('FLOWCTRL','next_qk_ready_wait','QK 等待','54779B'),
              ('MTE2', 'pv_receive','',NAVY),
              ('VECTOR', 'sum_inv','',NAVY)]
        for i,(lane,key,label,color) in enumerate(rows):
            y=680+i*57
            b.text(x,y-4,195,40,lane,28,True)
            spans=[(key,label)]
            if lane=='FLOWCTRL':spans.append(('pv_ready_wait','PV 等待'))
            for event,lab in spans:
                t0,t1=data[event];left=max(0,t0-start);right=t1-start
                b.shape(MSO_SHAPE.RECTANGLE,px+left*scale,y,(right-left)*scale,28,color,None)
                if lab:b.text(px+left*scale,y-2,(right-left)*scale,32,lab,25,True,'FFFFFF',center=True)
            if key=='pv_receive':
                b.text(px+(data[key][1]-start)*scale+8,y-4,145,36,'接收 PV',27)
            if key=='sum_inv':
                pos=px+(data[key][0]-start)*scale
                b.text(pos-15,y+29,160,36,'sum / VREC',27)
    timeline(80,before);timeline(1000,after)
    b.text(80,940,1760,42,'FLOWCTRL：WAIT_FLAG_DEVI    PC 0x10d11bb0',30,True,code=True)
    b.text(80,995,1760,55,'d 和 inv 只依赖 P        144.437 → 133.321 μs        耗时下降 7.70%',37,True)
    b.slide.notes_slide.notes_text_frame.text=(
        '对应IR004。send/wait/receive为教学伪代码。真实IR为sync_block_wait flag0及流水间wait_flag；'
        '机器指令WAIT_FLAG_DEVI PC0x10d11bb0处于FLOWCTRL，VREC处于VECTOR，FIXP为Cube搬出流水。'
        'P由UB经MTE3写GM，Cube经MTE2载入L1并经L0输入参与PV；PV从L0C经FIXP到GM，再经MTE2到UB。'
        '时间轴使用core0.veccore0原窗口，左右各以P发送起点为0，不比较不同重放的绝对时间原点。'
        '仅画所选事件，QK等待在左图起点前已经开始。sum/VREC原46.336..46.900，改后41.380..41.945。'
        '修改后实际重叠P发送和nextQK等待；没有把它误标为PV专属等待。'
        '原16行sum/VREC链前移，保留必需同步及全部计算，缓冲依赖审查通过。'
        '两图未画下一QK接收，以突出PV接收与独立分母的顺序。'
        '数据来自post-ir-003/qk-receive-mechanism.json与post-ir-004/sum-reciprocal-mechanism.json。')
    OUT.mkdir(parents=True,exist_ok=True)
    for slide in prs.slides:
        for shape in slide.shapes:
            assert shape.left>=0 and shape.top>=0
            assert shape.left+shape.width<=prs.slide_width+20
            assert shape.top+shape.height<=prs.slide_height+20
    prs.save(OUT/f'{STEM}.pptx')
    print('Saved 2 editable slides')


if __name__=='__main__':main()
