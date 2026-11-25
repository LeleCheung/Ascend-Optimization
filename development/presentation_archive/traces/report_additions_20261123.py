from pathlib import Path
from copy import deepcopy
from pptx import Presentation
from pptx.util import Pt
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR,MSO_AUTO_SIZE
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement
import json
ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'development/reports/report-additions-20261123'
DOC=Path('E:/0_XSAI/00_KernelAgent/0_HGKDA_docs/BAAI-reports/06_work')
source=next(DOC.glob('*演讲者.pptx'))
p=Presentation(source)
# Keep the user-edited WIP flow and its animation intact.
for i in reversed(range(len(p.slides))):
 if i!=16:
  e=p.slides._sldIdLst[i];p.part.drop_rel(e.rId);p.slides._sldIdLst.remove(e)
wip=p.slides[0]
navy='133F6D';ink='18232C';pale='E7EFF7';orange='B45D22'
def txt(s,x,y,w,h,t,size=20,bold=False,fg=ink,fill=None,center=False):
 sh=s.shapes.add_textbox(*map(Pt,(x,y,w,h)))
 if fill:sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(fill)
 tf=sh.text_frame;tf.word_wrap=False;tf.auto_size=MSO_AUTO_SIZE.NONE;tf.margin_left=tf.margin_right=Pt(12 if fill else 0);tf.margin_top=tf.margin_bottom=0;tf.vertical_anchor=MSO_ANCHOR.MIDDLE
 for i,line in enumerate(t.split('\n')):
  pp=tf.paragraphs[0] if i==0 else tf.add_paragraph();pp.space_before=pp.space_after=Pt(0);pp.alignment=PP_ALIGN.CENTER if center else PP_ALIGN.LEFT
  r=pp.add_run();r.text=line;r.font.name='Microsoft YaHei';r.font.size=Pt(size);r.font.bold=bold;r.font.color.rgb=RGBColor.from_string(fg)
  ea=OxmlElement('a:ea');ea.set('typeface','Microsoft YaHei');r._r.get_or_add_rPr().append(ea)
 return sh
def rule(s,y):
 z=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Pt(40),Pt(y),Pt(920),Pt(y));z.line.color.rgb=RGBColor.from_string('D5DEE6');z.line.width=Pt(1)
def blank(title):
 s=p.slides.add_slide(p.slide_layouts[6])
 for sh in list(s.shapes):sh._element.getparent().remove(sh._element)
 s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor.from_string('FFFFFF')
 txt(s,40,27,880,48,title,27,True);rule(s,76);return s
def note(s,t):
 frame=s.notes_slide.notes_text_frame
 if frame:frame.text=t
 else:
  # PowerPoint inserts text into the body placeholder during export.
  notes[str(s.slide_id)]=t
notes={}
title=next(sh for sh in wip.shapes if sh.has_text_frame and sh.text.startswith('Next Step'))
title.text_frame.paragraphs[0].runs[0].text='WIP：诊断后，如何选择修改层次？'
for r in title.text_frame.paragraphs[0].runs[1:]:r.text=''
title.width=Pt(880)
note(wip,'单算子 IR 已有 GQA 的验证案例。这里继续讨论把它变成可迁移的规则：同类问题重复出现时如何提炼通用 Pass；无法由 Triton 表达的硬件原语通过 Ascend C built-in 或 Triton TLE 暴露。后两项为后续研究，不视为已完成。')
agenda=blank('本次汇报')
groups={}
ag=[]
for i,(head,body) in enumerate([
('从常见算子优化 loop 出发','为什么还需要时序证据和编译产物'),
('HG-KDA 与阶段成果','系统 overview · FlagGems 26 算子'),
('GQA：贯穿三个技术点','成对 profiling · 热点定位 · 等待窗口'),
('WIP & Discussion','修改层次选择 · 应用与拓展方向')]):
 y=114+i*92
 before=len(agenda.shapes)
 txt(agenda,40,y,64,61,f'{i+1:02d}',24,True,'FFFFFF',navy,True)
 txt(agenda,126,y,760,33,head,24,True,navy)
 txt(agenda,126,y+35,760,31,body,18)
 ag.append([z.shape_id for z in list(agenda.shapes)[before:]])
note(agenda,'建议插在封面后。沿当前总稿顺序，系统 overview 后先展示阶段成绩，再用 GQA 串起三个技术点，最后留出讨论。')
loop=blank('算子优化 loop：还需要回答哪两类问题？')
txt(loop,40,92,880,40,'常见路径：真机性能反馈 + 知识与代码推理',20,True,navy)
for x,w,label in [(40,168,'Triton 源码'),(260,186,'编译 / 真机 profiling'),(498,185,'知识与代码推理'),(735,185,'修改 Triton')]:
 txt(loop,x,148,w,64,label,17,True,navy,pale,True)
for x in (218,456,693):txt(loop,x,148,33,64,'→',26,True,navy,center=True)
a=[];b=[]
def panel(x,head,lines,example,ending):
 before=len(loop.shapes)
 txt(loop,x,254,424,44,head,21,True,'FFFFFF',navy)
 txt(loop,x+12,313,400,64,lines,18)
 txt(loop,x+12,390,400,37,example,17,True,navy)
 txt(loop,x+12,442,400,52,ending,18,True,orange)
 return [sh.shape_id for sh in list(loop.shapes)[before:]]
a=panel(40,'① 从总量定位到融合区域','QK 和 PV 共用结果搬出路径。\n总搬运量里，哪一段占了主要成本？','GQA：按 PC 区分 QK / PV 搬出','模拟器热点 + 源码定位')
b=panel(496,'② 从源码意图检查实际指令','改了倒数表达，\n设备端是否真的用了倒数指令？','VRecOp → HIVMDecomposeOp → VDivOp','核对实际 IR，再选择修改层次')
note(loop,'建议用这一页作为原算子 loop 页的相邻讲解备选，不自动替换总稿。NVlabs KDA 和 Meta KernelAgent 是常见的真机反馈加知识/代码推理路线；不声称覆盖大多数所有研究。NCU 有 PM Sampling、PC/源码级 stall，不能说 profiler 一概没有时序。GQA 本例的 profiling 总量不能直接拆分 QK/PV 区域成本，模拟器按 PC 与源码关联补足。编译例子来自固定 AscendNPU-IR commit 02cd0c7 的 VRecOpHighPrecisionLowering；其注释和实现说明 VRecOp 被拆成 VBrcOp+VDivOp，本机产物核对与之对应。它支持预期倒数未保留下来的说法，并不证明整份 IR 完全未改变。来源及原文见 research-notes.md。')
disc=blank('Discussion：优先在哪些方向应用与拓展？')
dg=[]
items=[
('01','先服务哪些场景？','复杂融合算子、Attention、推理关键路径','先扩展 GQA 用例，再跨算子检验收益'),
('02','优先沉淀成什么能力？','通用编译 Pass，还是 Ascend C / Triton TLE？','从重复出现的问题中选择投入方向'),
('03','工具和层次由 Agent 自选吗？','何时 profiling 已够，何时需要模拟器或 IR？','比较自由选择与条件规则的收益和成本')]
for i,(num,h,question,detail) in enumerate(items):
 y=108+129*i
 before=len(disc.shapes)
 txt(disc,40,y+4,62,67,num,24,True,'FFFFFF',navy,True)
 txt(disc,127,y,788,38,h,23,True,navy)
 txt(disc,127,y+41,788,32,question,19)
 txt(disc,127,y+77,788,32,detail,18,True,orange)
 dg.append([sh.shape_id for sh in list(disc.shapes)[before:]])
note(disc,'面向研究院老师的讨论页。第一项聚焦近期应用优先级，不承诺未经验证的跨任务结果。第二项围绕 FlagTree/TLE 的实际硬件表达扩展能力讨论共建。第三项讨论自动决策的证据边界，建议固定起点、合同与预算做消融，记录收益、采集成本和失败率。')
entries=list(p.slides._sldIdLst)
for e in entries:p.slides._sldIdLst.remove(e)
for i in (1,2,0,3):p.slides._sldIdLst.append(entries[i])
groups={'1':ag,'2':[a,b],'4':dg}
REPORT.mkdir(parents=True,exist_ok=True)
path=REPORT/'汇报补充页-内容与动机-WIP-Discussion.pptx'
p.save(path)
(REPORT/'animation-spec.json').write_text(json.dumps({'groups':groups,'notes':notes},ensure_ascii=False,indent=2),encoding='utf8')
print(path)

