"""Convert the accepted evidence diagrams to editable native PowerPoint slides."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt

from build_hgkda_overview_20261122 import font
from build_simulator_flow_draft_20261122 import DELIVERY, MAIN

ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'development/reports/evidence-three-slides-20261123'
SCALE=.6  # SVG 1600x900 -> main deck 960x540pt.
PACK='HG-KDA-总图与技术点-三页'
SOURCES=[
    ('HG-KDA-overview-定稿','01_开场与总览',
     ROOT/'development/reports/evidence-overview-draft-20261123/HG-KDA-overview-角色命名草案.svg',
     '''本页沿用已确认总图。当前算子取得成对profiling、模拟器分析、知识库/代码推理三路依据；Analyst Agent分析证据并提出改法，Coder Agent实现修改，编译与验证后输出优化版本。虚线表示继续反馈迭代。

Analyst Agent和Coder Agent表示分析与实现职责，沿用既有分工；本图不额外声称独立多Agent编排。'''),
    ('成对profiling-定稿','06_成对Profiling',
     ROOT/'development/reports/paired-profile-flow-draft-20261123/成对profiling-简图草案.svg',
     '''技术点1：成对profiling。msprof op分别采集当前实现和CANN参考，按两组组织指标。

数据搬运：总字节数、搬运指令数与平均搬运粒度，回答搬了多少数据、分多少次搬、每次搬多少。对应通路的总字节数/指令总数才构成平均粒度，当前既有实现包含GM到UB的bytes_per_mte2。

流水活动：Vector、Scalar、MTE及适用Cube的时间与活跃比例；逐block保留均值、极值和记录数。各流水可重叠，比例不直接等于性能好坏。等待/冲突从图上省略，profiler仍可能提供其汇总统计。

差距标准化[0,1]是拟增加的表达层。现有compare.py已输出原始统计和当前/CANN比值，尚无统一归一化。可采用abs(c-r)/max(c,r)，两侧均0时为0，缺失记未采得；每条指标分别标准化，同时保留两侧原值、Delta与比值，不合并成单一好坏分数。

Analyst Agent解释证据，Coder Agent实现修改，下方独立的编译验证框包括编译、正确性及真机性能核验。结果进入迭代。'''),
    ('模拟器分析-定稿','03_模拟器',
     ROOT/'development/reports/simulator-flow-draft-20261122/模拟器分析-简图草案.svg',
     '''技术点2：热点区域定位。当前算子编译后的NPU二进制由CANN CA model执行；运行入口提供输入和上下文。Inst duration先定位Top-K耗时指令，编译器源码位置/内联调用栈将PC关联到源码行，AST提供循环、条件和函数区域。对应IR操作另结合编译loc和数据流核对。区域分析并不限于Top-K。

技术点3：流水时序分析。读取执行trace中的执行区间、同步和搬运事件，分析已观测窗口内的有效执行、等待、可观察空闲和重叠。时间空闲本身不直接证明完整因果；需要结合源码、IR与事件依赖。trace也提供源码位置，两条路径是分析目的的划分。

Analyst Agent结合证据与知识库提出改法，Coder Agent实现修改。下方淡蓝色编译验证框包括修改后重新编译、模拟器重放、正确性与真机性能核验；所有IR修改仍继续模拟器重放。虚线仅用迭代概括。

脚本对应：collect_simulator.py、simulator.py、hotspots.py、regions.py、timeline.py；GQA具体执行区间另由gqa_simulator_schedule.py提取。'''),
]


def points(d):
    tokens=re.findall(r'[MLHVZmlhvz]|-?\d+(?:\.\d+)?',d)
    result=[];i=0;command=None;x=y=0.
    while i<len(tokens):
        if tokens[i].isalpha():command=tokens[i];i+=1
        if command=='M' or command=='L':
            x,y=float(tokens[i]),float(tokens[i+1]);i+=2
        elif command=='H':x=float(tokens[i]);i+=1
        elif command=='V':y=float(tokens[i]);i+=1
        else:raise ValueError(f'Unsupported path command {command}: {d}')
        result.append((x,y))
        if command=='M':command='L'
    return result


def rectangle(slide,element):
    attrs=element.attrib
    x,y,w,h=(float(attrs.get(k,0)) for k in ('x','y','width','height'))
    if w==1600 and h==900:return
    shape=slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if attrs.get('rx') else MSO_SHAPE.RECTANGLE,
        Pt(x*SCALE),Pt(y*SCALE),Pt(w*SCALE),Pt(h*SCALE))
    shape.name='分区底色' if attrs.get('fill')=='#F3F4F5' else '流程方框'
    if attrs.get('rx'):shape.adjustments[0]=min(.2,float(attrs['rx'])/min(w,h))
    shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(attrs.get('fill','#FFFFFF').lstrip('#'))
    if 'stroke' in attrs:
        shape.line.color.rgb=RGBColor.from_string(attrs['stroke'].lstrip('#'))
        shape.line.width=Pt(float(attrs.get('stroke-width',1))*SCALE)
    else:shape.line.fill.background()
    shape._element.spPr.append(OxmlElement('a:effectLst'))


def line(slide,element,style):
    attrs=style|element.attrib
    pts=points(attrs['d'])
    for i,(a,b) in enumerate(zip(pts,pts[1:])):
        if a==b:continue
        shape=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
            Pt(a[0]*SCALE),Pt(a[1]*SCALE),Pt(b[0]*SCALE),Pt(b[1]*SCALE))
        shape.name='反馈迭代' if 'stroke-dasharray' in attrs else '流程箭头'
        shape.line.color.rgb=RGBColor.from_string(attrs.get('stroke','#606870').lstrip('#'))
        shape.line.width=Pt(float(attrs.get('stroke-width',2.8))*SCALE)
        if 'stroke-dasharray' in attrs:shape.line.dash_style=MSO_LINE_DASH_STYLE.DASH
        if attrs.get('marker-end') and i==len(pts)-2:
            end=OxmlElement('a:tailEnd');end.set('type','triangle');end.set('w','lg');end.set('len','lg')
            shape.line._get_or_add_ln().append(end)
        shape._element.spPr.append(OxmlElement('a:effectLst'))


def native_text(slide,element,bounds):
    value=element.text or ''
    size=float(element.get('font-size',26))
    # Use the measured glyph box, with room for PowerPoint's line metrics.
    x,y,w,h=bounds['x'],bounds['y'],bounds['width'],bounds['height']
    shape=slide.shapes.add_textbox(Pt((x-10)*SCALE),Pt((y-1)*SCALE),Pt((w+20)*SCALE),Pt((max(h,size*1.25)+3)*SCALE))
    shape.name=value
    frame=shape.text_frame;frame.clear();frame.word_wrap=False;frame.auto_size=MSO_AUTO_SIZE.NONE
    frame.margin_left=frame.margin_right=frame.margin_top=frame.margin_bottom=0
    frame.vertical_anchor=MSO_ANCHOR.TOP
    paragraph=frame.paragraphs[0];paragraph.alignment=PP_ALIGN.CENTER
    paragraph.line_spacing=1.;paragraph.space_before=paragraph.space_after=Pt(0)
    run=paragraph.add_run();run.text=value
    font(run,size*SCALE,element.get('fill','#133F6D').lstrip('#'),int(element.get('font-weight',600))>=600)


def add_diagram(deck,source,bounds,notes):
    slide=deck.slides.add_slide(deck.slide_layouts[6])
    slide.background.fill.solid();slide.background.fill.fore_color.rgb=RGBColor.from_string('FFFFFF')
    root=ET.fromstring(source.read_text(encoding='utf-8'))
    count=0
    def visit(element,style):
        nonlocal count
        tag=element.tag.rsplit('}',1)[-1]
        if tag in ('defs','title','desc') or element.get('class')=='impl-labels':return
        inherited=style|{k:v for k,v in element.attrib.items() if k in ('stroke','stroke-width','stroke-dasharray','fill')}
        if tag=='rect':rectangle(slide,element)
        elif tag=='path':line(slide,element,inherited)
        elif tag=='text':native_text(slide,element,bounds[count]);count+=1
        else:
            for child in element:visit(child,inherited)
    visit(root,{})
    assert count==len(bounds)
    note=slide.notes_slide
    if note.notes_text_frame is None:
        note.shapes._spTree.add_placeholder(note.shapes._next_shape_id,'Notes Text',PP_PLACEHOLDER.BODY,'horz','full',13)
    note.notes_text_frame.text=notes
    return slide


def new_deck():
    deck=Presentation(MAIN)
    for item in list(deck.slides._sldIdLst):
        deck.part.drop_rel(item.rId);deck.slides._sldIdLst.remove(item)
    deck.core_properties.author='HG-KDA'
    return deck


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    before=hashlib.sha256(MAIN.read_bytes()).hexdigest()
    pack=new_deck();deliveries=[];counts=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        for stem,folder,source,notes in SOURCES:
            page=browser.new_page(viewport={'width':1600,'height':900})
            page.goto(source.as_uri(),wait_until='load')
            bounds=page.locator('svg').evaluate('''svg=>[...svg.querySelectorAll('text')].filter(x=>!x.closest('.impl-labels')).map(x=>{const b=x.getBBox();return {text:x.textContent,x:b.x,y:b.y,width:b.width,height:b.height};})''')
            page.close()
            single=new_deck();slide=add_diagram(single,source,bounds,notes)
            add_diagram(pack,source,bounds,notes)
            single.core_properties.title=stem
            path=REPORT/(stem+'.pptx');single.save(path)
            target=DELIVERY/'material'/folder/path.name;shutil.copy2(path,target)
            deliveries.append({'file':path.name,'path':str(target)})
            counts.append({'slide':len(pack.slides),'shapes':len(slide.shapes),'pictures':sum(x.shape_type==13 for x in slide.shapes),'body_font_pt':26*SCALE})
        browser.close()
    pack.core_properties.title='HG-KDA 总图与技术点'
    path=REPORT/(PACK+'.pptx');pack.save(path)
    target=DELIVERY/'material/01_开场与总览'/path.name;shutil.copy2(path,target)
    deliveries.append({'file':path.name,'path':str(target)})
    assert hashlib.sha256(MAIN.read_bytes()).hexdigest()==before
    validation={'status':'generated','native_editable':True,'slides':counts,'main_ppt_sha256':before,'main_ppt_preserved':True,'delivery':deliveries,'normalization':'proposed; actual current runtime documented in speaker notes','overview':'confirmed layout preserved','profiling_groups':['数据搬运','流水活动'],'separate_compile_validation_boxes':True}
    (REPORT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(validation,ensure_ascii=False))


if __name__=='__main__':main()
