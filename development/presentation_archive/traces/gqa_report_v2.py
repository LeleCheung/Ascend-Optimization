"""用户确认的精简汇报：两页trace、原理、R7、R11、R22两页。"""
from pathlib import Path
import json
from html import escape
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.shapes import MSO_SHAPE
from .gqa_case_slides import Canvas, OUT, ROOT, NAVY, PALE, BORDER, INK
from .gqa_meeting_views import COLORS, mark

STEM='GQA-汇报精简版-v4'
DATA=ROOT/'development/reports/gqa-meeting-traces-20261122'


def overview():
    case=json.loads((DATA/'trace-data.json').read_text(encoding='utf-8'))
    geom=json.loads((DATA/'geomean-data.json').read_text(encoding='utf-8'))
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080"><rect width="1920" height="1080" fill="white"/>']
    def tx(x,y,s,size=28,color='#133F6D',bold=False):
        return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{700 if bold else 400}">{escape(s)}</text>'
    for offset,title,data,indices,compact in [(35,'GQA 四用例几何平均',geom,geom['displayed_indices'],False),
                                              (995,'GQA case 4 优化轨迹',case,case['compact_indices'],True)]:
        out.append(f'<g transform="translate({offset} 85)" font-family="Microsoft YaHei">')
        out.append(tx(25,50,title,38,bold=True))
        out.append(tx(25,110,f'{data["nodes"][-1]["score"]:.3f}× CANN',32,'#168358',True))
        left,right,top,bottom=90,830,220,760
        yy=lambda v:bottom-v/1.2*(bottom-top)
        xs=[left+(right-left)*(j/(len(indices)-1) if compact else i/23) for j,i in enumerate(indices)]
        for val in [0,.2,.4,.6,.8,1,1.2]:
            y=yy(val);dash='stroke-dasharray="7 6"' if val==1 else ''
            out.append(f'<path d="M{left} {y}H{right}" stroke="#dce3eb" stroke-width="1.3" {dash}/>')
            out.append(tx(0,y+9,f'{val:.1f}×',25))
        for j in range(1,len(indices)):
            x0,x1=xs[j-1:j+1];y0,y1=[yy(data['nodes'][i]['score']) for i in indices[j-1:j+1]]
            dx=(x1-x0)*.42
            out.append(f'<path d="M{x0} {y0}C{x0+dx} {y0} {x1-dx} {y1} {x1} {y1}" stroke="#168358" stroke-width="3.5" fill="none"/>')
        for j,i in enumerate(indices):
            n=data['nodes'][i];x=xs[j];y=yy(n['score'])
            out.append(f'<circle cx="{x}" cy="{y}" r="8" stroke="#{NAVY}" stroke-width="2" fill="white"/>' if i==0 else mark(x,y,n['location'],COLORS[n['display_category']],9))
            if compact or i in [0,6,12,18,23]:out.append(tx(x-12,820,str(i),25))
            if i==23 or compact and i in [7,11,22]:out.append(tx(x-45,y-25,f'{n["score"]:.3f}×',26,bold=True))
        out.append('</g>')
    out.append('</svg>')
    svg='\n'.join(out)
    (OUT/f'{STEM}-总览.svg').write_text(svg,encoding='utf-8')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser=p.chromium.launch();tab=browser.new_page(viewport={'width':1920,'height':1080},device_scale_factor=2)
        tab.set_content('<style>body{margin:0}</style>'+svg);tab.evaluate('document.fonts.ready')
        tab.locator('svg').screenshot(path=str(OUT/f'{STEM}-总览.png'));browser.close()


def title(c,s):c.text(80,60,1760,80,s,54,True,INK)
def note(c,s):c.slide.notes_slide.notes_text_frame.text=s
def header_box(c,x,y,w,h,header,body):
    c.box(x,y,w,h,'FFFFFF')
    c.shape(MSO_SHAPE.RECTANGLE,x+1,y+1,w-2,57,NAVY,None)
    c.text(x+12,y+5,w-24,47,header,30,True,'FFFFFF',True)
    c.text(x+12,y+70,w-24,h-82,body,38,True,INK,True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    prs=Presentation();prs.slide_width=Inches(13.333333);prs.slide_height=Inches(7.5)
    c=Canvas(prs)
    c.slide.shapes.add_picture(str(DATA/'compact.png'),0,0,width=prs.slide_width,height=prs.slide_height)
    note(c,'从case 4的六个进展点进入R7、R11和R22。')
    c=Canvas(prs);title(c,'GQA：三步计算与硬件执行')
    c.text(80,155,1760,50,'case 4：每 8 个 Q 头共享一组 K / V',34)
    for x,head,body in [(80,'① QK 得分','S = scale × QKᵀ'),
                        (680,'② 指数权重','P = exp(S − row_max(S))'),
                        (1280,'③ 加权汇总与归一化','O = PV ÷ row_sum(P)')]:
        c.box(x,245,560,165);c.text(x+20,258,520,45,head,32,True,center=True)
        c.text(x+20,320,520,55,body,29,center=True,code=True)
    for x,w,heading,lines in [(80,446,'Cube Unit',['QKᵀ','PV']), (680,560,'Memory',['GM / L1 / L0C / UB','MTE1 / MTE2 / MTE3 / FIXP']), (1394,446,'Vector Unit',['× scale    row_max    exp','row_sum    ÷ row_sum(P)'])]:
        c.box(x,555,w,280,'FFFFFF',NAVY)
        c.text(x,578,w,60,heading,38,True,center=True)
        for r,word in enumerate(lines):
            c.text(x+6,663+r*66,w-12,50,word,29,center=True,code=True)
    for x in [540,1254]:
        c.arrow(x,651,125)
        c.text(x,603,125,42,'QKᵀ / PV',29,center=True,code=True)
        sh=c.shape(MSO_SHAPE.LEFT_ARROW,x,751,125,24,NAVY,None)
        sh.adjustments[0]=.14;sh.adjustments[1]=.8
        c.text(x,779,125,42,'P',29,center=True,code=True)
    note(c,'输入输出BF16，D=128，scale=1/sqrt(D)。P为未归一化权重，d=row_sum(P)，O=PV/d。Cube执行QKᵀ与PV，Vector执行scale、row_max、exp、row_sum和除法。Memory区只列本例涉及的存储与搬运；同步留在R22。')
    # 2. R7：对比表 + 前后源码。
    c=Canvas(prs);title(c,'R7  成对 profiling：KV 持久复用')
    columns=[80,650,1050,1450];widths=[570,400,400,390]
    rows=[['数据搬运量（profiling counter）','修改前','CANN','修改后'],['GM → L1','112 MiB','约 64.3 MiB','52 MiB'],['L0C → GM','96 MiB','96 MiB','96 MiB']]
    for r,row in enumerate(rows):
        for j,v in enumerate(row):
            c.shape(MSO_SHAPE.RECTANGLE,columns[j],230+r*75,widths[j],75,NAVY if r==0 else ('E7EFF7' if r==1 else 'FFFFFF'),BORDER)
            c.text(columns[j]+15,235+r*75,widths[j]-30,65,v,34,r==0 or j==3,'FFFFFF' if r==0 else NAVY,True)
    for x,head,body in [(80,'修改前','# 每个 program 处理一个 query 块\nq = load(Q_tile)\nk = load(K_shared)\nv = load(V_shared)\nout = attention(q, k, v)'),
                        (1000,'修改后','# K/V 提到循环外\nk = load(K_shared)\nv = load(V_shared)\nfor tile in query_tiles:\n    q = load(Q_tile)\n    out = attention(q, k, v)')]:
        c.text(x,540,840,50,head,37,True);c.box(x,610,840,370)
        c.text(x+25,630,790,325,body,31,code=True)
    note(c,'表格数值是 profiling 的数据搬运计数器，单位MiB，箭头表示内存层级方向：GM→L1是输入侧搬入，L0C→GM是Cube结果搬出。16个program原来重复读取同一份K/V；改为同一program连续处理query块后，GM→L1 112→52MiB，L0C→GM保持96MiB，完整调用328.4468→268.7718us。')
    # 4. R11：CSV热点、args.code/AST查表、源码和验证。
    c=Canvas(prs);title(c,'R11  模拟器热点：QK 低精度交接')
    c.box(80,220,760,230);c.text(105,245,710,45,'CSV：热点指令',34,True)
    c.text(105,310,710,110,'PC 0x10d11568\nFIX_L0C_TO_DST\n流水：FIXP',30,True,code=True)
    from pptx.util import Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    c.text(910,202,930,50,'源码定位：按 PC 查表 / AST 分析',34,True)
    c.arrow(850,320,55)
    table=c.slide.shapes.add_table(2,2, Inches(920/144), Inches(277/144), Inches(920/144), Inches(150/144)).table
    for i, row in enumerate([['trace 中的 PC','源码位置（args.code）'],['0x10d11568','candidate.py:217']]):
        for j, value in enumerate(row):
            cell=table.cell(i,j);cell.text=value;cell.vertical_anchor=MSO_ANCHOR.MIDDLE
            cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string(NAVY if i==0 else PALE)
            para=cell.text_frame.paragraphs[0];para.alignment=PP_ALIGN.CENTER
            run=para.runs[0];run.font.name='Microsoft YaHei';run.font.size=Pt(15);run.font.bold=(i==0)
            run.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else NAVY)
    c.text(80,474,1760,45,'candidate.py:217',32,True,code=True)
    c.text(80,521,1760,54,'qk = tl.dot(q, tl.trans(k)) * qk_scale',30,code=True)
    c.text(80,594,1760,44,'修改后',32,True)
    c.text(80,645,1760,55,'qk = tl.dot(q, tl.trans(k)).to(tl.float16).to(tl.float32) * qk_scale',30,code=True)
    for x,w,head,body in [(80,370,'Cube','QK 累加为 FP32'),(793,335,'Memory','FP16   64 KiB / 块'),(1450,390,'Vector','扩为 FP32 后乘 scale')]:
        c.box(x,745,w,138,'FFFFFF',NAVY)
        c.text(x,755,w,50,head,34,True,center=True)
        c.text(x,815,w,50,body,29,center=True)
    c.arrow(467,817,307)
    c.text(465,750,310,50,'FIXP 舍入为 FP16',30,center=True)
    c.arrow(1143,817,290)
    c.text(1143,750,290,50,'读取 FP16',30,center=True)
    c.text(80,942,1760,60,'每块搬出 128 → 64 KiB      L0C → GM  96 → 64 MiB      完整调用 205.284 → 196.158 μs',30,True)
    note(c,'CSV给出PC、指令和流水统计；模拟器trace事件的args.code是编译时源码位置/内联栈字段，按PC查表得到candidate.py:217。AST只补充该语句所属的函数/循环区域。源码行定位不由AST从PC推导。修改后dot结果先转FP16再转回FP32，后续scale和softmax仍按FP32路径计算；真机重放中QK搬出指令的转换参数发生变化，L0C→GM总量从96降到64 MiB。FP32累加后按FP16搬出，Vector恢复为FP32再缩放。')
    # 5. R22：回退到上一版的硬件路径、前后代码和前后时间轴。
    c=Canvas(prs);title(c,'R22  等待窗口优化：分母与倒数前移')
    for x,w,h,b in [(80,260,'Vector / UB','P'),(430,180,'GM','P'),(700,360,'Cube / L0C','PV'),(1150,180,'GM','PV'),(1420,420,'Vector / UB','PV')]:
        header_box(c,x,205,w,120,h,b)
    for x,w,lab in [(350,70,'MTE3'),(620,70,'MTE2'),(1070,70,'FIXP'),(1340,70,'MTE2')]:
        c.text(x-10,165,w+20,32,lab,26,center=True);c.arrow(x,275,w)
    c.text(80,340,1780,45,'FLOWCTRL 等待 PV 时，VECTOR 处于空泡；PV 接收后才开始计算分母。'.replace('；','，'),32,True)
    c.text(80,383,1760,40,'WAIT_FLAG_DEVI   44.488–45.895 μs     VECTOR 空泡   40.866–45.898 μs',27,code=True)
    for x,head,body in [(80,'修改前','send_P_to_cube(P)\nPV = wait_and_receive_PV()\n\nd = row_sum(P)\ninv = 1 / d\nO = PV * inv'),
                        (1000,'IR 重排后','send_P_to_cube(P)\n\nd = row_sum(P)\ninv = 1 / d\n\nPV = wait_and_receive_PV()\nO = PV * inv')]:
        c.text(x,445,840,50,head,36,True);c.box(x,510,840,330)
        c.text(x+30,535,780,285,body,32,code=True)
    c.text(80,890,1760,45,'d 和 inv 只依赖 P，可以在等待 PV 时提前计算。',36,True)
    note(c,'P由Vector/UB经MTE3到GM，Cube经MTE2读取并计算PV，PV经FIXP到GM再经MTE2返回Vector/UB。同一核的 VECTOR idle_intervals 换算为40.866–45.898 μs，包含44.488–45.895 μs的PV就绪等待。随后接收和转换PV，46.336 μs开始归约。时间重叠提示调度机会，IR确认d和inv只依赖当前P，重排和后重放验证这部分工作可以提前。')
    c=Canvas(prs);title(c,'R22  重放验证：归约进入等待窗口')
    c.text(80,170,1760,45,'同一 Vector 核的重放窗口',31,True)
    c.text(80,235,1760,40,'修改前后时间轴：以各自 P 搬出起点为 0，单位 μs',29,True)
    root=ROOT/'development/reports/gqa-goal-20261121/gqa-ir-qk-receive-prefetch'
    pre=json.loads((root/'post-ir-003/qk-receive-mechanism.json').read_text(encoding='utf-8'))['example_us']
    post=json.loads((root/'post-ir-004/sum-reciprocal-mechanism.json').read_text(encoding='utf-8'))['example_us']
    pre['d_inv']=[pre['denominator_reduce'][0],46.900];post['d_inv']=[post['sum'][0],post['reciprocal'][1]]
    for x,head,data in [(80,'修改前',pre),(1000,'IR 重排后',post)]:
        c.text(x,315,840,45,head,35,True);px=x+200;scale=96;t0=data['p_send'][0]
        for tick in [0,2,4,6]:
            xx=px+tick*scale;c.shape(MSO_SHAPE.RECTANGLE,xx,410,1,300,BORDER,None);c.text(xx-15,370,40,32,str(tick),27,center=True)
        for i,(lane,key,label) in enumerate([('MTE3','p_send','P'),('FLOWCTRL','pv_ready_wait','PV 等待'),('MTE2','pv_receive','接收 PV'),('VECTOR','d_inv','d 和 inv')]):
            y=430+i*60
            if lane=='VECTOR' and data is pre:
                # Clip measured idle 40.866–45.898 to chart origin (P send start).
                idle_width=(45.898-t0)*scale
                c.shape(MSO_SHAPE.RECTANGLE,px,y,idle_width,28,'E7EFF7','D3DCE6')
                c.text(px,y,idle_width,28,'空泡',21,True,NAVY,True)
            if lane=='FLOWCTRL':
                a,b=data['next_qk_ready_wait'];left=max(0,a-t0)*scale;w=(b-t0)*scale-left
                c.shape(MSO_SHAPE.RECTANGLE,px+left,y,w,28,'54779B',None)
                c.text(px+left,y,w,28,'下一 QK 等待',21,True,'FFFFFF',True)
            c.text(x,y-2,190,35,lane,26,True);aa,bb=data[key];xx=px+max(0,aa-t0)*scale;ww=(bb-t0-max(0,aa-t0))*scale
            c.shape(MSO_SHAPE.RECTANGLE,xx,y,ww,28,'54779B' if lane=='FLOWCTRL' else NAVY,None)
            if ww>=105:c.text(xx,y,ww,28,label,21,True,'FFFFFF',True)
            else:c.text(xx+ww+8,y,150,28,label,21,True,NAVY)
    c.text(80,790,1760,45,'重排后，d 和 inv 与 P 搬出及下一 QK 就绪等待重叠。',29,True,code=True)
    c.text(80,900,1760,60,'144.437 → 133.321 μs        耗时下降 7.70%',42,True)
    note(c,'FLOWCTRL中的WAIT_FLAG_DEVI是等待指令，FIXP是Cube结果搬出流水，两者不同。修改后sum/VREC进入P发送和后续交接等待区间；VREC是inv=1/d的底层实现。正式调用144.43715→133.3206us，降低7.696462%。')
    assert len(prs.slides)==6
    for s in prs.slides:
        for sh in s.shapes:
            assert sh.left>=0 and sh.top>=0 and sh.left+sh.width<=prs.slide_width+30 and sh.top+sh.height<=prs.slide_height+30
    prs.save(OUT/f'{STEM}.pptx')
    (OUT/f'{STEM}-讲稿.md').write_text('\n\n'.join(f'## 第{i+1}页\n\n{s.notes_slide.notes_text_frame.text}' for i,s in enumerate(prs.slides))+'\n',encoding='utf-8')
    print('Saved 6-slide simplified deck and speaker notes')

if __name__=='__main__':main()
