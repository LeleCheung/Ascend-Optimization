"""Append accepted GQA continuation figures to the frozen group-meeting deck."""
import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

from .meeting_slides import BG, GREEN, INK, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--figure',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--summary',type=Path,help='Campaign-specific accepted mechanism and slide notes')
    args=parser.parse_args()
    campaign=args.campaign.resolve()
    selected=json.loads((campaign/'gqa/selection.json').read_bytes())
    before=selected['initial_retained_us'];after=selected['target_latency_us']
    summary=json.loads(args.summary.read_bytes()) if args.summary else {}
    p=Presentation(args.source)
    initial=len(p.slides)
    graph=p.slides.add_slide(p.slide_layouts[6])
    graph.background.fill.solid();graph.background.fill.fore_color.rgb=RGBColor.from_string(BG)
    graph.shapes.add_picture(str(args.figure),Inches(.35),Inches(.15),width=Inches(12.6))
    text(graph,'固定四项通过；每节点具体改法与来源保留；同实现测量隐藏，失败和退步展开详情',.65,7.02,11.8,.3,13)
    graph.notes_slide.notes_text_frame.text=('本次从784.63765μs开始。前三种机制完整合格但目标退步：'
        '残差累加链785.5762μs、首块剥离825.24775μs、BM32/BN2561042.3639μs。'
        '编译期几何及掩码消除774.8852μs，目标条件ABBA通过。随后增加非因果query行复用。'
        '图使用最新case4的CANN锚点统一历史；历史重算标≈，原始CANN结果冻结。'
        '最新两项实际改Triton源码，IR用于解释，持续重放所有完整候选。')
    if summary:
        graph.notes_slide.notes_text_frame.text=summary['graph_notes']
    result=p.slides.add_slide(p.slide_layouts[6])
    result.background.fill.solid();result.background.fill.fore_color.rgb=RGBColor.from_string(BG)
    text(result,summary.get('title','GQA续接：编译期几何＋query行复用'),.55,.3,12.1,.8,31,bold=True)
    text(result,f'{before:.3f} → {after:.3f} μs · 耗时下降{(1-after/before)*100:.2f}%',
         .7,1.25,11.9,.7,32,GREEN,True)
    text(result,summary.get('mechanism','先将shape／stride等参数特化并去冗余掩码，再增加非因果M方向复用\n'
         '保持BN128；BM64→128，目标程序数1024→512；核内AST保持'),
         .75,2.15,11.8,1.1,22)
    table=result.shapes.add_table(5,3,Inches(.8),Inches(3.45),Inches(11.7),Inches(2.15)).table
    rows=[['固定case','本次起点 μs','最新保留 μs']]+[[str(r['case_id']),f"{r['original_us']:.3f}",f"{r['candidate_us']:.3f}"] for r in selected['cases']]
    for i,row in enumerate(rows):
        for j,value in enumerate(row):
            cell=table.cell(i,j);cell.text=value
            cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string(GREEN if i==0 else 'FFFAF0')
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.name='Microsoft YaHei';run.font.size=Pt(19)
                    run.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else INK)
    text(result,summary.get('validation_text','四项官方checker全部通过；后profiling与模拟器完成，部分窗口分别解释'),.8,5.95,11.7,.55,20)
    text(result,summary.get('scope_text','当前结果覆盖case1／2／4／7，完整20项验收列为后续；原结果和负结果保留'),.8,6.65,11.7,.5,16)
    result.notes_slide.notes_text_frame.text=('完整结果与实际源码父链：../gqa-continuation-20261121/README.md。'
        '第三轮UB实际编译可行但目标变慢，资源估计与真实编译分别记录。'
        '第四轮小收益条件ABBA：785.398／774.206／773.946／785.314μs。'
        '第五轮目标三次计时约472μs，和774μs清晰分离；按条件策略无新增ABBA。'
        '每program计数、程序总数与完整调用分别比较；源码算术保持也可能有硬件舍入差异，采用原checker结论。')
    if summary:
        result.notes_slide.notes_text_frame.text=summary['result_notes']
    for number in range(initial+1,len(p.slides)+1):
        text(p.slides[number-1],str(number),12.3,7.12,.5,.25,11)
    for slide in p.slides:
        for shape in slide.shapes:
            assert shape.left>=0 and shape.top>=0
            assert shape.left+shape.width<=p.slide_width+Inches(.01)
            assert shape.top+shape.height<=p.slide_height+Inches(.01)
    p.save(args.output)
    receipt=dict(file=str(args.output.resolve()),slides=len(p.slides),added=list(range(initial+1,len(p.slides)+1)),
                 bounds_passed=True,native_render_passed=False,native_visual_review_passed=False)
    (campaign/'slides-validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(receipt)


if __name__=='__main__':main()
