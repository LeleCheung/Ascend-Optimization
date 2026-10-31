"""重跑历史真机/模拟器证据的离线分析；不启动NPU或模型调用。"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
TOOLS = PROJECT/'tools/profiling'
PAIRED = PROJECT/'operators/campaigns/paired-pointwise-20261022'
RAD = PROJECT/'operators/rad2deg_/reports/hgkda-stage2-20261015-rad2deg/round-02/profiles/instruction/case-579fd2242f0dbe2104cb'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def picture(path, title, subtitle, headers, rows, widths, footer):
    from PIL import Image, ImageDraw, ImageFont
    font_dir = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'
    def font(size, bold=False):
        return ImageFont.truetype(str(font_dir/('msyhbd.ttc' if bold else 'msyh.ttc')), size)
    width = sum(widths)+100
    height = 280+len(rows)*125+140
    im = Image.new('RGB', (width,height), 'white')
    draw = ImageDraw.Draw(im)
    draw.text((50,30),title,font=font(46,True),fill='#19375C')
    draw.text((50,105),subtitle,font=font(25),fill='#526174')
    def wrap(value, f, limit):
        parts=['']
        for ch in str(value):
            if ch=='\n': parts.append(''); continue
            if parts[-1] and draw.textlength(parts[-1]+ch,font=f)>limit: parts.append(ch)
            else: parts[-1]+=ch
        return '\n'.join(parts)
    y=175
    for r,row in enumerate([headers]+rows):
        h=90 if r==0 else 125
        x=50
        for c,value in enumerate(row):
            draw.rectangle((x,y,x+widths[c],y+h),fill='#19375C' if r==0 else '#F1F5F9' if r%2==0 else '#FFFFFF')
            draw.line((x,y+h,x+widths[c],y+h),fill='#DCE5EE',width=2)
            f=font(27,r==0)
            text=wrap(value,f,widths[c]-28)
            draw.multiline_text((x+14,y+18),text,font=f,fill='white' if r==0 else '#19375C',spacing=5)
            x+=widths[c]
        y+=h
    draw.multiline_text((50,y+24),wrap(footer,font(23),width-100),font=font(23),fill='#526174',spacing=6)
    im.save(path)

def compare_demo(output):
    archive = PAIRED/'raw-evidence/paired-final-20261022-v1.tar.gz'
    manifest = json.loads(archive.with_name(archive.name+'.manifest.json').read_bytes())
    archive_digest = sha(archive.read_bytes())
    assert archive_digest == manifest['transfer']['sha256'], '归档SHA不匹配'
    entries = {r['path']:r for r in manifest['contents']['files']}
    prefix = PurePosixPath('preflight/log10_/paired-v2-3')
    directory=output/'成对输入'
    extracted=[]
    with tarfile.open(archive,'r:gz') as stream:
        for member in stream:
            name=PurePosixPath(member.name)
            if not member.isfile() or not name.is_relative_to(prefix): continue
            relative=name.relative_to(prefix)
            if '..' in relative.parts or relative.is_absolute(): raise ValueError('不安全路径')
            needed=member.name.endswith('.csv') or relative.as_posix() in ('spec.json','reference/collection.json','candidate/collection.json')
            if not needed: continue
            raw=stream.extractfile(member).read()
            expected=entries[member.name]
            assert len(raw)==expected['size_bytes'] and sha(raw)==expected['sha256']
            target=directory.joinpath(*relative.parts)
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            extracted.append({'original':member.name,'sha256':sha(raw),'bytes':len(raw)})
    assert len(extracted)==19, '应有两侧各8份CSV、spec及两份collection'
    script=TOOLS/'hgkda/skills/hgkda/scripts/compare_opprof.py'
    full_path=output/'成对完整结果.json'
    command=[sys.executable,str(script),'--directory',str(directory),'--output',str(full_path)]
    run=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    assert run.returncode==0,run.stderr
    facts=json.loads(full_path.read_bytes())
    def metric(name):
        rows=[r for r in facts['differences'] if r['metric']==name and r['table'].startswith('Memory.csv/')]
        assert len(rows)==1,(name,len(rows))
        return rows[0]
    instructions=metric('aiv_mte2_instructions')
    data=metric('GM_to_UB_datas(KB)')
    derived=[r for k,r in facts['derived_differences'].items() if k.endswith('/bytes_per_mte2')]
    derived=[r for r in derived if r['candidate'] is not None and r['reference'] is not None]
    assert len(derived)==1
    granularity=derived[0]
    assert instructions['candidate']==2036 and instructions['reference']==760
    rows=[
        ['GM→UB搬运量（CSV:KB）',f"{data['candidate']:.1f}",f"{data['reference']:.1f}",f"{data['candidate_over_reference']:.3f}×"],
        ['AIV MTE2指令数',str(int(instructions['candidate'])),str(int(instructions['reference'])),f"{instructions['candidate_over_reference']:.3f}×"],
        ['平均搬运粒度（kB/条）',f"{granularity['candidate']/1000:.2f}",f"{granularity['reference']/1000:.2f}",f"{granularity['candidate_over_reference']:.3f}×"],
    ]
    subtitle='log10_｜4096×4096｜FP16｜历史种子与PyTorch同输入｜原CSV离线重算'
    footer='kB/条采用十进制（1kB=1000字节）。读取量相同，但候选搬运更零碎，提示检查分块与搬运组织；不能据此断言HBM饱和。'
    result={'status':'passed','kind':'历史真机采集的离线重算，不是本次重新采集','spec':facts['spec'],'command':command,
            'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'archive_sha256':archive_digest,
            'tool_sha256':sha(script.read_bytes()),'verified_inputs':extracted,'headers':['指标','候选','PyTorch','候选/PyTorch'],'rows':rows,
            'derived':granularity,'interpretation':footer,
            'scope':'诊断证据来自历史种子，不是当前最终0.875×候选，不能混用其身份。'}
    write_json(output/'成对演示核验.json',result)
    text=['[演示1] '+subtitle,'实际执行：'+subprocess.list2cmdline(command),'归档SHA及19个输入文件核对通过，compare_opprof.py退出码：0','']
    text+=[' | '.join(result['headers'])]+[' | '.join(row) for row in rows]+['',footer,result['scope']]
    transcript='\n'.join(text)+'\n'
    (output/'成对演示运行日志.txt').write_text(transcript,encoding='utf-8-sig')
    picture(output/'01-成对指标对比.png','成对指标对比：从原始CSV计算差异',subtitle,result['headers'],rows,[520,280,280,300],footer)
    return transcript

def hotspots_demo(output):
    request=json.loads((RAD/'request.json').read_bytes())
    listing_path=RAD/'artifacts/a008-ascend-instructions.json'
    listing=json.loads(listing_path.read_bytes())
    sources=request['implementation']['sources']
    source=next(r['content'] for r in sources if r['path']=='main.py')
    tool=TOOLS/'hgkda_adapter.py'
    adapter=load_module('hgkda_demo_adapter',tool)
    facts=adapter.instruction_hotspots(listing,source,count=12)
    assert facts['status']=='matched' and len(facts['kernels'])==1
    core=facts['kernels'][0]
    assert core['listed_instruction_count']==92 and core['mapped_instruction_count']==72
    hot=core['hotspots']
    assert len(hot)==12
    assert [r['instruction'] for r in hot[:5]]==['BAR','WAIT_FLAG','BAR','BAR','MOV_SRC_TO_DST_ALIGN']
    rows=[]
    for rank,row in enumerate(hot[:5],1):
        location=row.get('source',{})
        rows.append([str(rank),row['instruction'],f"{row['running_time_us']:.2f}",str(row.get('calls','')),
                     f"{location.get('file','')}:{location.get('line','')}\n{location.get('code','无源码映射')}"])
    result={'status':'passed','kind':'历史模拟器指令报告的离线重算，不是本次重新运行模拟器',
            'tool':'hgkda_adapter.py::instruction_hotspots(count=12)','tool_sha256':sha(tool.read_bytes()),
            'listing_sha256':sha(listing_path.read_bytes()),'source_sha256':sha(source.encode()),
            'request_sha256':sha((RAD/'request.json').read_bytes()),'original_request':str(RAD/'request.json'),
            'listing':str(listing_path),'result':facts,
            'scope':'历史诊断候选；源码来自同次KGS请求，不代表当前最终1.117×交付源码。'}
    write_json(output/'热点完整结果.json',result)
    (output/'热点输入源码.py').write_text(source,encoding='utf-8')
    text=['[演示2] rad2deg_历史模拟器指令热点，调用现有筛选函数重算',
          '执行：hgkda_adapter.py::instruction_hotspots(listing, source, count=12)',
          'kernel：'+core['kernel']+'；core：'+core['core'],
          '报告92条指令，72条有源码映射；本次保留每核前12条，下面显示前5条。','',
          '排名 | 指令 | 累计报告时间(us) | 调用次数 | 源码']
    for row in rows: text.append(' | '.join(row).replace('\n',' '))
    footer='WAIT_FLAG与搬运指令明确映射到tl.store；最大BAR映射到函数定义，另两条BAR映射到计算。不能全归因于写回。'
    text+=['',footer,'时间为该指令地址多次执行的累计模拟器报告时间，不能相加为真机总耗时。',result['scope']]
    transcript='\n'.join(text)+'\n'
    (output/'热点演示运行日志.txt').write_text(transcript,encoding='utf-8-sig')
    (output/'热点关联源码.txt').write_text('\n'.join(f'{i:>3} | {line}' for i,line in enumerate(source.splitlines(),1))+'\n',encoding='utf-8-sig')
    picture(output/'02-指令热点与源码.png','指令热点：匹配候选kernel、排序并关联源码',
            core['kernel']+'｜'+core['core']+'｜历史模拟器证据离线重算｜92条指令、72条有源码映射',
            ['排名','指令','累计时间(μs)','执行次数','源码关联'],rows,[100,440,200,150,800],footer)
    return transcript

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part',choices=['both','compare','hotspots'],default='both')
    parser.add_argument('--output',type=Path,help='新目录；已有目录拒绝覆盖')
    args=parser.parse_args()
    output=(args.output or HERE/('运行-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))).resolve()
    if output.exists(): raise ValueError('请使用新目录，保留旧演示结果：'+str(output))
    output.mkdir(parents=True)
    for fn in ([compare_demo,hotspots_demo] if args.part=='both' else [compare_demo] if args.part=='compare' else [hotspots_demo]):
        print(fn(output),flush=True)
    write_json(output/'运行状态.json',{'status':'passed','part':args.part,'recorded_at':datetime.datetime.now().astimezone().isoformat(),
                                  'script_sha256':sha(Path(__file__).read_bytes()),'mode':'离线重跑，未连接服务器、未占用NPU、未调用模型'})
    print('输出目录：'+str(output))

if __name__=='__main__':
    main()
