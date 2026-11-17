"""整理固定两臂试验与单列修复线；缺失或无效性能不补零。"""
import argparse
import hashlib
import html
import json
from pathlib import Path


def read(path):return json.loads(path.read_bytes())


def unwrap(value):return value.get('result',value)


def arm_result(run,case):
    source=run/'workspace/candidate.py'
    value=dict(arm=run.name,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    pre=run/'workspace/latest-preflight.json'
    value['preflight']=read(pre) if pre.exists() else None
    results=list(run.glob('formal-*/evaluation/result.json'))
    if results:
        assert len(results)==1,'多次正式评测必须显式选线'
        result=read(results[0]);value['formal']=result
        value['target']=next(x for x in result['per_workload'] if x['uuid']==case)
        audit=results[0].parent.parent/'audit-final.json'
        if audit.exists():value['audit']=read(audit)
    else:value.update(formal=None,target=None)
    calls=[]
    for p in sorted(run.glob('agent-calls/*/model-usage.json')):
        result=read(p)['result']
        calls.append(dict(invocation=p.parent.name,subtype=result.get('subtype'),
            duration_ms=result.get('duration_ms'),usage=result.get('usage'),cost_usd=result.get('total_cost_usd')))
    value['model_calls']=calls
    value['model_cost_usd']=sum(c['cost_usd'] or 0 for c in calls)
    profiles=[]
    for p in run.glob('paired-*/comparison.json'):
        document=unwrap(read(p));candidate=document['candidate']
        metrics={}
        for table in candidate['tables'].values():
            for name,stats in table['fields'].items():
                if name in ['aiv_vec_ratio','aiv_vec_misc_ratio','aiv_mte2_ratio']:
                    metrics[name]=stats['mean']
                if name in ['GM_to_UB_datas(KB)','aiv_mte2_instructions']:metrics[name]=stats['sum']
        profiles.append(dict(path=str(p),basic=candidate['basic'],metrics=metrics))
    value['profiles']=profiles
    return value


def text(x,y,value,size=34,color='#1b2434'):
    return f'<text x="{x}" y="{y}" text-anchor="middle" font-size="{size}" font-weight="700" fill="{color}">{html.escape(value)}</text>'


def draw(data):
    base,control,sim=data['arms'][:3]
    baseline=base['target']['latency_ms']*1000
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">',
           '<rect width="1600" height="900" fill="#fff7e8"/><g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif">',
           text(800,65,'同一起点，不同证据下的优化选择',48),
           text(800,120,'upsample_bilinear2d · FP16 · [4,32,128,128] → [4,32,256,256]',30),
           '<rect x="440" y="165" width="720" height="135" rx="12" fill="#e3f0df" stroke="#168358" stroke-width="3"/>',
           text(800,217,'共享 Triton 起点'),text(800,273,f'目标 case：{baseline:.1f} μs',42)]
    for row,x,color,title,methods in [
        (control,420,'#2368b1','成对 profiling＋源码／IR',['替换 gather','增加错位连续读取']),
        (sim,1180,'#c55a18','增加模拟器热点与时间轴',['移除 join 交织','改用奇偶列分别写回'])]:
        start=640 if x<800 else 960
        parts += [f'<path d="M {start} 300 L {x} 350" stroke="{color}" stroke-width="4" fill="none"/>',
                  f'<rect x="{x-330}" y="350" width="660" height="190" rx="12" fill="#fffaf0" stroke="{color}" stroke-width="3"/>',
                  text(x,400,title,34,color),text(x,458,methods[0],38),text(x,510,methods[1],38),
                  f'<path d="M {x} 540 L {x} 600" stroke="{color}" stroke-width="4"/>',
                  f'<rect x="{x-330}" y="600" width="660" height="180" rx="12" fill="#fffaf0" stroke="{color}" stroke-width="3"/>']
        target=row.get('target');formal=row.get('formal')
        if target and target['status']=='PASSED' and formal['status']=='PASSED':
            us=target['latency_ms']*1000
            parts += [text(x,656,f'目标 case：{us:.1f} μs',42),
                      text(x,709,f'相对起点：{baseline/us:.3f}×',36),
                      text(x,758,f"KGS {formal['num_passed']}/{formal['num_workloads']} · 双方补测通过",30)]
        else:
            parts += [text(x,656,'预检编译失败',42,color),text(x,709,'UB：需要 512 KiB',36),
                      text(x,758,'可用 192 KiB；无有效性能',30)]
    parts += [text(800,852,'原始对照：同起点／同提示词／同模型／相同预算',32),'</g></svg>']
    return '\n'.join(parts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',default='upsample_bilinear2d-timing-1')
    a=p.parse_args();root=a.run_directory;out=a.output;out.mkdir(parents=True,exist_ok=True)
    names=['baseline-compatibility','no-simulator','with-simulator']
    if (root/'simulator-followup').exists():names.append('simulator-followup')
    data=dict(schema='hgkda.contrast-report/v1',case=a.case,arms=[arm_result(root/n,a.case) for n in names],
        scope='单任务各一条路径的小规模试点；额外修复线单列，不参与原始同预算胜负比较')
    (out/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    svg=draw(data);(out/'comparison.svg').write_text(svg,encoding='utf8')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{background:#fff7e8;color:#1b2434;font:22px/1.6 Microsoft YaHei}main{max-width:1600px;margin:auto}svg{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#2368b1}</style><main>'
        '<p><a href="comparison.svg">SVG</a>　<a href="comparison.png">PNG</a>　<a href="results.json">完整数据</a></p>'+svg+
        '<p>额外编译修复线另存于数据中；原始对照不因追加修复重写。原始模型声明和工具读取记录不证明独立因果。</p>'
        '<details><summary>来源、完整评测、成本与指标</summary><pre>'+html.escape(json.dumps(data,ensure_ascii=False,indent=2))+'</pre></details></main>',encoding='utf8')
    print(json.dumps(dict(arms=len(names),output=str(out))))


if __name__=='__main__':main()
