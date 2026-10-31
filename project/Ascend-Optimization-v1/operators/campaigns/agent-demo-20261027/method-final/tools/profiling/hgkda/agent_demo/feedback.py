"""从新候选实测回看模型预测，自动生成演示用数值与真实反馈。"""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
from agent import invoke,save
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[3]

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);a=p.parse_args();b=a.batch.resolve()
    assert not (b/'feedback').exists();out=b/'feedback';out.mkdir()
    sys.path.insert(0,str(HERE.parent/'roofline'))
    from profile_data import parse
    from analyze import derive,load_reference
    empirical=load_reference(PROJECT/'operators/campaigns/roofline-20261027')
    s=importlib.util.spec_from_file_location('demo_pair_compare',HERE.parent/'skills/hgkda/scripts/compare_opprof.py');compare=importlib.util.module_from_spec(s);s.loader.exec_module(compare)
    pairs=[]
    for size in [512,1024]:
        directory=out/('float16-'+str(size));directory.mkdir()
        spec=dict(operator='matmul_bias_activation',shape=[size]*3,dtype='float16',seed=20261027,source_sha256=hashlib.sha256((b/'candidate.py').read_bytes()).hexdigest(),
            comparison='新候选与同批起点同输入，正式成绩独立KGS')
        save(directory/'spec.json',spec);spec_sha=hashlib.sha256((directory/'spec.json').read_bytes()).hexdigest()
        facts={}
        for side,origin in [('reference','seed-profile'),('candidate','candidate-profile')]:
            original=b/origin/('candidate-float16-'+str(size));target=directory/side;target.mkdir()
            (target/'profile').symlink_to(original/'profile',target_is_directory=True)
            discovery=json.loads((original/'discovery.json').read_bytes())
            save(target/'collection.json',dict(status='completed',spec_sha256=spec_sha,side=side,groups='Roofline',expected_kernel=discovery['selected'][0]))
            record=parse(original/'profile');facts[side]=dict(duration_us=record['duration_us'],roofline=derive(record,[size]*3,'float16',empirical),
                kernel=record['basic']['Op Name'],source_spec=json.loads((original/'spec.json').read_bytes()))
        full=compare.compare(directory);save(directory/'comparison.json',full)
        ir={}
        for label,origin in [('seed','seed-profile'),('candidate','candidate-profile')]:
            files=list((b/origin/('candidate-float16-'+str(size))/'compiled').rglob('bcmlir.txt'));assert len(files)==1
            text=files[0].read_text(encoding='utf8')
            ir[label]=dict(path=str(files[0]),sha256=hashlib.sha256(files[0].read_bytes()).hexdigest(),
                static_matmul=text.count('linalg.matmul'),static_copy=text.count('memref.copy'),static_alloc=text.count('memref.alloc'),
                scope='静态文本次数，非动态指令数')
        curves={side:{r['path']:r for r in v['roofline']['curves']} for side,v in facts.items()}
        ratios={key:curves['candidate'][key]['official_bytes']/curves['reference'][key]['official_bytes'] for key in curves['candidate'].keys()&curves['reference'].keys()}
        pairs.append(dict(size=size,facts=facts,traffic_candidate_over_seed=ratios,ir=ir,
            differences=[r for r in full['differences'] if any(x in r['metric'] for x in ['wait','stall','mte','cube','fixpipe'])],derived=full['derived_differences']))
    raw=json.loads((b/'candidate-instruction/summary.json').read_bytes());seed=json.loads((b/'agent-input.json').read_bytes())['instruction']
    result=dict(status='completed',comparison=json.loads((b/'validation/comparison.json').read_bytes()),pairs=pairs,
        seed_instruction=seed,candidate_instruction=raw,proposal=json.loads((b/'agent/proposal.json').read_bytes()),
        attribution='模板与控制器由Codex实现；CC＋DeepSeek实际选工具与动作，使用明确历史先验；本批新采集和评测，非盲发现')
    save(out/'facts.json',result)
    prompt=('你是HG-KDA实验复核角色。只输出JSON：confirmed（被数据支持的预测数组）、rejected（不符或未获支持的预测数组）、'
        'next_knowledge（可复用知识数组）、short_report（给导师的四句中文）。不要生成新实验，不夸大因果。'
        '候选与PyTorch分数不相除推算收益，直接使用actual_candidate_over_seed。'
        '模拟器child退出异常是部分证据，源代码debug映射不证明该源码表达式独占指令耗时。'
        '程序块平均时间不能当作总关键路径；IR文本次数不能当动态计数；部分指令时长不能相加。'
        '如果总流量基本不变，明确否定减少总搬运量这一解释；交错load只改变主循环，不自动消除尾处理等待。'
        '数据：\n'+json.dumps(result,ensure_ascii=False))
    answer=invoke(prompt,out/'agent-review')
    for key in ['confirmed','rejected','next_knowledge','short_report']:
        if not answer.get(key):raise ValueError('反馈缺少字段：'+key)
    save(out/'result.json',answer);print(json.dumps(answer,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
