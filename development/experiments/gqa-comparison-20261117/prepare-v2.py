from pathlib import Path
import json,shutil,hashlib
r=Path('/data/hanle/ascend-optimization/runs/gqa-attribution-20261117')
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
comparison=json.loads((r/'paired/comparison.json').read_bytes())
baseline=json.loads((r/'official-check-01.json').read_bytes())
summary={'source_sha256':baseline['source_sha256'],'case':4,'profiling':{},'full_call':[],'limitations':comparison['limitations']}
for side in ['candidate','reference']:
    facts=comparison[side];data={'basic':facts['basic'],'pipelines':{}}
    for table,group in facts['tables'].items():
        if table.startswith('PipeUtilization'):
            data['pipelines'][table]={k:v['mean'] for k,v in group['fields'].items() if k.endswith(('_ratio','_time'))}
    summary['profiling'][side]=data
for c in baseline['cases']:
    summary['full_call'].append({'case':c['case_id'],'dtype':c['dtype'],'shape':c['shape'],
      'reference_us':c['native_timing']['full_call_us'],'candidate_us':c['candidate_timing']['full_call_us'],
      'reference_passed':c['native_correctness']['official_checker']['passed'],'candidate_passed':c['candidate_correctness']['official_checker']['passed']})
summary['accuracy']='Official checker at pinned commit rounds FP64 golden to output dtype and applies small-value handling. Current four cases pass. Original direct-FP64 formula remains diagnostic, not acceptance.'
ir=r/'capture/candidate/actual-ir/_gqa_fwd_kernel.bcmlir'
lines=ir.read_text().splitlines();selected=set()
for i,line in enumerate(lines):
    if any(x in line for x in ['scf.for','linalg.matmul','linalg.transpose','memref.copy']):selected.update(range(max(0,i-1),min(len(lines),i+2)))
snippet='\n'.join(str(i+1)+': '+lines[i] for i in sorted(selected))
for arm in ['no-simulator','with-simulator']:
    old=r/arm;run=r/(arm+'-v2');w=run/'workspace';w.mkdir(parents=True,exist_ok=False)
    for name in ['candidate.py','contract.json','context-brief.json']:
        shutil.copyfile(old/'workspace'/name,w/name)
    assert hashlib.sha256((w/'candidate.py').read_bytes()).hexdigest()==baseline['source_sha256']
    shutil.copytree(old/'workspace/skills',w/'skills')
    shutil.copytree(old/'workspace/actual-ir',w/'actual-ir')
    save(w/'evidence-summary.json',summary)
    (w/'ir-excerpt.txt').write_text(snippet)
    save(w/'knowledge.json',{'retrieval':{'empty':True},'records':[],'scope':'No GQA-specific historical recipe supplied to either arm'})
    c=json.loads((old/'config.json').read_bytes());c.update(run_directory=str(run),paired_directory=str(r/'paired'),formal_cases=4)
    c['job']['checks']['supplementary']['previous_total']=4
    save(run/'config.json',c);save(run/'controller-state.json',{'status':'prepared'})
    shutil.copyfile(old/'model-settings.json',run/'model-settings.json')
prompt='''你负责修改GQA严格Triton候选。先用Read工具读取candidate.py、evidence-summary.json、ir-excerpt.txt。若存在simulator-summary.json，再用Read阅读它；原始IR仅在具体问题无法由摘录判断时按行读取。两组均未提供GQA专项知识。
请在最多6次读取后选定一个具体性能机制，使用Edit更新已有diagnosis.json，调用下方工具桥decision保存修改前记录，再使用Edit修改已有candidate.py。只用Bash运行工具桥decision；不使用Bash读取、搜索或写文件，不调用context或preflight，不访问工作区外路径。设备评测和只读代码审查由宿主完成。
目标：固定case4变快，全部case1/2/4/7继续通过官方checker。保持数学语义和输出dtype。只按通用shape/dtype等元数据选择路径，不按case编号或张量数值特判。不调用PyTorch/CANN计算，宿主仅分配与启动。当前代码四项已经通过官方检查；直接对未舍入FP64计算的旧MARE不是本轮验收条件。精度处理是否必要须由完整评测验证。
性能证据引用workspace/evidence-summary.json，category=paired_profiling；源码/IR引用workspace/candidate.py或workspace/ir-excerpt.txt，category=source_ir；模拟器材料仅在实际提供且使用时引用category=simulator。缺失知识库不要编造。请说明一个假设、可能的另一种解释、修改内容和预期，再小范围实现。
完成Edit后停止并交接；不要重复读遍全部IR或技能文档，不自行运行测试。下一轮会提供固定四项完整结果。
'''
(r/'comparison-prompt-v2.md').write_text(prompt)
m=json.loads((r/'contrast-manifest.json').read_bytes());m['phase']='comparison-v2';m['preparation_failure']='v1 no-simulator exhausted 36 turns reading long reports; zero candidate edits, retained separately with cost';m['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest();save(r/'contrast-manifest-v2.json',m)
for arm in ['no-simulator-v2','with-simulator-v2']:
    p=r/arm/'config.json';c=json.loads(p.read_bytes());c['method_manifest']=str(r/'contrast-manifest-v2.json');save(p,c)
print('v2 prepared')
