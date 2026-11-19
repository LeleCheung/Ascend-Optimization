from pathlib import Path
import json,shutil,hashlib,csv,sys
r=Path('/data/hanle/ascend-optimization/runs/gqa-attribution-20261117')
package=r/'method/hgkda';sys.path.insert(0,str(package/'src'))
old=Path('/data/hanle/ascend-optimization/runs/night-research-20261116/gqa')
source=old/'workspace/candidate.py';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
pair=r/'paired';pair.mkdir(exist_ok=False)
contract=json.loads((r/'contract.json').read_bytes());case=next(x for x in contract['cases'] if x['case_id']==4)
save(pair/'spec.json',{'source':str(source),'source_sha256':sha(source),'case':case,'seed':1733,'contract_sha256':sha(r/'contract.json')})
for side in ['candidate','reference']:
    dest=pair/side/'profile';dest.mkdir(parents=True)
    tables=list((r/'capture'/side/'raw').glob('OPPROF_*/*.csv'))
    for p in tables:shutil.copyfile(p,dest/p.name)
    with (dest/'OpBasicInfo.csv').open() as f:basic=list(csv.DictReader(f))
    assert len(basic)==1
    save(dest.parent/'collection.json',{'status':'completed','side':side,'spec_sha256':sha(pair/'spec.json'),'expected_kernel':basic[0]['Op Name'],'groups':'Roofline'})
from hgkda.compare import compare
comparison=compare(pair);comparison['scope']='fixed case 4; one full GQA device task each; full-call performance separately measured by walltime'
save(pair/'comparison.json',comparison)
baseline=json.loads((r/'official-check-01.json').read_bytes())
assert all(c[s+'_correctness']['official_checker']['passed'] for c in baseline['cases'] for s in ['native','candidate'])
manifest={'starting_source_sha256':sha(source),'model_settings_sha256':sha(old/'model-settings.json'),
 'method_files':[{'path':str(p.relative_to(package)),'sha256':sha(p)} for p in sorted(package.rglob('*')) if p.is_file() and '__pycache__' not in str(p)],
 'budget':{'model_calls_per_arm':2,'max_turns_per_call':36,'new_candidate_evaluations_per_arm':2},
 'target_case':4,'validation_cases':[1,2,4,7],
 'difference':'only simulator report supplied to with-simulator arm; source, profiling, IR, prompt, model and budgets shared',
 'acceptance':'all four cases official checker passed; target case faster; all-case timings retained',
 'initial_candidate_sha256':sha(source),'checker_commit':contract['checker_commit']}
save(r/'contrast-manifest.json',manifest)
for arm in ['no-simulator','with-simulator']:
    run=r/arm;w=run/'workspace';w.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source,w/'candidate.py')
    shutil.copytree(package/'skills',w/'skills')
    for name in ['contract.json','official-check-01.json']:shutil.copyfile(r/name,w/name)
    shutil.copyfile(pair/'comparison.json',w/'comparison.json')
    shutil.copytree(r/'capture/candidate/actual-ir',w/'actual-ir')
    shutil.copyfile(old/'model-settings.json',run/'model-settings.json')
    save(w/'task.json',manifest)
    save(w/'context-brief.json',{'source_sha256':sha(source),'evidence':['comparison.json','official-check-01.json','actual-ir/manifest.json'],
         'simulator': 'simulator-summary.json if supplied; do not infer unavailable evidence',
         'accuracy':'Official checker rounds FP64 golden to output dtype and applies small-value rules. Prior unrounded direct-FP64 formula is diagnostic only; all four current cases pass.'})
    config={'run_directory':str(run),'package':str(package),'operator':'gqa','context_kind':'bootstrap','active_source':'candidate.py',
      'phase':'performance_comparison','experiment_goal':'Optimize fixed GQA case 4 with supplied evidence; preserve correctness on all cases',
      'method_manifest':str(r/'contrast-manifest.json'),'job':{'operator':'gqa','checks':{'supplementary':{}}}}
    save(run/'config.json',config);save(run/'controller-state.json',{'status':'prepared'})
prompt='''请优化工作区candidate.py中的GQA严格Triton实现。首先阅读task.json、context-brief.json、comparison.json、official-check-01.json、相关实际IR。若工作区提供simulator-summary.json，再阅读其列出的模拟器热点与区域报告。不要读取其他实验目录或其他算子的历史最佳源码。
目标是降低固定case 4的完整调用耗时，同时保留case 1、2、4、7的正确性和性能结果。所有修改按输入元数据泛化，不能根据测试case编号或输入值特判；不能改变数学语义、输出dtype或容差。设备计算完全Triton，宿主仅元数据、分配、启动；不得调用PyTorch/CANN做计算。当前源码四项已通过固定官方检查器，旧未舍入FP64的诊断公式不是本轮验收条件。
先依据证据选择一个性能机制，写入diagnosis.json并调用decision登记，再通过Edit修改candidate.py。说明证据、替代解释、预期指标和验证方式。优先小范围修改，保持启动接口与正确路径。不要为了原始诊断MARE而添加不必要的计算，也不要未经验证移除精度处理。
本任务的设备检查由宿主调用固定GQA评测器完成，不调用FlagGems/KGS preflight工具。完成源码和决策落盘后返回简要说明，等待完整四项评测反馈。仅允许修改candidate.py和diagnosis.json。
'''
(r/'comparison-prompt.md').write_text(prompt)
print('prepared both arms')
