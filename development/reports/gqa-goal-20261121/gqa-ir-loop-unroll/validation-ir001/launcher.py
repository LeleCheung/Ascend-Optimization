import fcntl,json,subprocess,sys,shlex,shutil
from pathlib import Path
base=Path('/data/hanle/ascend-optimization/runs/gqa-goal-20261121')
run=base/'gqa-ir-loop-unroll'; val=run/'validation-ir001'; tools=base/'controller-tools'; sys.path.insert(0,str(tools))
from ir_campaign import read,save,sha
m=read(val/'manifest.json')
cmd=['/usr/local/python3.11.15/bin/python3.11',str(tools/'gqa_ir_bench_driver.py'),'--manifest',str(val/'manifest.json'),'--harness',str(base/'gqa/tools/attention_bench.py'),'--contract',m['contract'],'--candidate',m['source'],'--output',str(val/'results.json')]
shell=chr(10).join(['source /usr/local/Ascend/cann-9.0.0/set_env.sh','export ASCEND_RT_VISIBLE_DEVICES=7','export OMP_NUM_THREADS=1','export TRITON_DISABLE_LINE_INFO=false','exec '+shlex.join(cmd)])
with Path('/data/hanle/ascend-optimization/runs/physical-card-7-campaign.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 result=subprocess.run(['docker','exec','tle_yy','bash','-c',shell],timeout=1200)
save(val/'execution.json',dict(command=cmd,returncode=result.returncode))
r=read(val/'results.json')
qualified={x['case_id'] for x in r['cases']}=={1,2,4,7} and all(x['status']=='measured' and all(x[s+'_correctness']['official_checker']['passed'] for s in ['candidate','native']) for x in r['cases'])
target=next(x for x in r['cases'] if x['case_id']==4)
save(val/'reviewed-result.json',dict(status='four_pass_post_pending' if qualified else 'four_case_failure',source_sha256=r['source_sha256'],target_us=target.get('candidate_timing',{}).get('full_call_us'),original_fixed_source_us=262.27775,global_best=read(base/'gqa/selection.json'),post_status='pending' if qualified else 'not_qualified',source_frozen=True))
if not qualified: sys.exit(0)
post=run/'post-ir001'; (post/'workspace').mkdir(parents=True)
shutil.copyfile(m['source'],post/'workspace/candidate.py'); shutil.copyfile(m['contract'],post/'contract.json'); shutil.copytree(base/'gqa/tools',post/'tools')
for name in ['gqa_bcmlir_runtime.py','gqa_ir_bench_driver.py']: shutil.copyfile(tools/name,post/'tools'/name)
postmanifest=dict(m,source=str(post/'workspace/candidate.py'),cache=str(post/'compiled-cache'),receipts=str(post/'compiled-receipts.jsonl'))
save(post/'bcmlir-manifest.json',postmanifest)
c=read(base/'gqa/iterations/attempt-019/config.json'); c.update(run_directory=str(post),trace_round_id='ir-loop-unroll-001-post',bcmlir_manifest=str(post/'bcmlir-manifest.json'),existing_validation=dict(result=str(val/'results.json'),result_sha256=sha(val/'results.json'),cache=str(val/'cache')))
save(post/'config.json',c); save(post/'retained-parent.json',dict(source_sha256=m['source_sha256'],status='source_frozen_manual_ir_unroll_awaiting_post'))
command=[sys.executable,str(tools/'gqa_continue_observe.py'),'--run',str(post)]
with (post/'driver.log').open('x') as out: p=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT)
save(post/'driver-execution.json',dict(returncode=p.returncode,command=command))
if p.returncode:sys.exit(p.returncode)
from gqa_binding_facts import facts
f=facts(post); save(post/'binary-binding-and-facts.json',f)
v=read(val/'reviewed-result.json'); v.update(post_status='complete',binary_binding=str(post/'binary-binding-and-facts.json'),target_binary_sha256=f['target_binary_sha256']); save(val/'reviewed-result.json',v)
